import Foundation
import MLX

package enum FlintRouterKernel {
    package static let maxExperts = 256
    package static let maxTopK = 16
    package static let dtypes: Set<DType> = [.float16, .bfloat16, .float32]

    package static func supports(_ logits: MLXArray, expertBias: MLXArray?, k: Int) -> Bool {
        guard Device.defaultDevice() == .gpu, logits.ndim >= 1 else { return false }
        let experts = logits.dim(-1)
        let rows = experts > 0 ? logits.size / experts : 0
        let biasOK = expertBias.map { $0.ndim == 1 && $0.dim(0) == experts && dtypes.contains($0.dtype) } ?? true
        return experts >= 1 && experts <= maxExperts && rows >= 1 && rows <= Int(Int32.max) - 3
            && k >= 1 && k <= min(maxTopK, experts) && dtypes.contains(logits.dtype) && biasOK
    }

    package static func route(
        _ logits: MLXArray,
        expertBias: MLXArray? = nil,
        k: Int,
        normTopkProb: Bool = true,
        scoreFunction: String = "sigmoid",
        weightDType: DType = .float32
    ) -> FusedMoERouterOutput {
        if scoreFunction != "sigmoid" {
            if expertBias != nil {
                return referenceSoftmax(
                    logits, expertBias: expertBias, k: k, normTopkProb: normTopkProb, weightDType: weightDType)
            }
            return FusedMoERouterKernel.route(
                logits, k: k, normTopkProb: normTopkProb, softmaxBeforeTopK: true, weightDType: weightDType)
        }
        let experts = logits.ndim > 0 ? logits.dim(-1) : 0
        let clampedK = min(max(0, k), experts)
        let leading = Array(logits.shape.dropLast())
        let rows = experts > 0 ? logits.size / experts : 0
        let outShape = leading + [clampedK]
        guard rows > 0, clampedK > 0, supports(logits, expertBias: expertBias, k: clampedK), dtypes.contains(weightDType) else {
            return referenceSigmoid(
                logits, expertBias: expertBias, k: clampedK, normTopkProb: normTopkProb, weightDType: weightDType)
        }
        let biasVec = (expertBias ?? MLXArray.zeros([experts], dtype: .float32)).asType(.float32)
        let rowsPerGroup = min(rows, 4)
        let paddedRows = ((rows + rowsPerGroup - 1) / rowsPerGroup) * rowsPerGroup
        let raw = flintSigmoidRouterKernel(
            [logits.reshaped(rows, experts), biasVec],
            template: [
                ("OUT_T", weightDType),
                ("K", clampedK),
                ("NORM_TOPK", normTopkProb),
            ],
            grid: (32, paddedRows, 1),
            threadGroup: (32, rowsPerGroup, 1),
            outputShapes: [[rows, clampedK], [rows, clampedK]],
            outputDTypes: [.int32, weightDType]
        )
        return FusedMoERouterOutput(indices: raw[0].reshaped(outShape), weights: raw[1].reshaped(outShape))
    }

    package static func reference(
        _ logits: MLXArray,
        expertBias: MLXArray? = nil,
        k: Int,
        normTopkProb: Bool = true,
        scoreFunction: String = "sigmoid",
        weightDType: DType = .float32
    ) -> FusedMoERouterOutput {
        if scoreFunction != "sigmoid" {
            return referenceSoftmax(
                logits, expertBias: expertBias, k: k, normTopkProb: normTopkProb, weightDType: weightDType)
        }
        return referenceSigmoid(
            logits, expertBias: expertBias, k: k, normTopkProb: normTopkProb, weightDType: weightDType)
    }

    package static func referenceSigmoid(
        _ logits: MLXArray,
        expertBias: MLXArray? = nil,
        k: Int,
        normTopkProb: Bool = true,
        weightDType: DType = .float32
    ) -> FusedMoERouterOutput {
        if Device.defaultDevice() == .gpu,
           (logits.dtype == .float64 || expertBias?.dtype == .float64 || weightDType == .float64) {
            return Device.withDefaultDevice(.cpu) {
                referenceSigmoid(
                    logits, expertBias: expertBias, k: k, normTopkProb: normTopkProb, weightDType: weightDType)
            }
        }
        let experts = logits.ndim > 0 ? logits.dim(-1) : 0
        let clampedK = min(max(0, k), experts)
        let leading = Array(logits.shape.dropLast())
        let outShape = leading + [clampedK]
        guard experts > 0, clampedK > 0, logits.size > 0 else {
            return FusedMoERouterOutput(
                indices: MLXArray.zeros(outShape, type: Int32.self),
                weights: MLXArray.zeros(outShape, dtype: weightDType))
        }
        let z = logits.asType(.float32)
        let safeZ = MLX.where(isNaN(z), MLXArray(-Float.infinity), z)
        let scores = sigmoid(safeZ)
        let choiceScores = expertBias.map { scores + sanitizedBias($0) } ?? scores
        let indices = stopGradient(argSort(-choiceScores, axis: -1)[.ellipsis, ..<clampedK]).asType(.int32)
        var selected = takeAlong(scores, indices, axis: -1)
        if normTopkProb {
            selected = selected / (sum(selected, axis: -1, keepDims: true) + MLXArray(Float(1e-20)))
        }
        return FusedMoERouterOutput(indices: indices, weights: selected.asType(weightDType))
    }

    private static func referenceSoftmax(
        _ logits: MLXArray,
        expertBias: MLXArray?,
        k: Int,
        normTopkProb: Bool,
        weightDType: DType
    ) -> FusedMoERouterOutput {
        guard let expertBias else {
            return FusedMoERouterKernel.reference(
                logits, k: k, normTopkProb: normTopkProb, softmaxBeforeTopK: true, weightDType: weightDType)
        }
        if Device.defaultDevice() == .gpu,
           (logits.dtype == .float64 || expertBias.dtype == .float64 || weightDType == .float64) {
            return Device.withDefaultDevice(.cpu) {
                referenceSoftmax(
                    logits, expertBias: expertBias, k: k, normTopkProb: normTopkProb, weightDType: weightDType)
            }
        }
        let experts = logits.ndim > 0 ? logits.dim(-1) : 0
        let clampedK = min(max(0, k), experts)
        let leading = Array(logits.shape.dropLast())
        let outShape = leading + [clampedK]
        guard experts > 0, clampedK > 0, logits.size > 0 else {
            return FusedMoERouterOutput(
                indices: MLXArray.zeros(outShape, type: Int32.self),
                weights: MLXArray.zeros(outShape, dtype: weightDType))
        }
        let z = logits.asType(.float32)
        let safeZ = MLX.where(isNaN(z), MLXArray(-Float.infinity), z)
        let probs = FusedMoERouterKernel.safeSoftmax(safeZ)
        let biased = probs + sanitizedBias(expertBias)
        let indices = stopGradient(argSort(-biased, axis: -1)[.ellipsis, ..<clampedK]).asType(.int32)
        var selected = takeAlong(probs, indices, axis: -1)
        if normTopkProb {
            selected = selected / (sum(selected, axis: -1, keepDims: true) + MLXArray(Float(1e-20)))
        }
        return FusedMoERouterOutput(indices: indices, weights: selected.asType(weightDType))
    }

    private static func sanitizedBias(_ expertBias: MLXArray) -> MLXArray {
        let bias = expertBias.asType(.float32)
        return MLX.where(isFinite(bias), bias, MLXArray(Float(0)))
    }
}

private enum FlintSigmoidRouterMetal {
    static let source = """
        constexpr int MAX_PER_LANE = 8;
        constexpr int MAX_IDX = 2147483647;
        const int lane = int(thread_index_in_simdgroup);
        const int row = int(thread_position_in_grid.y);
        const int M_rows = int(logits_shape[0]);
        const int V = int(logits_shape[1]);
        if (row >= M_rows) {
            return;
        }

        float raw_scores[MAX_PER_LANE];
        float choice_vals[MAX_PER_LANE];
        uint active_mask = 0u;
        const size_t base = size_t(row) * size_t(V);
        #pragma unroll
        for (int slot = 0; slot < MAX_PER_LANE; slot++) {
            const int col = lane + 32 * slot;
            if (col < V) {
                const float raw = float(logits[base + size_t(col)]);
                const float s = isnan(raw) ? 0.0f : (1.0f / (1.0f + metal::precise::exp(-raw)));
                const float b = float(expert_bias[col]);
                raw_scores[slot] = s;
                choice_vals[slot] = s + ((isnan(b) || isinf(b)) ? 0.0f : b);
                active_mask |= (1u << uint(slot));
            } else {
                raw_scores[slot] = 0.0f;
                choice_vals[slot] = -INFINITY;
            }
        }

        float my_top_score = 0.0f;
        int my_top_idx = MAX_IDX;
        float score_sum = 0.0f;
        for (int r = 0; r < K; r++) {
            float lane_best = -INFINITY;
            #pragma unroll
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (((active_mask >> uint(slot)) & 1u) != 0u && choice_vals[slot] > lane_best) {
                    lane_best = choice_vals[slot];
                }
            }
            const float win_choice = simd_max(lane_best);
            int lane_win_idx = MAX_IDX;
            #pragma unroll
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (((active_mask >> uint(slot)) & 1u) != 0u && choice_vals[slot] == win_choice) {
                    const int col = lane + 32 * slot;
                    if (col < lane_win_idx) {
                        lane_win_idx = col;
                    }
                }
            }
            const int win_idx = simd_min(lane_win_idx);
            float lane_raw_score = 0.0f;
            #pragma unroll
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (((active_mask >> uint(slot)) & 1u) != 0u && (lane + 32 * slot) == win_idx) {
                    lane_raw_score = raw_scores[slot];
                    active_mask &= ~(1u << uint(slot));
                }
            }
            const float unbiased_score = simd_sum(lane_raw_score);
            if (lane == r) {
                my_top_score = unbiased_score;
                my_top_idx = win_idx;
            }
            score_sum += unbiased_score;
        }

        if (lane < K) {
            float weight = my_top_score;
            if (NORM_TOPK) {
                weight = weight / (score_sum + 1e-20f);
            }
            const size_t out_idx = size_t(row) * size_t(K) + size_t(lane);
            out_indices[out_idx] = my_top_idx;
            out_weights[out_idx] = static_cast<OUT_T>(weight);
        }
        """
}

private let flintSigmoidRouterKernel = MLXFast.metalKernel(
    name: "flint_moe_router_topk_sigmoid",
    inputNames: ["logits", "expert_bias"],
    outputNames: ["out_indices", "out_weights"],
    source: FlintSigmoidRouterMetal.source,
    header: "#include <metal_simdgroup>\n#include <metal_math>\n"
)
