import Foundation
import MLX

package enum FlintRouterKernel {
    package static let maxExperts = 256
    package static let maxTopK = 16
    package static let dtypes: Set<DType> = [.float16, .bfloat16, .float32]

    package static func supports(_ logits: MLXArray, expertBias: MLXArray?, k: Int) -> Bool {
        guard logits.ndim >= 1 else { return false }
        let experts = logits.dim(-1)
        let biasOK = expertBias.map { $0.ndim == 1 && $0.dim(0) == experts && dtypes.contains($0.dtype) } ?? true
        return experts >= 1 && experts <= maxExperts && k >= 1 && k <= min(maxTopK, experts)
            && dtypes.contains(logits.dtype) && biasOK
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
            return FusedMoERouterKernel.route(
                logits, k: k, normTopkProb: normTopkProb, softmaxBeforeTopK: true, weightDType: weightDType)
        }
        guard logits.ndim >= 1, k >= 1 else {
            return FusedMoERouterOutput(
                indices: MLXArray.zeros([0], type: Int32.self),
                weights: MLXArray.zeros([0], type: Float.self))
        }
        let experts = logits.dim(-1)
        let clampedK = min(k, experts)
        let leading = Array(logits.shape.dropLast())
        let rows = experts > 0 ? logits.size / experts : 0
        let outShape = leading + [clampedK]
        guard rows > 0, supports(logits, expertBias: expertBias, k: clampedK), dtypes.contains(weightDType) else {
            return referenceSigmoid(
                logits, expertBias: expertBias, k: clampedK, normTopkProb: normTopkProb, weightDType: weightDType)
        }
        let biasVec = (expertBias ?? MLXArray.zeros([experts], dtype: .float32)).asType(.float32)
        let rowsPerGroup = min(rows, 4)
        let paddedRows = ((rows + rowsPerGroup - 1) / rowsPerGroup) * rowsPerGroup
        let outputs = flintSigmoidRouterKernel(
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
        return FusedMoERouterOutput(
            indices: outputs[0].reshaped(outShape),
            weights: outputs[1].reshaped(outShape)
        )
    }

    package static func referenceSigmoid(
        _ logits: MLXArray,
        expertBias: MLXArray? = nil,
        k: Int,
        normTopkProb: Bool = true,
        weightDType: DType = .float32
    ) -> FusedMoERouterOutput {
        let experts = logits.ndim > 0 ? logits.dim(-1) : 0
        let clampedK = min(max(0, k), experts)
        let leading = Array(logits.shape.dropLast())
        let outShape = leading + [clampedK]
        guard experts > 0, clampedK > 0, logits.size > 0 else {
            return FusedMoERouterOutput(
                indices: MLXArray.zeros(outShape, type: Int32.self),
                weights: MLXArray.zeros(outShape).asType(weightDType))
        }
        let z = logits.asType(.float32)
        let safeZ = MLX.where(isNaN(z), MLXArray(-Float.infinity), z)
        let scores = sigmoid(safeZ)
        let choiceScores: MLXArray
        if let expertBias {
            choiceScores = scores + expertBias.asType(.float32)
        } else {
            choiceScores = scores
        }
        let indices = stopGradient(argSort(-choiceScores, axis: -1)[.ellipsis, ..<clampedK]).asType(.int32)
        var selected = takeAlong(scores, indices, axis: -1)
        if normTopkProb {
            selected = selected / (sum(selected, axis: -1, keepDims: true) + MLXArray(Float(1e-20)))
        }
        return FusedMoERouterOutput(indices: indices, weights: selected.asType(weightDType))
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
        int idxs[MAX_PER_LANE];
        const size_t base = size_t(row) * size_t(V);
        for (int slot = 0; slot < MAX_PER_LANE; slot++) {
            const int col = lane + 32 * slot;
            if (col < V) {
                const float raw = float(logits[base + size_t(col)]);
                const float s = isnan(raw) ? 0.0f : (1.0f / (1.0f + metal::precise::exp(-raw)));
                raw_scores[slot] = s;
                choice_vals[slot] = s + float(expert_bias[col]);
                idxs[slot] = col;
            } else {
                raw_scores[slot] = 0.0f;
                choice_vals[slot] = -INFINITY;
                idxs[slot] = MAX_IDX;
            }
        }

        float top_scores[K];
        int top_idxs[K];
        float score_sum = 0.0f;
        for (int r = 0; r < K; r++) {
            float lane_best = -INFINITY;
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (idxs[slot] < V && choice_vals[slot] > lane_best) {
                    lane_best = choice_vals[slot];
                }
            }
            const float win_choice = simd_max(lane_best);
            int lane_win_idx = MAX_IDX;
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (idxs[slot] < V && choice_vals[slot] == win_choice && idxs[slot] < lane_win_idx) {
                    lane_win_idx = idxs[slot];
                }
            }
            const int win_idx = simd_min(lane_win_idx);
            float lane_raw_score = 0.0f;
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (idxs[slot] == win_idx) {
                    lane_raw_score = raw_scores[slot];
                    idxs[slot] = MAX_IDX;
                    choice_vals[slot] = -INFINITY;
                }
            }
            const float unbiased_score = simd_sum(lane_raw_score);
            top_scores[r] = unbiased_score;
            top_idxs[r] = win_idx;
            score_sum += unbiased_score;
        }

        if (lane < K) {
            float weight = top_scores[lane];
            if (NORM_TOPK) {
                weight = weight / (score_sum + 1e-20f);
            }
            const size_t out_idx = size_t(row) * size_t(K) + size_t(lane);
            out_indices[out_idx] = top_idxs[lane];
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
