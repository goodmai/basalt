import Foundation
import MLX

public struct FusedMoERouterOutput {
    public let indices: MLXArray
    public let weights: MLXArray

    public init(indices: MLXArray, weights: MLXArray) {
        self.indices = indices
        self.weights = weights
    }
}

public enum FusedMoERouterKernel {
    public static let maxExperts = 256
    public static let maxTopK = 16
    public static let dtypes: Set<DType> = [.float16, .bfloat16, .float32]

    public static func supports(_ logits: MLXArray, k: Int) -> Bool {
        guard logits.ndim >= 1 else { return false }
        let experts = logits.dim(-1)
        return experts >= 1 && experts <= maxExperts && k >= 1 && k <= min(maxTopK, experts)
            && dtypes.contains(logits.dtype)
    }

    public static func route(
        _ logits: MLXArray, k: Int, normTopkProb: Bool = true,
        softmaxBeforeTopK: Bool = true, weightDType: DType? = nil
    ) -> FusedMoERouterOutput {
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
        let outType = weightDType ?? (dtypes.contains(logits.dtype) ? logits.dtype : .float32)
        guard rows > 0, supports(logits, k: clampedK), dtypes.contains(outType) else {
            return reference(
                logits, k: clampedK, normTopkProb: normTopkProb,
                softmaxBeforeTopK: softmaxBeforeTopK, weightDType: outType)
        }
        let rowsPerGroup = min(rows, 4)
        let paddedRows = ((rows + rowsPerGroup - 1) / rowsPerGroup) * rowsPerGroup
        let outputs = moeRouterTopkSoftmaxKernel(
            [logits.reshaped(rows, experts)],
            template: [
                ("OUT_T", outType),
                ("K", clampedK),
                ("NORM_TOPK", normTopkProb),
                ("SOFTMAX_FIRST", softmaxBeforeTopK),
            ],
            grid: (32, paddedRows, 1),
            threadGroup: (32, rowsPerGroup, 1),
            outputShapes: [[rows, clampedK], [rows, clampedK]],
            outputDTypes: [.int32, outType])
        return FusedMoERouterOutput(
            indices: outputs[0].reshaped(outShape),
            weights: outputs[1].reshaped(outShape))
    }

    public static func reference(
        _ logits: MLXArray, k: Int, normTopkProb: Bool = true,
        softmaxBeforeTopK: Bool = true, weightDType: DType? = nil
    ) -> FusedMoERouterOutput {
        let experts = logits.ndim > 0 ? logits.dim(-1) : 0
        let clampedK = min(max(0, k), experts)
        let leading = Array(logits.shape.dropLast())
        let outShape = leading + [clampedK]
        let outType = weightDType ?? (dtypes.contains(logits.dtype) ? logits.dtype : .float32)
        guard experts > 0, clampedK > 0, logits.size > 0 else {
            return FusedMoERouterOutput(
                indices: MLXArray.zeros(outShape, type: Int32.self),
                weights: MLXArray.zeros(outShape).asType(outType))
        }
        let z = logits.asType(.float32)
        let safeZ = MLX.where(isNaN(z), MLXArray(-Float.infinity), z)
        let indices = stopGradient(argSort(-safeZ, axis: -1)[.ellipsis, ..<clampedK]).asType(.int32)
        if softmaxBeforeTopK {
            let probs = softmax(safeZ, axis: -1, precise: true)
            var selected = takeAlong(probs, indices, axis: -1)
            if normTopkProb {
                let denom = sum(selected, axis: -1, keepDims: true)
                selected = MLX.where(denom .> 0, selected / denom, MLXArray(Float(0)))
            }
            return FusedMoERouterOutput(indices: indices, weights: selected.asType(outType))
        }
        let selectedLogits = takeAlong(safeZ, indices, axis: -1)
        let selected = softmax(selectedLogits, axis: -1, precise: true)
        return FusedMoERouterOutput(indices: indices, weights: selected.asType(outType))
    }
}

private enum FusedMoERouterMetal {
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

        float vals[MAX_PER_LANE];
        int idxs[MAX_PER_LANE];
        float lane_max = -INFINITY;
        const size_t base = size_t(row) * size_t(V);
        for (int slot = 0; slot < MAX_PER_LANE; slot++) {
            const int col = lane + 32 * slot;
            if (col < V) {
                const float raw = float(logits[base + size_t(col)]);
                const float clean = isnan(raw) ? -INFINITY : raw;
                vals[slot] = clean;
                idxs[slot] = col;
                lane_max = max(lane_max, clean);
            } else {
                vals[slot] = -INFINITY;
                idxs[slot] = MAX_IDX;
            }
        }

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (SOFTMAX_FIRST) {
            row_max = simd_max(lane_max);
            float lane_exp = 0.0f;
            if (row_max > -INFINITY && row_max < INFINITY) {
                for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                    if (idxs[slot] < V) {
                        lane_exp += metal::precise::exp(vals[slot] - row_max);
                    }
                }
            }
            row_sum = simd_sum(lane_exp);
        }

        float top_vals[K];
        int top_idxs[K];
        for (int r = 0; r < K; r++) {
            float lane_best = -INFINITY;
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (idxs[slot] < V && vals[slot] > lane_best) {
                    lane_best = vals[slot];
                }
            }
            const float win_val = simd_max(lane_best);
            int lane_win_idx = MAX_IDX;
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (idxs[slot] < V && vals[slot] == win_val && idxs[slot] < lane_win_idx) {
                    lane_win_idx = idxs[slot];
                }
            }
            const int win_idx = simd_min(lane_win_idx);
            for (int slot = 0; slot < MAX_PER_LANE; slot++) {
                if (idxs[slot] == win_idx) {
                    idxs[slot] = MAX_IDX;
                    vals[slot] = -INFINITY;
                }
            }
            top_vals[r] = win_val;
            top_idxs[r] = win_idx;
        }

        float probs[K];
        if (SOFTMAX_FIRST) {
            float p_sum = 0.0f;
            for (int r = 0; r < K; r++) {
                const float p = (row_sum > 0.0f) ? (metal::precise::exp(top_vals[r] - row_max) / row_sum) : (1.0f / float(K));
                probs[r] = p;
                p_sum += p;
            }
            if (NORM_TOPK) {
                const float inv = (p_sum > 0.0f) ? (1.0f / p_sum) : 0.0f;
                for (int r = 0; r < K; r++) {
                    probs[r] *= inv;
                }
            }
        } else {
            const float top_max = top_vals[0];
            float top_sum = 0.0f;
            for (int r = 0; r < K; r++) {
                const float p = (top_max > -INFINITY && top_max < INFINITY)
                    ? metal::precise::exp(top_vals[r] - top_max) : 1.0f;
                probs[r] = p;
                top_sum += p;
            }
            const float inv = (top_sum > 0.0f) ? (1.0f / top_sum) : 0.0f;
            for (int r = 0; r < K; r++) {
                probs[r] *= inv;
            }
        }

        if (lane < K) {
            const size_t out_idx = size_t(row) * size_t(K) + size_t(lane);
            out_indices[out_idx] = top_idxs[lane];
            out_weights[out_idx] = static_cast<OUT_T>(probs[lane]);
        }
        """
}

private let moeRouterTopkSoftmaxKernel = MLXFast.metalKernel(
    name: "fused_moe_router_topk_softmax",
    inputNames: ["logits"],
    outputNames: ["out_indices", "out_weights"],
    source: FusedMoERouterMetal.source,
    header: "#include <metal_simdgroup>\n#include <metal_math>\n")
