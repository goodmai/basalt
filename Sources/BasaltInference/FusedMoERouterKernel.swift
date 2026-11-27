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
    public static let maxExperts = 1024
    public static let maxTopK = 16
    public static let dtypes: Set<DType> = [.float16, .bfloat16, .float32]

    public static func supports(_ logits: MLXArray, k: Int) -> Bool {
        guard Device.defaultDevice() == .gpu, logits.ndim >= 1 else { return false }
        let experts = logits.dim(-1)
        let rows = experts > 0 ? logits.size / experts : 0
        return experts >= 1 && experts <= maxExperts && rows >= 1 && rows <= Int(Int32.max) - 3
            && k >= 1 && k <= min(maxTopK, experts) && dtypes.contains(logits.dtype)
    }

    public static func route(
        _ logits: MLXArray, k: Int, normTopkProb: Bool = true,
        softmaxBeforeTopK: Bool = true, weightDType: DType? = nil
    ) -> FusedMoERouterOutput {
        let experts = logits.ndim > 0 ? logits.dim(-1) : 0
        let clampedK = min(max(0, k), experts)
        let leading = Array(logits.shape.dropLast())
        let rows = experts > 0 ? logits.size / experts : 0
        let outShape = leading + [clampedK]
        let outType = weightDType ?? (dtypes.contains(logits.dtype) ? logits.dtype : .float32)
        guard rows > 0, clampedK > 0, supports(logits, k: clampedK), dtypes.contains(outType) else {
            return reference(
                logits, k: clampedK, normTopkProb: normTopkProb,
                softmaxBeforeTopK: softmaxBeforeTopK, weightDType: outType)
        }
        let rowsPerGroup = min(rows, 4)
        let paddedRows = ((rows + rowsPerGroup - 1) / rowsPerGroup) * rowsPerGroup
        let raw = moeRouterTopkSoftmaxKernel(
            [logits.reshaped(rows, experts)],
            template: [
                ("OUT_T", outType),
                ("K", clampedK),
                ("NORM_TOPK", normTopkProb),
                ("SOFTMAX_FIRST", softmaxBeforeTopK),
                ("PER_LANE", perLane(experts)),
            ],
            grid: (32, paddedRows, 1),
            threadGroup: (32, rowsPerGroup, 1),
            outputShapes: [[rows, clampedK], [rows, clampedK]],
            outputDTypes: [.int32, outType])
        return FusedMoERouterOutput(indices: raw[0].reshaped(outShape), weights: raw[1].reshaped(outShape))
    }

    static func perLane(_ experts: Int) -> Int {
        [8, 16, 32].first { $0 * 32 >= experts } ?? 32
    }

    static func safeSoftmax(_ x: MLXArray) -> MLXArray {
        let posInf = isPosInf(x)
        let hasPosInf = posInf.any(axis: -1, keepDims: true)
        let posInfMask = posInf.asType(.float32)
        let posInfCount = maximum(sum(posInfMask, axis: -1, keepDims: true), MLXArray(Float(1)))
        let posInfProbs = posInfMask / posInfCount
        let hasFinite = isFinite(x).any(axis: -1, keepDims: true)
        let finiteOrZero = MLX.where(hasFinite, x, MLXArray(Float(0)))
        let safeForExp = MLX.where(hasPosInf, MLXArray(Float(0)), finiteOrZero)
        return MLX.where(hasPosInf, posInfProbs, softmax(safeForExp, axis: -1, precise: true))
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
        if Device.defaultDevice() == .gpu, (logits.dtype == .float64 || outType == .float64) {
            return Device.withDefaultDevice(.cpu) {
                reference(
                    logits, k: k, normTopkProb: normTopkProb,
                    softmaxBeforeTopK: softmaxBeforeTopK, weightDType: weightDType)
            }
        }
        guard experts > 0, clampedK > 0, logits.size > 0 else {
            return FusedMoERouterOutput(
                indices: MLXArray.zeros(outShape, type: Int32.self),
                weights: MLXArray.zeros(outShape, dtype: outType))
        }
        let z = logits.asType(.float32)
        let safeZ = MLX.where(isNaN(z), MLXArray(-Float.infinity), z)
        let indices = stopGradient(argSort(-safeZ, axis: -1)[.ellipsis, ..<clampedK]).asType(.int32)
        if softmaxBeforeTopK {
            let probs = safeSoftmax(safeZ)
            var selected = takeAlong(probs, indices, axis: -1)
            if normTopkProb {
                let denom = sum(selected, axis: -1, keepDims: true)
                let safeDenom = maximum(denom, MLXArray(Float(1e-30)))
                selected = MLX.where(denom .> 0, selected / safeDenom, MLXArray(Float(0)))
            }
            return FusedMoERouterOutput(indices: indices, weights: selected.asType(outType))
        }
        let selectedLogits = takeAlong(safeZ, indices, axis: -1)
        let selected = safeSoftmax(selectedLogits)
        return FusedMoERouterOutput(indices: indices, weights: selected.asType(outType))
    }
}

private enum FusedMoERouterMetal {
    static let source = """
        constexpr int MAX_IDX = 2147483647;
        const int lane = int(thread_index_in_simdgroup);
        const int row = int(thread_position_in_grid.y);
        const int M_rows = int(logits_shape[0]);
        const int V = int(logits_shape[1]);
        if (row >= M_rows) {
            return;
        }

        float vals[PER_LANE];
        uint active_mask = 0u;
        float lane_max = -INFINITY;
        const size_t base = size_t(row) * size_t(V);
        #pragma unroll
        for (int slot = 0; slot < PER_LANE; slot++) {
            const int col = lane + 32 * slot;
            if (col < V) {
                const float raw = float(logits[base + size_t(col)]);
                const float clean = isnan(raw) ? -INFINITY : raw;
                vals[slot] = clean;
                active_mask |= (1u << uint(slot));
                lane_max = max(lane_max, clean);
            } else {
                vals[slot] = -INFINITY;
            }
        }

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (SOFTMAX_FIRST) {
            row_max = simd_max(lane_max);
            float lane_exp = 0.0f;
            if (row_max == INFINITY) {
                #pragma unroll
                for (int slot = 0; slot < PER_LANE; slot++) {
                    if (((active_mask >> uint(slot)) & 1u) != 0u && vals[slot] == INFINITY) {
                        lane_exp += 1.0f;
                    }
                }
            } else if (row_max > -INFINITY) {
                #pragma unroll
                for (int slot = 0; slot < PER_LANE; slot++) {
                    if (((active_mask >> uint(slot)) & 1u) != 0u) {
                        lane_exp += metal::precise::exp(vals[slot] - row_max);
                    }
                }
            }
            row_sum = simd_sum(lane_exp);
        }

        float top_max = -INFINITY;
        float my_top_val = -INFINITY;
        int my_top_idx = MAX_IDX;
        for (int r = 0; r < K; r++) {
            float lane_best = -INFINITY;
            #pragma unroll
            for (int slot = 0; slot < PER_LANE; slot++) {
                if (((active_mask >> uint(slot)) & 1u) != 0u && vals[slot] > lane_best) {
                    lane_best = vals[slot];
                }
            }
            const float win_val = simd_max(lane_best);
            int lane_win_idx = MAX_IDX;
            #pragma unroll
            for (int slot = 0; slot < PER_LANE; slot++) {
                if (((active_mask >> uint(slot)) & 1u) != 0u && vals[slot] == win_val) {
                    const int col = lane + 32 * slot;
                    if (col < lane_win_idx) {
                        lane_win_idx = col;
                    }
                }
            }
            const int win_idx = simd_min(lane_win_idx);
            if (win_idx < V && lane == (win_idx & 31)) {
                active_mask &= ~(1u << uint(win_idx >> 5));
            }
            if (r == 0) {
                top_max = win_val;
            }
            if (lane == r) {
                my_top_val = win_val;
                my_top_idx = win_idx;
            }
        }

        float my_prob = 0.0f;
        if (SOFTMAX_FIRST) {
            if (lane < K) {
                if (row_max == INFINITY) {
                    my_prob = (my_top_val == INFINITY) ? (1.0f / row_sum) : 0.0f;
                } else if (row_max > -INFINITY && row_sum > 0.0f) {
                    my_prob = metal::precise::exp(my_top_val - row_max) / row_sum;
                } else {
                    my_prob = 1.0f / float(V);
                }
            }
            if (NORM_TOPK) {
                const float p_sum = simd_sum(my_prob);
                if (lane < K) {
                    my_prob = (p_sum > 0.0f) ? (my_prob / p_sum) : 0.0f;
                }
            }
        } else {
            if (lane < K) {
                if (top_max == INFINITY) {
                    my_prob = (my_top_val == INFINITY) ? 1.0f : 0.0f;
                } else if (top_max > -INFINITY) {
                    my_prob = metal::precise::exp(my_top_val - top_max);
                } else {
                    my_prob = 1.0f;
                }
            }
            const float top_sum = simd_sum(my_prob);
            if (lane < K) {
                my_prob = (top_sum > 0.0f) ? (my_prob / top_sum) : 0.0f;
            }
        }

        if (lane < K) {
            const size_t out_idx = size_t(row) * size_t(K) + size_t(lane);
            out_indices[out_idx] = my_top_idx;
            out_weights[out_idx] = static_cast<OUT_T>(my_prob);
        }
        """
}

private let moeRouterTopkSoftmaxKernel = MLXFast.metalKernel(
    name: "fused_moe_router_topk_softmax",
    inputNames: ["logits"],
    outputNames: ["out_indices", "out_weights"],
    source: FusedMoERouterMetal.source,
    header: "#include <metal_simdgroup>\n#include <metal_math>\n")
