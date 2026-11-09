import MLX

enum QuestKernel {
    static let maxMiddleBlocks = 4096
    static let maxGroupWidth = 1024
    static let rowSimdgroups = 4
    static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]

    static func fits(queries: MLXArray, keys: MLXArray, values: MLXArray, middleBlocks: Int) -> Bool {
        let groups = queries.dim(1) / keys.dim(1)
        return keys.dim(3) % 32 == 0 && values.dim(3) % 32 == 0 && middleBlocks <= maxMiddleBlocks
            && groups * max(keys.dim(3), values.dim(3)) <= maxGroupWidth
            && dtypes.contains(keys.dtype) && queries.dtype == keys.dtype && values.dtype == keys.dtype
    }

    static func selectedBlocks(
        _ queries: MLXArray, upper: MLXArray, lower: MLXArray, first: Int, last: Int, topK: Int
    ) -> MLXArray {
        let (batch, kvHeads) = (upper.dim(0), upper.dim(1))
        let bounds = MLXArray([Int32(first), Int32(last)])
        let scores = blockScores(queries, upper: upper, lower: lower, bounds: bounds, middle: last - first)
        return selectKernel(
            [scores, bounds],
            template: [("K", topK), ("NMAX", maxMiddleBlocks)],
            grid: (256 * ((last - first + 255) / 256), batch * kvHeads, 1), threadGroup: (256, 1, 1),
            outputShapes: [[batch, kvHeads, topK]], outputDTypes: [.int32])[0]
    }

    private static func blockScores(_ queries: MLXArray, upper: MLXArray, lower: MLXArray, bounds: MLXArray, middle: Int) -> MLXArray {
        let (batch, kvHeads, dim) = (upper.dim(0), upper.dim(1), upper.dim(3))
        return scoreKernel(
            [queries, upper, lower, bounds],
            template: [("T", upper.dtype), ("D", dim), ("G", queries.dim(1) / kvHeads)],
            grid: (256 * ((middle + 7) / 8), batch * kvHeads, 1), threadGroup: (256, 1, 1),
            outputShapes: [[batch, kvHeads, middle]], outputDTypes: [.float32])[0]
    }

    static func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, selected: MLXArray, offset: Int, scale: Float
    ) -> MLXArray {
        let last = (offset - QuestKVCache.windowTokens) / QuestKVCache.blockTokens
        let (batch, kvHeads, dim, valueDim) = (keys.dim(0), keys.dim(1), keys.dim(3), values.dim(3))
        let groups = queries.dim(1) / kvHeads
        let sinkBlocks = QuestKVCache.sinkTokens / QuestKVCache.blockTokens
        let blocks = sinkBlocks + selected.dim(2) + QuestKVCache.windowTokens / QuestKVCache.blockTokens + 1
        let threads = 32 * rowSimdgroups
        let partial = sparseKernel(
            [queries, keys, values, selected, MLXArray([Int32(last), Int32(offset)]), MLXArray([scale])],
            template: [
                ("T", keys.dtype), ("D", dim), ("V", valueDim), ("G", groups), ("K", selected.dim(2)), ("NB", blocks),
                ("BN", rowSimdgroups), ("BT", QuestKVCache.blockTokens), ("SB", sinkBlocks),
            ],
            grid: (threads * blocks, batch * kvHeads, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[batch * kvHeads, blocks, groups, valueDim], [batch * kvHeads, blocks, groups, 2]],
            outputDTypes: [.float32, .float32])
        return combineKernel(
            partial,
            template: [("T", keys.dtype), ("V", valueDim), ("G", groups), ("S", blocks)],
            grid: (batch * kvHeads * groups * valueDim, 1, 1), threadGroup: (min(256, groups * valueDim), 1, 1),
            outputShapes: [[batch, kvHeads * groups, 1, valueDim]], outputDTypes: [keys.dtype])[0]
    }

    static let scoreSource = """
        const int bh = int(threadgroup_position_in_grid.y);
        const int lane = int(thread_index_in_simdgroup);
        const int j = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
        const int first = bounds[0];
        const int n = bounds[1] - first;
        const int nb = int(upper_shape[2]);
        if (j >= n) {
            return;
        }
        const device T* up = upper + (size_t(bh) * nb + first + j) * D;
        const device T* lo = lower + (size_t(bh) * nb + first + j) * D;
        float s = 0.0f;
        for (int d = lane; d < D; d += 32) {
            float p = 0.0f;
            float m = 0.0f;
            for (int g = 0; g < G; g++) {
                const float x = float(q[(bh * G + g) * D + d]);
                p += max(x, 0.0f);
                m += min(x, 0.0f);
            }
            s += p * float(up[d]) + m * float(lo[d]);
        }
        s = simd_sum(s);
        if (lane == 0) {
            scores[bh * n + j] = isnan(s) ? INFINITY : s;
        }
        """

    static let selectSource = """
        const int bh = int(threadgroup_position_in_grid.y);
        const int tid = int(thread_position_in_threadgroup.x);
        const int j = int(threadgroup_position_in_grid.x) * 256 + tid;
        const int first = bounds[0];
        const int n = bounds[1] - first;
        const int quads = (n + 3) / 4;
        threadgroup float4 score[NMAX / 4];
        threadgroup float* flat = (threadgroup float*)score;
        for (int i = tid; i < quads * 4; i += 256) {
            flat[i] = i < n ? scores[bh * n + i] : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j >= n) {
            return;
        }
        const float v = flat[j];
        int4 rank = int4(0);
        for (int i = 0; i < quads; i++) {
            const float4 u = score[i];
            const int4 at = int4(4 * i) + int4(0, 1, 2, 3);
            rank += select(int4(0), int4(1), u > v || (u == v && at < j));
        }
        const int total = rank.x + rank.y + rank.z + rank.w;
        if (total < K) {
            chosen[bh * K + total] = first + j;
        }
        """

    static let sparseSource = """
        constexpr int E = D / 32;
        constexpr int EV = V / 32;
        const int lb = int(threadgroup_position_in_grid.x);
        const int bh = int(threadgroup_position_in_grid.y);
        const int tid = int(thread_position_in_threadgroup.x);
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int last = bounds[0];
        const int offset = bounds[1];
        const int blocks = SB + K + (offset - last * BT + BT - 1) / BT;
        const int capacity = int(keys_shape[2]);
        float o[G][EV];
        float m[G];
        float l[G];
        for (int g = 0; g < G; g++) {
            for (int i = 0; i < EV; i++) {
                o[g][i] = 0.0f;
            }
            m[g] = -FLT_MAX;
            l[g] = 0.0f;
        }
        if (lb < blocks) {
            const int block = lb < SB ? lb : (lb < SB + K ? selected[bh * K + lb - SB] : last + lb - SB - K);
            const int end = min(block * BT + BT, offset);
            const device T* kh = keys + size_t(bh) * capacity * D + lane * E;
            const device T* vh = values + size_t(bh) * capacity * V + lane * EV;
            float qr[G][E];
            for (int g = 0; g < G; g++) {
                for (int i = 0; i < E; i++) {
                    qr[g][i] = scale[0] * float(q[(bh * G + g) * D + lane * E + i]);
                }
            }
            for (int row = block * BT + sg; row < end; row += BN) {
                float k[E];
                for (int i = 0; i < E; i++) {
                    k[i] = float(kh[size_t(row) * D + i]);
                }
                float v[EV];
                for (int i = 0; i < EV; i++) {
                    v[i] = float(vh[size_t(row) * V + i]);
                }
                for (int g = 0; g < G; g++) {
                    float s = 0.0f;
                    for (int i = 0; i < E; i++) {
                        s += qr[g][i] * k[i];
                    }
                    s = simd_sum(s);
                    const float mm = max(m[g], s);
                    const float factor = fast::exp(m[g] - mm);
                    const float e = fast::exp(s - mm);
                    m[g] = mm;
                    l[g] = l[g] * factor + e;
                    for (int i = 0; i < EV; i++) {
                        o[g][i] = o[g][i] * factor + e * v[i];
                    }
                }
            }
        }
        threadgroup float maxima[BN * G];
        threadgroup float sums[BN * G];
        threadgroup float outputs[BN * G * V];
        for (int g = 0; g < G; g++) {
            if (lane == 0) {
                maxima[sg * G + g] = m[g];
                sums[sg * G + g] = l[g];
            }
            for (int i = 0; i < EV; i++) {
                outputs[(sg * G + g) * V + lane * EV + i] = o[g][i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const size_t at = size_t(bh) * NB + lb;
        for (int index = tid; index < G * V; index += BN * 32) {
            const int g = index / V;
            float top = -FLT_MAX;
            for (int s = 0; s < BN; s++) {
                top = max(top, maxima[s * G + g]);
            }
            float value = 0.0f;
            float total = 0.0f;
            for (int s = 0; s < BN; s++) {
                const float w = fast::exp(maxima[s * G + g] - top);
                value += w * outputs[s * G * V + index];
                total += w * sums[s * G + g];
            }
            partial[at * G * V + index] = value;
            if (index % V == 0) {
                stats[(at * G + g) * 2] = top;
                stats[(at * G + g) * 2 + 1] = total;
            }
        }
        """

    static let combineSource = """
        const int index = int(thread_position_in_grid.x);
        const int bh = index / (G * V);
        const int g = (index / V) % G;
        const int d = index % V;
        float top = -FLT_MAX;
        for (int s = 0; s < S; s++) {
            top = max(top, stats[((bh * S + s) * G + g) * 2]);
        }
        float value = 0.0f;
        float total = 0.0f;
        for (int s = 0; s < S; s++) {
            const int at = (bh * S + s) * G + g;
            const float w = fast::exp(stats[at * 2] - top);
            value += w * partial[at * V + d];
            total += w * stats[at * 2 + 1];
        }
        out[index] = static_cast<T>(total == 0.0f ? 0.0f : value / total);
        """
}

private let scoreKernel = MLXFast.metalKernel(
    name: "quest_block_scores", inputNames: ["q", "upper", "lower", "bounds"], outputNames: ["scores"],
    source: QuestKernel.scoreSource)

private let selectKernel = MLXFast.metalKernel(
    name: "quest_select_blocks", inputNames: ["scores", "bounds"], outputNames: ["chosen"],
    source: QuestKernel.selectSource)

private let sparseKernel = MLXFast.metalKernel(
    name: "quest_sparse_sdpa", inputNames: ["q", "keys", "values", "selected", "bounds", "scale"],
    outputNames: ["partial", "stats"], source: QuestKernel.sparseSource)

private let combineKernel = MLXFast.metalKernel(
    name: "quest_sparse_combine", inputNames: ["partial", "stats"], outputNames: ["out"],
    source: QuestKernel.combineSource)
