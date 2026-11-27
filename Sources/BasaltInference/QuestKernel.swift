import MLX

enum QuestKernel {
    static let maxMiddleBlocks = 4096
    static let maxGroupWidth = 1024
    static let rowSimdgroups = 4
    static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]

    static func fits(queries: MLXArray, keys: MLXArray, values: MLXArray, middleBlocks: Int) -> Bool {
        guard Device.defaultDevice() == .gpu,
              middleBlocks > 0, middleBlocks <= maxMiddleBlocks,
              dtypes.contains(keys.dtype), queries.dtype == keys.dtype, values.dtype == keys.dtype
        else { return false }
        return validAttentionShapes(queries: queries, keys: keys, values: values)
    }

    private static func validAttentionShapes(queries: MLXArray, keys: MLXArray, values: MLXArray) -> Bool {
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else { return false }
        let (batch, kvHeads, dim, valueDim) = (keys.dim(0), keys.dim(1), keys.dim(3), values.dim(3))
        guard batch > 0, kvHeads > 0, dim > 0, valueDim > 0 else { return false }
        guard [queries.dim(0), queries.dim(3)] == [batch, dim],
              [values.dim(0), values.dim(1)] == [batch, kvHeads],
              queries.dim(1) > 0, queries.dim(1).isMultiple(of: kvHeads)
        else { return false }
        let groups = queries.dim(1) / kvHeads
        return dim.isMultiple(of: 32) && valueDim.isMultiple(of: 32) && groups * max(dim, valueDim) <= maxGroupWidth
    }

    static func selectedBlocks(
        _ queries: MLXArray, upper: MLXArray, lower: MLXArray, first: Int, last: Int, topK: Int
    ) -> MLXArray {
        guard canSelectOnGPU(queries, upper: upper, lower: lower, first: first, last: last, topK: topK) else {
            return selectedBlocksReference(queries, upper: upper, lower: lower, first: first, last: last, topK: topK)
        }
        let (batch, kvHeads) = (upper.dim(0), upper.dim(1))
        let middle = last - first
        let bounds = MLXArray([Int32(first), Int32(last), Int32(topK)])
        let scores = blockScores(queries, upper: upper, lower: lower, bounds: bounds, middle: middle)
        return selectKernel(
            [scores, bounds],
            template: [("NMAX", maxMiddleBlocks)],
            grid: (256 * ((middle + 255) / 256), batch * kvHeads, 1), threadGroup: (256, 1, 1),
            outputShapes: [[batch, kvHeads, topK]], outputDTypes: [.int32])[0]
    }

    private static func canSelectOnGPU(
        _ queries: MLXArray, upper: MLXArray, lower: MLXArray, first: Int, last: Int, topK: Int
    ) -> Bool {
        guard Device.defaultDevice() == .gpu,
              queries.ndim == 4, upper.ndim == 4, lower.ndim == 4,
              dtypes.contains(upper.dtype), queries.dtype == upper.dtype, lower.dtype == upper.dtype
        else { return false }
        let (batch, kvHeads, dim) = (upper.dim(0), upper.dim(1), upper.dim(3))
        let middle = last - first
        guard batch > 0, kvHeads > 0, dim > 0, first >= 0, last <= upper.dim(2),
              middle > 0, middle <= maxMiddleBlocks, topK > 0, topK <= middle
        else { return false }
        return lower.shape == upper.shape
            && [queries.dim(0), queries.dim(2), queries.dim(3)] == [batch, 1, dim]
            && queries.dim(1) > 0 && queries.dim(1).isMultiple(of: kvHeads)
    }

    private static func selectedBlocksReference(
        _ queries: MLXArray, upper: MLXArray, lower: MLXArray, first: Int, last: Int, topK: Int
    ) -> MLXArray {
        if Device.defaultDevice() == .gpu, [queries, upper, lower].contains(where: { $0.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) {
                selectedBlocksReference(queries, upper: upper, lower: lower, first: first, last: last, topK: topK)
            }
        }
        let batch = upper.ndim >= 1 ? upper.dim(0) : 0
        let kvHeads = upper.ndim >= 2 ? upper.dim(1) : 0
        let middle = last - first
        guard queries.ndim == 4, upper.ndim == 4, lower.ndim == 4,
              batch > 0, kvHeads > 0, upper.dim(3) > 0,
              lower.shape == upper.shape, queries.dim(0) == batch, queries.dim(3) == upper.dim(3),
              queries.dim(1) > 0, queries.dim(1).isMultiple(of: kvHeads),
              queries.dim(2) == 1, first >= 0, last <= upper.dim(2), middle > 0, topK > 0
        else {
            return MLXArray.zeros([max(0, batch), max(0, kvHeads), max(0, topK)], type: Int32.self)
        }
        let k = min(topK, middle)
        if k == middle {
            return broadcast(MLXArray(Int32(first) ..< Int32(last)).reshaped(1, 1, middle), to: [batch, kvHeads, middle])
        }
        let q = queries.asType(.float32).reshaped(batch, kvHeads, queries.dim(1) / kvHeads, upper.dim(3))
        let raw = QuestKVCache.blockScores(
            q,
            upper: upper[0..., 0..., first ..< last].asType(.float32),
            lower: lower[0..., 0..., first ..< last].asType(.float32))
        let score = MLX.where(isNaN(raw), MLXArray(Float.infinity), raw)
        return argPartition(-score, kth: k - 1, axis: -1)[.ellipsis, ..<k].asType(.int32) + Int32(first)
    }

    private static func blockScores(_ queries: MLXArray, upper: MLXArray, lower: MLXArray, bounds: MLXArray, middle: Int) -> MLXArray {
        let (batch, kvHeads) = (upper.dim(0), upper.dim(1))
        return scoreKernel(
            [queries, upper, lower, bounds],
            template: [("T", upper.dtype)],
            grid: (256 * ((middle + 7) / 8), batch * kvHeads, 1), threadGroup: (256, 1, 1),
            outputShapes: [[batch, kvHeads, middle]], outputDTypes: [.float32])[0]
    }

    static func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, selected: MLXArray, offset: Int, scale: Float
    ) -> MLXArray {
        guard canAttendOnGPU(queries: queries, keys: keys, values: values, selected: selected, offset: offset) else {
            return gatheredAttentionReference(
                queries: queries, keys: keys, values: values, selected: selected, offset: offset, scale: scale)
        }
        let last = (offset - QuestKVCache.windowTokens) / QuestKVCache.blockTokens
        let (batch, kvHeads, dim, valueDim) = (keys.dim(0), keys.dim(1), keys.dim(3), values.dim(3))
        let groups = queries.dim(1) / kvHeads
        let sinkBlocks = QuestKVCache.sinkTokens / QuestKVCache.blockTokens
        let blocks = sinkBlocks + selected.dim(2) + QuestKVCache.windowTokens / QuestKVCache.blockTokens + 1
        let threads = 32 * rowSimdgroups
        let partial = sparseKernel(
            [queries, keys, values, selected.asType(.int32), MLXArray([Int32(last), Int32(offset), Int32(blocks)]), MLXArray([scale])],
            template: [
                ("T", keys.dtype), ("D", dim), ("V", valueDim), ("G", groups),
                ("BN", rowSimdgroups), ("BT", QuestKVCache.blockTokens), ("SB", sinkBlocks),
            ],
            grid: (threads * blocks, batch * kvHeads, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[batch * kvHeads, blocks, groups, valueDim], [batch * kvHeads, blocks, groups, 2]],
            outputDTypes: [.float32, .float32])
        return combineKernel(
            partial,
            template: [("T", keys.dtype), ("V", valueDim), ("G", groups)],
            grid: (batch * kvHeads * groups * valueDim, 1, 1), threadGroup: (min(256, groups * valueDim), 1, 1),
            outputShapes: [[batch, kvHeads * groups, 1, valueDim]], outputDTypes: [keys.dtype])[0]
    }

    private static func canAttendOnGPU(
        queries: MLXArray, keys: MLXArray, values: MLXArray, selected: MLXArray, offset: Int
    ) -> Bool {
        guard Device.defaultDevice() == .gpu,
              dtypes.contains(keys.dtype), queries.dtype == keys.dtype, values.dtype == keys.dtype,
              validAttentionShapes(queries: queries, keys: keys, values: values)
        else { return false }
        let minOffset = QuestKVCache.sinkTokens + QuestKVCache.windowTokens
        guard queries.dim(2) == 1, selected.ndim == 3, selected.dim(2) > 0,
              selected.dtype.isInteger, selected.dtype != .int64, selected.dtype != .uint64,
              offset >= minOffset, offset <= keys.dim(2), offset <= values.dim(2)
        else { return false }
        return selected.dim(0) == keys.dim(0) && selected.dim(1) == keys.dim(1)
    }
}

extension QuestKernel {
    private static func gatheredAttentionReference(
        queries: MLXArray, keys: MLXArray, values: MLXArray, selected: MLXArray, offset: Int, scale: Float
    ) -> MLXArray {
        if Device.defaultDevice() == .gpu, [queries, keys, values, selected].contains(where: { $0.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) {
                gatheredAttentionReference(
                    queries: queries, keys: keys, values: values, selected: selected, offset: offset, scale: scale)
            }
        }
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4,
              queries.size > 0, keys.size > 0, values.size > 0, keys.dim(1) > 0, offset > 0,
              values.dim(0) == keys.dim(0), values.dim(1) == keys.dim(1),
              queries.dim(0) == keys.dim(0), queries.dim(3) == keys.dim(3),
              queries.dim(1) > 0, queries.dim(1).isMultiple(of: keys.dim(1)),
              offset <= keys.dim(2), offset <= values.dim(2)
        else {
            let vDim = values.ndim >= 4 ? values.dim(3) : 0
            let (b, h, l) = queries.ndim >= 3 ? (queries.dim(0), queries.dim(1), queries.dim(2)) : (0, 0, 0)
            return MLXArray.zeros([b, h, l, vDim], dtype: queries.dtype)
        }
        let (batch, kvHeads) = (keys.dim(0), keys.dim(1))
        let (rows, valid) = gatheredRows(batch: batch, kvHeads: kvHeads, selected: selected, offset: offset)
        let count = rows.dim(2)
        let gatheredK = takeAlong(keys, broadcast(rows, to: [batch, kvHeads, count, keys.dim(3)]), axis: 2)
        let gatheredV = takeAlong(values, broadcast(rows, to: [batch, kvHeads, count, values.dim(3)]), axis: 2)
        let groups = queries.dim(1) / kvHeads
        let maskHead = repeated(valid.expandedDimensions(axis: 2), count: groups, axis: 1)
        let sdpaMask = broadcast(maskHead, to: [batch, queries.dim(1), queries.dim(2), count])
        let qIsRealFloat = dtypes.contains(queries.dtype) || queries.dtype == .float64
        let computeType: DType = qIsRealFloat && keys.dtype == queries.dtype && values.dtype == queries.dtype
            ? queries.dtype : .float32
        let qIn = queries.dtype == computeType ? queries : queries.asType(computeType)
        let kIn = gatheredK.dtype == computeType ? gatheredK : gatheredK.asType(computeType)
        let vIn = gatheredV.dtype == computeType ? gatheredV : gatheredV.asType(computeType)
        let out = MLXFast.scaledDotProductAttention(
            queries: qIn, keys: kIn, values: vIn, scale: scale, mask: .array(sdpaMask))
        return qIsRealFloat && out.dtype != queries.dtype ? out.asType(queries.dtype) : out
    }

    private static func gatheredRows(batch: Int, kvHeads: Int, selected: MLXArray, offset: Int) -> (rows: MLXArray, valid: MLXArray) {
        let blockTokens = QuestKVCache.blockTokens
        let sinkEnd = min(offset, QuestKVCache.sinkTokens)
        let last = max(sinkEnd / blockTokens, (offset - QuestKVCache.windowTokens) / blockTokens)
        let windowStart = max(sinkEnd, min(offset, last * blockTokens))
        let fixedParts = [
            sinkEnd > 0 ? MLXArray(Int32(0) ..< Int32(sinkEnd)) : nil,
            offset > windowStart ? MLXArray(Int32(windowStart) ..< Int32(offset)) : nil,
        ].compactMap { $0 }
        let fixed = fixedParts.count == 2 ? concatenated(fixedParts, axis: 0) : (fixedParts.first ?? MLXArray([Int32]()))
        let fixedRows = broadcast(fixed.reshaped(1, 1, -1), to: [batch, kvHeads, fixed.dim(0)])
        let fixedValid = MLXArray.full([batch, kvHeads, fixed.dim(0)], values: MLXArray(true))
        guard selected.ndim == 3, selected.dim(0) == batch, selected.dim(1) == kvHeads,
              selected.dim(2) > 0, selected.dtype.isInteger else {
            return (fixedRows.expandedDimensions(axis: -1), fixedValid)
        }
        let maxBlocks = Int64((offset + blockTokens - 1) / blockTokens)
        let blk64 = selected.asType(.int64)
        let blkValid = (blk64 .>= Int64(0)) .&& (blk64 .< maxBlocks)
        let safeBlk = MLX.where(blkValid, blk64, MLXArray(Int64(0)))
        let blockSteps = MLXArray((0 ..< blockTokens).map(Int64.init)).reshaped(1, 1, 1, -1)
        let rawRows = safeBlk.expandedDimensions(axis: -1) * Int64(blockTokens) + blockSteps
        let rowValid = (blkValid.expandedDimensions(axis: -1) .&& (rawRows .< Int64(offset))).reshaped(batch, kvHeads, -1)
        let safeRows = MLX.where(rowValid, rawRows.reshaped(batch, kvHeads, -1), MLXArray(Int64(0))).asType(.int32)
        return (
            concatenated([fixedRows, safeRows], axis: -1).expandedDimensions(axis: -1),
            concatenated([fixedValid, rowValid], axis: -1))
    }

    static let scoreSource = """
        const int bh = int(threadgroup_position_in_grid.y);
        const int lane = int(thread_index_in_simdgroup);
        const int j = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
        const int first = bounds[0];
        const int n = bounds[1] - first;
        const int nb = int(upper_shape[2]);
        const int D = int(upper_shape[3]);
        const int G = int(q_shape[1]) / int(upper_shape[1]);
        if (j >= n) {
            return;
        }
        const size_t bound_base = (size_t(bh) * size_t(nb) + size_t(first + j)) * size_t(D);
        const size_t q_bh_base = size_t(bh) * size_t(G) * size_t(D);
        float s = 0.0f;
        for (int d = lane; d < D; d += 32) {
            float p = 0.0f;
            float m = 0.0f;
            for (int g = 0; g < G; g++) {
                const float x = float(q[q_bh_base + size_t(g) * size_t(D) + size_t(d)]);
                p += isnan(x) ? x : max(x, 0.0f);
                m += isnan(x) ? x : min(x, 0.0f);
            }
            s += p * float(upper[bound_base + size_t(d)]) + m * float(lower[bound_base + size_t(d)]);
        }
        s = simd_sum(s);
        if (lane == 0) {
            scores[size_t(bh) * size_t(n) + size_t(j)] = isnan(s) ? INFINITY : s;
        }
        """

    static let selectSource = """
        const int bh = int(threadgroup_position_in_grid.y);
        const int tid = int(thread_position_in_threadgroup.x);
        const int j = int(threadgroup_position_in_grid.x) * 256 + tid;
        const int first = bounds[0];
        const int n = bounds[1] - first;
        const int K = bounds[2];
        const int quads = (n + 3) / 4;
        threadgroup float4 score[NMAX / 4];
        threadgroup float* flat = (threadgroup float*)score;
        for (int i = tid; i < quads * 4; i += 256) {
            flat[i] = i < n ? scores[size_t(bh) * size_t(n) + size_t(i)] : -INFINITY;
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
            chosen[size_t(bh) * size_t(K) + size_t(total)] = first + j;
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
        const int NB = bounds[2];
        const int K = int(selected_shape[2]);
        const int blocks = SB + K + (offset - last * BT + BT - 1) / BT;
        const size_t k_capacity = size_t(keys_shape[2]);
        const size_t v_capacity = size_t(values_shape[2]);
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
            const int block = lb < SB ? lb : (lb < SB + K ? selected[size_t(bh) * size_t(K) + size_t(lb - SB)] : last + lb - SB - K);
            const int max_blocks = (offset + BT - 1) / BT;
            if (block >= 0 && block < max_blocks) {
                const int start = block * BT;
                const int end = min(start + BT, offset);
                const size_t k_head_base = size_t(bh) * k_capacity * size_t(D) + size_t(lane * E);
                const size_t v_head_base = size_t(bh) * v_capacity * size_t(V) + size_t(lane * EV);
                const size_t q_head_base = size_t(bh) * size_t(G) * size_t(D) + size_t(lane * E);
                float qr[G][E];
                for (int g = 0; g < G; g++) {
                    for (int i = 0; i < E; i++) {
                        qr[g][i] = scale[0] * float(q[q_head_base + size_t(g) * size_t(D) + size_t(i)]);
                    }
                }
                for (int row = start + sg; row < end; row += BN) {
                    const size_t k_row_base = k_head_base + size_t(row) * size_t(D);
                    float k[E];
                    for (int i = 0; i < E; i++) {
                        k[i] = float(keys[k_row_base + size_t(i)]);
                    }
                    const size_t v_row_base = v_head_base + size_t(row) * size_t(V);
                    float v[EV];
                    for (int i = 0; i < EV; i++) {
                        v[i] = float(values[v_row_base + size_t(i)]);
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
        const size_t at = size_t(bh) * size_t(NB) + size_t(lb);
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
            partial[at * size_t(G * V) + size_t(index)] = value;
            if (index % V == 0) {
                stats[(at * size_t(G) + size_t(g)) * 2] = top;
                stats[(at * size_t(G) + size_t(g)) * 2 + 1] = total;
            }
        }
        """

    static let combineSource = """
        const size_t index = size_t(thread_position_in_grid.x);
        const size_t gv = size_t(G) * size_t(V);
        const size_t bh = index / gv;
        const size_t g = (index / size_t(V)) % size_t(G);
        const size_t d = index % size_t(V);
        const int S = int(stats_shape[1]);
        float top = -FLT_MAX;
        for (int s = 0; s < S; s++) {
            const size_t at = (bh * size_t(S) + size_t(s)) * size_t(G) + g;
            top = max(top, stats[at * 2]);
        }
        float value = 0.0f;
        float total = 0.0f;
        for (int s = 0; s < S; s++) {
            const size_t at = (bh * size_t(S) + size_t(s)) * size_t(G) + g;
            const float w = fast::exp(stats[at * 2] - top);
            value += w * partial[at * size_t(V) + d];
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
