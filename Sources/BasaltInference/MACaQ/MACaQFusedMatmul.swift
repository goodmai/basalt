import BasaltCore
import Foundation
import MLX
import MLXNN

package enum MACaQActivationMode: String, Sendable, Codable, Equatable, CaseIterable {
    case fp16 = "w4a16"
    case w4a8 = "w4a8"
    case w4a4 = "w4a4"

    package var kernelCode: Int {
        switch self {
        case .fp16: 0
        case .w4a8: 1
        case .w4a4: 2
        }
    }

    package static var environmentDefault: MACaQActivationMode {
        guard let raw = ProcessInfo.processInfo.environment["BASALT_MACAQ_ACTIVATIONS"]?.lowercased(),
              let parsed = MACaQActivationMode(rawValue: raw) else {
            return .fp16
        }
        return parsed
    }
}

package struct MACaQFusedLinearInputs {
    package var packed: MACaQPackedWeights
    package var preScale: MLXArray
    package var rowScale: MLXArray?
    package var bias: MLXArray?
    package var spec: MACaQLayerSpec
    package var activationMode: MACaQActivationMode

    package var groupSize: Int { spec.groupSize }
    package var bits: Int { spec.bits }
    package var mode: QuantizationMode { spec.quantizationMode }
    package var hadamardBlock: Int { spec.hadamardBlock }

    package init(
        packed: MACaQPackedWeights,
        preScale: MLXArray,
        spec: MACaQLayerSpec,
        rowScale: MLXArray? = nil,
        bias: MLXArray? = nil,
        activationMode: MACaQActivationMode = .fp16
    ) {
        self.packed = packed
        self.preScale = preScale
        self.spec = spec
        self.rowScale = rowScale
        self.bias = bias
        self.activationMode = activationMode
    }
}

package enum MACaQFusedMatmul {
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let supportedBlocks: Set<Int> = [32, 64]
    package static let supportedBits: Set<Int> = [2, 3, 4, 8]
    package static let supportedGroups: Set<Int> = [32, 64, 128]
    static let rowsPerThreadgroup = 4

    package static func supportsLinear(_ x: MLXArray, inputs: MACaQFusedLinearInputs) -> Bool {
        guard x.ndim >= 1,
              supportedDTypes.contains(x.dtype),
              supportedBlocks.contains(inputs.hadamardBlock),
              supportedBits.contains(inputs.bits),
              supportedGroups.contains(inputs.groupSize),
              inputs.mode == .affine || (inputs.mode == .mxfp4 && inputs.bits == 4 && inputs.groupSize == 32)
        else { return false }
        let k = x.dim(-1)
        let rows = k > 0 ? (x.size / k) : 0
        guard rows > 0, k.isMultiple(of: inputs.hadamardBlock), k.isMultiple(of: inputs.groupSize),
              inputs.preScale.size == k
        else { return false }
        if inputs.activationMode == .fp16 && inputs.rowScale == nil && rows > 8 {
            return false
        }
        return true
    }

    static func cleanDType(_ array: MLXArray) -> MLXArray {
        supportedDTypes.contains(array.dtype) ? array : array.asType(.float32)
    }

    package static func linear(_ x: MLXArray, inputs: MACaQFusedLinearInputs) -> MLXArray {
        let origType = x.dtype
        let k = x.dim(-1)
        let rows = x.size / k
        let nOut = inputs.packed.weight.dim(0)
        let mTile = selectMTile(rows: rows)
        let mBlocks = (rows + mTile - 1) / mTile
        let nGroups = (nOut + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        let dummy = MLXArray.zeros([2], dtype: origType)
        let preClean = cleanDType(inputs.preScale)
        let scalesClean = inputs.mode == .mxfp4 ? inputs.packed.scales.asType(.uint8) : cleanDType(inputs.packed.scales)
        let biasesClean = inputs.packed.biases.map(cleanDType) ?? dummy
        let rowScaleClean = inputs.rowScale.map(cleanDType) ?? dummy
        let biasClean = inputs.bias.map(cleanDType) ?? dummy
        let shapeInfo = MLXArray([Int32(rows), Int32(nOut), Int32(k), Int32(inputs.packed.weight.dim(1))])
        let out = fusedLinearKernel(
            [x.reshaped(rows, k), inputs.packed.weight, scalesClean, biasesClean, preClean.reshaped(k), rowScaleClean.reshaped(rowScaleClean.size), biasClean.reshaped(biasClean.size), shapeInfo],
            template: [
                ("T", origType), ("TP", preClean.dtype), ("TS", scalesClean.dtype),
                ("TQB", biasesClean.dtype), ("TRS", rowScaleClean.dtype), ("TB", biasClean.dtype),
                ("B", inputs.hadamardBlock), ("BITS", inputs.bits), ("G", inputs.groupSize),
                ("IS_MXFP4", inputs.mode == .mxfp4), ("ACT_MODE", inputs.activationMode.kernelCode),
                ("HAS_ROW_SCALE", inputs.rowScale != nil), ("HAS_BIAS", inputs.bias != nil), ("MTILE", mTile),
            ],
            grid: (32, nGroups * rowsPerThreadgroup, mBlocks),
            threadGroup: (32, rowsPerThreadgroup, 1),
            outputShapes: [[rows, nOut]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(Array(x.shape.dropLast()) + [nOut])
    }

    static func selectMTile(rows: Int) -> Int {
        if rows <= 1 { return 1 }
        if rows == 2 { return 2 }
        if rows <= 4 { return 4 }
        return 8
    }
}

extension MACaQFusedMatmul {
    package static func supportsGather(_ x: MLXArray, indices: MLXArray, switchLayer: MACaQSwitchLinear) -> Bool {
        guard x.ndim >= 2,
              supportedDTypes.contains(x.dtype),
              supportedBlocks.contains(switchLayer.hadamardBlock),
              supportedBits.contains(switchLayer.bits),
              supportedGroups.contains(switchLayer.groupSize),
              switchLayer.mode == .affine || (switchLayer.mode == .mxfp4 && switchLayer.bits == 4 && switchLayer.groupSize == 32)
        else { return false }
        let k = x.dim(-1)
        let rows = x.dim(-2)
        let flatTasks = indices.size * max(0, rows)
        return rows > 0
            && flatTasks > 0
            && flatTasks <= 64
            && k > 0
            && k.isMultiple(of: switchLayer.hadamardBlock)
            && k.isMultiple(of: switchLayer.groupSize)
    }

    package static func gather(
        _ x: MLXArray,
        indices: MLXArray,
        weight: MLXArray,
        switchLayer: MACaQSwitchLinear,
        bias: MLXArray?
    ) -> MLXArray {
        let origType = x.dtype
        let batchShape = indices.shape
        let rows = x.dim(-2)
        let k = x.dim(-1)
        let targetXShape = batchShape + [rows, k]
        let xExp = x.shape == targetXShape ? x : broadcast(x, to: targetXShape)
        let idxExp = rows == 1
            ? indices.asType(.int32)
            : broadcast(indices.asType(.int32).expandedDimensions(axis: -1), to: batchShape + [rows])
        let flatTasks = indices.size * rows
        let nOut = weight.dim(1)
        let nGroups = (nOut + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        let dummy = MLXArray.zeros([2], dtype: origType)
        let preClean = cleanDType(switchLayer.preScale)
        let scalesClean = switchLayer.mode == .mxfp4 ? switchLayer.scales.asType(.uint8) : cleanDType(switchLayer.scales)
        let biasesClean = switchLayer.biases.map(cleanDType) ?? dummy
        let biasClean = bias.map(cleanDType) ?? dummy
        let shapeInfo = MLXArray([Int32(flatTasks), Int32(weight.dim(0)), Int32(nOut), Int32(k), Int32(weight.dim(2))])
        let out = fusedGatherKernel(
            [xExp.reshaped(flatTasks, k), idxExp.reshaped(flatTasks), weight, scalesClean, biasesClean, preClean, biasClean, shapeInfo],
            template: [
                ("T", origType), ("TP", preClean.dtype), ("TS", scalesClean.dtype),
                ("TQB", biasesClean.dtype), ("TB", biasClean.dtype),
                ("B", switchLayer.hadamardBlock), ("BITS", switchLayer.bits), ("G", switchLayer.groupSize),
                ("IS_MXFP4", switchLayer.mode == .mxfp4), ("PER_EXPERT_PRE", switchLayer.preScale.ndim == 2), ("HAS_BIAS", bias != nil),
            ],
            grid: (32, nGroups * rowsPerThreadgroup, flatTasks),
            threadGroup: (32, rowsPerThreadgroup, 1),
            outputShapes: [[flatTasks, nOut]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(batchShape + [rows, nOut])
    }
}

private enum MACaQFusedMatmulMSL {
    static let helpers = """
        inline float macaq_micro_act32(float v, int act_mode) {
            if (act_mode == 0) { return v; }
            const float amax = simd_max(fabs(v));
            if (!(amax > 0.0f)) { return 0.0f; }
            if (act_mode == 1) {
                const float sc = amax / 448.0f;
                return basalt_quant_dequant_fp8_e4m3(v, sc);
            }
            const uint8_t eb = basalt_compute_e8m0(amax, 6.0f);
            const float sc = basalt_e8m0_scale(eb);
            if (!(sc > 0.0f)) { return 0.0f; }
            const uint nib = basalt_quant_e2m1(v, 1.0f / sc);
            return basalt_e2m1_val(nib) * sc;
        }

        template <int BITS, int G, bool IS_MXFP4, typename PtrS, typename PtrQB>
        inline float macaq_decode_weight(
            const device uint32_t* w_row,
            PtrS s_row,
            PtrQB b_row,
            int col
        ) {
            if (IS_MXFP4) {
                const uint nib = basalt_extract_code(w_row, col, 4);
                const float sc = basalt_e8m0_scale(uint8_t(s_row[col >> 5]));
                return basalt_e2m1_val(nib) * sc;
            }
            const uint q = basalt_extract_code(w_row, col, BITS);
            const int g = col / G;
            return fma(float(q), float(s_row[g]), float(b_row[g]));
        }
        """

    static let linearSource = """
        const int M = int(shapes[0]);
        const int N = int(shapes[1]);
        const int K = int(shapes[2]);
        const int WPR = int(shapes[3]);
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int n = int(threadgroup_position_in_grid.y) * 4 + sg;
        const int m0 = int(threadgroup_position_in_grid.z) * MTILE;
        if (n >= N || m0 >= M) { return; }
        const int KG = K / G;
        const device uint32_t* w_row = w + size_t(n) * size_t(WPR);
        auto s_row = scales + size_t(n) * size_t(KG);
        auto b_row = IS_MXFP4 ? qbiases : (qbiases + size_t(n) * size_t(KG));
        float acc[MTILE];
        for (int r = 0; r < MTILE; ++r) { acc[r] = 0.0f; }
        if (B == 32) {
            for (int base = 0; base < K; base += 32) {
                const int col = base + lane;
                const float ps = float(pre_scale[col]);
                const float wv = macaq_decode_weight<BITS, G, IS_MXFP4>(w_row, s_row, b_row, col);
                for (int r = 0; r < MTILE; ++r) {
                    const int mr = m0 + r;
                    const float xv = (mr < M) ? (float(x[size_t(mr) * size_t(K) + size_t(col)]) * ps) : 0.0f;
                    const float xr = macaq_micro_act32(basalt_wht32_simd(xv, lane), ACT_MODE);
                    acc[r] = fma(xr, wv, acc[r]);
                }
            }
        } else {
            for (int base = 0; base < K; base += 64) {
                const int col0 = base + lane;
                const int col1 = col0 + 32;
                const float ps0 = float(pre_scale[col0]);
                const float ps1 = float(pre_scale[col1]);
                const float wv0 = macaq_decode_weight<BITS, G, IS_MXFP4>(w_row, s_row, b_row, col0);
                const float wv1 = macaq_decode_weight<BITS, G, IS_MXFP4>(w_row, s_row, b_row, col1);
                for (int r = 0; r < MTILE; ++r) {
                    const int mr = m0 + r;
                    const float x0 = (mr < M) ? (float(x[size_t(mr) * size_t(K) + size_t(col0)]) * ps0) : 0.0f;
                    const float x1 = (mr < M) ? (float(x[size_t(mr) * size_t(K) + size_t(col1)]) * ps1) : 0.0f;
                    float xr0 = 0.0f;
                    float xr1 = 0.0f;
                    basalt_wht64_simd(x0, x1, lane, xr0, xr1);
                    xr0 = macaq_micro_act32(xr0, ACT_MODE);
                    xr1 = macaq_micro_act32(xr1, ACT_MODE);
                    acc[r] = fma(xr1, wv1, fma(xr0, wv0, acc[r]));
                }
            }
        }
        for (int r = 0; r < MTILE; ++r) {
            const float total = simd_sum(acc[r]);
            const int mr = m0 + r;
            if (lane == 0 && mr < M) {
                out[size_t(mr) * size_t(N) + size_t(n)] = basalt_dual_scale_epilogue<T, HAS_ROW_SCALE, HAS_BIAS>(
                    total, row_scale, bias, n);
            }
        }
        """

    static let gatherSource = """
        const int R = int(shapes[0]);
        const int E = int(shapes[1]);
        const int N = int(shapes[2]);
        const int K = int(shapes[3]);
        const int WPR = int(shapes[4]);
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int n = int(threadgroup_position_in_grid.y) * 4 + sg;
        const int task = int(threadgroup_position_in_grid.z);
        if (n >= N || task >= R) { return; }
        const int e = int(rhs_indices[task]);
        if (e < 0 || e >= E) {
            if (lane == 0) { out[size_t(task) * size_t(N) + size_t(n)] = static_cast<T>(0.0f); }
            return;
        }
        const int KG = K / G;
        const size_t row_idx = size_t(e) * size_t(N) + size_t(n);
        const device uint32_t* w_row = w + row_idx * size_t(WPR);
        auto s_row = scales + row_idx * size_t(KG);
        auto b_row = IS_MXFP4 ? qbiases : (qbiases + row_idx * size_t(KG));
        auto p_row = PER_EXPERT_PRE ? (pre_scale + size_t(e) * size_t(K)) : pre_scale;
        const device T* x_row = x + size_t(task) * size_t(K);
        float acc = 0.0f;
        if (B == 32) {
            for (int base = 0; base < K; base += 32) {
                const int col = base + lane;
                const float xr = basalt_wht32_simd(float(x_row[col]) * float(p_row[col]), lane);
                const float wv = macaq_decode_weight<BITS, G, IS_MXFP4>(w_row, s_row, b_row, col);
                acc = fma(xr, wv, acc);
            }
        } else {
            for (int base = 0; base < K; base += 64) {
                const int col0 = base + lane;
                const int col1 = col0 + 32;
                float xr0 = 0.0f;
                float xr1 = 0.0f;
                basalt_wht64_simd(
                    float(x_row[col0]) * float(p_row[col0]),
                    float(x_row[col1]) * float(p_row[col1]),
                    lane, xr0, xr1);
                const float wv0 = macaq_decode_weight<BITS, G, IS_MXFP4>(w_row, s_row, b_row, col0);
                const float wv1 = macaq_decode_weight<BITS, G, IS_MXFP4>(w_row, s_row, b_row, col1);
                acc = fma(xr1, wv1, fma(xr0, wv0, acc));
            }
        }
        const float total = simd_sum(acc);
        if (lane == 0) {
            float val = total;
            if (HAS_BIAS) { val += float(bias[row_idx]); }
            out[size_t(task) * size_t(N) + size_t(n)] = static_cast<T>(val);
        }
        """
}

private let fusedLinearKernel = MLXFast.metalKernel(
    name: "macaq_fused_linear_qmm",
    inputNames: ["x", "w", "scales", "qbiases", "pre_scale", "row_scale", "bias", "shapes"],
    outputNames: ["out"],
    source: MACaQFusedMatmulMSL.linearSource,
    header: BasaltMetalSubstrate.combinedHeader + MACaQFusedMatmulMSL.helpers
)

private let fusedGatherKernel = MLXFast.metalKernel(
    name: "macaq_fused_gather_qmm",
    inputNames: ["x", "rhs_indices", "w", "scales", "qbiases", "pre_scale", "bias", "shapes"],
    outputNames: ["out"],
    source: MACaQFusedMatmulMSL.gatherSource,
    header: BasaltMetalSubstrate.combinedHeader + MACaQFusedMatmulMSL.helpers
)
