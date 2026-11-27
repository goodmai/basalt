import Foundation
import MLX
import MLXNN
import BasaltCore

enum ThreeBitQMV {
    static let environmentKey = "BASALT_Q3_KERNEL"
    static let groupSizes: Set<Int> = [32, 64, 128]
    static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]
    static let rowsPerSimdgroup = 4

    static var enabled: Bool { ProcessInfo.processInfo.environment[environmentKey] == "1" }

    static func fits(bits: Int, mode: QuantizationMode, groupSize: Int, inputs: Int, outputs: Int = 1, hasBiases: Bool) -> Bool {
        bits == 3 && mode == .affine && hasBiases && groupSizes.contains(groupSize)
            && inputs > 0 && outputs > 0 && inputs % groupSize == 0
    }

    static func handles(_ x: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int? = nil) -> Bool {
        guard Device.defaultDevice() == .gpu, x.ndim >= 2, x.dim(-2) == 1 else { return false }
        guard scales.ndim >= 2, biases.ndim >= 2, scales.shape == biases.shape else { return false }
        guard dtypes.contains(x.dtype), scales.dtype == x.dtype, biases.dtype == x.dtype else { return false }
        return fitsScaleShape(inputs: x.dim(-1), scales: scales, groupSize: groupSize)
    }

    private static func fitsScaleShape(inputs: Int, scales: MLXArray, groupSize: Int?) -> Bool {
        let outputs = scales.dim(-2)
        let groups = scales.dim(-1)
        guard inputs > 0, outputs > 0, groups > 0, inputs % groups == 0 else { return false }
        let resolved = inputs / groups
        if let groupSize { return resolved == groupSize && groupSizes.contains(groupSize) }
        return groupSizes.contains(resolved)
    }

    static func handlesSingleRow(_ x: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int? = nil) -> Bool {
        scales.ndim == 2 && handles(x, scales: scales, biases: biases, groupSize: groupSize) && x.size == x.dim(-1)
    }

    static func handlesGather(_ x: MLXArray, indices: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int? = nil) -> Bool {
        scales.ndim == 3 && scales.dim(0) > 0
            && handles(x, scales: scales, biases: biases, groupSize: groupSize)
            && x.ndim == indices.ndim + 2
            && Array(x.shape.prefix(indices.ndim)) == indices.shape
    }

    static func matmul(_ x: MLXArray, indices: MLXArray?, weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int) -> MLXArray {
        if Device.defaultDevice() == .gpu, [x, indices, weight, scales, biases].contains(where: { $0?.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) {
                matmul(x, indices: indices, weight: weight, scales: scales, biases: biases, groupSize: groupSize)
            }
        }
        let inputs = x.ndim > 0 ? x.dim(-1) : 0
        let outputs = weight.ndim >= 2 ? weight.dim(-2) : 0
        let rows = x.ndim >= 2 ? x.dim(-2) : 0
        let outShape: [Int]
        let gatherBatchCount: Int
        if let indices {
            let xBatch = x.ndim >= 2 ? Array(x.shape.dropLast(2)) : []
            let batchShape = BasaltMetalSubstrate.broadcastShape(xBatch, indices.shape) ?? indices.shape
            gatherBatchCount = batchShape.reduce(1, *)
            outShape = batchShape + [max(0, rows), max(0, outputs)]
        } else {
            gatherBatchCount = 1
            outShape = (x.ndim > 0 ? Array(x.shape.dropLast()) : []) + [outputs]
        }
        let numExperts = indices != nil ? (weight.ndim > 0 ? weight.dim(0) : 0) : 1
        let hasIndices = indices.map { $0.size > 0 } ?? true
        guard inputs > 0, outputs > 0, x.size > 0, numExperts > 0, hasIndices, gatherBatchCount > 0,
              (indices == nil || rows > 0) else {
            return MLXArray.zeros(outShape, dtype: x.dtype)
        }
        guard canDispatchKernel(x, indices: indices, weight: weight, scales: scales, biases: biases, groupSize: groupSize) else {
            return fallbackMatmul(x, indices: indices, weight: weight, scales: scales, biases: biases, groupSize: groupSize)
        }
        let tasks = x.size / inputs
        let index = indices?.asType(.int64).reshaped(tasks) ?? MLXArray.zeros([1], dtype: .int64)
        let simdgroups = (outputs + rowsPerSimdgroup - 1) / rowsPerSimdgroup
        let result = kernel(
            [x.reshaped(tasks, inputs), index, weight, scales, biases, MLXArray([Int32(outputs), Int32(inputs), Int32(numExperts)])],
            template: [("T", x.dtype), ("G", groupSize), ("GATHER", indices != nil)],
            grid: (32, 2 * ((simdgroups + 1) / 2), tasks),
            threadGroup: (32, 2, 1),
            outputShapes: [[tasks, outputs]],
            outputDTypes: [x.dtype])[0]
        return result.reshaped(outShape)
    }

    private static func canDispatchKernel(
        _ x: MLXArray, indices: MLXArray?, weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int
    ) -> Bool {
        let inputs = x.dim(-1)
        let expectedNdim = indices == nil ? 2 : 3
        guard Device.defaultDevice() == .gpu, groupSizes.contains(groupSize), inputs % groupSize == 0,
              weight.ndim == expectedNdim, weight.dtype == .uint32,
              scales.ndim == expectedNdim, biases.ndim == expectedNdim
        else { return false }
        let outputs = weight.dim(-2)
        let expectedScaleShape = indices == nil ? [outputs, inputs / groupSize] : [weight.dim(0), outputs, inputs / groupSize]
        guard weight.dim(-1) == (inputs * 3) / 32,
              scales.shape == expectedScaleShape, biases.shape == expectedScaleShape,
              dtypes.contains(x.dtype), scales.dtype == x.dtype, biases.dtype == x.dtype
        else { return false }
        return validGatherLayout(x, indices: indices)
    }

    private static func validGatherLayout(_ x: MLXArray, indices: MLXArray?) -> Bool {
        guard let indices else { return true }
        return indices.dtype.isInteger && indices.size > 0 && x.ndim == indices.ndim + 2 && x.dim(-2) == 1
            && Array(x.shape.prefix(indices.ndim)) == indices.shape
    }

    private static func fallbackMatmul(
        _ x: MLXArray, indices: MLXArray?, weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int
    ) -> MLXArray {
        let hasF64 = x.dtype == .float64 || scales.dtype == .float64 || biases.dtype == .float64
        let xIn = dtypes.contains(x.dtype) ? x : x.asType(.float32)
        let sIn = dtypes.contains(scales.dtype) ? scales : scales.asType(.float32)
        let bIn = dtypes.contains(biases.dtype) ? biases : biases.asType(.float32)
        let result: MLXArray
        if let indices {
            let numExperts = weight.dim(0)
            let idx64 = indices.asType(.int64)
            let valid = (idx64 .>= Int64(0)) .&& (idx64 .< Int64(numExperts))
            let safeIndices = which(valid, idx64, Int64(0)).asType(.int32)
            let raw = MLX.gatherQuantizedMM(
                xIn, weight, scales: sIn, biases: bIn, rhsIndices: safeIndices,
                transpose: true, groupSize: groupSize, bits: 3, mode: .affine)
            let mask = valid.reshaped(indices.shape + [1, 1])
            result = which(mask, raw, MLXArray.zeros(raw.shape, dtype: raw.dtype))
        } else {
            result = MLX.quantizedMM(
                xIn, weight, scales: sIn, biases: bIn,
                transpose: true, groupSize: groupSize, bits: 3, mode: .affine)
        }
        if hasF64 { return result.asType(.float64) }
        return dtypes.contains(x.dtype) ? result : result.asType(x.dtype)
    }

    static let header = """
        inline float basalt_q3_scale_x(thread float* xt) {
            float sum = 0.0f;
            for (int j = 0; j < 16; j++) { sum += xt[j]; }
            xt[1] *= 0.125f; xt[2] *= 1.0f / 64.0f; xt[3] *= 1.0f / 512.0f; xt[4] *= 1.0f / 4096.0f;
            xt[6] *= 0.25f; xt[7] *= 1.0f / 32.0f; xt[8] *= 1.0f / 256.0f; xt[9] *= 1.0f / 2048.0f;
            xt[11] *= 0.5f; xt[12] *= 1.0f / 16.0f; xt[13] *= 1.0f / 128.0f; xt[14] *= 1.0f / 1024.0f; xt[15] *= 1.0f / 8192.0f;
            return sum;
        }
        inline float basalt_q3_dot(uint h0, uint h1, uint h2, thread const float* x) {
            float a = float(h0 & 0x0007u) * x[0] + float(h0 & 0x0038u) * x[1] + float(h0 & 0x01c0u) * x[2]
                + float(h0 & 0x0e00u) * x[3] + float(h0 & 0x7000u) * x[4];
            a += (float(h0 >> 15) + float(h1 & 0x0003u) * 2.0f) * x[5];
            a += float(h1 & 0x001cu) * x[6] + float(h1 & 0x00e0u) * x[7] + float(h1 & 0x0700u) * x[8] + float(h1 & 0x3800u) * x[9];
            a += (float(h1 >> 14) + float(h2 & 0x0001u) * 4.0f) * x[10];
            a += float(h2 & 0x000eu) * x[11] + float(h2 & 0x0070u) * x[12] + float(h2 & 0x0380u) * x[13]
                + float(h2 & 0x1c00u) * x[14] + float(h2 & 0xe000u) * x[15];
            return a;
        }
        """

    static let source = """
        const int N = int(shapes[0]);
        const int K = int(shapes[1]);
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int task = int(threadgroup_position_in_grid.z);
        const int n0 = (int(threadgroup_position_in_grid.y) * 2 + sg) * 4;
        if (n0 >= N) { return; }
        const int E = int(shapes[2]);
        const int64_t e = GATHER ? int64_t(idx[task]) : 0;
        const int KW = (K * 3) / 32;
        const int KG = K / G;
        const int last = min(3, N - 1 - n0);
        if (GATHER && (e < 0 || e >= int64_t(E))) {
            if (lane == 0) {
                for (int r = 0; r <= last; r++) {
                    out[size_t(task) * size_t(N) + size_t(n0 + r)] = static_cast<T>(0.0f);
                }
            }
            return;
        }
        const size_t row0 = size_t(e) * size_t(N) + size_t(n0);
        const int halfIdx = lane & 1;
        const size_t xBase = size_t(task) * size_t(K) + size_t(lane * 16);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float xt[16];
        for (int k = 0; k < K; k += 512) {
            if (k + lane * 16 < K) {
                const size_t xOff = xBase + size_t(k);
                for (int j = 0; j < 16; j++) { xt[j] = float(x[xOff + size_t(j)]); }
                const float sum = basalt_q3_scale_x(xt);
                const int col = k + lane * 16;
                const int g = col / G;
                const size_t blkOff = size_t(col >> 5) * size_t(3);
                for (int r = 0; r < 4; r++) {
                    const size_t row = row0 + size_t(min(r, last));
                    const size_t wOff = row * size_t(KW) + blkOff;
                    const uint a = w[wOff + size_t(halfIdx)];
                    const uint b = w[wOff + size_t(halfIdx + 1)];
                    const uint h0 = halfIdx == 0 ? (a & 0xffffu) : (a >> 16);
                    const uint h1 = halfIdx == 0 ? (a >> 16) : (b & 0xffffu);
                    const uint h2 = halfIdx == 0 ? (b & 0xffffu) : (b >> 16);
                    const float d = basalt_q3_dot(h0, h1, h2, xt);
                    const size_t sgOff = row * size_t(KG) + size_t(g);
                    acc[r] += float(scales[sgOff]) * d + float(biases[sgOff]) * sum;
                }
            }
        }
        for (int r = 0; r < 4; r++) {
            const float v = simd_sum(acc[r]);
            if (lane == 0 && r <= last) { out[size_t(task) * size_t(N) + size_t(n0 + r)] = static_cast<T>(v); }
        }
        """

    static let logger = BasaltLogger(module: "ThreeBitQMV")

    static func installIfEnabled(in model: Module, enabled: Bool = false) {
        guard enabled || Self.enabled else { return }
        logger.info("Experimental 3-bit decode kernel (#653): \(install(in: model)) MACaQ 3-bit affine layers run one-row forwards through it, the rest stay on stock quantized matmul")
    }

    @discardableResult
    static func install(in model: Module, enabled: Bool = true) -> Int {
        var count = 0
        let leaves = model.leafModules().flattened().map(\.1)
        for module in leaves.isEmpty ? [model] : leaves {
            if let layer = module as? MACaQQuantizedLinear,
               fits(bits: layer.bits, mode: layer.mode, groupSize: layer.groupSize, inputs: layer.shape.1, outputs: layer.shape.0, hasBiases: layer.biases != nil) {
                layer.useQ3Kernel = enabled
                count += 1
            } else if let layer = module as? MACaQSwitchLinear,
                      fits(bits: layer.bits, mode: layer.mode, groupSize: layer.groupSize, inputs: layer.scales.dim(-1) * layer.groupSize, outputs: layer.scales.ndim >= 2 ? layer.scales.dim(-2) : 0, hasBiases: layer.biases != nil) {
                layer.useQ3Kernel = enabled
                count += 1
            }
        }
        return count
    }
}

private let kernel = MLXFast.metalKernel(
    name: "basalt_affine3_qmv", inputNames: ["x", "idx", "w", "scales", "biases", "shapes"], outputNames: ["out"],
    source: ThreeBitQMV.source, header: ThreeBitQMV.header)
