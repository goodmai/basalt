import Foundation
import MLX
import MLXNN
import BasaltCore

enum SmallMQuantizedMatmul {
    static let environmentKey = "BASALT_SMALLM_KERNEL"
    static let rows = 2 ... 16
    static let groupSizes: Set<Int> = [32, 64, 128]
    static let chunk = 64
    static let splitK = 4
    static let tilesPerSimdgroup = 4
    static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]

    static var enabled: Bool { ProcessInfo.processInfo.environment[environmentKey] == "1" }

    static func fits(_ layer: QuantizedLinear) -> Bool {
        let (outputs, inputs) = layer.shape
        guard outputs > 0, inputs > 0, layer.mode == .affine, layer.bits == 4,
              let biases = layer.biases, layer.globalScale == nil
        else { return false }
        return fitsGeometry(
            inputs: inputs, outputs: outputs, groupSize: layer.groupSize,
            weight: layer.weight, scales: layer.scales, biases: biases)
    }

    private static func fitsGeometry(
        inputs: Int, outputs: Int, groupSize: Int, weight: MLXArray, scales: MLXArray, biases: MLXArray
    ) -> Bool {
        guard outputs > 0, inputs > 0, groupSizes.contains(groupSize), weight.ndim == 2, weight.dtype == .uint32,
              inputs % chunk == 0, inputs % groupSize == 0, outputs % (8 * tilesPerSimdgroup) == 0,
              weight.dim(1) == inputs / 8
        else { return false }
        let expected = [outputs, inputs / groupSize]
        return scales.shape == expected && biases.shape == expected
    }

    static func handles(_ x: MLXArray, layer: QuantizedLinear) -> Bool {
        guard Device.defaultDevice() == .gpu, !layer.training, x.ndim >= 1, fits(layer) else { return false }
        let inputs = x.dim(-1)
        guard inputs == layer.shape.1, rows.contains(x.size / inputs) else { return false }
        return dtypes.contains(x.dtype) && layer.scales.dtype == x.dtype && layer.biases?.dtype == x.dtype
    }

    static func matmul(_ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int) -> MLXArray {
        if Device.defaultDevice() == .gpu, [x, weight, scales, biases].contains(where: { $0.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) {
                matmul(x, weight: weight, scales: scales, biases: biases, groupSize: groupSize)
            }
        }
        let inputs = x.ndim > 0 ? x.dim(-1) : 0
        let outputs = weight.ndim > 0 ? weight.dim(0) : 0
        let outShape = (x.ndim > 0 ? Array(x.shape.dropLast()) : []) + [outputs]
        guard inputs > 0, outputs > 0, x.size > 0 else {
            return MLXArray.zeros(outShape, dtype: x.dtype)
        }
        let rowCount = x.size / inputs
        guard Device.defaultDevice() == .gpu,
              rows.contains(rowCount),
              dtypes.contains(x.dtype),
              scales.dtype == x.dtype,
              biases.dtype == x.dtype,
              fitsGeometry(inputs: inputs, outputs: outputs, groupSize: groupSize, weight: weight, scales: scales, biases: biases)
        else {
            let hasF64 = x.dtype == .float64 || scales.dtype == .float64 || biases.dtype == .float64
            let raw = MLX.quantizedMM(
                dtypes.contains(x.dtype) ? x : x.asType(.float32), weight,
                scales: dtypes.contains(scales.dtype) ? scales : scales.asType(.float32),
                biases: dtypes.contains(biases.dtype) ? biases : biases.asType(.float32),
                transpose: true, groupSize: groupSize, bits: 4, mode: .affine)
            if hasF64 { return raw.asType(.float64) }
            return dtypes.contains(x.dtype) ? raw : raw.asType(x.dtype)
        }
        let mTiles = (rowCount + 7) / 8
        let result = kernel(
            [x.reshaped(rowCount, inputs), weight, scales, biases],
            template: [("T", x.dtype), ("M", rowCount), ("MT", mTiles), ("G", groupSize), ("S", splitK), ("NT", tilesPerSimdgroup)],
            grid: (32, splitK * outputs / (8 * tilesPerSimdgroup), 1),
            threadGroup: (32, splitK, 1),
            outputShapes: [[rowCount, outputs]],
            outputDTypes: [x.dtype])[0]
        return result.reshaped(outShape)
    }

    static let source = """
        constexpr int CH = 64;
        const int K = int(x_shape[1]);
        const int N = int(w_shape[0]);
        const int KW = K / 8;
        const int KG = K / G;
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int n0 = int(threadgroup_position_in_grid.y) * (8 * NT);
        const int qid = lane / 4;
        const int sm = (qid & 4) + (lane / 2) % 4;
        const int sn = (qid & 2) * 2 + (lane % 2) * 2;
        simdgroup_float8x8 acc[MT][NT];
        for (int mt = 0; mt < MT; mt++) {
            for (int j = 0; j < NT; j++) {
                acc[mt][j] = simdgroup_float8x8(0.0f);
            }
        }
        for (int c = sg * CH; c < K; c += S * CH) {
            const int kx = c + 16 * (sm / 2) + (sm % 2);
            const int kw = c + 16 * (sn / 2);
            float xa[MT][8];
            float xb[MT][8];
            for (int mt = 0; mt < MT; mt++) {
                const int r0 = mt * 8 + sn;
                for (int t = 0; t < 8; t++) {
                    xa[mt][t] = r0 < M ? float(x[size_t(r0) * size_t(K) + size_t(kx + 2 * t)]) : 0.0f;
                    xb[mt][t] = (r0 + 1) < M ? float(x[size_t(r0 + 1) * size_t(K) + size_t(kx + 2 * t)]) : 0.0f;
                }
            }
            for (int j = 0; j < NT; j++) {
                const int n = n0 + 8 * j + sm;
                const size_t wIdx = size_t(n) * size_t(KW) + size_t(kw / 8);
                const uint q0 = w[wIdx];
                const uint q1 = w[wIdx + 1];
                const float s = scales[size_t(n) * size_t(KG) + size_t(kw / G)];
                const float b = biases[size_t(n) * size_t(KG) + size_t(kw / G)];
                for (int t = 0; t < 8; t++) {
                    const uint word = t < 4 ? q0 : q1;
                    const uint shift = 8u * uint(t % 4);
                    simdgroup_float8x8 wt;
                    wt.thread_elements()[0] = s * float((word >> shift) & 15u) + b;
                    wt.thread_elements()[1] = s * float((word >> (shift + 4u)) & 15u) + b;
                    for (int mt = 0; mt < MT; mt++) {
                        simdgroup_float8x8 xt;
                        xt.thread_elements()[0] = xa[mt][t];
                        xt.thread_elements()[1] = xb[mt][t];
                        simdgroup_multiply_accumulate(acc[mt][j], wt, xt, acc[mt][j]);
                    }
                }
            }
        }
        threadgroup float partial[S * MT * NT * 64];
        for (int mt = 0; mt < MT; mt++) {
            for (int j = 0; j < NT; j++) {
                const int base = ((sg * MT + mt) * NT + j) * 64 + sm * 8 + sn;
                partial[base] = acc[mt][j].thread_elements()[0];
                partial[base + 1] = acc[mt][j].thread_elements()[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (int mt = 0; mt < MT; mt++) {
                for (int j = 0; j < NT; j++) {
                    for (int e = 0; e < 2; e++) {
                        float v = 0.0f;
                        for (int p = 0; p < S; p++) {
                            v += partial[((p * MT + mt) * NT + j) * 64 + sm * 8 + sn + e];
                        }
                        const int row = mt * 8 + sn + e;
                        if (row < M) {
                            out[size_t(row) * size_t(N) + size_t(n0 + 8 * j + sm)] = static_cast<T>(v);
                        }
                    }
                }
            }
        }
        """
}

private let kernel = MLXFast.metalKernel(
    name: "smallm_affine4_mma", inputNames: ["x", "w", "scales", "biases"], outputNames: ["out"],
    source: SmallMQuantizedMatmul.source, header: "#include <metal_simdgroup_matrix>\n")

final class SmallMQuantizedLinear: QuantizedLinear {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hasF64 = [x, weight, scales, biases, bias, globalScale].contains(where: { $0?.dtype == .float64 })
        if hasF64 {
            return Device.defaultDevice() == .gpu ? Device.withDefaultDevice(.cpu) { float64Fallback(x) } : float64Fallback(x)
        }
        let inputs = x.ndim > 0 ? x.dim(-1) : 0
        let outputs = weight.ndim > 0 ? weight.dim(0) : 0
        guard inputs > 0, outputs > 0, x.size > 0 else {
            let outShape = (x.ndim > 0 ? Array(x.shape.dropLast()) : []) + [outputs]
            let zeros = MLXArray.zeros(outShape, dtype: x.dtype)
            return bias.map { zeros + $0 } ?? zeros
        }
        let dts = SmallMQuantizedMatmul.dtypes
        let needsSafeFallback = !dts.contains(x.dtype)
            || (mode == .affine && (!dts.contains(scales.dtype) || biases.map { !dts.contains($0.dtype) } == true))
        if needsSafeFallback {
            return float64Fallback(x)
        }
        guard !training, let biases, SmallMQuantizedMatmul.handles(x, layer: self) else { return super.callAsFunction(x) }
        let result = SmallMQuantizedMatmul.matmul(x, weight: weight, scales: scales, biases: biases, groupSize: groupSize)
        guard let bias else { return result }
        return result + bias
    }

    private static func safeFloatOperand(_ array: MLXArray) -> MLXArray {
        SmallMQuantizedMatmul.dtypes.contains(array.dtype) ? array : array.asType(.float32)
    }

    private func float64Fallback(_ x: MLXArray) -> MLXArray {
        let inputs = x.ndim > 0 ? x.dim(-1) : 0
        let outputs = weight.ndim > 0 ? weight.dim(0) : 0
        let outShape = (x.ndim > 0 ? Array(x.shape.dropLast()) : []) + [outputs]
        guard inputs > 0, outputs > 0, x.size > 0 else {
            let zeros = MLXArray.zeros(outShape, dtype: x.dtype)
            return bias.map { zeros + $0 } ?? zeros
        }
        let mmF64 = [x, scales, biases, globalScale].contains(where: { $0?.dtype == .float64 })
        let xIn = Self.safeFloatOperand(x)
        let sIn = mode == .affine ? Self.safeFloatOperand(scales) : scales
        let bIn = mode == .affine ? biases.map(Self.safeFloatOperand) : nil
        var raw = MLX.quantizedMM(xIn, weight, scales: sIn, biases: bIn, transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        if let gs = globalScale {
            raw = (raw * (gs.asType(.float32) / Float(448 * 6))).asType(raw.dtype)
        }
        let out = mmF64 ? raw.asType(.float64) : (SmallMQuantizedMatmul.dtypes.contains(x.dtype) ? raw : raw.asType(x.dtype))
        return bias.map { out + $0 } ?? out
    }

    static let logger = BasaltLogger(module: "SmallMQuantizedLinear")

    static func installIfEnabled(in model: Module, enabled: Bool = false) {
        let diffusionAuto = model is any DiffusionLanguageModel
            && ProcessInfo.processInfo.environment["BASALT_DISABLE_SMALLM_KERNEL"] != "1"
        guard enabled || SmallMQuantizedMatmul.enabled || diffusionAuto else { return }
        logger.info("Experimental small-M kernel (#208): \(install(in: model)) 4-bit affine linear layers run 2-16 rows through it, the rest stay on stock quantizedMatmul")
    }

    @discardableResult
    static func install(in model: Module) -> Int {
        if let macaq = model as? MACaQQuantizedLinear, SmallMQuantizedMatmul.fits(macaq) {
            macaq.useSmallMKernel = true
            return 1
        }
        var macaqCount = 0
        let updates = model.leafModules().flattened().compactMap { path, module -> (String, Module)? in
            if let macaq = module as? MACaQQuantizedLinear, SmallMQuantizedMatmul.fits(macaq) {
                macaq.useSmallMKernel = true
                macaqCount += 1
                return nil
            }
            guard let layer = module as? QuantizedLinear, type(of: layer) == QuantizedLinear.self,
                  SmallMQuantizedMatmul.fits(layer) else { return nil }
            return (path, SmallMQuantizedLinear(replacing: layer))
        }
        if !updates.isEmpty { model.update(modules: ModuleChildren.unflattened(updates)) }
        return updates.count + macaqCount
    }

    init(replacing layer: QuantizedLinear) {
        super.init(
            weight: layer.weight, bias: layer.bias, scales: layer.scales, biases: layer.biases,
            groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode, globalScale: layer.globalScale)
        freeze()
        train(false)
    }
}
