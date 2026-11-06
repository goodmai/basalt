import Foundation
import MLX
import MLXNN
import BasaltCore

extension MLXInferenceEngine {
    public static func enablesSmallMKernel() {
        setenv(SmallMQuantizedMatmul.environmentKey, "1", 1)
    }
}

enum SmallMQuantizedMatmul {
    static let environmentKey = "BASALT_SMALLM_KERNEL"
    static let rows = 4 ... 8
    static let groupSizes: Set<Int> = [32, 64, 128]
    static let chunk = 64
    static let splitK = 4
    static let tilesPerSimdgroup = 4
    static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]

    static var enabled: Bool { ProcessInfo.processInfo.environment[environmentKey] == "1" }

    static func fits(_ layer: QuantizedLinear) -> Bool {
        let (outputs, inputs) = layer.shape
        return layer.mode == .affine && layer.bits == 4 && layer.biases != nil && layer.globalScale == nil
            && groupSizes.contains(layer.groupSize) && layer.weight.ndim == 2
            && inputs % chunk == 0 && outputs % (8 * tilesPerSimdgroup) == 0
    }

    static func handles(_ x: MLXArray, layer: QuantizedLinear) -> Bool {
        let inputs = x.dim(-1)
        return inputs > 0 && rows.contains(x.size / inputs) && dtypes.contains(x.dtype)
            && layer.scales.dtype == x.dtype && layer.biases?.dtype == x.dtype
    }

    static func matmul(_ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int) -> MLXArray {
        let inputs = x.dim(-1)
        let outputs = weight.dim(0)
        let rows = x.size / inputs
        let result = kernel(
            [x.reshaped(rows, inputs), weight, scales, biases],
            template: [("T", x.dtype), ("M", rows), ("G", groupSize), ("S", splitK), ("NT", tilesPerSimdgroup)],
            grid: (32, splitK * outputs / (8 * tilesPerSimdgroup), 1),
            threadGroup: (32, splitK, 1),
            outputShapes: [[rows, outputs]],
            outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    static let source = """
        constexpr int CH = 64;
        const int K = int(x_shape[1]);
        const int N = int(w_shape[0]);
        const int KB = K / 2;
        const int KG = K / G;
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int n0 = int(threadgroup_position_in_grid.y) * (8 * NT);
        const int qid = lane / 4;
        const int sm = (qid & 4) + (lane / 2) % 4;
        const int sn = (qid & 2) * 2 + (lane % 2) * 2;
        const device uint8_t* wb = (const device uint8_t*)w;
        simdgroup_float8x8 acc[NT];
        for (int j = 0; j < NT; j++) {
            acc[j] = simdgroup_float8x8(0.0f);
        }
        for (int c = sg * CH; c < K; c += S * CH) {
            const int kx = c + 16 * (sm / 2) + (sm % 2);
            const int kw = c + 16 * (sn / 2);
            float xa[8];
            float xb[8];
            for (int t = 0; t < 8; t++) {
                xa[t] = sn < M ? float(x[sn * K + kx + 2 * t]) : 0.0f;
                xb[t] = sn + 1 < M ? float(x[(sn + 1) * K + kx + 2 * t]) : 0.0f;
            }
            for (int j = 0; j < NT; j++) {
                const int n = n0 + 8 * j + sm;
                const uint2 q = *((const device uint2*)(wb + n * KB + kw / 2));
                const float s = scales[n * KG + kw / G];
                const float b = biases[n * KG + kw / G];
                for (int t = 0; t < 8; t++) {
                    const uint word = t < 4 ? q.x : q.y;
                    const uint shift = 8u * uint(t % 4);
                    simdgroup_float8x8 wt;
                    wt.thread_elements()[0] = s * float((word >> shift) & 15u) + b;
                    wt.thread_elements()[1] = s * float((word >> (shift + 4u)) & 15u) + b;
                    simdgroup_float8x8 xt;
                    xt.thread_elements()[0] = xa[t];
                    xt.thread_elements()[1] = xb[t];
                    simdgroup_multiply_accumulate(acc[j], wt, xt, acc[j]);
                }
            }
        }
        threadgroup float partial[S * NT * 64];
        for (int j = 0; j < NT; j++) {
            partial[(sg * NT + j) * 64 + sm * 8 + sn] = acc[j].thread_elements()[0];
            partial[(sg * NT + j) * 64 + sm * 8 + sn + 1] = acc[j].thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (int j = 0; j < NT; j++) {
                for (int e = 0; e < 2; e++) {
                    float v = 0.0f;
                    for (int p = 0; p < S; p++) {
                        v += partial[(p * NT + j) * 64 + sm * 8 + sn + e];
                    }
                    if (sn + e < M) {
                        out[(sn + e) * N + n0 + 8 * j + sm] = static_cast<T>(v);
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
    init(replacing layer: QuantizedLinear) {
        super.init(
            weight: layer.weight, bias: layer.bias, scales: layer.scales, biases: layer.biases,
            groupSize: layer.groupSize, bits: layer.bits)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let biases, SmallMQuantizedMatmul.handles(x, layer: self) else { return super.callAsFunction(x) }
        let result = SmallMQuantizedMatmul.matmul(x, weight: weight, scales: scales, biases: biases, groupSize: groupSize)
        guard let bias else { return result }
        return result + bias
    }

    static let logger = BasaltLogger(module: "SmallMQuantizedLinear")

    static func installIfEnabled(in model: Module) {
        guard SmallMQuantizedMatmul.enabled else { return }
        logger.info("Experimental small-M kernel (#208): \(install(in: model)) 4-bit affine linear layers run 4-8 rows through it, the rest stay on stock quantizedMatmul")
    }

    @discardableResult
    static func install(in model: Module) -> Int {
        let updates = model.leafModules().flattened().compactMap { path, module -> (String, Module)? in
            guard let layer = module as? QuantizedLinear, type(of: layer) == QuantizedLinear.self,
                  SmallMQuantizedMatmul.fits(layer) else { return nil }
            return (path, SmallMQuantizedLinear(replacing: layer))
        }
        if !updates.isEmpty { model.update(modules: ModuleChildren.unflattened(updates)) }
        return updates.count
    }
}
