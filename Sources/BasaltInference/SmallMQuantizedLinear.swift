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
    static let rows = 2 ... 8
    static let groupSizes: Set<Int> = [32, 64, 128]
    static let blockSize = 512
    static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]

    static var enabled: Bool { ProcessInfo.processInfo.environment[environmentKey] == "1" }

    static func fits(_ layer: QuantizedLinear) -> Bool {
        let (outputs, inputs) = layer.shape
        return layer.mode == .affine && layer.bits == 4 && layer.biases != nil && layer.globalScale == nil
            && groupSizes.contains(layer.groupSize) && layer.weight.ndim == 2
            && inputs % blockSize == 0 && outputs % 8 == 0
    }

    static func handles(_ x: MLXArray, layer: QuantizedLinear) -> Bool {
        let inputs = x.dim(-1)
        return inputs > 0 && rows.contains(x.size / inputs) && dtypes.contains(x.dtype)
            && layer.scales.dtype == x.dtype && layer.biases?.dtype == x.dtype
    }

    static func resultsPerSimdgroup(rows: Int) -> Int {
        rows <= 4 ? 4 : 2
    }

    static func matmul(_ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int) -> MLXArray {
        let inputs = x.dim(-1)
        let outputs = weight.dim(0)
        let rows = x.size / inputs
        let perSimdgroup = resultsPerSimdgroup(rows: rows)
        let result = kernel(
            [x.reshaped(rows, inputs), weight, scales, biases],
            template: [("T", x.dtype), ("M", rows), ("R", perSimdgroup), ("G", groupSize)],
            grid: (32, outputs / perSimdgroup, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows, outputs]],
            outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    static let source = """
        constexpr int BLOCK = 512;
        constexpr int VPT = 16;
        const int K = int(x_shape[1]);
        const int N = int(w_shape[0]);
        const int KW = K / 4;
        const int KG = K / G;
        const int lane = int(thread_index_in_simdgroup);
        const int outRow = int(thread_position_in_grid.y) * R;
        const device uint16_t* ws = (const device uint16_t*)w + outRow * KW + lane * 4;
        const device T* sl = scales + outRow * KG + lane / (G / VPT);
        const device T* bl = biases + outRow * KG + lane / (G / VPT);
        const device T* xl = x + lane * VPT;
        float result[M][R];
        for (int m = 0; m < M; m++) {
            for (int r = 0; r < R; r++) {
                result[m][r] = 0.0f;
            }
        }
        for (int k = 0; k < K; k += BLOCK) {
            float sum[M];
            float accum[M][R];
            for (int m = 0; m < M; m++) {
                sum[m] = 0.0f;
                for (int r = 0; r < R; r++) {
                    accum[m][r] = 0.0f;
                }
            }
            for (int i = 0; i < VPT / 4; i++) {
                float xq[M][4];
                for (int m = 0; m < M; m++) {
                    const device T* xi = xl + m * K + 4 * i;
                    sum[m] += float(xi[0]) + float(xi[1]) + float(xi[2]) + float(xi[3]);
                    xq[m][0] = xi[0];
                    xq[m][1] = xi[1] / 16.0f;
                    xq[m][2] = xi[2] / 256.0f;
                    xq[m][3] = xi[3] / 4096.0f;
                }
                for (int r = 0; r < R; r++) {
                    const uint16_t wv = ws[r * KW + i];
                    for (int m = 0; m < M; m++) {
                        accum[m][r] += (xq[m][0] * (wv & 0x000f) + xq[m][1] * (wv & 0x00f0)
                            + xq[m][2] * (wv & 0x0f00) + xq[m][3] * (wv & 0xf000));
                    }
                }
            }
            for (int r = 0; r < R; r++) {
                const float s = sl[r * KG];
                const float b = bl[r * KG];
                for (int m = 0; m < M; m++) {
                    result[m][r] += s * accum[m][r] + sum[m] * b;
                }
            }
            ws += BLOCK / 4;
            sl += BLOCK / G;
            bl += BLOCK / G;
            xl += BLOCK;
        }
        for (int m = 0; m < M; m++) {
            for (int r = 0; r < R; r++) {
                const float v = simd_sum(result[m][r]);
                if (lane == 0) {
                    out[m * N + outRow + r] = static_cast<T>(v);
                }
            }
        }
        """
}

private let kernel = MLXFast.metalKernel(
    name: "smallm_affine4_qmv", inputNames: ["x", "w", "scales", "biases"], outputNames: ["out"],
    source: SmallMQuantizedMatmul.source)

final class SmallMQuantizedLinear: QuantizedLinear {
    init(replacing layer: QuantizedLinear) {
        super.init(
            weight: layer.weight, bias: layer.bias, scales: layer.scales, biases: layer.biases,
            groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode)
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
        logger.info("Experimental small-M kernel (#208): \(install(in: model)) 4-bit affine linear layers run 2-8 rows through it, the rest stay on stock quantizedMatmul")
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
