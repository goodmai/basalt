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

    static func fits(bits: Int, mode: QuantizationMode, groupSize: Int, inputs: Int, hasBiases: Bool) -> Bool {
        bits == 3 && mode == .affine && hasBiases && groupSizes.contains(groupSize) && inputs > 0 && inputs % 16 == 0
    }

    static func handles(_ x: MLXArray, scales: MLXArray, biases: MLXArray) -> Bool {
        x.ndim >= 2 && x.dim(-2) == 1 && dtypes.contains(x.dtype) && scales.dtype == x.dtype && biases.dtype == x.dtype
    }

    static func handlesSingleRow(_ x: MLXArray, scales: MLXArray, biases: MLXArray) -> Bool {
        handles(x, scales: scales, biases: biases) && x.size == x.dim(-1)
    }

    static func handlesGather(_ x: MLXArray, indices: MLXArray, scales: MLXArray, biases: MLXArray) -> Bool {
        handles(x, scales: scales, biases: biases) && x.ndim == indices.ndim + 2
            && Array(x.shape.prefix(indices.ndim)) == indices.shape
    }

    static func matmul(_ x: MLXArray, indices: MLXArray?, weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int) -> MLXArray {
        let inputs = x.dim(-1)
        let tasks = x.size / inputs
        let outputs = weight.dim(-2)
        let index = indices?.asType(.int32).reshaped(tasks) ?? MLXArray.zeros([1], dtype: .int32)
        let simdgroups = (outputs + rowsPerSimdgroup - 1) / rowsPerSimdgroup
        let result = kernel(
            [x.reshaped(tasks, inputs), index, weight, scales, biases, MLXArray([Int32(outputs), Int32(inputs)])],
            template: [("T", x.dtype), ("G", groupSize), ("GATHER", indices != nil)],
            grid: (32, 2 * ((simdgroups + 1) / 2), tasks),
            threadGroup: (32, 2, 1),
            outputShapes: [[tasks, outputs]],
            outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    static let header = """
        template <typename TX>
        inline float basalt_q3_load(const device TX* x, thread float* xt) {
            float sum = 0.0f;
            for (int j = 0; j < 16; j++) { xt[j] = float(x[j]); sum += xt[j]; }
            xt[1] *= 0.125f; xt[2] *= 1.0f / 64.0f; xt[3] *= 1.0f / 512.0f; xt[4] *= 1.0f / 4096.0f;
            xt[6] *= 0.25f; xt[7] *= 1.0f / 32.0f; xt[8] *= 1.0f / 256.0f; xt[9] *= 1.0f / 2048.0f;
            xt[11] *= 0.5f; xt[12] *= 1.0f / 16.0f; xt[13] *= 1.0f / 128.0f; xt[14] *= 1.0f / 1024.0f; xt[15] *= 1.0f / 8192.0f;
            return sum;
        }
        inline float basalt_q3_dot(const device uchar* w, thread const float* x) {
            const device ushort* h = (const device ushort*)w;
            const uint h0 = h[0], h1 = h[1], h2 = h[2];
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
        const int e = GATHER ? int(idx[task]) : 0;
        const int KB = K * 3 / 8;
        const int KG = K / G;
        const int last = min(3, N - 1 - n0);
        const size_t row0 = size_t(e) * size_t(N) + size_t(n0);
        const device uchar* wb = (const device uchar*)w + row0 * size_t(KB) + lane * 6;
        const device T* sb = scales + row0 * size_t(KG);
        const device T* bb = biases + row0 * size_t(KG);
        const device T* xr = x + size_t(task) * size_t(K) + lane * 16;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float xt[16];
        for (int k = 0; k < K; k += 512) {
            if (k + lane * 16 < K) {
                const float sum = basalt_q3_load(xr + k, xt);
                const int g = (k + lane * 16) / G;
                for (int r = 0; r < 4; r++) {
                    const int src = min(r, last);
                    const float d = basalt_q3_dot(wb + size_t(src) * size_t(KB) + (k * 3 / 8), xt);
                    acc[r] += float(sb[src * KG + g]) * d + float(bb[src * KG + g]) * sum;
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
               fits(bits: layer.bits, mode: layer.mode, groupSize: layer.groupSize, inputs: layer.shape.1, hasBiases: layer.biases != nil) {
                layer.useQ3Kernel = enabled
                count += 1
            } else if let layer = module as? MACaQSwitchLinear,
                      fits(bits: layer.bits, mode: layer.mode, groupSize: layer.groupSize, inputs: layer.scales.dim(-1) * layer.groupSize, hasBiases: layer.biases != nil) {
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
