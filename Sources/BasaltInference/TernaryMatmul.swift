import Foundation
import MLX
import MLXNN

struct TernaryMatmul: Sendable {
    static let maxRows = 4
    static let rowsPerSimdgroup = 4
    static let wordsPerThread = 1
    static let simdgroups = 2
    static let rowsPerThreadgroup = rowsPerSimdgroup * simdgroups
    static let tolerance: Float = 1.0 / 256

    let inputs: Int
    let outputs: Int

    init?(_ layer: QuantizedLinear, environment: [String: String] = ProcessInfo.processInfo.environment) {
        guard environment["BASALT_TERNARY_MATMUL"] == "1", layer.bits == 2, layer.groupSize == 128, layer.mode == .affine,
              layer.bias == nil, layer.globalScale == nil, let biases = layer.biases,
              layer.weight.dtype == .uint32, layer.scales.dtype == .float16, biases.dtype == .float16,
              layer.weight.ndim == 2, layer.scales.ndim == 2, layer.scales.shape == biases.shape else { return nil }
        let (outputs, words) = layer.weight.shape2
        guard outputs > 0, outputs % Self.rowsPerThreadgroup == 0, words % 8 == 0,
              layer.scales.shape == [outputs, words / 8],
              arrayEqual(biases, -layer.scales).item(Bool.self) else { return nil }
        self.inputs = words * 16
        self.outputs = outputs
    }

    func callAsFunction(_ x: MLXArray, weight: MLXArray, scales: MLXArray) -> MLXArray? {
        guard x.dtype == .float16, x.ndim >= 1, x.dim(-1) == inputs else { return nil }
        let leading = Array(x.shape.dropLast())
        let rows = leading.reduce(1, *)
        guard (1...Self.maxRows).contains(rows) else { return nil }
        let threads = 32 * Self.simdgroups
        let y = Self.kernel(
            [x.reshaped(rows, inputs), weight, scales],
            template: [
                ("M", rows), ("R", Self.rowsPerSimdgroup), ("P", Self.wordsPerThread), ("NSG", Self.simdgroups),
                ("KW", inputs / 16), ("N", outputs),
            ],
            grid: (outputs / Self.rowsPerThreadgroup * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [.float16])[0]
        return y.reshaped(leading + [outputs])
    }

    static let kernel = MLXFast.metalKernel(
        name: "basalt_ternary_qmv", inputNames: ["x", "w", "scales"], outputNames: ["y"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const int row0 = (int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup)) * R;
            constexpr int G = KW / 8;
            float acc[M][R];
            for (int m = 0; m < M; m++) for (int r = 0; r < R; r++) acc[m][r] = 0.0f;
            const device uint32_t* wrow = w + row0 * KW;
            const device half* srow = scales + row0 * G;
            for (int base = int(lane) * P; base < KW; base += 32 * P) {
                float xv[M][16 * P];
                for (int m = 0; m < M; m++) {
                    const device half4* xs = (const device half4*)(x + m * KW * 16 + base * 16);
                    for (int i = 0; i < 4 * P; i++) {
                        float4 v = float4(xs[i]);
                        xv[m][4 * i] = v.x; xv[m][4 * i + 1] = v.y; xv[m][4 * i + 2] = v.z; xv[m][4 * i + 3] = v.w;
                    }
                }
                const int g = base / 8;
                for (int r = 0; r < R; r++) {
                    float part[M];
                    for (int m = 0; m < M; m++) part[m] = 0.0f;
                    for (int p = 0; p < P; p++) {
                        const uint32_t word = wrow[r * KW + base + p];
                        for (int j = 0; j < 16; j++) {
                            const float c = float((word >> (2 * j)) & 3u) - 1.0f;
                            for (int m = 0; m < M; m++) part[m] = fma(c, xv[m][16 * p + j], part[m]);
                        }
                    }
                    const float s = float(srow[r * G + g]);
                    for (int m = 0; m < M; m++) acc[m][r] = fma(s, part[m], acc[m][r]);
                }
            }
            for (int m = 0; m < M; m++) {
                for (int r = 0; r < R; r++) {
                    const float v = simd_sum(acc[m][r]);
                    if (lane == 0) y[m * N + row0 + r] = half(v);
                }
            }
            """)
}
