import Foundation
import MLX
import MLXNN

struct TernaryMatmul: Sendable {
    static let maxRows = 4
    static let geometry = Geometry(ProcessInfo.processInfo.environment["BASALT_TERNARY_MATMUL_GEOMETRY"])
    static var rowsPerSimdgroup: Int { geometry.rows }
    static var wordsPerThread: Int { geometry.words }
    static var simdgroups: Int { geometry.simdgroups }
    static var rowsPerThreadgroup: Int { rowsPerSimdgroup * simdgroups }
    static let tolerance: Float = 1.0 / 256

    struct Geometry: Sendable, Equatable {
        let rows: Int
        let simdgroups: Int
        let words: Int

        init(rows: Int = 4, simdgroups: Int = 2, words: Int = 1) {
            self.rows = rows
            self.simdgroups = simdgroups
            self.words = words
        }

        init(_ spec: String?) {
            let parts = (spec ?? "").split(separator: ",").compactMap { Int($0) }
            guard parts.count == 3, [1, 2, 4, 8].contains(parts[0]), [1, 2, 4, 8].contains(parts[1]), [1, 2, 4, 8].contains(parts[2]) else {
                self.init()
                return
            }
            self.init(rows: parts[0], simdgroups: parts[1], words: parts[2])
        }
    }

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
                float xsum[M][P];
                for (int m = 0; m < M; m++) {
                    const device half4* xs = (const device half4*)(x + m * KW * 16 + base * 16);
                    for (int p = 0; p < P; p++) {
                        float total = 0.0f;
                        for (int i = 0; i < 4; i++) {
                            float4 v = float4(xs[4 * p + i]);
                            total += v.x + v.y + v.z + v.w;
                            xv[m][16 * p + 4 * i] = v.x;
                            xv[m][16 * p + 4 * i + 1] = v.y * 0.25f;
                            xv[m][16 * p + 4 * i + 2] = v.z * 0.0625f;
                            xv[m][16 * p + 4 * i + 3] = v.w * 0.015625f;
                        }
                        xsum[m][p] = total;
                    }
                }
                const int g = base / 8;
                for (int r = 0; r < R; r++) {
                    float part[M];
                    for (int m = 0; m < M; m++) part[m] = 0.0f;
                    for (int p = 0; p < P; p++) {
                        const uchar4 bytes = as_type<uchar4>(wrow[r * KW + base + p]);
                        for (int i = 0; i < 4; i++) {
                            const uint b = bytes[i];
                            const float c0 = float(b & 0x03u), c1 = float(b & 0x0cu), c2 = float(b & 0x30u), c3 = float(b & 0xc0u);
                            for (int m = 0; m < M; m++) {
                                const int k = 16 * p + 4 * i;
                                part[m] = fma(c0, xv[m][k], fma(c1, xv[m][k + 1], fma(c2, xv[m][k + 2], fma(c3, xv[m][k + 3], part[m]))));
                            }
                        }
                        for (int m = 0; m < M; m++) part[m] -= xsum[m][p];
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
