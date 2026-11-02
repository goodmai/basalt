import Foundation
import Testing
import MLX
import MLXNN
@testable import BasaltInference

enum TernaryPack {
    struct Packed {
        var weight: MLXArray
        var scales: MLXArray
        var dense: MLXArray

        var biases: MLXArray { -scales }

        func layer(bias: MLXArray? = nil, biases: MLXArray? = nil, bits: Int = 2) -> QuantizedLinear {
            QuantizedLinear(
                weight: weight, bias: bias, scales: scales, biases: biases ?? self.biases,
                groupSize: 128, bits: bits, mode: .affine)
        }
    }

    static func make(outputs: Int, inputs: Int, seed: UInt64) -> Packed {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let codes = MLXRandom.randInt(0 ..< 3, [outputs, inputs]).asType(.uint32)
            let scales = MLXRandom.uniform(low: 0.005, high: 0.05, [outputs, inputs / 128]).asType(.float16)
            let shifts = MLXArray((0..<16).map { UInt32(2 * $0) })
            let weight = (codes.reshaped(outputs, inputs / 16, 16) << shifts).sum(axis: -1).asType(.uint32)
            let perWeight = repeated(scales.asType(.float32), count: 128, axis: 1)
            let dense = (codes.asType(.float32) - 1) * perWeight
            eval(weight, scales, dense)
            return Packed(weight: weight, scales: scales, dense: dense)
        }
    }

    static func stock(_ x: MLXArray, _ packed: Packed) -> MLXArray {
        quantizedMM(x, packed.weight, scales: packed.scales, biases: packed.biases, transpose: true, groupSize: 128, bits: 2)
    }
}

@Suite("Ternary 2-bit matmul kernel (Prism packs, biases == -scales)")
struct TernaryMatmulTests {
    @Test("The test packing is MLX's: dequantized weights reproduce the ternary float matrix")
    func packingMatchesMLX() {
        let packed = TernaryPack.make(outputs: 16, inputs: 256, seed: 1)
        let dequantized = dequantized(packed.weight, scales: packed.scales, biases: packed.biases, groupSize: 128, bits: 2, dtype: .float32)
        #expect(abs(dequantized - packed.dense).max().item(Float.self) == 0)
    }

    @Test(
        "The kernel matches quantizedMatmul for 1 to 4 rows",
        arguments: [(8, 128), (64, 640), (256, 1152), (1024, 2048), (4096, 512)])
    func matchesStock(outputs: Int, inputs: Int) throws {
        let packed = TernaryPack.make(outputs: outputs, inputs: inputs, seed: UInt64(outputs + inputs))
        let plan = try #require(TernaryMatmul(packed.layer(), environment: ["BASALT_TERNARY_MATMUL": "1"]))
        for rows in 1...4 {
            let x = MLXRandom.normal([rows, inputs], key: MLXRandom.key(UInt64(rows))).asType(.float16)
            let expected = TernaryPack.stock(x, packed)
            let actual = try #require(plan(x, weight: packed.weight, scales: packed.scales))
            #expect(actual.shape == [rows, outputs] && actual.dtype == .float16)
            let spread = abs(expected).max().item(Float.self)
            #expect(abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self) <= spread * TernaryMatmul.tolerance)
        }
    }

    @Test("Leading batch dimensions are flattened into rows and restored")
    func batchedInput() throws {
        let packed = TernaryPack.make(outputs: 32, inputs: 256, seed: 3)
        let plan = try #require(TernaryMatmul(packed.layer(), environment: ["BASALT_TERNARY_MATMUL": "1"]))
        let x = MLXRandom.normal([1, 3, 256], key: MLXRandom.key(5)).asType(.float16)
        let actual = try #require(plan(x, weight: packed.weight, scales: packed.scales))
        #expect(actual.shape == [1, 3, 32])
        let expected = TernaryPack.stock(x, packed)
        #expect(abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self) <= abs(expected).max().item(Float.self) * TernaryMatmul.tolerance)
    }

    @Test("Layers and activations outside the contract are declined")
    func refusals() throws {
        let packed = TernaryPack.make(outputs: 16, inputs: 256, seed: 4)
        #expect(TernaryMatmul(packed.layer(biases: packed.biases + 0.001), environment: ["BASALT_TERNARY_MATMUL": "1"]) == nil)
        #expect(TernaryMatmul(packed.layer(bias: MLXArray.zeros([16], dtype: .float16)), environment: ["BASALT_TERNARY_MATMUL": "1"]) == nil)
        #expect(TernaryMatmul(packed.layer(), environment: ["BASALT_TERNARY_MATMUL": "0"]) == nil)
        #expect(TernaryMatmul(packed.layer(), environment: [:]) == nil)
        let four = QuantizedLinear(weight: MLXRandom.normal([16, 256]), bias: nil, groupSize: 128, bits: 4)
        #expect(TernaryMatmul(four, environment: ["BASALT_TERNARY_MATMUL": "1"]) == nil)
        #expect(TernaryMatmul(TernaryPack.make(outputs: 12, inputs: 256, seed: 6).layer(), environment: ["BASALT_TERNARY_MATMUL": "1"]) == nil)

        let plan = try #require(TernaryMatmul(packed.layer(), environment: ["BASALT_TERNARY_MATMUL": "1"]))
        #expect(plan(MLXArray.zeros([5, 256], dtype: .float16), weight: packed.weight, scales: packed.scales) == nil)
        #expect(plan(MLXArray.zeros([1, 256], dtype: .float32), weight: packed.weight, scales: packed.scales) == nil)
        #expect(plan(MLXArray.zeros([1, 128], dtype: .float16), weight: packed.weight, scales: packed.scales) == nil)
    }
}

@Suite("Ternary matmul kernel vs quantizedMM on Qwen3.8-27B shapes", .serialized)
struct TernaryMatmulBenchmark {
    static let enabled = ProcessInfo.processInfo.environment["BASALT_BENCH"] != nil

    static func medianMicros(_ runs: Int, _ body: () -> MLXArray) -> Double {
        let clock = ContinuousClock()
        for _ in 0..<5 { eval(body()) }
        let samples = (0..<runs).map { _ in
            let elapsed = clock.measure { eval(body()) }
            return Double(elapsed.components.attoseconds) / 1e12 + Double(elapsed.components.seconds) * 1e6
        }.sorted()
        return samples[samples.count / 2]
    }

    @Test("Median wall-clock per call, kernel vs stock, M = 1, 2, 4", .enabled(if: enabled, "set BASALT_BENCH=1"))
    func shapes() throws {
        let shapes = [(5120, 17408), (5120, 10240), (5120, 6144), (5120, 12288), (17408, 5120), (6144, 5120), (5120, 248320)]
        print("K      N       M  stock µs  kernel µs  speedup")
        for (inputs, outputs) in shapes {
            let weight = MLXRandom.randInt(0 ..< Int(Int32.max), [outputs, inputs / 16]).asType(.uint32)
            let scales = MLXRandom.uniform(low: 0.005, high: 0.05, [outputs, inputs / 128]).asType(.float16)
            let biases = -scales
            eval(weight, scales, biases)
            let plan = try #require(TernaryMatmul(QuantizedLinear(weight: weight, bias: nil, scales: scales, biases: biases, groupSize: 128, bits: 2, mode: .affine), environment: ["BASALT_TERNARY_MATMUL": "1"]))
            for rows in [1, 2, 4] {
                let x = MLXRandom.normal([rows, inputs]).asType(.float16)
                eval(x)
                let stock = Self.medianMicros(50) { quantizedMM(x, weight, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2) }
                let kernel = Self.medianMicros(50) { plan(x, weight: weight, scales: scales)! }
                print(String(format: "%-6d %-7d %d  %8.1f  %9.1f  %6.2fx", inputs, outputs, rows, stock, kernel, stock / kernel))
            }
        }
    }
}

@Suite("Ternary matmul geometry override (#229)")
struct TernaryMatmulGeometryTests {
    @Test("BASALT_TERNARY_MATMUL_GEOMETRY parses rows,simdgroups,words and falls back on anything else")
    func parse() {
        #expect(TernaryMatmul.Geometry("2,8,4") == TernaryMatmul.Geometry(rows: 2, simdgroups: 8, words: 4))
        #expect(TernaryMatmul.Geometry(nil) == TernaryMatmul.Geometry())
        #expect(TernaryMatmul.Geometry("3,2,1") == TernaryMatmul.Geometry())
        #expect(TernaryMatmul.Geometry("4,2") == TernaryMatmul.Geometry())
    }
}
