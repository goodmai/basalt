import Foundation
import MLX
import Testing
@testable import BasaltInference

@Suite("NF4 codec: the QLoRA NormalFloat codebook, exact round trips and the error on Gaussian weights against a pure-Swift reference")
struct NF4CodecTests {
    static func normalQuantile(_ p: Double) -> Double {
        var (low, high) = (-10.0, 10.0)
        for _ in 0 ..< 200 {
            let middle = (low + high) / 2
            if 0.5 * erfc(-middle / 2.0.squareRoot()) < p { low = middle } else { high = middle }
        }
        return (low + high) / 2
    }

    static func referenceCodebook() -> [Double] {
        let offset = 0.5 * (1.0 / 32 + 1.0 / 30)
        func steps(_ count: Int) -> [Double] { (0 ..< count).map { 1 - offset + Double($0) * (offset - 0.5) / Double(count) } }
        let positive = steps(8).map(normalQuantile)
        let negative = steps(7).map { -normalQuantile($0) }
        let values = (negative + [0] + positive).sorted()
        let peak = values.map(\.magnitude).max() ?? 1
        return values.map { $0 / peak }
    }

    static func referenceRoundTrip(_ values: [Float], groupSize: Int) -> [Float] {
        stride(from: 0, to: values.count, by: groupSize).flatMap { start in
            let group = values[start ..< start + groupSize]
            let scale = Float(Float16(group.map(\.magnitude).max() ?? 0))
            return group.map { value in
                let normalized = scale > 0 ? value / scale : 0
                let level = NF4Codec.codebook.min { abs($0 - normalized) < abs($1 - normalized) } ?? 0
                return Float(Float16(level * scale))
            }
        }
    }

    @Test("The codebook is the 16 normal quantiles of QLoRA: sorted, an exact zero at index 7, ±1 at the ends, each level within 1e-6 of the quantile definition")
    func codebook() {
        let codebook = NF4CodecTests.referenceCodebook()
        #expect(NF4Codec.codebook.count == 16)
        #expect(NF4Codec.codebook == NF4Codec.codebook.sorted())
        #expect(NF4Codec.codebook[7] == 0 && NF4Codec.codebook.first == -1 && NF4Codec.codebook.last == 1)
        #expect(zip(NF4Codec.codebook, codebook).allSatisfy { abs(Double($0) - $1) < 1e-6 }, "\(codebook)")
    }

    @Test("Every codebook level times a power-of-two absmax decodes to itself, eight 4-bit codes to a uint32 and one fp16 scale per group", arguments: [32, 64])
    func exactRoundTrip(groupSize: Int) {
        let levels = (0 ..< groupSize).map { NF4Codec.codebook[$0 % 16] * 8 }
        let weight = MLXArray(levels + levels.reversed()).reshaped(2, groupSize)
        let (codes, scales) = NF4Codec.quantized(weight, groupSize: groupSize)
        #expect(codes.dtype == .uint32 && codes.shape == [2, groupSize / 8])
        #expect(scales.dtype == .float16 && scales.shape == [2, 1])
        let restored = NF4Codec.dequantized(codes, scales: scales, groupSize: groupSize).asType(.float32).asArray(Float.self)
        #expect(restored == weight.asType(.float16).asType(.float32).asArray(Float.self))
    }

    @Test("On Gaussian weights the MLX codec equals a pure-Swift nearest-level reference, and the relative MSE is about the expected 0.0076 (g32) and 0.0085 (g64)", arguments: [(32, 0.0076), (64, 0.0085)])
    func gaussian(groupSize: Int, expected: Double) {
        let weight = MLXRandom.normal([128, 512], key: MLXRandom.key(7))
        let values = weight.asArray(Float.self)
        let (codes, scales) = NF4Codec.quantized(weight, groupSize: groupSize)
        let restored = NF4Codec.dequantized(codes, scales: scales, groupSize: groupSize).asType(.float32).asArray(Float.self)
        let reference = Self.referenceRoundTrip(values, groupSize: groupSize)
        let mismatches = zip(restored, reference).filter { abs($0 - $1) > 1e-3 * max(1, abs($1)) }.count
        #expect(mismatches <= values.count / 10_000, "\(mismatches) of \(values.count) differ from the reference")
        let squared = zip(restored, values).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) }
        let mse = squared / values.reduce(0.0) { $0 + Double($1 * $1) }
        #expect(abs(mse - expected) < 0.15 * expected, "relative MSE \(mse)")
    }

    @Test("NF4 bytes are half a byte per weight plus two bytes per group: 4.5 bits at g32, 4.25 at g64")
    func bytes() {
        #expect(NF4Codec.bytes(elements: 1024, groupSize: 32) * 8 == 1024 * 9 / 2)
        #expect(NF4Codec.bytes(elements: 1024, groupSize: 64) * 8 == 1024 * 17 / 4)
    }
}
