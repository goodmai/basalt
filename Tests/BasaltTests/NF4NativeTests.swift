import Foundation
import MLX
import Testing
@testable import BasaltInference

@Suite("Native nf4 mode: the QLoRA NormalFloat codebook, exact round trips, the phase (b) layout and the error on Gaussian weights against a pure-Swift reference (#357)")
struct NF4NativeTests {
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

    static func pack(_ weight: MLXArray, groupSize: Int) -> (MLXArray, MLXArray) {
        let (codes, scales, _) = quantized(weight, groupSize: groupSize, bits: 4, mode: .nf4)
        return (codes, scales)
    }

    static func unpack(_ codes: MLXArray, _ scales: MLXArray, groupSize: Int) -> [Float] {
        dequantized(codes, scales: scales, biases: nil, groupSize: groupSize, bits: 4, mode: .nf4).asType(.float32).asArray(Float.self)
    }

    static var codebook: [Float] {
        let codes = MLXArray((0 ..< 32).map { UInt32($0 % 16) }).reshaped(4, 8)
        let packed = (codes * MLXArray((0 ..< 8).map { UInt32(1) << UInt32(4 * $0) })).sum(axis: -1).asType(.uint32).reshaped(1, 4)
        return Array(unpack(packed, MLXArray([Float16(1)]).reshaped(1, 1), groupSize: 32).prefix(16))
    }

    static func referenceRoundTrip(_ values: [Float], groupSize: Int) -> [Float] {
        let levels = referenceCodebook().map(Float.init)
        return stride(from: 0, to: values.count, by: groupSize).flatMap { start in
            let group = values[start ..< start + groupSize]
            let scale = Float(Float16(group.map(\.magnitude).max() ?? 0))
            return group.map { value in
                let normalized = scale > 0 ? value / scale : 0
                let level = levels.min { abs($0 - normalized) < abs($1 - normalized) } ?? 0
                return Float(Float16(level * scale))
            }
        }
    }

    static func phaseBCodes(_ weight: MLXArray, groupSize: Int) -> (MLXArray, MLXArray) {
        let groups = weight.asType(.float32).reshaped(-1, groupSize)
        let scales = abs(groups).max(axis: -1, keepDims: true).asType(.float16)
        let scale = scales.asType(.float32)
        let normalized = groups / which(scale .> 0, scale, 1)
        let levels = referenceCodebook().map(Float.init)
        var codes = MLXArray.zeros(normalized.shape, dtype: .uint32)
        for (low, high) in zip(levels, levels.dropFirst()) { codes = codes + (normalized .> ((low + high) / 2)).asType(.uint32) }
        let packed = (codes.reshaped(-1, 8) * MLXArray((0 ..< 8).map { UInt32(1) << UInt32(4 * $0) })).sum(axis: -1).asType(.uint32)
        let leading = Array(weight.shape.dropLast())
        return (packed.reshaped(leading + [weight.dim(-1) / 8]), scales.reshaped(leading + [weight.dim(-1) / groupSize]))
    }

    @Test("Codes 0…15 decode to the 16 normal quantiles of QLoRA: sorted, an exact zero at index 7, ±1 at the ends, each within 1e-6 of the definition")
    func codebook() {
        let reference = Self.referenceCodebook()
        let levels = Self.codebook
        #expect(levels == levels.sorted() && levels[7] == 0 && levels.first == -1 && levels.last == 1)
        #expect(zip(levels, reference).allSatisfy { abs(Double($0) - $1) < 1e-3 }, "\(levels)")
    }

    @Test("Every level times a power-of-two absmax decodes to itself: eight codes to a uint32, one fp16 scale per group", arguments: [32, 64, 128])
    func exactRoundTrip(groupSize: Int) {
        let levels = (0 ..< groupSize).map { Float(Self.referenceCodebook()[$0 % 16]) * 8 }
        let weight = MLXArray(levels + levels.reversed()).reshaped(2, groupSize).asType(.float16)
        let (codes, scales) = Self.pack(weight, groupSize: groupSize)
        #expect(codes.dtype == .uint32 && codes.shape == [2, groupSize / 8])
        #expect(scales.dtype == .float16 && scales.shape == [2, 1])
        #expect(Self.unpack(codes, scales, groupSize: groupSize) == weight.asType(.float32).asArray(Float.self))
    }

    @Test("Phase (b) checkpoints load unchanged: the native packing equals the phase (b) codec bit for bit", arguments: [32, 64])
    func phaseBLayout(groupSize: Int) {
        let weight = MLXRandom.normal([96, 256], key: MLXRandom.key(9))
        let (codes, scales) = Self.pack(weight, groupSize: groupSize)
        let (oldCodes, oldScales) = Self.phaseBCodes(weight, groupSize: groupSize)
        let differing = (codes .!= oldCodes).asType(.int32).sum().item(Int.self)
        #expect(differing <= codes.size / 1000, "\(differing) of \(codes.size) words differ")
        #expect(scales.asArray(Float16.self) == oldScales.asArray(Float16.self))
    }

    @Test("On Gaussian weights the native codec equals a pure-Swift nearest-level reference, and the relative MSE is about 0.0076 (g32) and 0.0085 (g64)", arguments: [(32, 0.0076), (64, 0.0085)])
    func gaussian(groupSize: Int, expected: Double) {
        let weight = MLXRandom.normal([128, 512], key: MLXRandom.key(7))
        let values = weight.asArray(Float.self)
        let (codes, scales) = Self.pack(weight, groupSize: groupSize)
        let restored = Self.unpack(codes, scales, groupSize: groupSize)
        let reference = Self.referenceRoundTrip(values, groupSize: groupSize)
        let mismatches = zip(restored, reference).filter { abs($0 - $1) > 1e-3 * max(1, abs($1)) }.count
        #expect(mismatches <= values.count / 10_000, "\(mismatches) of \(values.count) differ from the reference")
        let squared = zip(restored, values).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) }
        let mse = squared / values.reduce(0.0) { $0 + Double($1 * $1) }
        #expect(abs(mse - expected) < 0.15 * expected, "relative MSE \(mse)")
    }
}
