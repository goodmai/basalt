import Foundation
import MLX
import Testing
@testable import BasaltInference

@Suite("nf4 small-M and gather kernels against the dequantized matmul, relative max error under 1e-2 (#357 phase d)")
struct NF4KernelParityTests {
    static let tolerance: Float = 1e-2

    static func relativeError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
        (abs(actual.asType(.float32) - expected).max() / abs(expected).max()).item(Float.self)
    }

    static func matmulError(x: MLXArray, codes: MLXArray, scales: MLXArray, groupSize: Int) -> Float {
        let dense = NF4Metal.dequantized(codes, scales: scales, groupSize: groupSize).asType(.float32)
        let kernel = NF4Metal.matmul(x, codes: codes, scales: scales, groupSize: groupSize)
        return relativeError(kernel, matmul(x.asType(.float32), dense.T))
    }

    static func gatherError(x: MLXArray, codes: MLXArray, scales: MLXArray, experts: [UInt32], groupSize: Int) -> Float {
        let kernel = NF4Metal.gatherMatmul(x, codes: codes, scales: scales, rhsIndices: MLXArray(experts), groupSize: groupSize)
        let expected = concatenated(experts.enumerated().map { row, expert in
            let index = Int(expert)
            let dense = NF4Metal.dequantized(codes[index], scales: scales[index], groupSize: groupSize).asType(.float32)
            return matmul(x[row].asType(.float32), dense.T).expandedDimensions(axis: 0)
        }, axis: 0)
        return relativeError(kernel, expected)
    }

    @Test("quantizedMM nf4 at M = 1, 2, 4 and 8: K = 1024 in whole 256-wide blocks and K = 1056 with a partial last block, N a multiple of 8",
          arguments: [(1024, 64, 32), (1056, 72, 32), (1024, 64, 64), (1024, 64, 128)])
    func smallM(inputs: Int, outputs: Int, groupSize: Int) {
        MLXRandom.seed(UInt64(inputs + outputs + groupSize))
        let (codes, scales) = NF4Metal.quantized(MLXRandom.normal([outputs, inputs]) * 0.05, groupSize: groupSize)
        for m in [1, 2, 4, 8] {
            let x = MLXRandom.normal([m, inputs]).asType(.bfloat16)
            let error = Self.matmulError(x: x, codes: codes, scales: scales, groupSize: groupSize)
            #expect(error < Self.tolerance, "K \(inputs) N \(outputs) g\(groupSize) M \(m): \(error)")
        }
    }

    @Test("quantizedMM nf4 on Gemma 4 12B MLP shapes at M = 1, 16 and 512: decode, a verify window and prefill",
          arguments: [(3840, 15360), (15360, 3840)])
    func modelShapes(inputs: Int, outputs: Int) {
        MLXRandom.seed(UInt64(inputs ^ outputs))
        let (codes, scales) = NF4Metal.quantized(MLXRandom.normal([outputs, inputs]) * 0.02, groupSize: 32)
        for m in [1, 16, 512] {
            let x = MLXRandom.normal([m, inputs]).asType(.bfloat16)
            let error = Self.matmulError(x: x, codes: codes, scales: scales, groupSize: 32)
            #expect(error < Self.tolerance, "K \(inputs) N \(outputs) M \(m): \(error)")
        }
    }

    @Test("gatherQuantizedMM nf4 on [topK, 1, K] matches each picked expert's dequantized matmul", arguments: [1024, 1056])
    func gather(inputs: Int) {
        MLXRandom.seed(UInt64(inputs))
        let (codes, scales) = NF4Metal.quantized(MLXRandom.normal([8, 72, inputs]) * 0.05, groupSize: 32)
        let experts: [UInt32] = [5, 0, 3, 7]
        let x = MLXRandom.normal([experts.count, 1, inputs]).asType(.bfloat16)
        let error = Self.gatherError(x: x, codes: codes, scales: scales, experts: experts, groupSize: 32)
        #expect(error < Self.tolerance, "gather K \(inputs): \(error)")
    }
}
