import Testing
import MLX
import MLXLMCommon

@Suite("The fused sigmoid-bias router kernel is bit-identical to the reference chain (#221)")
struct FusedRouterTests {
    @Test("Indices and weights match bitwise on 10k random rows, ties included",
          arguments: [(32, 4, true), (64, 4, true), (64, 8, false), (256, 8, true)])
    func fusedMatchesReference(experts: Int, k: Int, normalize: Bool) throws {
        MLXRandom.seed(UInt64(experts * 100 + k))
        let rows = 10_000
        let gates = (MLXRandom.normal([rows, experts]) * 4).asType(.bfloat16)
        let tied = concatenated([gates[..<(rows / 2)], (gates[(rows / 2)...] * 2).round() / 2], axis: 0)
        let bias = (MLXRandom.normal([experts]) * 0.1).round() / 10
        let reference = sigmoidBiasRouterTopK(tied, bias: bias, k: k, normalize: normalize, fused: false)
        let fused = sigmoidBiasRouterTopK(tied, bias: bias, k: k, normalize: normalize, fused: true)
        eval(reference.indices, reference.weights, fused.indices, fused.weights)
        #expect(fused.indices.dtype == .uint32)
        #expect(arrayEqual(fused.indices, reference.indices.asType(.uint32)).item(Bool.self))
        #expect(arrayEqual(fused.weights, reference.weights).item(Bool.self))
    }
}
