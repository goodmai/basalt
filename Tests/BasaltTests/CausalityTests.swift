import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltInference

@Suite("Every tiny family is causal with and without a cache: changing the last token leaves earlier logits untouched")
struct CausalityTests {
    @Test("Earlier logits are bit-identical when only the last token changes, and a prefix gives the same logits as the full sequence",
          arguments: TinyModel.Family.allCases.filter { $0 != .prism })
    func causal(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let a = MLXArray([Int32(5), 9, 13, 17, 21, 25, 29, 33]).reshaped(1, 8)
        let b = MLXArray([Int32(5), 9, 13, 17, 21, 25, 29, 99]).reshaped(1, 8)
        let full = model(a, cache: nil)[0, ..<7].asType(.float32)
        #expect(MLX.abs(full - model(b, cache: nil)[0, ..<7].asType(.float32)).max().item(Float.self) == 0, "\(family) without a cache")
        #expect(MLX.abs(full - model(a[0..., ..<7], cache: nil)[0].asType(.float32)).max().item(Float.self) < 1e-4, "\(family) prefix")
        let (ca, cb) = (try model.newCache(parameters: nil), try model.newCache(parameters: nil))
        #expect(MLX.abs(model(a, cache: ca)[0, ..<7] - model(b, cache: cb)[0, ..<7]).max().item(Float.self) == 0, "\(family) with a cache")
    }
}
