import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("MLXLMCommon SplitFeedForwardPrefill fork contract on Gemma 4 unified")
struct SplitFeedForwardPrefillContractTests {
    @Test("Delegating each dense layer MLP through SplitFeedForwardPrefill reproduces the stock forward bitwise")
    func prefillContractMatchesStock() async throws {
        let raw = try TinyModel.make(.gemma4Unified)
        let model = try #require(raw as? any SplitFeedForwardPrefill & Module)
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        let tokens = MLXArray((0 ..< 12).map { Int32(($0 * 37) % 250 + 1) }).expandedDimensions(axis: 0)
        let stock = raw(tokens, cache: try raw.newCache(parameters: nil)).asType(.float32)
        let delegated = try await model.prefill(text: tokens, cache: try raw.newCache(parameters: nil)) { layer, x in
            let mlp = try #require(modules[model.feedForwardPath(layer: layer)] as? any UnaryLayer)
            return mlp(x)
        }.asType(.float32)
        #expect(!model.denseFeedForwardLayers.isEmpty)
        #expect(arrayEqual(delegated, stock).item(Bool.self))
    }
}
