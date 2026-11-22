import Foundation
import Testing
import Hummingbird
import MLX
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltInference
@testable import BasaltServer

@Suite("Round 5 audit cloud & cluster regression tests (#692)", .serialized)
struct Round5CloudClusterRegressionTests {
    @Test("GenerationResponse.cloudReply preserves sub-second Duration (#692)")
    func cloudReplyPreservesSubSecondDuration() {
        let reply = GenerationResponse.cloudReply(
            text: "ok",
            promptTokens: 10,
            completionTokens: 50,
            finishReason: "stop",
            elapsed: .milliseconds(500)
        )
        #expect(abs(reply.generationTime - 0.5) < 0.01)
        #expect(abs(reply.tokensPerSecond - 100.0) < 2.0)
    }

    @Test("ModelCatalog uses separate input and output token pricing (#692)")
    func modelCatalogSeparateInputOutputPricing() {
        var catalog = ModelCatalog()
        catalog.merge([
            OpenRouterClient.OpenRouterModel(
                id: "test/asymmetric",
                name: "Asymmetric",
                contextLength: 8192,
                pricing: .init(prompt: "0.000003", completion: "0.000015")
            )
        ])
        let cost = catalog.estimateCost(model: "test/asymmetric", promptTokens: 1_000_000, completionTokens: 0)
        #expect(abs(cost - 3.0) < 1e-6)
    }

    @Test("OpenRouterClient.ChatResponse decodes null message content (#692)")
    func openRouterDecodesNullContent() throws {
        let json = """
        {"id":"r1","model":"m","choices":[{"message":{"role":"assistant","content":null},"finish_reason":"tool_calls"}]}
        """
        let decoded = try JSONDecoder().decode(OpenRouterClient.ChatResponse.self, from: Data(json.utf8))
        #expect(decoded.choices.first?.message.content == nil)
    }

    @Test("ClusterFrameAuth rejects equivalent IPv6/IPv4 wildcard bind hosts (#692)")
    func clusterFrameAuthRejectsWildcardEquivalents() {
        for host in ["::0", "0::0", "::ffff:0.0.0.0", "0", "0000:0000:0000:0000:0000:0000:0000:0000"] {
            #expect(throws: ClusterError.self) {
                _ = try ClusterFrameAuth.validateBindHost(host)
            }
        }
    }
}

@Suite("Round 5 audit KV & server regression tests (#693, #694)", .serialized)
struct Round5KVServerRegressionTests {
    @Test("Inkling2WindowKVCache ignores negative trim and clears tensors on full trim (#693)")
    func inkling2WindowCacheNegativeAndFullTrim() {
        let cache = Inkling2WindowKVCache(window: 8)
        let k = MLXArray.ones([1, 2, 4, 8], dtype: .float16)
        let v = MLXArray.ones([1, 2, 4, 8], dtype: .float16)
        _ = cache.update(keys: k, values: v)
        #expect(cache.trim(-3) == 0)
        #expect(cache.offset == 4)
        #expect(cache.trim(4) == 4)
        #expect(cache.offset == 0)
        #expect(cache.innerState().isEmpty)
    }

    @Test("KEqVRowKVCache handles zero keyNormWeight without NaN and clamps negative trim (#693)")
    func kEqVRowCacheZeroNormAndNegativeTrim() throws {
        let layout = try #require(KEqVRowLayout(headDim: 4, partialRotaryFactor: 0.5, kvHeads: 1, globalLayers: 1))
        let normW = MLXArray([Float(0.0), 2.0, 1.0, 1.0])
        let cache = KEqVRowKVCache(layout: layout, keyNormWeight: normW)
        #expect(cache.trim(-5) == 0)
        #expect(cache.offset == 0)
    }

    @Test("EvictingKVCache handles kept == 0 eviction without crashing (#693)")
    func evictingKVCacheZeroKeptEviction() {
        let zones = KVZones(sinkTokens: 1, windowTokens: 2, migrationStep: 1, evictedMiddleFraction: 0.9)
        let cache = EvictingKVCache(zones: zones)
        let k = MLXArray.ones([1, 1, 4, 4], dtype: .float16)
        let v = MLXArray.ones([1, 1, 4, 4], dtype: .float16)
        _ = cache.update(keys: k, values: v)
        #expect(cache.offset == 4)
    }

    @Test("APIKeyAuthenticator publicPaths includes health probes and JSONNestingLimit caps unknown paths (#694)")
    func serverMiddlewarePublicPathsAndBodyLimit() {
        let paths = APIKeyAuthenticator<BasicRequestContext>.publicPaths
        #expect(paths.contains("/v1/health/live"))
        #expect(paths.contains("/v1/health/ready"))
        #expect(JSONNestingLimitMiddleware<BasicRequestContext>.bodyLimit(for: "/v1/unknown") == 4 * 1024 * 1024)
        #expect(JSONNestingLimitMiddleware<BasicRequestContext>.bodyLimit(for: "/v1/chat/completions") == ChatPipeline.bodyLimit)
    }

    @Test("AnthropicMessagesStream does not emit phantom empty text block when stopping during thinking (#694)")
    func anthropicStreamClosingDuringThinking() {
        var stream = AnthropicMessagesStream(model: "claude")
        _ = stream.frames(for: .reasoning("thinking..."))
        let events = stream.closing()
        let joined = events.joined()
        #expect(!joined.contains("\"type\":\"text\""))
        #expect(joined.contains("\"content_block_stop\""))
    }
}
