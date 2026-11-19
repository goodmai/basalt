import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import NIOCore
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Per-Token X-Ray Attribution & Hallucination Risk (#612)", .serialized)
struct TokenAttributionXrayTests {
    @Test("Hallucination scorer distinguishes grounded, uncertain, and high-risk late-onset tokens")
    func hallucinationScorerStates() {
        let grounded = TokenHallucinationScorer.evaluate(
            probability: 0.96, entropy: 0.12, margin: 0.92, onsetLayer: 1, totalLayers: 8
        )
        #expect(grounded.state == .grounded)
        #expect(grounded.risk < 0.15)

        let uncertain = TokenHallucinationScorer.evaluate(
            probability: 0.42, entropy: 1.4, margin: 0.25, onsetLayer: 4, totalLayers: 8
        )
        #expect(uncertain.state == .uncertain)

        let highRisk = TokenHallucinationScorer.evaluate(
            probability: 0.12, entropy: 3.6, margin: 0.03, onsetLayer: 7, totalLayers: 8
        )
        #expect(highRisk.state == .highRisk)
        #expect(highRisk.risk >= 0.62)

        let summary = AttributedTokenTraceSummary(tokens: [
            OutputTokenTelemetry(index: 0, token: "ok", tokenId: 10, logprob: -0.04, entropy: 0.1, margin: 0.9),
            OutputTokenTelemetry(index: 1, token: "?", tokenId: 11, logprob: -2.40, entropy: 3.5, margin: 0.02),
        ])
        #expect(summary.highRiskTokenIndices == [1])
        #expect(summary.minProbability < summary.meanProbability)
    }

    @Test("POST /v1/xray/tokens attributes probabilities, heads, MoE experts, and logit-lens onset")
    func xrayTokensMoERoute() async throws {
        let engine = try await TinyModel.engine(.lfm2Moe)
        let anatomy = try await engine.anatomy()
        let server = RESTServer(
            orchestrator: ModelOrchestratorActor(engine: engine),
            config: try ServerConfig(modelPath: "tiny-lfm2-moe")
        )
        let app = Application(router: server.makeRouter())
        var xrayReq = AttributedTokenTraceRequest(
            prompt: "Explain sparse mixture of experts.", maxTokens: 4, temperature: 0.0, topLogprobs: 3, topHeads: 2
        )
        xrayReq.includeAttentionEntropy = true
        let bodyData = try JSONEncoder().encode(xrayReq)

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/xray/tokens", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: bodyData)
            ) { res in
                #expect(res.status == .ok)
                let report = try JSONDecoder().decode(AttributedTokenTraceReport.self, from: Data(res.body.readableBytesView))
                try Self.assertValidReport(report, layerCount: anatomy.layers.count)
            }
        }
    }

    private static func assertValidReport(_ report: AttributedTokenTraceReport, layerCount: Int) throws {
        #expect(report.promptTokens > 0)
        #expect(report.completionTokens == report.tokens.count)
        #expect(!report.tokens.isEmpty && report.tokens.count <= 4)
        #expect((0.0 ... 1.0).contains(report.summary.meanProbability))
        for tok in report.tokens {
            #expect(abs(tok.probability - Foundation::exp(tok.logprob)) < 1e-4)
            #expect(tok.entropy >= 0 && (0.0 ... 1.0).contains(tok.margin))
            #expect(tok.topLogprobs.count == 3)
            let layers = try #require(tok.layers)
            #expect(layers.count == layerCount)
            #expect(layers.contains { !$0.experts.isEmpty } && layers.contains { !$0.topHeads.isEmpty })
            #expect(layers.last?.lensTopTokenId == tok.tokenId)
            let onset = try #require(tok.onsetLayer)
            #expect((0 ..< layerCount).contains(onset))
        }
    }
}

@Suite("OpenAI & llama.cpp Chat/Generate Logprobs Routes (#612)", .serialized)
struct ChatAndGenerateLogprobsTests {
    private static func postJSON(
        _ client: some TestClientProtocol, uri: String, body: [String: Any]
    ) async throws -> (HTTPResponse.Status, Data) {
        let payload = try JSONSerialization.data(withJSONObject: body)
        var captured: (HTTPResponse.Status, Data) = (.internalServerError, Data())
        try await client.execute(
            uri: uri, method: .post,
            headers: [.contentType: "application/json"],
            body: ByteBuffer(bytes: payload)
        ) { res in
            captured = (res.status, Data(res.body.readableBytesView))
        }
        return captured
    }

    @Test("Chat completions supports logprobs, n_probs, xray_trace, and SSE streaming")
    func chatCompletionLogprobModes() async throws {
        let engine = try await TinyModel.engine(.qwen3Small)
        let server = RESTServer(orchestrator: ModelOrchestratorActor(engine: engine), config: try ServerConfig(modelPath: "tiny-qwen3"))
        let app = Application(router: server.makeRouter())

        try await app.test(.router) { client in
            let (s1, d1) = try await Self.postJSON(client, uri: "/v1/chat/completions", body: [
                "messages": [["role": "user", "content": "Hello"]], "max_tokens": 3, "temperature": 0.0, "logprobs": true, "top_logprobs": 4,
            ])
            #expect(s1 == .ok)
            let r1 = try JSONDecoder().decode(OpenAIController.ChatCompletionResponse.self, from: d1)
            let c1 = try #require(r1.choices.first?.logprobs?.content)
            #expect(!c1.isEmpty && c1[0].topLogprobs.count == 4 && c1[0].layers == nil)

            let (s2, d2) = try await Self.postJSON(client, uri: "/v1/chat/completions", body: [
                "messages": [["role": "user", "content": "Hello"]], "max_tokens": 2, "temperature": 0.0, "n_probs": 3,
            ])
            #expect(s2 == .ok)
            let r2 = try JSONDecoder().decode(OpenAIController.ChatCompletionResponse.self, from: d2)
            let c2 = try #require(r2.choices.first?.logprobs?.content)
            #expect(!c2.isEmpty && c2[0].topLogprobs.count == 3)

            let (s3, d3) = try await Self.postJSON(client, uri: "/v1/chat/completions", body: [
                "messages": [["role": "user", "content": "Hello"]], "max_tokens": 2, "temperature": 0.0, "xray_trace": true, "top_logprobs": 2,
            ])
            #expect(s3 == .ok)
            let r3 = try JSONDecoder().decode(OpenAIController.ChatCompletionResponse.self, from: d3)
            let lp3 = try #require(r3.choices.first?.logprobs)
            let layers3 = try #require(lp3.content?.first?.layers)
            #expect(!layers3.isEmpty && lp3.summary != nil)

            let (s4, d4) = try await Self.postJSON(client, uri: "/v1/chat/completions", body: [
                "messages": [["role": "user", "content": "Hello"]], "max_tokens": 2, "temperature": 0.0, "stream": true, "logprobs": true,
            ])
            #expect(s4 == .ok)
            let sse = String(decoding: d4, as: UTF8.self)
            #expect(sse.contains("\"logprobs\"") && sse.contains("\"hallucination_risk\"") && sse.contains("[DONE]"))
        }
    }

    @Test("Generate endpoint populates tokenTelemetry and rejects invalid logprob bounds")
    func generateAndBoundsValidation() async throws {
        let engine = try await TinyModel.engine(.qwen3Small)
        let server = RESTServer(orchestrator: ModelOrchestratorActor(engine: engine), config: try ServerConfig(modelPath: "tiny-qwen3"))
        let app = Application(router: server.makeRouter())

        try await app.test(.router) { client in
            var genReq = GenerationRequest(prompt: "Hello", maxTokens: 2, temperature: 0.0)
            genReq.logprobs = true
            genReq.topLogprobs = 2
            try await client.execute(
                uri: "/v1/generate", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: try JSONEncoder().encode(genReq))
            ) { res in
                #expect(res.status == .ok)
                let genRes = try JSONDecoder().decode(GenerationResponse.self, from: Data(res.body.readableBytesView))
                let tel = try #require(genRes.tokenTelemetry)
                #expect(!tel.isEmpty && tel[0].topLogprobs.count == 2)
            }

            let (bad1, _) = try await Self.postJSON(client, uri: "/v1/chat/completions", body: [
                "messages": [["role": "user", "content": "Hi"]], "n_probs": 25,
            ])
            #expect(bad1 == .badRequest)

            let (bad2, _) = try await Self.postJSON(client, uri: "/api/v1/xray/tokens", body: [
                "prompt": "   ", "max_tokens": 4,
            ])
            #expect(bad2 == .badRequest)
        }
    }
}
