import Foundation
import Hummingbird
import HummingbirdTesting
import MLX
import MLXLMCommon
import NIOCore
import Testing
@testable import BasaltBench
@testable import BasaltCore
@testable import BasaltInference

@Suite("BasaltBench metric, comparator, runner, and dataset regressions (A-B5, A-B6, A-B7, A-B8, A-B12)")
struct BenchMetricsDefectRegressionTests {
    static func makeSpec() -> ComposeSpec {
        ComposeSpec(name: "bench-defects", models: ["chat": ComposeSpec.Model(source: .local("org/m"))], answer: "chat")
    }

    static func makeProvenance() -> BasaltProvenance {
        BasaltProvenance(gitCommit: "a", composeSHA256: "b", chip: "M4", unifiedMemoryGB: 24, timestampISO8601: "2026-11-25T00:00:00Z")
    }

    @Test("A-B5: basalt:throughput streams for real TTFT, excludes prefill from decode_tps, and never fabricates 0.15 * totalMs")
    func throughputDoesNotFabricateTTFTAndExcludesPrefill() async throws {
        let endpoint = BenchmarkEndpoint.mock { req in
            if req.path.hasSuffix("/stats") {
                return BenchmarkHTTPResponse(statusCode: 404, body: Data())
            }
            let reqJSON = try #require(req.body.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] })
            #expect(reqJSON["stream"] as? Bool == true)
            let payload: [String: Any] = [
                "choices": [["message": ["role": "assistant", "content": "ok"]]],
                "usage": ["prompt_tokens": 100, "completion_tokens": 20],
            ]
            let body = try JSONSerialization.data(withJSONObject: payload)
            return BenchmarkHTTPResponse(body: body, ttftMs: nil, totalMs: 500.0)
        }
        let task = BenchmarkTaskSpec(adapter: "basalt:throughput", target: "model:chat", params: ["iterations": .int(1)])
        let ctx = BenchmarkTaskContext(spec: Self.makeSpec(), suiteName: "s", task: task, defaults: BenchmarkDefaultsSpec(), endpoint: endpoint)
        let out = try await BasaltThroughputAdapter().run(context: ctx)
        #expect(out.metrics["ttft_ms"] == nil)
        #expect(out.samples.first?.ttftMs == nil)

        let timedEndpoint = BenchmarkEndpoint.mock { req in
            if req.path.hasSuffix("/stats") { return BenchmarkHTTPResponse(statusCode: 404, body: Data()) }
            let payload: [String: Any] = [
                "choices": [["message": ["role": "assistant", "content": "ok"]]],
                "usage": ["prompt_tokens": 100, "completion_tokens": 20],
            ]
            return BenchmarkHTTPResponse(body: try JSONSerialization.data(withJSONObject: payload), ttftMs: 800.0, totalMs: 1000.0)
        }
        let timedCtx = BenchmarkTaskContext(spec: Self.makeSpec(), suiteName: "s", task: task, defaults: BenchmarkDefaultsSpec(), endpoint: timedEndpoint)
        let timedOut = try await BasaltThroughputAdapter().run(context: timedCtx)
        #expect(timedOut.metrics["ttft_ms"] == 800.0)
        #expect(abs((timedOut.metrics["decode_tps"] ?? 0) - 100.0) < 1e-6)
    }

    @Test("A-B5: BenchmarkEndpoint.live streaming TTFT fires on first reasoning_content or content delta")
    func streamingTTFTStartsAtReasoningOrContentDelta() async throws {
        let router = Router()
        router.post("/v1/chat/completions") { _, _ in
            let sse = """
            data: {"choices":[{"delta":{"reasoning_content":"step 1"}}]}

            data: {"choices":[{"delta":{"content":"hello "}}]}

            data: {"choices":[{"delta":{"content":"world"}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}

            data: [DONE]

            """
            return Response(status: .ok, headers: [.contentType: "text/event-stream"], body: .init(byteBuffer: ByteBuffer(string: sse)))
        }
        router.post("/v1/reasoning-only") { _, _ in
            let sse = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"only reasoning\"}}]}\n\ndata: [DONE]\n\n"
            return Response(status: .ok, headers: [.contentType: "text/event-stream"], body: .init(byteBuffer: ByteBuffer(string: sse)))
        }
        try await Application(router: router).test(.live) { client in
            let port = try #require(client.port)
            let live = BenchmarkEndpoint.live(baseURL: try #require(URL(string: "http://localhost:\(port)")))
            let (json, reply) = try await live.postJSON("/v1/chat/completions", payload: ["model": "m", "stream": true])
            #expect(reply.ttftMs != nil && (reply.ttftMs ?? 0) > 0)
            #expect(BenchmarkStats.extractChatContent(json) == "hello world")
            #expect(BenchmarkStats.extractUsage(json).completion == 2)
            let (_, reasoningReply) = try await live.postJSON("/v1/reasoning-only", payload: ["model": "m", "stream": true])
            #expect(reasoningReply.ttftMs != nil && (reasoningReply.ttftMs ?? 0) > 0)
        }
    }

    @Test("A-B6: basalt:perplexity scores held-out text via /v1/score mean_nll and fails when mean_nll is absent")
    func perplexityScoresHeldOutTextViaScoreEndpoint() async throws {
        let scoring = BenchmarkEndpoint.mock { req in
            #expect(req.path == "/v1/score")
            let body = try #require(req.body.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] })
            #expect(body["completion_text"] as? String == "held-out text")
            return BenchmarkHTTPResponse(body: try JSONSerialization.data(withJSONObject: ["mean_nll": 2.0]), totalMs: 12.0)
        }
        let task = BenchmarkTaskSpec(adapter: "basalt:perplexity", target: "model:chat", params: ["text": .string("held-out text")])
        let ctx = BenchmarkTaskContext(spec: Self.makeSpec(), suiteName: "s", task: task, defaults: BenchmarkDefaultsSpec(), endpoint: scoring)
        let out = try await BasaltPerplexityAdapter().run(context: ctx)
        #expect(abs((out.metrics["ppl"] ?? 0) - exp(2.0)) < 1e-6)

        let missing = BenchmarkEndpoint.mock { _ in
            BenchmarkHTTPResponse(body: try JSONSerialization.data(withJSONObject: ["positions": []]), totalMs: 10.0)
        }
        let missingCtx = BenchmarkTaskContext(spec: Self.makeSpec(), suiteName: "s", task: task, defaults: BenchmarkDefaultsSpec(), endpoint: missing)
        await #expect(throws: BenchmarkError.self) {
            _ = try await BasaltPerplexityAdapter().run(context: missingCtx)
        }

        let model = try TinyModel.make(.qwen3)
        let tokenizer = ByteTokenizer()
        let text = "abcdef"
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let prompt = try MLXInferenceEngine.labPrompt(messages: [["role": "user", "content": "hi"]], tokenizer)
        let fullSeq = prompt + ids
        let fullLogits = model(MLXArray(fullSeq.dropLast().map(Int32.init))[.newAxis], cache: nil)[0].asType(.float32)
        let slice = fullLogits[(prompt.count - 1)...]
        let logProbs = slice - logSumExp(slice, axis: -1, keepDims: true)
        let targets = MLXArray(ids.map(Int32.init))[0..., .newAxis]
        let refNLL = -Double(takeAlong(logProbs, targets, axis: -1).mean().item(Float.self))

        let report = try MLXInferenceEngine.score(
            ScoreRequest(messages: [["role": "user", "content": "hi"]], completion: ids, topLogprobs: 2),
            model: model,
            tokenizer: tokenizer
        )
        let textReport = try MLXInferenceEngine.score(
            ScoreRequest(messages: [["role": "user", "content": "hi"]], completion: [], topLogprobs: 2, completionText: text),
            model: model,
            tokenizer: tokenizer
        )
        let chunked = try MLXInferenceEngine.scorePositions(prompt: prompt, completion: ids, k: 2, model: model, chunk: 2)
        #expect(abs((report.meanNLL ?? 0) - refNLL) < 1e-4)
        #expect(abs((textReport.meanNLL ?? 0) - refNLL) < 1e-4)
        #expect(abs(chunked.meanNLL - refNLL) < 1e-4)
    }
}

extension BenchMetricsDefectRegressionTests {

    @Test("A-B7: BenchmarkComparator treats metricx24 as lower-is-better")
    func metricX24IsLowerIsBetter() {
        let prov = Self.makeProvenance()
        let base = BenchmarkRunArtifact(
            manifest: BenchmarkManifest(runID: "base", composeName: "c", suite: "s", endpoint: "http://127.0.0.1"),
            provenance: prov,
            scorecard: BenchmarkScorecard(metrics: ["metricx24": 2.0], passed: true),
            samples: []
        )
        let worse = BenchmarkRunArtifact(
            manifest: BenchmarkManifest(runID: "worse", composeName: "c", suite: "s", endpoint: "http://127.0.0.1"),
            provenance: prov,
            scorecard: BenchmarkScorecard(metrics: ["metricx24": 4.0], passed: true),
            samples: []
        )
        let regressed = BenchmarkComparator.compare(baseline: base, candidate: worse)
        #expect(regressed.deltas.first?.regressed == true && regressed.deltas.first?.improved == false)

        let better = BenchmarkRunArtifact(
            manifest: BenchmarkManifest(runID: "better", composeName: "c", suite: "s", endpoint: "http://127.0.0.1"),
            provenance: prov,
            scorecard: BenchmarkScorecard(metrics: ["metricx24": 1.0], passed: true),
            samples: []
        )
        let improved = BenchmarkComparator.compare(baseline: base, candidate: better)
        #expect(improved.deltas.first?.improved == true && improved.deltas.first?.regressed == false)
    }

    @Test("A-B8 & A-B12: multi-task suites preserve per-task metrics, gateless failing suites fail, explicit missing datasets throw, and basalt:mom fails on empty route")
    func multiTaskMetricsGatelessFailAndExplicitDatasetValidation() async throws {
        var spec = Self.makeSpec()
        spec.benchmarks = ComposeBenchmarksSpec(
            defaults: BenchmarkDefaultsSpec(gpuLock: false),
            suites: [
                "multi": BenchmarkSuiteSpec(
                    tasks: [
                        BenchmarkTaskSpec(adapter: "basalt:mom", target: "pipeline"),
                        BenchmarkTaskSpec(adapter: "basalt:system-one", target: "pipeline"),
                    ]
                ),
            ]
        )
        let failingEndpoint = BenchmarkEndpoint.mock { req in
            if req.path == "/v1/systemone" {
                let body = try JSONSerialization.data(withJSONObject: ["answers": ["route": ["choice": "wrong", "confidence": 0.9]]])
                return BenchmarkHTTPResponse(body: body, totalMs: 5.0)
            }
            let body = try JSONSerialization.data(withJSONObject: ["answers": [:], "system2": ["text": "answered without route"]])
            return BenchmarkHTTPResponse(body: body, totalMs: 5.0)
        }
        let req = BenchmarkSuiteRunRequest(spec: spec, composeYAML: "", suiteName: "multi", endpoint: failingEndpoint, runID: "r1")
        let artifact = try await BenchmarkRunner().run(req)
        #expect(!artifact.scorecard.passed)
        #expect(artifact.scorecard.metrics["task[0].accuracy"] == 0.0)
        #expect(artifact.scorecard.metrics["task[1].accuracy"] == 0.0)

        let mixedEndpoint = BenchmarkEndpoint.mock { req in
            if req.path == "/v1/systemone" {
                let body = try JSONSerialization.data(withJSONObject: ["answers": ["route": ["choice": "wrong", "confidence": 0.9]]])
                return BenchmarkHTTPResponse(body: body, totalMs: 30.0)
            }
            let body = try JSONSerialization.data(withJSONObject: ["answers": ["route": ["choice": "code", "confidence": 0.9]], "system2": ["text": "ok"]])
            return BenchmarkHTTPResponse(body: body, totalMs: 10.0)
        }
        let mixed = try await BenchmarkRunner().run(
            BenchmarkSuiteRunRequest(spec: spec, composeYAML: "", suiteName: "multi", endpoint: mixedEndpoint, runID: "r2")
        )
        #expect(mixed.scorecard.metrics["task[0].accuracy"] == 1.0)
        #expect(mixed.scorecard.metrics["task[1].accuracy"] == 0.0)
        #expect(abs((mixed.scorecard.metrics["accuracy"] ?? -1) - (1.0 / 3.0)) < 1e-9)
        #expect(abs((mixed.scorecard.metrics["p50_ms"] ?? -1) - 30.0) < 1e-9)
    }

    @Test("A-B8 & A-B12: multi-task perplexity averages flat ppl across tasks and missing explicit datasets throw")
    func multiTaskPerplexityAverageAndMissingDatasetsThrow() async throws {
        var pplSpec = Self.makeSpec()
        pplSpec.benchmarks = ComposeBenchmarksSpec(
            defaults: BenchmarkDefaultsSpec(gpuLock: false),
            suites: [
                "ppl-multi": BenchmarkSuiteSpec(tasks: [
                    BenchmarkTaskSpec(adapter: "basalt:perplexity", target: "model:chat", params: ["text": .string("a")]),
                    BenchmarkTaskSpec(adapter: "basalt:perplexity", target: "model:chat", params: ["text": .string("b")]),
                ]),
            ]
        )
        let pplEndpoint = BenchmarkEndpoint.mock { req in
            let body = try #require(req.body.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] })
            let nll = (body["completion_text"] as? String) == "a" ? 1.0 : 2.0
            return BenchmarkHTTPResponse(body: try JSONSerialization.data(withJSONObject: ["mean_nll": nll]), totalMs: 10.0)
        }
        let pplRun = try await BenchmarkRunner().run(
            BenchmarkSuiteRunRequest(spec: pplSpec, composeYAML: "", suiteName: "ppl-multi", endpoint: pplEndpoint, runID: "r3")
        )
        #expect(abs((pplRun.scorecard.metrics["task[0].ppl"] ?? 0) - exp(1.0)) < 1e-6)
        #expect(abs((pplRun.scorecard.metrics["task[1].ppl"] ?? 0) - exp(2.0)) < 1e-6)
        #expect(abs((pplRun.scorecard.metrics["ppl"] ?? 0) - (exp(1.0) + exp(2.0)) / 2.0) < 1e-6)

        let missingPath = "/nonexistent-\(UUID().uuidString).jsonl"
        let missingTask = BenchmarkTaskSpec(adapter: "basalt:system-one", dataset: missingPath, target: "pipeline")
        let ctx = BenchmarkTaskContext(spec: pplSpec, suiteName: "ppl-multi", task: missingTask, defaults: BenchmarkDefaultsSpec(), endpoint: pplEndpoint)
        await #expect(throws: BenchmarkError.self) {
            _ = try await BasaltSystemOneAdapter().run(context: ctx)
        }
        await #expect(throws: BenchmarkError.self) {
            _ = try await TranslationBenchmarkAdapter().run(context: ctx)
        }
        await #expect(throws: BenchmarkError.self) {
            _ = try await ExternalHarnessAdapter(id: "arc-agi", summary: "s").run(context: ctx)
        }
    }
}
