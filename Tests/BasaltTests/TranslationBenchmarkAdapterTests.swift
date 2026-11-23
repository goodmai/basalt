import BasaltBench
import Foundation
import Testing
@testable import BasaltCLI
@testable import BasaltCore

@Suite("Translation benchmark adapter: chrF++, COMET, MetricX-24, lite & full modes (#726)")
struct TranslationBenchmarkAdapterTests {
    @Test("TranslationMetrics computes exact chrF++, techFidelity, proxy COMET/MetricX-24, and judge JSON")
    func metricsMathAndJudgeParsing() {
        let jaRef = "`basalt fit` を実行してモデルを確認します。"
        #expect(abs(TranslationMetrics.chrfPP(hypothesis: jaRef, reference: jaRef) - 100.0) < 1e-6)
        #expect(TranslationMetrics.chrfPP(hypothesis: "zzzz", reference: jaRef) < 5.0)
        #expect(TranslationMetrics.chrfPP(hypothesis: "", reference: jaRef) == 0.0)

        let src = "Run `basalt serve --rest` and open https://bas-alt.vercel.app for docs."
        #expect(TranslationMetrics.techFidelity(source: src, hypothesis: "Запустите `basalt serve --rest` и откройте https://bas-alt.vercel.app") == 100.0)
        #expect(TranslationMetrics.techFidelity(source: src, hypothesis: "Запустите сервер без ссылки") == 0.0)

        let proxy = TranslationMetrics.scoreProxy(source: src, hypothesis: jaRef, reference: jaRef)
        #expect(proxy.chrfPP > 99.0)
        #expect(proxy.comet >= 0.7)
        #expect(proxy.metricX24 <= 7.5)

        let judged = TranslationMetrics.parseJudgeJSON("Result: {\"comet\": 0.91, \"metricx24\": 2.15}")
        #expect(judged?.comet == 0.91)
        #expect(judged?.metricX24 == 2.15)
        #expect(TranslationMetrics.parseJudgeJSON("not json") == nil)
    }

    @Test("TranslationBenchmarkAdapter runs lite mode with built-in probes and full mode with JSONL + LLM QE judge")
    func liteAndFullModesEndToEnd() async throws {
        let yamlURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "docs/examples/compose/translate.yaml")
        let yaml = try String(contentsOf: yamlURL, encoding: .utf8)
        let spec = try ComposeFile.parse(yaml, file: yamlURL.path, context: ComposeContext(memoryGB: 24, chip: "Apple M4 Pro").with(base: yamlURL.path))
        #expect(spec.benchmarks?.suites["lite"] != nil)
        #expect(spec.benchmarks?.suites["full"] != nil)

        let endpoint = BenchmarkEndpoint(baseURL: URL(string: "http://127.0.0.1:8080")!) { req in
            let obj = (req.body.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
            let msgs = (obj["messages"] as? [[String: Any]]) ?? []
            let prompt = (msgs.last?["content"] as? String) ?? ""
            let replyText: String
            if prompt.contains("Score the translation quality strictly as JSON") {
                replyText = "{\"comet\": 0.88, \"metricx24\": 2.6}"
            } else if prompt.contains("translate.yaml") {
                let sys = (msgs.first?["content"] as? String) ?? ""
                if sys.contains("Japanese") {
                    replyText = TranslationBenchmarkAdapter.builtinProbes[0].reference
                } else if sys.contains("Korean") {
                    replyText = TranslationBenchmarkAdapter.builtinProbes[1].reference
                } else if sys.contains("Indonesian") {
                    replyText = TranslationBenchmarkAdapter.builtinProbes[2].reference
                } else {
                    replyText = TranslationBenchmarkAdapter.builtinProbes[3].reference
                }
            } else {
                replyText = "Запустите `basalt fit`."
            }
            let body = try JSONSerialization.data(withJSONObject: [
                "choices": [["message": ["role": "assistant", "content": replyText]]],
                "usage": ["prompt_tokens": 20, "completion_tokens": 18],
            ])
            return BenchmarkHTTPResponse(body: body, ttftMs: 10, totalMs: 25)
        }

        let adapter = try #require(BenchmarkAdapterRegistry.standard.adapter(for: "basalt:translation"))

        let liteTask = try #require(spec.benchmarks?.suites["lite"]?.tasks.first)
        let liteOut = try await adapter.run(context: BenchmarkTaskContext(
            spec: spec, suiteName: "lite", task: liteTask, defaults: BenchmarkDefaultsSpec(), endpoint: endpoint
        ))
        #expect(liteOut.samples.count == 4)
        #expect((liteOut.metrics["chrf_pp"] ?? 0) > 95.0)
        #expect((liteOut.metrics["tech_fidelity"] ?? 0) == 100.0)
        #expect((liteOut.metrics["comet"] ?? 0) > 0.9)
        #expect((liteOut.metrics["metricx24"] ?? 25) < 2.5)

        let tmp = FileManager.default.temporaryDirectory.appending(path: "tr-full-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let line = try String(decoding: JSONEncoder().encode(TranslationBenchmarkAdapter.builtinProbes[0]), as: UTF8.self)
        try Data((line + "\n").utf8).write(to: tmp)

        var fullTask = try #require(spec.benchmarks?.suites["full"]?.tasks.first)
        fullTask.dataset = tmp.path
        let fullOut = try await adapter.run(context: BenchmarkTaskContext(
            spec: spec, suiteName: "full", task: fullTask, defaults: BenchmarkDefaultsSpec(), endpoint: endpoint
        ))
        #expect(fullOut.samples.count == 1)
        #expect(fullOut.metrics["comet"] == 0.88)
        #expect(fullOut.metrics["metricx24"] == 2.6)
    }
}
