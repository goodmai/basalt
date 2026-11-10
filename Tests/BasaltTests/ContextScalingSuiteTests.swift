import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("PerformanceBenchmark --suite fae-scaling: rows, model context and export (#242)")
struct ContextScalingSuiteTests {
    func response(prompt: Int, cached: Int, ttft: Double, tps: Double, peak: Int) -> GenerationResponse {
        GenerationResponse(
            generatedText: "", promptTokens: prompt, cachedPromptTokens: cached, completionTokens: 128, tokensPerSecond: tps,
            generationTime: 1, timeToFirstToken: ttft, memory: .init(peakBytes: peak, activeBytes: 0, cacheBytes: 0), finishReason: .length)
    }

    @Test("Rows average the runs, take prefill over the uncached prompt and render one Markdown line per context")
    func rowArithmetic() {
        let row = ContextScalingRow(context: 4096, runs: [
            response(prompt: 4100, cached: 0, ttft: 2, tps: 40, peak: 10 << 20),
            response(prompt: 4100, cached: 100, ttft: 4, tps: 60, peak: 12 << 20),
        ])
        #expect(row.promptTokens == 4100)
        #expect(row.cachedPromptTokens == 100)
        #expect(row.ttftSeconds == 3)
        #expect(row.prefillTokensPerSecond == (2050 + 1000) / 2)
        #expect(row.decodeTokensPerSecond == 50)
        #expect(row.completionTokens == 128)
        #expect(row.peakMemoryMB == 12)
        #expect(ContextScalingSuite.markdown([row]).last == "| 4096 | 4100 | 3.00 | 1525 | 50.0 | 128 | 12 |")
        #expect(ContextScalingRow(context: 1024, runs: [response(prompt: 1000, cached: 0, ttft: 0, tps: 9, peak: 0)]).prefillTokensPerSecond == 0)
    }

    @Test("The model's context comes from text_config, then the top level, and is unknown without a config")
    func modelContext() throws {
        let directory = FileManager.default.temporaryDirectory.appending(component: "fae-scaling-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(ContextScalingSuite.modelContext(at: directory) == nil)
        let config = directory.appending(component: "config.json")
        try Data(#"{"max_position_embeddings": 32768, "text_config": {"max_position_embeddings": 262144}}"#.utf8).write(to: config)
        #expect(ContextScalingSuite.modelContext(at: directory) == 262144)
        try Data(#"{"max_position_embeddings": 32768}"#.utf8).write(to: config)
        #expect(ContextScalingSuite.modelContext(at: directory) == 32768)
    }

    @Test("The report exports rows and the contexts it did not measure as JSON and Markdown")
    func report() throws {
        let row = ContextScalingRow(context: 1024, runs: [response(prompt: 1030, cached: 0, ttft: 0.5, tps: 30, peak: 8 << 30)])
        let report = ContextScalingReport(model: "org/model", chip: "Apple M4 Pro", fitLevel: "Good", rows: [row], notMeasured: ["131072 (fit check)"])
        let decoded = try JSONDecoder().decode(ContextScalingReport.self, from: JSONEncoder().encode(report))
        #expect(decoded.rows == [row] && decoded.notMeasured == ["131072 (fit check)"])
        let lines = report.markdown.split(separator: "\n").map(String.init)
        #expect(lines.first == "## fae-scaling: org/model")
        #expect(lines.contains("| 1024 | 1030 | 0.50 | 2060 | 30.0 | 128 | 8192 |"))
        #expect(lines.last == "Not measured: 131072 (fit check)")
        #expect(ContextScalingSuite.faeContexts == [1024, 4096, 16384, 32768, 65536, 131072])
        #expect(ContextScalingSuite(runs: 0, tokens: 64, temperature: 0.7).runs == 1)
    }
}
