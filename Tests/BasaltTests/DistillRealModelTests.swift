import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("distill on real models: a real teacher behind an API, multi-step word problems, student accuracy and teacher NLL before and after (#393)", .serialized)
struct DistillRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let models = NF4Run.models("BASALT_E2E_DISTILL_MODELS")
    static let gatePrompts = ["Explain what a hash table is in two sentences.", "Write a haiku about autumn.", "What is the capital of Australia?", "Summarize why the sky looks blue."]

    static func problems(count: Int, seed: UInt64) -> [DistillExample] {
        var generator = SplitMix64(seed: seed)
        let items = ["boxes of pens", "crates of apples", "packs of cards", "bags of rice"]
        return (0 ..< count).map { _ in
            let item = items[Int.random(in: 0 ..< items.count, using: &generator)]
            let (a, b, c, d) = (Int.random(in: 12...97, using: &generator), Int.random(in: 3...19, using: &generator), Int.random(in: 40...400, using: &generator), Int.random(in: 2...9, using: &generator))
            let prompt = "A shop receives \(a) \(item) with \(b) items in each. It sells \(c) items, then splits the rest equally into \(d) displays and keeps the remainder in storage. How many items are in storage? Show brief working, then end with a final line 'Answer: <number>'."
            return DistillExample(prompt: prompt, answer: String((a * b - c) % d))
        }
    }

    static func accuracy(_ engine: MLXInferenceEngine, _ examples: [DistillExample]) async throws -> Float {
        var correct = 0
        for example in examples {
            let reply = try await engine.generate(request: GenerationRequest(prompt: example.prompt, maxTokens: 384, temperature: 0)).generatedText
            if TrainingTask(prompt: example.prompt, answer: example.answer ?? "").accepts(reply) { correct += 1 }
        }
        return Float(correct) / Float(max(examples.count, 1))
    }

    static func request(prompts: URL) throws -> DistillRequest {
        let url = try #require(env["BASALT_E2E_DISTILL_TEACHER_URL"]), model = try #require(env["BASALT_E2E_DISTILL_TEACHER_MODEL"])
        var teacher = TeacherSpec(url: url, model: model)
        (teacher.maxTokens, teacher.concurrency, teacher.reasoningEffort, teacher.temperature) = (1024, 8, env["BASALT_E2E_DISTILL_EFFORT"], 0.7)
        var request = DistillRequest(prompts: prompts.path, teacher: teacher, gate: GateRequest(heldOut: gatePrompts))
        (request.filter, request.iterations, request.batchSize, request.rank, request.learningRate, request.maxSeqLen, request.heldOutFraction) = (.verify, 300, 2, 16, 1e-4, 1024, 0.1)
        return request
    }

    @Test("600 seeded word problems: the teacher answers once (cached across students), verified answers train a LoRA, accuracy and NLL are logged before and after",
          .enabled(if: !models.isEmpty && env["BASALT_E2E_DISTILL_TEACHER_URL"] != nil, "set BASALT_E2E=1, BASALT_E2E_DISTILL_MODELS, BASALT_E2E_DISTILL_TEACHER_URL/_MODEL, BASALT_TEACHER_URL and BASALT_TEACHER_API_KEY"),
          arguments: models)
    func distill(entry: String) async throws {
        let path = NF4Run.path(entry)
        let log = try await ModelRunLog("distill", modelPath: path)
        defer { log.releaseGPU() }
        let examples = Self.problems(count: 600, seed: 393)
        let prompts = FileManager.default.temporaryDirectory.appending(path: "distill-problems-393.jsonl")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let lines = try examples.map { String(decoding: try encoder.encode($0), as: UTF8.self) }
        try Data(lines.joined(separator: "\n").utf8).write(to: prompts)
        var request = try Self.request(prompts: prompts)
        request.output = FileManager.default.temporaryDirectory.appending(path: "distill-adapter-\(UUID().uuidString)").path
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let heldOut = Array(examples.suffix(DistillCore.heldOutCount(examples.count, fraction: request.heldOutFraction)))
        let before = try await log.step("accuracy before") { try await Self.accuracy(engine, heldOut) }
        let report = try await log.step("distill") { try await engine.distill(request) }
        log.record("teacher", "\(report.teacherCalls) calls, \(report.cached) cached, \(report.teacherInputTokens) prompt + \(report.teacherOutputTokens) completion tokens, stopped by \(report.stoppedBy.rawValue)")
        log.record("data", "kept \(report.kept), held out \(report.heldOut), rejected \(report.rejected.sorted { $0.key < $1.key })")
        log.record("training", "losses \(report.losses.map { String(format: "%.3f", $0) }.joined(separator: " "))")
        let gate = report.verdict?.report.map { "passed \($0.passed), \($0)" } ?? "none"
        log.record("gate", "overlay \(report.verdict?.overlay?.id.description ?? "none"), \(gate)")
        if report.verdict?.overlay == nil, let directory = report.adapterDirectory {
            _ = try await engine.applyAdapter(.directory(directory), gate: nil)
        }
        let after = try await log.step("accuracy after") { try await Self.accuracy(engine, heldOut) }
        log.record("RESULT", "held-out accuracy \(before) -> \(after) on \(heldOut.count), teacher NLL \(report.teacherNLLBefore ?? .nan) -> \(report.teacherNLLAfter ?? .nan)")
        #expect((report.teacherNLLAfter ?? .infinity) < (report.teacherNLLBefore ?? 0))
    }
}
