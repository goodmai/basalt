import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import Synchronization
@testable import BasaltCore
@testable import BasaltInference

@Suite("distill: a teacher behind an API answers, a LoRA learns its completions only, cached and budgeted (#393)", .serialized)
struct DistillTests {
    static let canary = "sk-canary-0123456789abcdef"

    static func scratch() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "distill-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func openAI(_ text: String, finish: String = "stop") -> Data {
        Data(#"{"choices": [{"message": {"content": "\#(text)"}, "finish_reason": "\#(finish)"}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}"#.utf8)
    }

    static func anthropic(_ text: String) -> Data {
        Data(#"{"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "\#(text)"}], "stop_reason": "end_turn", "usage": {"input_tokens": 7, "output_tokens": 3}}"#.utf8)
    }

    static func prompts(_ count: Int, in directory: URL) throws -> String {
        let url = directory.appending(path: "prompts.jsonl")
        let lines = (0 ..< count).map { #"{"prompt": "Name the colour number \#($0).", "answer": "blue"}"# }
        try Data(lines.joined(separator: "\n").utf8).write(to: url)
        return url.path
    }

    static func request(_ host: String, prompts: String, output: URL) -> DistillRequest {
        var request = DistillRequest(prompts: prompts, teacher: TeacherSpec(url: "https://\(host)/v1", model: "teacher"), gate: GateRequest(heldOut: ["Hello."], maxKL: 100, maxPerplexityRise: 100))
        (request.iterations, request.rank, request.learningRate, request.maxSeqLen, request.heldOutFraction) = (30, 4, 3e-3, 256, 0.25)
        request.output = output.path
        return request
    }

    static func baseWeights(_ container: ModelContainer) async -> [String: [Float]] {
        await container.perform { context in
            Dictionary(uniqueKeysWithValues: context.model.parameters().flattened().map { ($0.0, $0.1.asType(.float32).asArray(Float.self)) })
        }
    }

    @Test("The mask covers exactly the completion targets, and the loss is the mean cross-entropy over them")
    func completionOnlyMask() throws {
        let row = DistillRow(tokens: [5, 6, 7, 8, 9, 10], promptCount: 3)
        let (inputs, targets, mask) = MLXInferenceEngine.distillBatch([row, DistillRow(tokens: [1, 2, 3, 4], promptCount: 2)])
        #expect(inputs.shape == [2, 5])
        #expect(targets.asArray(Int32.self) == [6, 7, 8, 9, 10, 2, 3, 4, 0, 0])
        #expect(mask.asArray(Float.self) == [0, 0, 1, 1, 1, 0, 1, 1, 0, 0])
        let model = try TinyModel.make(.qwen3)
        let logits = model(inputs, cache: nil).asType(.float32)
        let ce: [Float] = crossEntropy(logits: logits, targets: targets, reduction: .none).asArray(Float.self)
        let weights: [Float] = mask.asArray(Float.self)
        var expected: Float = 0
        for (value, weight) in zip(ce, weights) where weight == 1 { expected += value / 5 }
        #expect(abs(MLXInferenceEngine.maskedLoss(model, inputs, targets, mask).item(Float.self) - expected) < 1e-4)
    }

    @Test("OpenAI and Anthropic replies decode with text, reasoning, truncation and usage; the key goes in the provider's header")
    func providers() async throws {
        let openAIHost = StubURLProtocol.host { _, _ in (200, [:], Self.openAI("hi", finish: "length")) }
        let anthropicHost = StubURLProtocol.host { _, _ in (200, [:], Self.anthropic("yo")) }
        let session = URLSession(configuration: StubURLProtocol.configuration)
        let openAI = try TeacherClient("https://\(openAIHost)/v1", model: "m", environment: [TeacherClient.urlVariable: "https://\(openAIHost)/v1", TeacherClient.keyVariable: Self.canary], session: session)
        let anthropic = try TeacherClient("https://\(anthropicHost)/v1", model: "m", provider: .anthropic, environment: [TeacherClient.urlVariable: "https://\(anthropicHost)", TeacherClient.keyVariable: Self.canary], session: session)
        #expect(try await openAI.complete("q", maxTokens: 8, temperature: 0) == .init(text: "hi", reasoning: nil, truncated: true, inputTokens: 10, outputTokens: 20))
        #expect(try await anthropic.complete("q", maxTokens: 8, temperature: 0) == .init(text: "yo", reasoning: "hmm", truncated: false, inputTokens: 7, outputTokens: 3))
        let openAIRequest = try #require(StubURLProtocol.requests(for: openAIHost).first?.0)
        let anthropicRequest = try #require(StubURLProtocol.requests(for: anthropicHost).first?.0)
        #expect(openAIRequest.url?.path == "/v1/chat/completions" && openAIRequest.value(forHTTPHeaderField: "Authorization") == "Bearer \(Self.canary)")
        #expect(anthropicRequest.url?.path == "/v1/messages" && anthropicRequest.value(forHTTPHeaderField: "x-api-key") == Self.canary)
        #expect(anthropicRequest.value(forHTTPHeaderField: "anthropic-version") == "2023-06-01")
    }

    @Test("A reasoning effort goes into reasoning_effort for OpenAI-compatible teachers and output_config.effort for Anthropic, and nowhere when unset")
    func reasoningEffort() throws {
        func body(_ provider: TeacherProvider, _ effort: String?) throws -> [String: Any] {
            let request = try TeacherClient("https://t.test/v1", model: "m", provider: provider, environment: [:]).request("q", maxTokens: 8, temperature: 0, reasoningEffort: effort)
            let data = try #require(request.httpBody)
            return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        }
        #expect(try body(.openai, "low")["reasoning_effort"] as? String == "low")
        #expect((try body(.anthropic, "low")["output_config"] as? [String: String]) == ["effort": "low"])
        #expect(try body(.openai, nil)["reasoning_effort"] == nil)
    }

    @Test("A 429 is retried; a 400 is an error that names neither the key nor the body; a key bound to another host is not sent")
    func retriesAndKeys() async throws {
        let calls = Mutex(0)
        let flaky = StubURLProtocol.host { _, _ in calls.withLock { $0 += 1; return $0 } == 1 ? (429, [:], Data()) : (200, [:], Self.openAI("ok")) }
        let refusing = StubURLProtocol.host { _, _ in (400, [:], Data("bad \(Self.canary)".utf8)) }
        let session = URLSession(configuration: StubURLProtocol.configuration)
        #expect(try await TeacherClient("https://\(flaky)/v1", model: "m", environment: [:], session: session).complete("q", maxTokens: 8, temperature: 0).text == "ok")
        let client = try TeacherClient("https://\(refusing)/v1", model: "m", environment: [TeacherClient.urlVariable: "https://elsewhere.test", TeacherClient.keyVariable: Self.canary], session: session)
        #expect(client.key == nil)
        let error = await #expect(throws: BasaltError.self) { try await client.complete("q", maxTokens: 8, temperature: 0) }
        #expect(!String(describing: error).contains(Self.canary))
        #expect(StubURLProtocol.requests(for: refusing).allSatisfy { $0.0.value(forHTTPHeaderField: "Authorization") == nil })
    }

    @Test("Retry waits follow Retry-After or the rate-limit reset when given, back off exponentially otherwise, and never exceed 30 s")
    func retryDelay() {
        #expect(TeacherClient.delay(attempt: 1, advised: nil) == 1)
        #expect(TeacherClient.delay(attempt: 3, advised: nil) == 4)
        #expect(TeacherClient.delay(attempt: 1, advised: 7) == 7)
        #expect(TeacherClient.delay(attempt: 1, advised: 11802) == 30)
        #expect(TeacherClient.delay(attempt: 9, advised: nil) == 30)
    }

    @Test("Empty, truncated, duplicate and unverified answers are rejected by reason; the held-out split is the tail")
    func filters() {
        var seen = Set<String>()
        func reason(_ text: String, truncated: Bool = false, answer: String? = nil, filter: DistillFilter = .verify) -> String? {
            DistillCore.rejection(.init(text: text, reasoning: nil, truncated: truncated, inputTokens: 0, outputTokens: 0), example: .init(prompt: "p", answer: answer), filter: filter, seen: &seen)
        }
        #expect(reason("  ") == "empty")
        #expect(reason("long", truncated: true) == "truncated")
        #expect(reason("red", answer: "blue") == "unverified")
        #expect(reason("Answer: blue.", answer: "blue") == nil)
        #expect(reason("Answer: blue.", answer: "blue") == "duplicate")
        #expect(reason("red", answer: "blue", filter: .none) == nil)
        #expect(DistillCore.heldOutCount(20, fraction: 0.1) == 2 && DistillCore.heldOutCount(2, fraction: 0.5) == 1 && DistillCore.heldOutCount(3, fraction: 0.99) == 2)
        #expect(throws: BasaltError.self) { try DistillCore.examples(jsonl: #"{"prompt": "only one"}"#) }
        #expect(throws: BasaltError.self) { try DistillCore.examples(jsonl: "{\"prompt\": \"a\"}\nnot json") }
    }

    @Test("includeReasoning puts the teacher's reasoning before its answer, inside the model's thinking markers when it has them; without it, or without reasoning, the answer alone is the target")
    func reasoningTarget() {
        let completion = TeacherClient.Completion(text: " Answer: 4 ", reasoning: " 2 + 2 is 4 ", truncated: false, inputTokens: 0, outputTokens: 0)
        #expect(DistillCore.target(completion, includeReasoning: true) == "2 + 2 is 4\n\nAnswer: 4")
        #expect(DistillCore.target(completion, includeReasoning: false) == "Answer: 4")
        #expect(DistillCore.target(completion, includeReasoning: true, markers: LiquidProfile().thinkingMarkers) == "<think>\n2 + 2 is 4\n</think>\n\nAnswer: 4")
        #expect(DistillCore.target(completion, includeReasoning: false, markers: LiquidProfile().thinkingMarkers) == "Answer: 4")
        #expect(DistillCore.target(.init(text: "x", reasoning: nil, truncated: false, inputTokens: 0, outputTokens: 0), includeReasoning: true) == "x")
    }

    @Test("A 20-prompt run on a tiny model lowers the held-out teacher NLL, leaves the base weights, saves a loadable adapter, caches every answer and never writes the key")
    func endToEnd() async throws {
        let directory = try Self.scratch()
        let host = StubURLProtocol.host { _, body in
            let prompt = body.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }.flatMap { ($0["messages"] as? [[String: Any]])?.first?["content"] as? String } ?? ""
            return (200, [:], Self.openAI("The colour is blue, as in \(prompt.count % 3 == 0 ? "sky" : "sea")."))
        }
        let session = URLSession(configuration: StubURLProtocol.configuration)
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await Self.baseWeights(container)
        let request = Self.request(host, prompts: try Self.prompts(20, in: directory), output: directory.appending(path: "adapter"))
        let environment = [TeacherClient.urlVariable: "https://\(host)", TeacherClient.keyVariable: Self.canary]
        let report = try await engine.distill(request, session: session, home: directory, environment: environment)
        #expect(report.teacherCalls == 20 && report.cached == 0 && report.kept == 15 && report.heldOut == 5)
        #expect(!report.losses.isEmpty && report.losses.count <= MLXInferenceEngine.reportedLosses)
        let nllBefore = try #require(report.teacherNLLBefore), nllAfter = try #require(report.teacherNLLAfter)
        #expect(nllAfter < nllBefore, "\(nllBefore) -> \(nllAfter)")
        _ = try LoRAContainer.from(directory: directory.appending(path: "adapter"))
        if let overlay = report.verdict?.overlay { try await engine.undo(overlay.id) }
        #expect(await Self.baseWeights(container) == before)
        let cache = try String(contentsOf: URL(fileURLWithPath: try #require(report.cacheDirectory)).appending(component: "teacher.jsonl"), encoding: .utf8)
        #expect(cache.split(separator: "\n").count == 20 && !cache.contains(Self.canary))
        #expect(!String(decoding: try JSONEncoder().encode(report), as: UTF8.self).contains(Self.canary))

        var again = request
        again.output = directory.appending(path: "adapter-2").path
        let second = try await engine.distill(again, session: session, home: directory, environment: environment)
        #expect(second.teacherCalls == 0 && second.cached == 20)
        #expect(StubURLProtocol.requests(for: host).count == 20)
    }

    @Test("At budgetUSD the run stops calling the teacher, keeps what it has, and says why")
    func budget() async throws {
        let directory = try Self.scratch()
        let host = StubURLProtocol.host { _, _ in (200, [:], Self.openAI("blue")) }
        var request = Self.request(host, prompts: try Self.prompts(12, in: directory), output: directory.appending(path: "adapter"))
        (request.teacher.maxTokens, request.teacher.outputPricePerMillion, request.teacher.concurrency, request.budgetUSD) = (20, 50_000, 2, 4.5)
        let cache = DistillCache(directory: directory.appending(path: "cache"))
        let examples = try DistillCore.examples(jsonl: try String(contentsOfFile: request.prompts, encoding: .utf8))
        let teacher = try TeacherClient(request.teacher.url, model: "m", environment: [:], session: URLSession(configuration: StubURLProtocol.configuration))
        let report = try await MLXInferenceEngine.collect(request, examples: examples, teacher: teacher, cache: cache)
        #expect(report.stoppedBy == .budget)
        #expect(report.teacherCalls == 4 && cache.load().count == 4)
        #expect(report.spentUSD <= 4.5)
    }

    @Test("Out-of-range settings and a budget without prices are refused before any call")
    func refused() {
        var base = DistillRequest(prompts: "p", teacher: TeacherSpec(url: "https://x.test/v1", model: "m"), gate: GateRequest(heldOut: ["a"]))
        #expect(throws: Never.self) { try MLXInferenceEngine.validate(base) }
        let changes: [(inout DistillRequest) -> Void] = [
            { $0.budgetUSD = 5 }, { $0.teacher.model = "" }, { $0.teacher.maxTokens = 1 }, { $0.teacher.concurrency = 0 },
            { $0.samplesPerPrompt = 9 }, { $0.heldOutFraction = 1 }, { $0.rank = 0 }, { $0.learningRate = .nan }, { $0.teacher.temperature = 3 },
        ]
        for change in changes {
            var request = base
            change(&request)
            #expect(throws: BasaltError.self) { try MLXInferenceEngine.validate(request) }
        }
        base.teacher.outputPricePerMillion = 1
        base.budgetUSD = 5
        #expect(throws: Never.self) { try MLXInferenceEngine.validate(base) }
    }
}
