import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference

@Suite("max_tokens is capped by memory, --max-tokens is a ceiling (#85)", .serialized)
struct MemoryCapTests {
    static let qwen3_4BConfig = #"{"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2560}"#
    static let bytesPerToken: Int64 = 2 * 36 * 8 * 128 * 2
    static let gib: Int64 = 1 << 30

    static func modelDirectory(weightsBytes: UInt64 = 0) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qwen3-4b-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(qwen3_4BConfig.utf8).write(to: dir.appendingPathComponent("config.json"))
        if weightsBytes > 0 {
            let weights = dir.appendingPathComponent("model.safetensors")
            FileManager.default.createFile(atPath: weights.path, contents: nil)
            let handle = try FileHandle(forWritingTo: weights)
            try handle.truncate(atOffset: weightsBytes)
            try handle.close()
        }
        return dir
    }

    static func orchestrator(
        kvBudget: Int64?, promptTokens: Int? = nil, maxTokens: Int = 65_536, weightsBytes: UInt64 = 0,
        free: Int64 = 64 << 30
    ) async throws -> (ModelOrchestratorActor, MockInferenceEngine, URL) {
        let engine = MockInferenceEngine()
        await engine.setMemory(kvBudget: kvBudget, promptTokens: promptTokens)
        let dir = try modelDirectory(weightsBytes: weightsBytes)
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: maxTokens, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: dir.path)
        await orchestrator.setFreeMemoryProbe { free }
        return (orchestrator, engine, dir)
    }

    @Test("24 GB Mac, Qwen3-4B: a 40k-token prompt with max_tokens 65536 is capped to what the KV room holds")
    func longPromptIsCapped() async throws {
        let budget = 7 * Self.gib
        let (orchestrator, engine, dir) = try await Self.orchestrator(kvBudget: budget, promptTokens: 40_000)
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "long", maxTokens: 65_536))
        let fitting = Int(budget / Self.bytesPerToken)
        #expect(await engine.getLastRequest()?.maxTokens == fitting - 40_000)
        #expect(await orchestrator.effectiveMaxTokens(for: GenerationRequest(prompt: "long", maxTokens: 65_536)) == fitting - 40_000)
    }

    @Test("A prompt that does not fit at all is refused with 413 before generation")
    func promptAloneDoesNotFit() async throws {
        let (orchestrator, engine, dir) = try await Self.orchestrator(kvBudget: 1 * Self.gib, promptTokens: 40_000)
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try await orchestrator.generate(request: GenerationRequest(prompt: "long"))
            Issue.record("expected 413")
        } catch {
            guard case .contextLengthExceeded(let fitting, let requested) = error else {
                Issue.record("unexpected \(error)")
                return
            }
            #expect(fitting == Int(Self.gib / Self.bytesPerToken))
            #expect(requested == 40_001)
            #expect(error.httpStatus == 413)
        }
        #expect(await engine.getRequests().isEmpty)
    }

    @Test("--max-tokens is an upper limit, not a floor the memory budget raises")
    func maxTokensIsACeiling() async throws {
        let (orchestrator, engine, dir) = try await Self.orchestrator(kvBudget: 100 * Self.gib, promptTokens: 10, maxTokens: 2_048)
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "hi", maxTokens: 10_000))
        #expect(await engine.getLastRequest()?.maxTokens == 2_048)
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "hi"))
        #expect(await engine.getLastRequest()?.maxTokens == 2_048)
    }

    @Test("Loaded weights are not subtracted from free memory a second time")
    func weightsCountedOnce() async throws {
        let (orchestrator, _, dir) = try await Self.orchestrator(
            kvBudget: nil, promptTokens: 1_000, weightsBytes: 8 << 30, free: 10 * Self.gib)
        defer { try? FileManager.default.removeItem(at: dir) }
        let fitting = Int(Int64(Double(10 * Self.gib) * TokenBudgetCalculator.safetyMargin) / Self.bytesPerToken)
        #expect(await orchestrator.effectiveMaxTokens(for: GenerationRequest(prompt: "p")) == fitting - 1_000)
    }

    @Test("The prompt length comes from the engine's tokenizer when it has one, else UTF-8 bytes / 4")
    func tokenizerCountWins() async throws {
        let budget = 2 * Self.gib
        let fitting = Int(budget / Self.bytesPerToken)
        let (counted, _, dir1) = try await Self.orchestrator(kvBudget: budget, promptTokens: 5_000)
        let (estimated, _, dir2) = try await Self.orchestrator(kvBudget: budget, promptTokens: nil)
        defer { [dir1, dir2].forEach { try? FileManager.default.removeItem(at: $0) } }
        let prompt = String(repeating: "abcd", count: 1_000)
        #expect(await counted.effectiveMaxTokens(for: GenerationRequest(prompt: prompt)) == fitting - 5_000)
        #expect(await estimated.effectiveMaxTokens(for: GenerationRequest(prompt: prompt)) == fitting - 1_000)
    }

    @Test("The MLX engine's KV room is the GPU working set minus the weights and the prompt-cache budget")
    func engineBudget() async throws {
        let engine = MLXInferenceEngine(promptCacheBytes: 64 << 20)
        #expect(await engine.kvMemoryBudgetBytes == nil)
        await engine.attach(try TinyModel.container())
        let budget = try #require(await engine.kvMemoryBudgetBytes)
        let workingSet = Int64(PromptCacheBudget.gpuWorkingSet())
        #expect(budget > 0)
        #expect(budget <= workingSet - (64 << 20))
        #expect(await engine.promptTokenCount(for: GenerationRequest(prompt: "abcd")) == 4)
    }
}
