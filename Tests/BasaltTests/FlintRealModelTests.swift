import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Recursive Flint real-model 3-bit and 4-bit verification on Metal (#518)", .serialized)
struct FlintRealModelTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_FLINT_MODEL_PATH"]
    }

    @Test("Real Flint checkpoint passes shard integrity, loads within 24 GB unified memory, emits <think> reasoning and structured <tool_call><arg_key> tool calls, and supports 8-bit KV quantization",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_FLINT_MODEL_PATH"))
    func realFlintEndToEnd() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("flint-e2e", modelPath: path)
        defer { log.releaseGPU() }

        MLXInferenceEngine.renewsResidency()
        var config = EngineConfiguration.standard
        config.wiresWorkingSet = true
        let engine = MLXInferenceEngine(config)
        let loadStart = ContinuousClock.now
        try await log.step("load") { try await engine.load(modelPath: path) }
        let loadSeconds = (ContinuousClock.now - loadStart) / .seconds(1)
        let container = try #require(await engine.container)
        let streamed = await container.perform { ExpertStreamingLoader.store(in: $0.model) != nil }
        let activeAfterLoadMB = Memory.activeMemory / 1_048_576
        let peakAfterLoadMB = Memory.peakMemory / 1_048_576
        log.record("memory", String(format: "load %.2fs streamed %@ activeMB %d peakMB %d", loadSeconds, "\(streamed)", activeAfterLoadMB, peakAfterLoadMB))

        let codePrompt = "Write a concise Swift 6 function `clamp(_ x: Int, min lo: Int, max hi: Int) -> Int` in one code block."
        var codeReq = GenerationRequest(
            prompt: codePrompt, maxTokens: 192, temperature: 1.0, topP: 0.95,
            messages: [ChatMessageInput(role: .user, text: codePrompt)]
        )
        codeReq.topK = 20
        let codeReply = try await log.step("coding") { try await engine.generate(request: codeReq) }
        let combined = (codeReply.reasoningText ?? "") + codeReply.generatedText
        log.record("coding-metrics", String(
            format: "promptTok %d genTok %d decode %.2f tok/s ttft %.0f ms reasoningChars %d answerChars %d",
            codeReply.promptTokens, codeReply.completionTokens,
            codeReply.tokensPerSecond, codeReply.timeToFirstToken * 1000,
            codeReply.reasoningText?.count ?? 0, codeReply.generatedText.count))
        #expect(!combined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(codeReply.tokensPerSecond > 0)

        try await Self.assertToolAndQuantizedKV(engine, log: log)
    }

    private static func assertToolAndQuantizedKV(_ engine: MLXInferenceEngine, log: ModelRunLog) async throws {
        let toolDef = ToolDefinition(
            name: "read_swift_file",
            description: "Read lines from a Swift source file on disk.",
            parametersJSON: """
                {"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"}},"required":["path"]}
                """
        )
        let toolPrompt = "Call the `read_swift_file` tool to read `Package.swift` starting at line 1. Do not answer without calling the tool."
        let toolReq = GenerationRequest(
            prompt: toolPrompt, maxTokens: 256, temperature: 0.0, topP: 1.0,
            tools: [toolDef],
            messages: [ChatMessageInput(role: .user, text: toolPrompt)]
        )
        let toolReply = try await log.step("tool-call") { try await engine.generate(request: toolReq) }
        let calls = toolReply.toolCalls ?? []
        log.record("tool-metrics", "calls \(calls.count) first \(calls.first?.name ?? "none") args \(calls.first?.arguments ?? "none") repairs \(toolReply.toolCallRepairs) raw \(toolReply.generatedText.debugDescription)")
        #expect(calls.first?.name == "read_swift_file")
        #expect(calls.first?.arguments.contains("Package.swift") == true)

        let kvPrompt = "State in one short sentence what sliding window attention saves."
        let kv8Req = GenerationRequest(
            prompt: kvPrompt, maxTokens: 128, temperature: 0.0, topP: 1.0,
            messages: [ChatMessageInput(role: .user, text: kvPrompt)]
        ).withKVPrecision(.q8)
        let kv8Reply = try await log.step("kv-bits-8") { try await engine.generate(request: kv8Req) }
        let kv8Combined = (kv8Reply.reasoningText ?? "") + kv8Reply.generatedText
        log.record("kv8-metrics", String(
            format: "genTok %d decode %.2f tok/s peakMB %d text %@",
            kv8Reply.completionTokens, kv8Reply.tokensPerSecond, Memory.peakMemory / 1_048_576, kv8Combined.debugDescription))
        #expect(!kv8Combined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }
}
