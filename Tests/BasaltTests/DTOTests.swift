import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("GenerationRequest validation")
struct GenerationRequestValidationTests {

    @Test("Valid request passes through unchanged")
    func validRequest() throws {
        let req = GenerationRequest(prompt: "Hello world", maxTokens: 512, temperature: 0.7)
        let validated = try req.validated()
        #expect(validated.prompt == "Hello world")
        #expect(validated.maxTokens == 512)
    }

    @Test("Empty prompt throws invalidRequestStructure")
    func emptyPrompt() {
        let req = GenerationRequest(prompt: "", maxTokens: 100)
        #expect(throws: BasaltError.self) {
            try req.validated()
        }
    }

    @Test("maxTokens = 0 throws contextLengthExceeded")
    func zeroMaxTokens() {
        let req = GenerationRequest(prompt: "Hello", maxTokens: 0)
        #expect(throws: BasaltError.self) {
            try req.validated()
        }
    }

    @Test("maxTokens > 131072 throws contextLengthExceeded")
    func tooManyTokens() {
        let req = GenerationRequest(prompt: "Hello", maxTokens: 200_000)
        #expect(throws: BasaltError.self) {
            try req.validated()
        }
    }

    @Test("temperature < 0 throws invalidRequestStructure")
    func negativeTemperature() {
        let req = GenerationRequest(prompt: "Hi", maxTokens: 100, temperature: -0.1)
        #expect(throws: BasaltError.self) {
            try req.validated()
        }
    }

    @Test("temperature = 2.0 is valid boundary")
    func temperatureBoundary() throws {
        let req = GenerationRequest(prompt: "Hi", maxTokens: 100, temperature: 2.0)
        let validated = try req.validated()
        #expect(validated.temperature == 2.0)
    }
}

@Suite("GenerationResponse Codable")
struct GenerationResponseCodableTests {

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    @Test("Round-trip encode/decode preserves all fields")
    func roundTrip() throws {
        let original = GenerationResponse(
            generatedText: "Test output",
            promptTokens: 5,
            completionTokens: 10,
            tokensPerSecond: 24.5,
            generationTime: 0.408,
            timeToFirstToken: 0.1,
            memory: .init(peakBytes: 2048, activeBytes: 1024, cacheBytes: 512),
            finishReason: .stop
        )
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(GenerationResponse.self, from: data)

        #expect(decoded.generatedText   == original.generatedText)
        #expect(decoded.promptTokens    == original.promptTokens)
        #expect(decoded.completionTokens == original.completionTokens)
        #expect(decoded.tokensPerSecond == original.tokensPerSecond)
        #expect(decoded.generationTime  == original.generationTime)
        #expect(decoded.timeToFirstToken == original.timeToFirstToken)
        #expect(decoded.memory.peakBytes == original.memory.peakBytes)
        #expect(decoded.finishReason    == original.finishReason)
    }

    @Test("cachedPromptTokens round-trips, and a payload from a server without it decodes as 0")
    func cachedPromptTokensDecoding() throws {
        let original = GenerationResponse(
            generatedText: "x", promptTokens: 40, cachedPromptTokens: 32, completionTokens: 1,
            tokensPerSecond: 1, generationTime: 1, timeToFirstToken: 0.1,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop)
        let data = try encoder.encode(original)
        #expect(try decoder.decode(GenerationResponse.self, from: data).cachedPromptTokens == 32)

        var object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["cachedPromptTokens"] = nil
        let older = try JSONSerialization.data(withJSONObject: object)
        let decoded = try decoder.decode(GenerationResponse.self, from: older)
        #expect(decoded.cachedPromptTokens == 0)
        #expect(decoded.promptTokens == 40)
    }

    @Test("finishReason .length round-trips correctly")
    func finishReasonLength() throws {
        let resp = GenerationResponse(
            generatedText: "", promptTokens: 0, completionTokens: 1024,
            tokensPerSecond: 10, generationTime: 102.4, 
            timeToFirstToken: 0.05, 
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
            finishReason: .length
        )
        let data  = try encoder.encode(resp)
        let dec   = try decoder.decode(GenerationResponse.self, from: data)
        #expect(dec.finishReason == .length)
    }

    @Test("GenerationRequest Codable round-trip")
    func requestRoundTrip() throws {
        let req = GenerationRequest(prompt: "Hello", maxTokens: 256, temperature: 0.5, topP: 0.9)
        let data = try encoder.encode(req)
        let dec  = try decoder.decode(GenerationRequest.self, from: data)
        #expect(dec.prompt      == req.prompt)
        #expect(dec.maxTokens   == req.maxTokens)
        #expect(dec.temperature == req.temperature)
        #expect(dec.topP        == req.topP)
    }

    @Test("minP and dynamicTemperature: absent in old JSON, kept by decode, validated and withMaxTokens")
    func samplingFieldsBackCompat() throws {
        let old = try decoder.decode(GenerationRequest.self, from: Data(#"{"prompt": "hi"}"#.utf8))
        #expect(old.minP == nil)
        #expect(old.dynamicTemperature == nil)

        let new = try decoder.decode(GenerationRequest.self,
                                     from: Data(#"{"prompt": "hi", "minP": 0.05, "dynamicTemperature": true}"#.utf8))
        #expect(new.minP == 0.05)
        #expect(new.dynamicTemperature == true)

        let validated = try new.validated()
        #expect(validated.minP == 0.05)
        #expect(validated.dynamicTemperature == true)
        let capped = validated.withMaxTokens(8)
        #expect(capped.minP == 0.05)
        #expect(capped.dynamicTemperature == true)
    }

    @Test("minP outside [0, 1] is rejected")
    func minPRange() throws {
        #expect(throws: BasaltError.self) { try GenerationRequest(prompt: "hi", minP: 1.5).validated() }
        #expect(throws: BasaltError.self) { try GenerationRequest(prompt: "hi", minP: -0.1).validated() }
        #expect(try GenerationRequest(prompt: "hi", minP: 0).validated().minP == 0)
    }

    @Test("ErrorResponse Codable round-trip")
    func errorResponseRoundTrip() throws {
        let error = ErrorResponse(error: "Test Error", code: 404)
        let data = try encoder.encode(error)
        let dec = try decoder.decode(ErrorResponse.self, from: data)
        #expect(dec.error == "Test Error")
        #expect(dec.code == 404)
    }
}

@Suite("ServiceRequestDetector")
struct ServiceRequestDetectorTests {

    @Test("Detects thread title prompts with short maxTokens")
    func detectsTitleRequests() {
        let titlePrompt = "Generate a short 3-5 word title for this conversation:\nUser: How to setup Swift on macOS?"
        #expect(ServiceRequestDetector.isTitleOrServiceRequest(prompt: titlePrompt, maxTokens: 32))
        #expect(ServiceRequestDetector.isTitleOrServiceRequest(prompt: titlePrompt, maxTokens: 64))

        let summarizePrompt = "Summarize the above session in 3 words"
        #expect(ServiceRequestDetector.isTitleOrServiceRequest(prompt: summarizePrompt, maxTokens: 20))
    }

    @Test("Ignores standard user prompts even with short maxTokens")
    func ignoresStandardUserPrompts() {
        let prompt = "What is the capital of France?"
        #expect(!ServiceRequestDetector.isTitleOrServiceRequest(prompt: prompt, maxTokens: 32))
        #expect(!ServiceRequestDetector.isTitleOrServiceRequest(prompt: prompt, maxTokens: 2048))
    }

    @Test("compactTitlePrompt truncates long context to tail")
    func compactsLongTitlePrompt() {
        let longHistory = String(repeating: "Conversation history line. ", count: 100)
        let prompt = "Please create a title for this chat:\n\(longHistory)"
        let compacted = ServiceRequestDetector.compactTitlePrompt(prompt)
        #expect(compacted.count < prompt.count)
        #expect(compacted.hasPrefix("Generate a short 3-5 word title for this discussion:\n"))
    }

    @Test("Ignores bare title: and code prompts")
    func ignoresBareTitleInCodePrompts() {
        let htmlPrompt = "Can you help me write an HTML file with <title>My App</title>?"
        #expect(!ServiceRequestDetector.isTitleOrServiceRequest(prompt: htmlPrompt, maxTokens: 1024))

        let yamlPrompt = "apiVersion: v1\nmetadata:\n  title: Sample\n"
        #expect(!ServiceRequestDetector.isTitleOrServiceRequest(prompt: yamlPrompt, maxTokens: 1024))
    }
}

@Suite("Coding Agent Options & Responses DTOs")
struct CodingAgentDTOTests {

    @Test("ReasoningPolicy rawValue mapping")
    func reasoningPolicyMapping() {
        #expect(ReasoningPolicy(rawValue: "off") == .disabled)
        #expect(ReasoningPolicy(rawValue: "none") == .disabled)
        #expect(ReasoningPolicy(rawValue: "false") == .disabled)
        // Level names pass through; each profile maps them (Qwen: high → xhigh).
        #expect(ReasoningPolicy(rawValue: "High") == .effort("high"))
        #expect(ReasoningPolicy(rawValue: "xhigh") == .effort("xhigh"))
        #expect(ReasoningPolicy(rawValue: "medium") == .effort("medium"))
        #expect(ReasoningPolicy(rawValue: "low") == .effort("low"))
        #expect(ReasoningPolicy(rawValue: "auto") == .modelDefault)
        #expect(ReasoningPolicy(rawValue: nil) == nil)
    }

    @Test("QwenProfile handles reasoning policy in templateContext")
    func qwenProfileReasoningContext() {
        let qwen = QwenProfile()
        let disabledCtx = qwen.templateContext(for: .disabled)
        #expect(disabledCtx["enable_thinking"] as? Bool == false)

        let xhighCtx = qwen.templateContext(for: .effort("xhigh"))
        #expect(xhighCtx["reasoning_effort"] as? String == "xhigh")

        let lowCtx = qwen.templateContext(for: .effort("low"))
        #expect(lowCtx["reasoning_effort"] as? String == "low")
    }

    @Test("ResponsesRequest decodes input string and reasoning_effort")
    func responsesRequestDecoding() throws {
        let json = """
        {
            "model": "qwen3-coder",
            "input": "Refactor this function",
            "max_output_tokens": 1024,
            "reasoning_effort": "low",
            "stream": true
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(OpenAIController.ResponsesRequest.self, from: json)
        #expect(req.model == "qwen3-coder")
        #expect(req.maxOutputTokens == 1024)
        #expect(req.reasoningEffort == "low")
        #expect(req.stream == true)
        if case .text(let t) = req.input {
            #expect(t == "Refactor this function")
        } else {
            Issue.record("Expected text input")
        }
    }

    @Test("ResponsesRequest decodes input structured item array")
    func responsesRequestDecodingStructuredItems() throws {
        let json = """
        {
            "model": "qwen3-coder",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Refactor this function"
                        }
                    ]
                }
            ],
            "reasoning_effort": "low"
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(OpenAIController.ResponsesRequest.self, from: json)
        #expect(req.model == "qwen3-coder")
        #expect(req.reasoningEffort == "low")
        guard case .messages(let msgs) = req.input, let first = msgs.first else {
            Issue.record("Expected messages input")
            return
        }
        #expect(first.role == "user")
        #expect(first.content?.textValue == "Refactor this function")
    }

    @Test("ResponseInput throws decoding error for malformed input")
    func responseInputThrowsOnMalformed() {
        let invalidJson = """
        {
            "input": 12345
        }
        """.data(using: .utf8)!

        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(OpenAIController.ResponsesRequest.self, from: invalidJson)
        }
    }

    @Test("MLXInferenceEngine builds templateContext with request reasoningEffort")
    func buildTemplateContextFromRequest() {
        let qwen = QwenProfile()
        let reqLow = GenerationRequest(prompt: "hello", reasoningEffort: "low")
        let ctxLow = MLXInferenceEngine.buildTemplateContext(request: reqLow, profile: qwen)
        #expect(ctxLow?["reasoning_effort"] as? String == "low")

        let reqNone = GenerationRequest(prompt: "hello", reasoningEffort: "none")
        let ctxNone = MLXInferenceEngine.buildTemplateContext(request: reqNone, profile: qwen)
        #expect(ctxNone?["enable_thinking"] as? Bool == false)

        let reqDisable = GenerationRequest(prompt: "hello", enableThinking: false)
        let ctxDisable = MLXInferenceEngine.buildTemplateContext(request: reqDisable, profile: qwen)
        #expect(ctxDisable?["enable_thinking"] as? Bool == false)

        // Request-level effort overrides engine default
        let reqHigh = GenerationRequest(prompt: "hello", reasoningEffort: "high")
        let ctxOverride = MLXInferenceEngine.buildTemplateContext(
            request: reqHigh,
            defaultReasoningEffort: "low",
            profile: qwen
        )
        #expect(ctxOverride?["reasoning_effort"] as? String == "xhigh")
    }
}

