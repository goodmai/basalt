import Testing
import Foundation
@testable import GemmaServerCore

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
        #expect(throws: GemmaServerError.self) {
            try req.validated()
        }
    }

    @Test("maxTokens = 0 throws contextLengthExceeded")
    func zeroMaxTokens() {
        let req = GenerationRequest(prompt: "Hello", maxTokens: 0)
        #expect(throws: GemmaServerError.self) {
            try req.validated()
        }
    }

    @Test("maxTokens > 65536 throws contextLengthExceeded")
    func tooManyTokens() {
        let req = GenerationRequest(prompt: "Hello", maxTokens: 100_000)
        #expect(throws: GemmaServerError.self) {
            try req.validated()
        }
    }

    @Test("temperature < 0 throws invalidRequestStructure")
    func negativeTemperature() {
        let req = GenerationRequest(prompt: "Hi", maxTokens: 100, temperature: -0.1)
        #expect(throws: GemmaServerError.self) {
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
            finishReason: .stop
        )
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(GenerationResponse.self, from: data)

        #expect(decoded.generatedText   == original.generatedText)
        #expect(decoded.promptTokens    == original.promptTokens)
        #expect(decoded.completionTokens == original.completionTokens)
        #expect(decoded.tokensPerSecond == original.tokensPerSecond)
        #expect(decoded.generationTime  == original.generationTime)
        #expect(decoded.finishReason    == original.finishReason)
    }

    @Test("finishReason .length round-trips correctly")
    func finishReasonLength() throws {
        let resp = GenerationResponse(
            generatedText: "", promptTokens: 0, completionTokens: 1024,
            tokensPerSecond: 10, generationTime: 102.4, finishReason: .length
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
}
