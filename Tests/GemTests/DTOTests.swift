import Testing
import Foundation
@testable import GemCore

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
        #expect(throws: GemError.self) {
            try req.validated()
        }
    }

    @Test("maxTokens = 0 throws contextLengthExceeded")
    func zeroMaxTokens() {
        let req = GenerationRequest(prompt: "Hello", maxTokens: 0)
        #expect(throws: GemError.self) {
            try req.validated()
        }
    }

    @Test("maxTokens > 131072 throws contextLengthExceeded")
    func tooManyTokens() {
        let req = GenerationRequest(prompt: "Hello", maxTokens: 200_000)
        #expect(throws: GemError.self) {
            try req.validated()
        }
    }

    @Test("temperature < 0 throws invalidRequestStructure")
    func negativeTemperature() {
        let req = GenerationRequest(prompt: "Hi", maxTokens: 100, temperature: -0.1)
        #expect(throws: GemError.self) {
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

    @Test("ErrorResponse Codable round-trip")
    func errorResponseRoundTrip() throws {
        let error = ErrorResponse(error: "Test Error", code: 404)
        let data = try encoder.encode(error)
        let dec = try decoder.decode(ErrorResponse.self, from: data)
        #expect(dec.error == "Test Error")
        #expect(dec.code == 404)
    }
}
