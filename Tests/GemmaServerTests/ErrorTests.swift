import Testing
@testable import GemmaServerCore

@Suite("GemmaServerError HTTP status mapping")
struct ErrorHTTPStatusTests {

    @Test("modelNotFound → 404")
    func modelNotFound() {
        let err = GemmaServerError.modelNotFound(identifier: "x")
        #expect(err.httpStatus == 404)
    }

    @Test("weightsCorrupted → 500")
    func weightsCorrupted() {
        #expect(GemmaServerError.weightsCorrupted(path: "/p", reason: "bad").httpStatus == 500)
    }

    @Test("outOfMemory → 507")
    func outOfMemory() {
        #expect(GemmaServerError.outOfMemory(requestedMB: 100, availableMB: 50).httpStatus == 507)
    }

    @Test("inferenceHardwareFailure → 503")
    func inferenceHardwareFailure() {
        #expect(GemmaServerError.inferenceHardwareFailure(reason: "crash").httpStatus == 503)
    }

    @Test("invalidRequestStructure → 400")
    func invalidRequestStructure() {
        #expect(GemmaServerError.invalidRequestStructure(details: "bad field").httpStatus == 400)
    }

    @Test("contextLengthExceeded → 413")
    func contextLengthExceeded() {
        #expect(GemmaServerError.contextLengthExceeded(maxTokens: 1024, requested: 2048).httpStatus == 413)
    }
}

@Suite("GemmaServerError descriptions")
struct ErrorDescriptionTests {

    @Test("All error cases have non-nil localizedDescription")
    func allHaveDescriptions() {
        let cases: [GemmaServerError] = [
            .modelNotFound(identifier: "model-id"),
            .weightsCorrupted(path: "/path", reason: "truncated"),
            .outOfMemory(requestedMB: 8192, availableMB: 4096),
            .inferenceHardwareFailure(reason: "Metal error"),
            .invalidRequestStructure(details: "missing field"),
            .contextLengthExceeded(maxTokens: 32_768, requested: 50_000),
        ]
        for err in cases {
            #expect(err.errorDescription != nil, "Expected description for \(err)")
            #expect(!(err.errorDescription?.isEmpty ?? true))
        }
    }

    @Test("outOfMemory description contains MB values")
    func outOfMemoryContainsValues() {
        let err = GemmaServerError.outOfMemory(requestedMB: 8192, availableMB: 4096)
        let desc = err.errorDescription ?? ""
        #expect(desc.contains("8192"))
        #expect(desc.contains("4096"))
    }

    @Test("modelNotFound description contains identifier")
    func modelNotFoundContainsId() {
        let id = "mlx-community/gemma-4-e2b-it-4bit"
        let err = GemmaServerError.modelNotFound(identifier: id)
        #expect(err.errorDescription?.contains(id) == true)
    }
}
