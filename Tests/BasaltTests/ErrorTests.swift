import Testing
import Foundation
import Hummingbird
import HTTPTypes
import Synchronization
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("BasaltError HTTP status mapping")
struct ErrorHTTPStatusTests {

    @Test("modelNotFound → 404")
    func modelNotFound() {
        let err = BasaltError.modelNotFound(identifier: "x")
        #expect(err.httpStatus == 404)
    }

    @Test("weightsCorrupted → 500")
    func weightsCorrupted() {
        #expect(BasaltError.weightsCorrupted(path: "/p", reason: "bad").httpStatus == 500)
    }

    @Test("outOfMemory → 507")
    func outOfMemory() {
        #expect(BasaltError.outOfMemory(requestedMB: 100, availableMB: 50).httpStatus == 507)
    }

    @Test("inferenceHardwareFailure → 503")
    func inferenceHardwareFailure() {
        #expect(BasaltError.inferenceHardwareFailure(reason: "crash").httpStatus == 503)
    }

    @Test("invalidRequestStructure → 400")
    func invalidRequestStructure() {
        #expect(BasaltError.invalidRequestStructure(details: "bad field").httpStatus == 400)
    }

    @Test("contextLengthExceeded → 413")
    func contextLengthExceeded() {
        #expect(BasaltError.contextLengthExceeded(maxTokens: 1024, requested: 2048).httpStatus == 413)
    }
}

@Suite("BasaltError descriptions")
struct ErrorDescriptionTests {

    @Test("All error cases have non-nil localizedDescription")
    func allHaveDescriptions() {
        let cases: [BasaltError] = [
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
        let err = BasaltError.outOfMemory(requestedMB: 8192, availableMB: 4096)
        let desc = err.errorDescription ?? ""
        #expect(desc.contains("8192"))
        #expect(desc.contains("4096"))
    }

    @Test("modelNotFound description contains identifier")
    func modelNotFoundContainsId() {
        let id = "mlx-community/gemma-4-e2b-it-4bit"
        let err = BasaltError.modelNotFound(identifier: id)
        #expect(err.errorDescription?.contains(id) == true)
    }
}

@Suite("ProviderErrorResponse shape and type mapping (#76)")
struct ProviderErrorResponseTests {

    final class BufferAccumulator: Sendable {
        let mutex = Mutex(ByteBuffer())
    }

    struct TestBodyWriter: ResponseBodyWriter {
        let accumulator = BufferAccumulator()
        mutating func write(_ buffer: ByteBuffer) async throws {
            var b = buffer
            _ = accumulator.mutex.withLock { $0.writeBuffer(&b) }
        }
        consuming func finish(_ trailingHeaders: HTTPFields?) async throws {}
    }

    private func extractData(_ response: Response) async throws -> Data {
        let writer = TestBodyWriter()
        try await response.body.write(writer)
        return writer.accumulator.mutex.withLock { Data($0.readableBytesView) }
    }

    @Test("Path to ProviderErrorShape resolution")
    func pathResolution() {
        #expect(ProviderErrorShape.forPath("/v1/messages") == .anthropic)
        #expect(ProviderErrorShape.forPath("/v1/messages/extra") == .anthropic)
        #expect(ProviderErrorShape.forPath("/v1/chat/completions") == .openAI)
        #expect(ProviderErrorShape.forPath("/v1/responses") == .openAI)
        #expect(ProviderErrorShape.forPath("/v1/models") == .openAI)
        #expect(ProviderErrorShape.forPath("/api/v1/generate") == .native)
        #expect(ProviderErrorShape.forPath("/api/v1/generate/stream") == .native)
        #expect(ProviderErrorShape.forPath("/api/v1/health") == .native)
    }

    @Test("Anthropic error types and Retry-After headers for each BasaltError case")
    func anthropicErrorMapping() async throws {
        let expectations: [(BasaltError, Int, String, String?)] = [
            (.modelNotFound(identifier: "m"), 404, "not_found_error", nil),
            (.modelNotCached(identifier: "m"), 404, "not_found_error", nil),
            (.authenticationFailed(details: "d"), 401, "authentication_error", nil),
            (.invalidRequestStructure(details: "d"), 400, "invalid_request_error", nil),
            (.modelConfigurationInvalid(path: "p", reason: "r"), 422, "invalid_request_error", nil),
            (.contextLengthExceeded(maxTokens: 10, requested: 20), 413, "invalid_request_error", nil),
            (.rateLimitExceeded(retryAfter: "7"), 429, "rate_limit_error", "7"),
            (.modelSwitching(from: "a", to: "b"), 503, "overloaded_error", "1"),
            (.serverBusy(retryAfterSeconds: 4), 503, "overloaded_error", "4"),
            (.inferenceHardwareFailure(reason: "r"), 503, "overloaded_error", "1"),
            (.outOfMemory(requestedMB: 100, availableMB: 10), 507, "overloaded_error", "1"),
            (.weightsCorrupted(path: "p", reason: "r"), 500, "api_error", nil),
            (.modelInferenceError(details: "d"), 502, "api_error", nil),
            (.inferenceError("msg"), 500, "api_error", nil)
        ]

        for (err, expectedStatus, expectedType, expectedRetryAfter) in expectations {
            let res = ProviderErrorResponse.response(for: err, shape: .anthropic)
            #expect(res.status.code == expectedStatus)
            if let expectedRetryAfter {
                #expect(res.headers[.retryAfter] == expectedRetryAfter)
            } else {
                #expect(res.headers[.retryAfter] == nil)
            }
            let data = try await extractData(res)
            let obj = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(obj["type"] as? String == "error")
            let errorObj = try #require(obj["error"] as? [String: Any])
            #expect(errorObj["type"] as? String == expectedType)
            #expect(errorObj["message"] != nil)
        }
    }

    @Test("OpenAI error shape has top-level type and nested message, type")
    func openAIErrorMapping() async throws {
        let res429 = ProviderErrorResponse.response(status: .tooManyRequests, message: "Limit", shape: .openAI, retryAfter: "12")
        #expect(res429.status == .tooManyRequests)
        #expect(res429.headers[.retryAfter] == "12")
        let data429 = try await extractData(res429)
        let obj429 = try #require(try JSONSerialization.jsonObject(with: data429) as? [String: Any])
        #expect(obj429["type"] as? String == "error")
        let err429 = try #require(obj429["error"] as? [String: Any])
        #expect(err429["type"] as? String == "rate_limit_error")
        #expect(err429["message"] as? String == "Limit")

        let res503 = ProviderErrorResponse.response(for: BasaltError.modelSwitching(from: "a", to: "b"), shape: .openAI)
        #expect(res503.status == .serviceUnavailable)
        #expect(res503.headers[.retryAfter] == "1")
        let data503 = try await extractData(res503)
        let obj503 = try #require(try JSONSerialization.jsonObject(with: data503) as? [String: Any])
        let err503 = try #require(obj503["error"] as? [String: Any])
        #expect(err503["type"] as? String == "server_error")
    }

    @Test("Native error shape matches ErrorResponse DTO")
    func nativeErrorMapping() async throws {
        let res = ProviderErrorResponse.response(status: .notFound, message: "Missing", shape: .native)
        #expect(res.status == .notFound)
        let data = try await extractData(res)
        let dto = try JSONDecoder().decode(ErrorResponse.self, from: data)
        #expect(dto.error == "Missing")
        #expect(dto.code == 404)
    }
}
