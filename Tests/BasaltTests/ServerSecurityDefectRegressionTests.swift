import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import NIOCore
import Synchronization
import Testing
@testable import BasaltCloud
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Server security and protocol defect regressions (A-S1, A-S2, A-S3, A-S8, A-S9, A-S12)")
struct ServerSecurityDefectRegressionTests {
    private actor StubConversionService: ModelConversionService {
        private(set) var lastRequest: ModelConversionRequest?

        func plan(
            source: String,
            target: QuantizationTarget,
            budgetGB: Double?,
            pvTune: Bool,
            cacheRoot: URL
        ) async throws -> ModelConversionPlan {
            ModelConversionPlan(sourceId: source, sourceFormat: "mlx", target: target.specString, supportsPVTune: false)
        }

        func convert(
            _ request: ModelConversionRequest,
            cacheRoot: URL,
            onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
        ) async throws -> ModelConversionResult {
            lastRequest = request
            return ModelConversionResult(outputId: request.outputId, bytesBefore: 100, bytesAfter: 50, shards: 1)
        }
    }

    private actor ErrorFinishEngine: InferenceEngine {
        var isLoaded: Bool { true }
        var throwOnStream = false

        func setThrowOnStream(_ value: Bool) { throwOnStream = value }
        func load(modelPath: String) async throws(BasaltError) {}
        func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
            GenerationResponse(
                generatedText: "ok", promptTokens: 1, completionTokens: 1,
                tokensPerSecond: 10, generationTime: 0.1, timeToFirstToken: 0.05,
                memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
                finishReason: .stop
            )
        }

        func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
            if throwOnStream { throw .modelNotLoaded }
            let (stream, continuation) = AsyncStream<StreamChunk>.makeStream()
            continuation.yield(.text("partial"))
            continuation.yield(.metadata(GenerationResponse(
                generatedText: "partial", promptTokens: 1, completionTokens: 1,
                tokensPerSecond: 10, generationTime: 0.1, timeToFirstToken: 0.05,
                memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
                finishReason: .error
            )))
            continuation.finish()
            return stream
        }
    }

    static func makeServer<E: InferenceEngine>(
        apiKey: String? = nil,
        virtualKeys: [VirtualKey] = [],
        rateLimit: Int? = nil,
        rateBurst: Int? = nil,
        engine: E = MockInferenceEngine(),
        conversion: (any ModelConversionService)? = nil
    ) throws -> Application<RouterResponder<BasicRequestContext>> {
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        var config = try ServerConfig(
            modelPath: "org/model",
            host: "127.0.0.1",
            apiKey: apiKey,
            access: .init(rateLimitPerMinute: rateLimit, rateBurst: rateBurst, virtualKeys: virtualKeys)
        )
        config.modelId = "org/model"
        var server = RESTServer(orchestrator: orchestrator, config: config)
        if let conversion {
            let hub = ModelHubOperations(converter: conversion) { _, _ in URL(fileURLWithPath: "/tmp") }
            server = server.withModelHub(hub)
        }
        return Application(router: server.makeRouter())
    }
}

extension ServerSecurityDefectRegressionTests {
    @Test("A-S1: /v1/generate, /api/v1/generate, /api/v1/generate/stream, and MCP basalt_generate reject overflowing/negative maxTokens; ModelCatalog.estimateCost never overflows")
    func generateEndpointsValidateMaxTokensAndCostNeverOverflows() async throws {
        let catalog = ModelCatalog()
        #expect(catalog.estimateCost(model: "gpt-4", promptTokens: 100, completionTokens: Int.max).isFinite)
        #expect(catalog.estimateCost(model: "openai/gpt-4-turbo", promptTokens: -50, completionTokens: -1000) == 0.0)

        #expect(throws: MCPFailure.self) {
            _ = try MCPArguments.generation(["prompt": "hi", "maxTokens": Int.max])
        }
        #expect(throws: MCPFailure.self) {
            _ = try MCPArguments.generation(["prompt": "hi", "maxTokens": 1_000_000])
        }

        let app = try Self.makeServer()
        let jsonHeaders: HTTPFields = [.contentType: "application/json"]
        try await app.test(.router) { client in
            for path in ["/v1/generate", "/api/v1/generate", "/api/v1/generate/stream"] {
                for badMax in [Int.max, -100, 1_000_000] {
                    let body = ByteBuffer(string: #"{"prompt":"hi","maxTokens":\#(badMax)}"#)
                    try await client.execute(uri: path, method: .post, headers: jsonHeaders, body: body) { res in
                        #expect(res.status == .badRequest || res.status == .contentTooLarge, "\(path) maxTokens=\(badMax) got \(res.status)")
                    }
                }
            }
        }
    }

    @Test("A-S2: virtual keys are forbidden on admin/control routes and cannot collide with master rate bucket")
    func virtualKeysBlockedFromAdminRoutesAndRateIsolated() async throws {
        let vkMaster = VirtualKey(key: "vk-impersonate-master-1234", id: "master", maxBudget: 0)
        let app = try Self.makeServer(apiKey: "real-master-secret-key", virtualKeys: [vkMaster], rateLimit: 60, rateBurst: 1)
        let vkAuth: HTTPFields = [.authorization: "Bearer vk-impersonate-master-1234", .contentType: "application/json"]
        let masterAuth: HTTPFields = [.authorization: "Bearer real-master-secret-key", .contentType: "application/json"]
        try await app.test(.router) { client in
            for (method, path) in [
                (HTTPRequest.Method.delete, "/v1/models/org/model"),
                (.post, "/v1/models/load"),
                (.post, "/v1/models/download"),
                (.post, "/v1/models/publish"),
                (.post, "/v1/models/convert"),
                (.post, "/api/v1/models/convert"),
                (.post, "/v1/compose/apply"),
                (.post, "/v1/compose/diff"),
                (.post, "/v1/mcp/approvals/req-1/approve"),
            ] {
                let body = method == .post ? ByteBuffer(string: "{}") : nil
                try await client.execute(uri: path, method: method, headers: vkAuth, body: body) { res in
                    #expect(res.status == .forbidden, "\(method) \(path) returned \(res.status)")
                }
            }
            try await client.execute(uri: "/v1/models", method: .get, headers: vkAuth) { res in
                #expect(res.status == .ok)
            }
            try await client.execute(uri: "/v1/models", method: .get, headers: vkAuth) { res in
                #expect(res.status == .tooManyRequests)
            }
            try await client.execute(uri: "/v1/models", method: .get, headers: masterAuth) { res in
                #expect(res.status == .ok)
            }
        }
    }

    @Test("A-S3: REST /v1/models/convert preserves hf: prefix on adapter so CWD directories cannot be loaded")
    func restConvertPreservesHFAdapterPrefix() async throws {
        let stub = StubConversionService()
        let app = try Self.makeServer(conversion: stub)
        var convReq = ModelConversionRequest(
            source: "org/base",
            outputId: "org/out",
            target: .affine(bits: 4, groupSize: 64),
            stream: false
        )
        convReq.adapter = "hf:org/lora"
        let body = ByteBuffer(bytes: try JSONEncoder().encode(convReq))
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/models/convert", method: .post, headers: [.contentType: "application/json"], body: body) { res in
                #expect(res.status == .ok)
            }
        }
        let recorded = await stub.lastRequest
        #expect(recorded?.adapter == "hf:org/lora")
    }

    @Test("A-S8, A-S9, A-S12: oversized JSON returns JSON 413, /api/v1/generate/stream fails before 200 & omits [DONE] on error, no-auth strips budget/rate headers")
    func middleware413StreamErrorAndHeaderSanitization() async throws {
        let errorEngine = ErrorFinishEngine()
        let app = try Self.makeServer(engine: errorEngine)
        let headers: HTTPFields = [.contentType: "application/json"]
        let tooBig = ByteBuffer(string: #"{"prompt":""# + String(repeating: "a", count: 1_100_000) + #""}"#)
        try await app.test(.router) { client in
            try await client.execute(uri: "/api/v1/generate", method: .post, headers: headers, body: tooBig) { res in
                #expect(res.status == .contentTooLarge)
                #expect(res.headers[.contentType]?.contains("application/json") == true)
            }
            try await client.execute(
                uri: "/api/v1/generate/stream", method: .post, headers: headers,
                body: ByteBuffer(string: #"{"prompt":"hi","maxTokens":8}"#)
            ) { res in
                let text = String(buffer: res.body)
                #expect(res.status == .ok && text.contains("event: error") && !text.contains("data: [DONE]"))
            }
            await errorEngine.setThrowOnStream(true)
            try await client.execute(
                uri: "/api/v1/generate/stream", method: .post, headers: headers,
                body: ByteBuffer(string: #"{"prompt":"hi","maxTokens":8}"#)
            ) { res in
                #expect(res.status == .serviceUnavailable)
            }
        }

        let seenBudget = Mutex<String?>(nil)
        let seenRate = Mutex<String?>(nil)
        let guardRouter = Router()
        guardRouter.add(middleware: RequestGuardMiddleware(bindHost: "127.0.0.1", bindPort: 8080))
        guardRouter.get("/probe") { request, _ in
            seenBudget.withLock { $0 = request.headers[APIKeyAuthenticator<BasicRequestContext>.budgetKeyHeader] }
            seenRate.withLock { $0 = request.headers[APIKeyAuthenticator<BasicRequestContext>.rateKeyHeader] }
            return Response(status: .ok)
        }
        try await Application(router: guardRouter).test(.router) { client in
            var spoofHeaders = HTTPFields()
            spoofHeaders[APIKeyAuthenticator<BasicRequestContext>.budgetKeyHeader] = "spoof-tenant"
            spoofHeaders[APIKeyAuthenticator<BasicRequestContext>.rateKeyHeader] = "master"
            try await client.execute(uri: "/probe", method: .get, headers: spoofHeaders) { res in
                #expect(res.status == .ok)
            }
        }
        #expect(seenBudget.withLock { $0 } == nil && seenRate.withLock { $0 } == nil)
    }
}
