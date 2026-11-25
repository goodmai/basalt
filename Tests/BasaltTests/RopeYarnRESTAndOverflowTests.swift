import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

private actor RoPEStubEngine: InferenceEngine {
    var isLoaded: Bool { loadedPath != nil }
    private var loadedPath: String?
    private var statusByPath: [String: ModelRoPEStatus]

    init(statusByPath: [String: ModelRoPEStatus]) {
        self.statusByPath = statusByPath
    }

    var ropeStatus: ModelRoPEStatus {
        guard let loadedPath else { return .empty }
        return statusByPath[loadedPath] ?? .empty
    }

    func load(modelPath: String) async throws(BasaltError) {
        loadedPath = modelPath
    }

    func unload() async {
        loadedPath = nil
    }

    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        GenerationResponse(
            generatedText: "ok", promptTokens: 1, completionTokens: 1,
            tokensPerSecond: 100, generationTime: 0.01, timeToFirstToken: 0.01,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
            finishReason: .stop
        )
    }

    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        AsyncStream { c in c.finish() }
    }
}

@Suite("--rope-yarn P1 REST reporting, hot-swap warnings, and context-overflow warning (#729 P1)", .serialized)
struct RopeYarnRESTAndOverflowTests {
    static func jsonObject(_ buffer: ByteBuffer) throws -> [String: Any] {
        let data = Data(buffer.readableBytesView)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test("ModelRoPEStatus formats yarn label, computes context overflow warning, and round-trips Codable")
    func ropeStatusValueCore() throws {
        let yarnInt = ModelRoPEStatus(factor: 2.0, nativeContext: 262_144, effectiveContext: 524_288)
        #expect(yarnInt.ropeLabel == "yarn ×2")
        #expect(yarnInt.contextOverflowWarning(promptTokens: 500_000, maxTokens: 24_288) == nil)
        let warn = try #require(yarnInt.contextOverflowWarning(promptTokens: 500_000, maxTokens: 30_000))
        #expect(warn.contains("530000") && warn.contains("524288") && warn.contains("yarn ×2"))

        let yarnFrac = ModelRoPEStatus(factor: 2.5, nativeContext: 1000, effectiveContext: 2500)
        #expect(yarnFrac.ropeLabel == "yarn ×2.5")

        let stock = ModelRoPEStatus(factor: nil, nativeContext: 1024, effectiveContext: 1024, loadWarnings: ["stock fallback"])
        #expect(stock.ropeLabel == nil)
        let stockWarn = try #require(stock.contextOverflowWarning(promptTokens: 900, maxTokens: 200))
        #expect(stockWarn.contains("1024") && stockWarn.contains("stock RoPE"))
        #expect(ModelRoPEStatus.empty.contextOverflowWarning(promptTokens: 999_999, maxTokens: 100) == nil)

        let encoded = try JSONEncoder().encode(stock)
        let decoded = try JSONDecoder().decode(ModelRoPEStatus.self, from: encoded)
        #expect(decoded == stock)
    }

    @Test("MLXInferenceEngine.ropeSourceWithStatus reports active YaRN on qwen3_5 and warns + falls back to stock RoPE on gemma4")
    func engineRopeStatusAndFallback() async throws {
        let qwenDir = try RopeYarnFlagTests.directory(RopeYarnFlagTests.qwen)
        let gemmaDir = try RopeYarnFlagTests.directory([
            "model_type": "gemma4",
            "text_config": ["max_position_embeddings": 262_144],
        ])
        defer {
            try? FileManager.default.removeItem(at: qwenDir)
            try? FileManager.default.removeItem(at: gemmaDir)
        }

        let plain = MLXInferenceEngine(EngineConfiguration())
        let plainRes = await plain.ropeSourceWithStatus(qwenDir, modelType: "qwen3_5", modelPath: qwenDir.path)
        #expect(plainRes.url == qwenDir && plainRes.status.ropeLabel == nil && plainRes.status.effectiveContext == 262_144)
        #expect(plainRes.status.loadWarnings.isEmpty)

        var cfg = EngineConfiguration()
        cfg.ropeYarn = 2
        let yarn = MLXInferenceEngine(cfg)
        let qwenRes = await yarn.ropeSourceWithStatus(qwenDir, modelType: "qwen3_5", modelPath: qwenDir.path)
        #expect(qwenRes.url != qwenDir && qwenRes.status.ropeLabel == "yarn ×2")
        #expect(qwenRes.status.nativeContext == 262_144 && qwenRes.status.effectiveContext == 524_288)
        #expect(qwenRes.status.loadWarnings.isEmpty)

        let gemmaRes = await yarn.ropeSourceWithStatus(gemmaDir, modelType: "gemma4", modelPath: "mlx-community/gemma-4-26B-A4B")
        #expect(gemmaRes.url == gemmaDir && gemmaRes.status.ropeLabel == nil && gemmaRes.status.effectiveContext == 262_144)
        #expect(gemmaRes.status.loadWarnings.count == 1)
        #expect(gemmaRes.status.loadWarnings[0].contains("gemma-4-26B-A4B") && gemmaRes.status.loadWarnings[0].contains("p-RoPE"))
    }

    @Test("GET /v1/models, GET /v1/models/current, and POST /v1/models/load report rope + effective_context for YaRN and warnings without rope on Gemma 4 fallback")
    func restModelsRoPEReportingAndHotSwapWarnings() async throws {
        let qwenDir = try RopeYarnFlagTests.directory(RopeYarnFlagTests.qwen)
        let gemmaDir = try RopeYarnFlagTests.directory(["model_type": "gemma4", "text_config": ["max_position_embeddings": 262_144]])
        defer {
            try? FileManager.default.removeItem(at: qwenDir)
            try? FileManager.default.removeItem(at: gemmaDir)
        }
        let qwenStatus = ModelRoPEStatus(factor: 2.0, nativeContext: 262_144, effectiveContext: 524_288)
        let gemmaWarning = "--rope-yarn 2.0 does not apply to mlx-community/gemma-4-26B: Gemma 4 uses p-RoPE"
        let gemmaStatus = ModelRoPEStatus(factor: nil, nativeContext: 262_144, effectiveContext: 262_144, loadWarnings: [gemmaWarning])
        let engine = RoPEStubEngine(statusByPath: [qwenDir.path: qwenStatus, gemmaDir.path: gemmaStatus])
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: qwenDir.path)
        await orchestrator.setModelId("Qwen/Qwen3.8-27B")

        var activeEntry: [String: Any] = ["id": "qwen"]
        ModelsController.applyRoPEFields(qwenStatus, to: &activeEntry)
        #expect(activeEntry["rope"] as? String == "yarn ×2" && activeEntry["effective_context"] as? Int == 524_288)
        var stockEntry: [String: Any] = ["id": "gemma"]
        ModelsController.applyRoPEFields(gemmaStatus, to: &stockEntry)
        #expect(stockEntry["rope"] == nil && stockEntry["effective_context"] == nil)

        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: qwenDir.path))
        let app = Application(router: server.makeRouter())
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/models/current", method: .get) { res in
                let body = try Self.jsonObject(res.body)
                #expect(res.status == .ok)
                #expect(body["rope"] as? String == "yarn ×2")
                #expect(body["effective_context"] as? Int == 524_288)
            }
            try await client.execute(
                uri: "/v1/models/load", method: .post, headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"Qwen/Qwen3.8-27B"}"#)
            ) { res in
                let body = try Self.jsonObject(res.body)
                #expect(res.status == .ok && body["rope"] as? String == "yarn ×2" && body["warnings"] == nil)
            }
            try await orchestrator.switchModel(to: gemmaDir.path)
            await orchestrator.setModelId("mlx-community/gemma-4-26B")
            try await client.execute(uri: "/v1/models/current", method: .get) { res in
                let body = try Self.jsonObject(res.body)
                #expect(res.status == .ok && body["rope"] == nil && body["effective_context"] == nil)
            }
            try await client.execute(
                uri: "/v1/models/load", method: .post, headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"mlx-community/gemma-4-26B"}"#)
            ) { res in
                let body = try Self.jsonObject(res.body)
                #expect(res.status == .ok && body["rope"] == nil)
                #expect(body["warnings"] as? [String] == [gemmaWarning])
            }
            try await client.execute(uri: "/v1/models", method: .get) { res in
                let body = try Self.jsonObject(res.body)
                let data = try #require(body["data"] as? [[String: Any]])
                let loaded = data.first { $0["is_loaded"] as? Bool == true }
                #expect(loaded?["rope"] == nil)
            }
        }
    }
}
