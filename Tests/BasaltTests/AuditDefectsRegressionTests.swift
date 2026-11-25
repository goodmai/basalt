import Foundation
import Testing
import Hummingbird
import HummingbirdTesting
import HTTPTypes
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltInference
@testable import BasaltCLI
@testable import BasaltAppCore

@Suite("Audit security and cloud regression tests (#664, #667)", .serialized)
struct AuditSecurityAndCloudRegressionTests {
    @Test("REST /v1/models/convert* rejects local filesystem paths in source and pv_teacher (#664)")
    func convertEndpointsRejectLocalPaths() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        try await orchestrator.loadModel(path: "mock")
        let hub = ModelHubOperations(converter: StubConverter()) { _, _ in FileManager.default.temporaryDirectory }
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock")).withModelHub(hub)
        let app = Application(router: server.makeRouter())
        try await app.test(.router) { client in
            let planStatus = try await client.execute(
                uri: "/v1/models/convert/plan?source=/etc/passwd&target=macaq:affine:4:64",
                method: .get
            ) { $0.status }
            #expect(planStatus == .badRequest)

            let badSource = Data(#"{"source":"/etc/passwd","output_id":"local/out","target":"macaq:affine:4:64","stream":false}"#.utf8)
            let sourceStatus = try await client.execute(
                uri: "/v1/models/convert",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: badSource)
            ) { $0.status }
            #expect(sourceStatus == .badRequest)

            let badTeacher = Data(#"{"source":"org/model","output_id":"local/out","target":"macaq:affine:4:64","pv_tune":true,"pv_teacher":"/etc/passwd","stream":false}"#.utf8)
            let teacherStatus = try await client.execute(
                uri: "/v1/models/convert",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: badTeacher)
            ) { $0.status }
            #expect(teacherStatus == .badRequest)
        }
    }

    @Test("HuggingFaceHub.collectPublishableFiles skips symlinks escaping the root directory (#664)")
    func collectPublishableFilesBlocksEscapingSymlinks() throws {
        let base = FileManager.default.temporaryDirectory.appending(path: "basalt-symlink-test-\(UUID().uuidString)")
        let root = base.appending(path: "checkpoint")
        let secret = base.appending(path: "secret.txt")
        defer { try? FileManager.default.removeItem(at: base) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("{\"model_type\":\"qwen3\"}".utf8).write(to: root.appending(path: "config.json"))
        try Data("top-secret".utf8).write(to: secret)
        try FileManager.default.createSymbolicLink(at: root.appending(path: "leak.txt"), withDestinationURL: secret)

        let files = try HuggingFaceHub.collectPublishableFiles(in: root)
        #expect(files.map(\.relativePath) == ["config.json"])
    }

    @Test("ModelRouter.cloudMessages redacts credentials across prompts, tool results, and tool calls (#664)")
    func cloudMessagesRedactsSecrets() {
        let synthetic = ["sk", "ant", "api03", String(repeating: "a", count: 32)].joined(separator: "-")
        let flat = ModelRouter.cloudMessages(GenerationRequest(prompt: "use \(synthetic)"))
        #expect(flat.count == 1)
        #expect(!flat[0].content.contains(synthetic))

        let chat = GenerationRequest(
            prompt: "",
            messages: [
                ChatMessageInput(role: .user, text: "key=\(synthetic)"),
                ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "1", name: "auth", arguments: "{\"k\":\"\(synthetic)\"}")]),
                ChatMessageInput(role: .tool, text: "returned \(synthetic)", toolName: "auth"),
            ]
        )
        let messages = ModelRouter.cloudMessages(chat)
        #expect(messages.count == 3)
        for msg in messages {
            #expect(!msg.content.contains(synthetic))
        }
    }

    @Test("CostTracker ignores non-finite costs and evicts only oldest sessions at cap (#664)")
    func costTrackerNonFiniteAndPartialEviction() async throws {
        let file = FileManager.default.temporaryDirectory.appending(path: "basalt-cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let tracker = try CostTracker(storageURL: file, budget: .init(dailyLimit: 10_000, monthlyLimit: 100_000))

        try await tracker.recordUsage(model: "m", cost: .infinity, scopes: [.session("s-inf")])
        try await tracker.recordUsage(model: "m", cost: .nan, scopes: [.session("s-nan")])
        #expect(await tracker.getStats().dailyUsage == 0)

        for i in 0 ..< CostTracker.maxTrackedSessions {
            try await tracker.recordUsage(model: "m", cost: 0.001, scopes: [.session("s-\(i)")])
        }
        let recent = "s-\(CostTracker.maxTrackedSessions - 1)"
        try await tracker.recordUsage(model: "m", cost: 0.001, scopes: [.session("s-overflow")])
        #expect(await tracker.trackedSessionCount <= CostTracker.maxTrackedSessions)
        #expect(await tracker.spent(in: .session(recent)) > 0)
        #expect(await tracker.spent(in: .session("s-0")) == 0)
    }

    @Test("RateLimitMiddleware ignores spoofed X-Forwarded-For and unauthenticated headers when auth is off (#664)")
    func rateLimitIgnoresSpoofedHeadersWithoutAuth() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        let config = try ServerConfig(modelPath: "stub", host: "127.0.0.1", access: .init(rateLimitPerMinute: 6, rateBurst: 2))
        let server = RESTServer(orchestrator: orchestrator, config: config, rateClock: { 1_000 })
        let app = Application(router: server.makeRouter())
        try await app.test(.router) { client in
            for i in 0 ..< 2 {
                let status = try await client.execute(
                    uri: "/v1/models",
                    method: .get,
                    headers: [
                        HTTPField.Name("x-forwarded-for")!: "10.0.0.\(i)",
                        .authorization: "Bearer spoof-\(i)",
                    ]
                ) { $0.status }
                #expect(status == .ok)
            }
            let third = try await client.execute(
                uri: "/v1/models",
                method: .get,
                headers: [
                    HTTPField.Name("x-forwarded-for")!: "10.0.0.99",
                    .authorization: "Bearer spoof-99",
                ]
            ) { $0.status }
            #expect(third == .tooManyRequests)
        }
    }

    @Test("JSONNestingLimitMiddleware allows multipart/form-data with NUL bytes and uses route-aware body limits (#667)")
    func jsonNestingAllowsMultipartAndRouteLimits() async throws {
        let router = Router()
        router.add(middleware: JSONNestingLimitMiddleware())
        router.post("/v1/images/edits") { _, _ in Response(status: .ok) }
        let app = Application(router: router)
        try await app.test(.router) { client in
            let body = ByteBuffer(bytes: [0x2D, 0x2D, 0x00, 0x89, 0x50, 0x4E, 0x47, 0x00])
            let status = try await client.execute(
                uri: "/v1/images/edits",
                method: .post,
                headers: [.contentType: "multipart/form-data; boundary=test"],
                body: body
            ) { $0.status }
            #expect(status == .ok)
        }
        #expect(JSONNestingLimitMiddleware<BasicRequestContext>.bodyLimit(for: "/v1/images/edits") == ImagesController.bodyLimit)
        #expect(JSONNestingLimitMiddleware<BasicRequestContext>.bodyLimit(for: "/v1/pipeline") == PipelineController.bodyLimit)
        #expect(JSONNestingLimitMiddleware<BasicRequestContext>.bodyLimit(for: "/v1/systemone") == PipelineController.bodyLimit)
    }

    @Test("ModelRouter.relayCloudStream reports usage metadata even when finish_reason is omitted (#667)")
    func relayCloudStreamReportsUsageWithoutFinishReason() async throws {
        let source = AsyncThrowingStream<OpenRouterClient.ChatStreamResponse, Error> { continuation in
            continuation.yield(OpenRouterClient.ChatStreamResponse(
                id: "1",
                model: "test/model",
                choices: [.init(delta: .init(role: "assistant", content: "hi"), finishReason: nil)],
                usage: .init(promptTokens: 12, completionTokens: 4, totalTokens: 16)
            ))
            continuation.finish()
        }
        actor UsageBox {
            var recorded: GenerationResponse?
            func set(_ value: GenerationResponse) { recorded = value }
        }
        let box = UsageBox()
        let stream = ModelRouter.relayCloudStream(source, startTime: .now) { meta in
            await box.set(meta)
        }
        var gotMetadata = false
        for try await chunk in stream {
            if case .metadata(let meta) = chunk {
                gotMetadata = true
                #expect(meta.promptTokens == 12 && meta.completionTokens == 4 && meta.finishReason == .stop)
            }
        }
        #expect(gotMetadata)
        let recorded = await box.recorded
        #expect(recorded?.promptTokens == 12 && recorded?.completionTokens == 4)
    }
}

@Suite("Audit inference and app/CLI regression tests (#666, #668)", .serialized)
struct AuditInferenceAndAppRegressionTests {
    @Test("SageMXAttentionMetal aligns causal mask to bottom-right when LQ < LK (#666)")
    func sageMXCausalMaskWithKVOffset() {
        let q = MLXArray.ones([1, 2, 3, 64], type: Float.self)
        let k = MLXArray.ones([1, 2, 8, 64], type: Float.self)
        let positions = MLXArray(Array(0 ..< 8).map(Float.init)).reshaped(1, 1, 8, 1)
        let v = broadcast(positions, to: [1, 2, 8, 64])
        let scale: Float = 1.0 / sqrt(64.0)
        let opts = SageMXAttentionMetal.Options(scale: scale, causal: true, usePNQ: false, meanCenter: false)

        let ref = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .causal)
        eval(q, k, v, ref)
        let cpuArr = SageMXAttentionMetal.cpuAttention(
            q: q.asArray(Float.self),
            k: k.asArray(Float.self),
            v: v.asArray(Float.self),
            shape: .init(heads: 2, lQ: 3, lK: 8, d: 64),
            options: opts
        )
        let cpu = MLXArray(cpuArr, [1, 2, 3, 64])
        let gpu = SageMXAttentionMetal.attention(q: q, k: k, v: v, options: opts)
        eval(cpu, gpu)

        let cpuDiff = Float(max(abs(cpu - ref)).item(Float.self))
        let gpuDiff = Float(max(abs(gpu - ref)).item(Float.self))
        #expect(cpuDiff < 1e-4)
        #expect(gpuDiff < 1e-4)
    }

    @Test("HeadSplitKVCache.trim works on streaming-only layers and short prompts under sinkTokens (#666)")
    func headSplitTrimStreamingOnlyAndShortPrompts() {
        let streamOnly = HeadSplitKVCache(retrieval: [])
        let q = MLXArray.zeros([1, 2, 4, 8], type: Float.self)
        let k = MLXArray.zeros([1, 2, 4, 8], type: Float.self)
        let v = MLXArray.zeros([1, 2, 4, 8], type: Float.self)
        _ = streamOnly.updateAndAttend(queries: q, keys: k, values: v, scale: 1, mask: .causal)
        #expect(streamOnly.offset == 4)
        #expect(streamOnly.trim(2) == 2)
        #expect(streamOnly.offset == 2 && streamOnly.streamingTokens == 2)

        let mixed = HeadSplitKVCache(retrieval: [0])
        _ = mixed.updateAndAttend(queries: q, keys: k, values: v, scale: 1, mask: .causal)
        #expect(mixed.trim(2) == 2)
        #expect(mixed.offset == 2 && mixed.retrievalTokens == 2 && mixed.streamingTokens == 2)
    }

    @Test("BlockSparseRoutingMetal handles lK not divisible by blockSize and EngineConfiguration clamps topK (#666)")
    func blockSparseNonMultipleAndTopKClamp() {
        let rng = MLXRandom.RandomState(seed: 7)
        let q = MLXRandom.normal([1, 2, 1, 32], key: rng).asType(.float16)
        let k = MLXRandom.normal([1, 2, 70, 32], key: rng).asType(.float16)
        let v = MLXRandom.normal([1, 2, 70, 32], key: rng).asType(.float16)
        let selected = BlockSparseRoutingMetal.selectMoBABlocks(q: q, k: k, blockSize: 16, topBlocks: 2)
        eval(selected)
        #expect(selected.shape == [2, 1, 2])

        let cfg = BlockSparseRoutingMetal.Config(blockSize: 16, topBlocks: 2, slidingWindow: 16, scale: 1.0 / sqrt(32.0))
        let out = BlockSparseRoutingMetal.nsaThreeBranchAttention(q: q, k: k, v: v, config: cfg)
        eval(out)
        #expect(out.shape == q.shape)

        var engineCfg = EngineConfiguration()
        engineCfg.mobaTopK = 0
        engineCfg.questTopK = 0
        let normalized = engineCfg.normalized
        #expect(normalized.mobaTopK == 1)
        #expect(normalized.questTopK == 1)
    }

    @Test("PaddleOCRVLText.positions applies delta shift in batch ropeOffset mode (#666)")
    func paddleOCRPositionsBatchDelta() {
        let rowA = KVCacheSimple()
        let rowB = KVCacheSimple()
        _ = rowA.update(keys: MLXArray.zeros([1, 1, 5, 4], type: Float.self), values: MLXArray.zeros([1, 1, 5, 4], type: Float.self))
        _ = rowB.update(keys: MLXArray.zeros([1, 1, 3, 4], type: Float.self), values: MLXArray.zeros([1, 1, 3, 4], type: Float.self))
        let cache = BatchKVCache()
        cache.append([rowA, rowB])
        let pos = PaddleOCRVLText.positions(after: cache, delta: MLXArray(Int32(3)), batch: 2, length: 1)
        eval(pos)
        #expect(pos[0, 0, 0].item(Int32.self) == 8)
        #expect(pos[0, 1, 0].item(Int32.self) == 6)
    }

    @Test("LibraryReducer.listFailed stops retry storm and CLI validation catches missing --out and local paths (#668)")
    func appAndCLIRegressions() throws {
        var state = LibraryState()
        let effects = LibraryReducer.reduce(&state, .listFailed("engine offline"))
        #expect(effects.isEmpty)
        #expect(state.problem == "engine offline")

        #expect(throws: Error.self) {
            _ = try ConvertSubcommand.parse(["org/model", "--quantize", "affine:4:64", "--register-cache", "local/out"])
        }
        let ok = try ConvertSubcommand.parse(["org/model", "--to", "macaq:affine:4:64", "--register-cache", "local/out"])
        #expect(ok.registerCache == "local/out")

        let dir = FileManager.default.temporaryDirectory.appending(path: "basalt-local-model-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        #expect(ChatCommand.resolvedLocalPath(for: dir.path) == dir.path)
        #expect(ChatCommand.resolvedLocalPath(for: "mlx-community/nonexistent-repo-xyz") == nil)
    }
}

private struct StubConverter: ModelConversionService {
    func plan(source: String, target: QuantizationTarget, budgetGB: Double?, pvTune: Bool, cacheRoot: URL) async throws -> ModelConversionPlan {
        let fp = ModelConversionPlan.MemoryFootprint(
            sourceBytes: 1024,
            estimatedOutputBytes: 512,
            peakConversionRAMBytes: 2048,
            availableRAMBytes: 16_384
        )
        return ModelConversionPlan(
            sourceId: source,
            sourceFormat: "dense:bf16",
            target: target.specString,
            supportsPVTune: target.supportsPVTune,
            footprint: fp
        )
    }

    func convert(
        _ request: ModelConversionRequest,
        cacheRoot: URL,
        onProgress: @escaping @Sendable (BasaltCore::ModelConversionProgress) -> Void
    ) async throws -> BasaltCore::ModelConversionResult {
        BasaltCore::ModelConversionResult(
            outputId: request.outputId,
            bytesBefore: 1024,
            bytesAfter: 512,
            shards: 1,
            pvTune: nil
        )
    }
}
