import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import MLX
import MLXLMCommon
import MLXNN
import NIOCore
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

private actor HarnessMockAdapterEngine: InferenceEngine {
    let inner = MockInferenceEngine()
    let served: [String]
    private(set) var switches: [String?] = []
    private var active: String?

    init(adapters: [String]) {
        served = adapters
    }

    func load(modelPath: String) async throws(BasaltError) {
        try await inner.load(modelPath: modelPath)
    }

    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        try await inner.generate(request: request)
    }

    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        try await inner.generateStream(request: request)
    }

    var isLoaded: Bool { get async { await inner.isLoaded } }
    var adapterNames: [String] { served }
    var activeAdapter: String? { active }

    func useAdapter(_ name: String?) async throws(BasaltError) {
        switches.append(name)
        active = name
    }
}

private enum FrozenHarnessFixtures {
    struct Case: Sendable {
        let label: String
        let fingerprint: HarnessRequestFingerprint
        let expected: AgentHarnessKind?
    }

    static let claudeCodeCLI = HarnessRequestFingerprint(
        userAgent: "claude-cli/1.0.35 (external, cli)",
        endpointPath: "/v1/messages",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "You are Claude Code, Anthropic's official CLI for Claude.",
        anthropicVersionHeader: "2023-06-01"
    )

    static let claudeCodeSDK = HarnessRequestFingerprint(
        userAgent: "Anthropic/JS 0.39.0",
        endpointPath: "/v1/messages",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "x-anthropic-billing-header: cc\nYou are Claude Code, Anthropic's official CLI for Claude.",
        anthropicVersionHeader: "2023-06-01"
    )

    static let genericAnthropicSDK = HarnessRequestFingerprint(
        userAgent: "Anthropic/JS 0.39.0",
        endpointPath: "/v1/messages",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "You are a concise technical assistant.",
        anthropicVersionHeader: "2023-06-01"
    )

    static let openCodeUA = HarnessRequestFingerprint(
        userAgent: "opencode/0.1.48",
        endpointPath: "/v1/chat/completions",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "Follow repository conventions.",
        anthropicVersionHeader: nil
    )

    static let openCodeSystem = HarnessRequestFingerprint(
        userAgent: "ai-sdk/4.1.0",
        endpointPath: "/v1/chat/completions",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "You are opencode, the best coding agent on the planet.",
        anthropicVersionHeader: nil
    )

    static let piCodingAgent = HarnessRequestFingerprint(
        userAgent: "pi-coding-agent/0.52.0",
        endpointPath: "/v1/chat/completions",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "You are an expert coding assistant operating inside pi, a coding agent harness.",
        anthropicVersionHeader: nil
    )

    static let piShortUA = HarnessRequestFingerprint(
        userAgent: "pi/0.40.0 (darwin arm64)",
        endpointPath: "/v1/chat/completions",
        explicitAdapterHeader: nil,
        systemPromptPrefix: nil,
        anthropicVersionHeader: nil
    )

    static let hermesAgent = HarnessRequestFingerprint(
        userAgent: "nous-hermes-agent/0.4.2",
        endpointPath: "/v1/chat/completions",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "You are Hermes Agent, an autonomous coding assistant.",
        anthropicVersionHeader: nil
    )

    static let genericCurl = HarnessRequestFingerprint(
        userAgent: "curl/8.7.1",
        endpointPath: "/v1/chat/completions",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "You are a helpful assistant.",
        anthropicVersionHeader: nil
    )

    static let fastAPIClient = HarnessRequestFingerprint(
        userAgent: "fastapi/0.115.0 python-httpx/0.27.2",
        endpointPath: "/v1/chat/completions",
        explicitAdapterHeader: nil,
        systemPromptPrefix: "Answer briefly.",
        anthropicVersionHeader: nil
    )

    static let allCases: [Case] = [
        Case(label: "claude-cli UA", fingerprint: claudeCodeCLI, expected: .claudeCode),
        Case(label: "claude-code SDK + system marker", fingerprint: claudeCodeSDK, expected: .claudeCode),
        Case(label: "generic Anthropic SDK", fingerprint: genericAnthropicSDK, expected: nil),
        Case(label: "opencode UA", fingerprint: openCodeUA, expected: .openCode),
        Case(label: "opencode system marker", fingerprint: openCodeSystem, expected: .openCode),
        Case(label: "pi-coding-agent UA", fingerprint: piCodingAgent, expected: .pi),
        Case(label: "pi/ product token UA", fingerprint: piShortUA, expected: .pi),
        Case(label: "nous-hermes-agent UA", fingerprint: hermesAgent, expected: .hermes),
        Case(label: "generic curl", fingerprint: genericCurl, expected: nil),
        Case(label: "fastapi does not match pi/", fingerprint: fastAPIClient, expected: nil),
    ]
}

@Suite("HarnessRoute detection, precedence, thinking-off default, and CLI/REST wiring (#675 P1.2)")
struct HarnessRoutingTests {
    static let routingTable = HarnessRoutingTable(
        harnessToAdapter: [
            .claudeCode: "cc",
            .openCode: "oc",
            .pi: "pi",
            .hermes: "hm",
        ],
        defaultAdapter: nil,
        thinkingOffWhenHarnessAdapterActive: true
    )

    @Test("Frozen session fixtures detect Claude Code, OpenCode, Pi, and Hermes deterministically without false positives on generic clients")
    func frozenFixtureDetection() {
        for item in FrozenHarnessFixtures.allCases {
            #expect(HarnessRoute.detectHarness(item.fingerprint) == item.expected, "\(item.label)")
        }
    }

    @Test("Explicit X-Basalt-Adapter overrides harness map and defaultAdapter; none and base force base model with thinking enabled")
    func explicitHeaderPrecedenceAndForceBase() {
        let withDefault = HarnessRoutingTable(
            harnessToAdapter: [.claudeCode: "cc", .openCode: "oc"],
            defaultAdapter: "fallback",
            thinkingOffWhenHarnessAdapterActive: true
        )

        let explicitOverride = HarnessRequestFingerprint(
            userAgent: "claude-cli/1.0.35 (external, cli)",
            endpointPath: "/v1/messages",
            explicitAdapterHeader: "pi",
            systemPromptPrefix: nil,
            anthropicVersionHeader: "2023-06-01"
        )
        #expect(HarnessRoute.resolve(explicitOverride, table: withDefault) == HarnessRouteDecision(
            adapterName: "pi",
            detectedHarness: .claudeCode,
            defaultThinkingOff: false,
            promptCacheNamespace: "adapter:pi"
        ))

        let plainTableExplicit = HarnessRequestFingerprint(
            userAgent: "curl/8.0",
            endpointPath: "/v1/chat/completions",
            explicitAdapterHeader: "base",
            systemPromptPrefix: nil,
            anthropicVersionHeader: nil
        )
        #expect(HarnessRoute.resolve(plainTableExplicit, table: .empty) == HarnessRouteDecision(
            adapterName: "base",
            detectedHarness: nil,
            defaultThinkingOff: false,
            promptCacheNamespace: "adapter:base"
        ))

        for forcedBase in ["none", "NONE", "base", "Base"] {
            let forced = HarnessRequestFingerprint(
                userAgent: "claude-cli/1.0.35 (external, cli)",
                endpointPath: "/v1/messages",
                explicitAdapterHeader: forcedBase,
                systemPromptPrefix: nil,
                anthropicVersionHeader: "2023-06-01"
            )
            #expect(HarnessRoute.resolve(forced, table: withDefault) == HarnessRouteDecision(
                adapterName: nil,
                detectedHarness: .claudeCode,
                defaultThinkingOff: false,
                promptCacheNamespace: "base"
            ), "\(forcedBase)")
        }
    }

    @Test("Detected harness routes through table, unknown clients fall back to defaultAdapter or base, and thinkingOffWhenHarnessAdapterActive toggles thinking default")
    func harnessMapDefaultFallbackAndThinkingToggle() {
        let withDefault = HarnessRoutingTable(
            harnessToAdapter: [.claudeCode: "cc", .openCode: "oc"],
            defaultAdapter: "fallback",
            thinkingOffWhenHarnessAdapterActive: true
        )

        #expect(HarnessRoute.resolve(FrozenHarnessFixtures.openCodeUA, table: withDefault) == HarnessRouteDecision(
            adapterName: "oc",
            detectedHarness: .openCode,
            defaultThinkingOff: true,
            promptCacheNamespace: "adapter:oc"
        ))

        #expect(HarnessRoute.resolve(FrozenHarnessFixtures.genericCurl, table: withDefault) == HarnessRouteDecision(
            adapterName: "fallback",
            detectedHarness: nil,
            defaultThinkingOff: true,
            promptCacheNamespace: "adapter:fallback"
        ))

        #expect(HarnessRoute.resolve(FrozenHarnessFixtures.genericCurl, table: Self.routingTable) == HarnessRouteDecision(
            adapterName: nil,
            detectedHarness: nil,
            defaultThinkingOff: false,
            promptCacheNamespace: "base"
        ))

        let thinkingKept = HarnessRoutingTable(
            harnessToAdapter: [.pi: "pi"],
            defaultAdapter: nil,
            thinkingOffWhenHarnessAdapterActive: false
        )
        let keptDecision = HarnessRoute.resolve(FrozenHarnessFixtures.piCodingAgent, table: thinkingKept)
        #expect(keptDecision.adapterName == "pi")
        #expect(!keptDecision.defaultThinkingOff)
        #expect(keptDecision.promptCacheNamespace == "adapter:pi")
    }

    @Test("HarnessRoutingTable parses valid mappings, round-trips Codable, and refuses unknown harnesses, duplicates, empty names, and unregistered adapters")
    func parseValidateAndRefusals() throws {
        let parsed = try HarnessRoutingTable.parseMappings([
            "claude-code=cc", "opencode=oc", "pi=pi-ad", "hermes=hm",
        ])
        #expect(parsed == [.claudeCode: "cc", .openCode: "oc", .pi: "pi-ad", .hermes: "hm"])

        let table = HarnessRoutingTable(harnessToAdapter: parsed, defaultAdapter: "oc", thinkingOffWhenHarnessAdapterActive: true)
        try table.validate(registeredAdapters: ["cc", "oc", "pi-ad", "hm"])
        let encoded = try JSONEncoder().encode(table)
        #expect(try JSONDecoder().decode(HarnessRoutingTable.self, from: encoded) == table)

        let invalidSpecs: [[String]] = [
            ["cursor=cc"],
            ["claude-code=a", "claude-code=b"],
            ["pi="],
            ["pi=   "],
            ["=cc"],
            ["opencode"],
        ]
        for specs in invalidSpecs {
            #expect(throws: BasaltError.self) {
                _ = try HarnessRoutingTable.parseMappings(specs)
            }
        }

        #expect(throws: BasaltError.self) {
            try HarnessRoutingTable(harnessToAdapter: [.pi: "missing"]).validate(registeredAdapters: ["cc"])
        }
        #expect(throws: BasaltError.self) {
            try HarnessRoutingTable(defaultAdapter: "missing").validate(registeredAdapters: ["cc"])
        }
        #expect(throws: BasaltError.self) {
            try HarnessRoutingTable(defaultAdapter: "  ").validate(registeredAdapters: ["cc"])
        }
    }

    @Test("ServeCommand parses --harness-adapter and --default-adapter and refuses unconfigured or malformed harness options")
    func serveCommandFlagsAndValidation() throws {
        let valid = try ServeCommand.parse([
            "--adapter", "cc=/tmp/cc",
            "--adapter", "oc=/tmp/oc",
            "--harness-adapter", "claude-code=cc",
            "--harness-adapter", "opencode=oc",
            "--default-adapter", "oc",
        ])
        let table = try valid.harnessRoutingTable()
        #expect(table.harnessToAdapter == [.claudeCode: "cc", .openCode: "oc"])
        #expect(table.defaultAdapter == "oc")
        #expect(table.thinkingOffWhenHarnessAdapterActive)

        let badInvocations: [[String]] = [
            ["--adapter", "cc=/tmp/cc", "--harness-adapter", "unknown=cc"],
            ["--adapter", "a=/tmp/a", "--adapter", "b=/tmp/b", "--harness-adapter", "pi=a", "--harness-adapter", "pi=b"],
            ["--harness-adapter", "claude-code=cc"],
            ["--adapter", "cc=/tmp/cc", "--default-adapter", "missing"],
            ["--adapter", "cc=/tmp/cc", "--default-adapter", ""],
        ]
        for args in badInvocations {
            #expect(throws: (any Error).self) {
                _ = try ServeCommand.parse(args)
            }
        }
    }
}

extension HarnessRoutingTests {
    private static func makeRESTApp(
        table: HarnessRoutingTable,
        adapters: [String] = ["cc", "oc", "pi", "hm"]
    ) async throws -> (Application<RouterResponder<BasicRequestContext>>, HarnessMockAdapterEngine) {
        let engine = HarnessMockAdapterEngine(adapters: adapters)
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 4096, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        var config = try ServerConfig(modelPath: "org/mock-model")
        config.harnessRouting = table
        let server = RESTServer(orchestrator: orchestrator, config: config)
        return (Application(router: server.makeRouter()), engine)
    }

    private static func post(
        app: Application<RouterResponder<BasicRequestContext>>,
        uri: String,
        body: String,
        headers: HTTPFields
    ) async throws -> HTTPResponse.Status {
        var requestHeaders = headers
        requestHeaders[.contentType] = "application/json"
        let finalHeaders = requestHeaders
        return try await app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: finalHeaders, body: ByteBuffer(string: body)) { res in
                res.status
            }
        }
    }

    @Test("REST /v1/chat/completions auto-routes OpenCode, Pi, and Hermes, isolates cache namespaces, defaults thinking off, and honors explicit overrides")
    func restChatCompletionsRouting() async throws {
        let (app, engine) = try await Self.makeRESTApp(table: Self.routingTable)
        let plainBody = #"{"model": "local", "messages": [{"role": "user", "content": "hi"}]}"#

        var ocHeaders = HTTPFields()
        ocHeaders[.userAgent] = "opencode/0.1.48"
        #expect(try await Self.post(app: app, uri: "/v1/chat/completions", body: plainBody, headers: ocHeaders) == .ok)
        let ocReq = try #require(await engine.inner.lastRequest)
        #expect(ocReq.adapter == "oc")
        #expect(ocReq.cacheKey == "shared|adapter:oc")
        #expect(ocReq.enableThinking == false)

        let explicitEnableBody = #"{"model": "local", "enable_thinking": true, "messages": [{"role": "user", "content": "hi"}]}"#
        #expect(try await Self.post(app: app, uri: "/v1/chat/completions", body: explicitEnableBody, headers: ocHeaders) == .ok)
        let overrideReq = try #require(await engine.inner.lastRequest)
        #expect(overrideReq.adapter == "oc")
        #expect(overrideReq.enableThinking == true)

        var piHeaders = HTTPFields()
        piHeaders[.userAgent] = "pi-coding-agent/0.52.0"
        let effortBody = #"{"model": "local", "reasoning_effort": "medium", "messages": [{"role": "user", "content": "hi"}]}"#
        #expect(try await Self.post(app: app, uri: "/v1/chat/completions", body: effortBody, headers: piHeaders) == .ok)
        let piReq = try #require(await engine.inner.lastRequest)
        #expect(piReq.adapter == "pi")
        #expect(piReq.cacheKey == "shared|adapter:pi")
        #expect(piReq.enableThinking == true)
        #expect(piReq.reasoningEffort == "medium")

        var forcedBaseHeaders = ocHeaders
        forcedBaseHeaders[GenerationRequest.adapterHeader] = "none"
        #expect(try await Self.post(app: app, uri: "/v1/chat/completions", body: plainBody, headers: forcedBaseHeaders) == .ok)
        let baseReq = try #require(await engine.inner.lastRequest)
        #expect(baseReq.adapter == nil)
        #expect(baseReq.cacheKey == "shared")
        #expect(baseReq.enableThinking == true)

        var serviceHeaders = ocHeaders
        serviceHeaders[HTTPField.Name("x-service-request")!] = "true"
        #expect(try await Self.post(app: app, uri: "/v1/chat/completions", body: plainBody, headers: serviceHeaders) == .ok)
        let serviceReq = try #require(await engine.inner.lastRequest)
        #expect(serviceReq.cacheKey == nil)
    }

    @Test("REST /v1/messages auto-routes Claude Code, isolates cache namespace, defaults thinking off, and respects explicit thinking config")
    func restAnthropicMessagesRouting() async throws {
        let (app, engine) = try await Self.makeRESTApp(table: Self.routingTable)

        var ccHeaders = HTTPFields()
        ccHeaders[.userAgent] = "Anthropic/JS 0.39.0"
        ccHeaders[GenerationRequest.anthropicVersionHeader] = "2023-06-01"
        let ccBody = #"{"max_tokens": 2048, "system": "You are Claude Code, Anthropic's official CLI for Claude.", "messages": [{"role": "user", "content": "hi"}]}"#
        #expect(try await Self.post(app: app, uri: "/v1/messages", body: ccBody, headers: ccHeaders) == .ok)
        let ccReq = try #require(await engine.inner.lastRequest)
        #expect(ccReq.adapter == "cc")
        #expect(ccReq.cacheKey == "shared|adapter:cc")
        #expect(ccReq.enableThinking == false)

        let ccThinkingBody = #"{"max_tokens": 4096, "system": "You are Claude Code, Anthropic's official CLI for Claude.", "thinking": {"type": "enabled", "budget_tokens": 1024}, "messages": [{"role": "user", "content": "hi"}]}"#
        #expect(try await Self.post(app: app, uri: "/v1/messages", body: ccThinkingBody, headers: ccHeaders) == .ok)
        let ccThinkingReq = try #require(await engine.inner.lastRequest)
        #expect(ccThinkingReq.adapter == "cc")
        #expect(ccThinkingReq.enableThinking == true)
        #expect(ccThinkingReq.thinkingBudget == 1024)

        var curlHeaders = HTTPFields()
        curlHeaders[.userAgent] = "curl/8.7.1"
        curlHeaders[GenerationRequest.anthropicVersionHeader] = "2023-06-01"
        let genericBody = #"{"max_tokens": 256, "system": "Be concise.", "messages": [{"role": "user", "content": "hi"}]}"#
        #expect(try await Self.post(app: app, uri: "/v1/messages", body: genericBody, headers: curlHeaders) == .ok)
        let genericReq = try #require(await engine.inner.lastRequest)
        #expect(genericReq.adapter == nil)
        #expect(genericReq.cacheKey == "shared")
        #expect(genericReq.enableThinking == true)
    }
}

@Suite("SuperAdapterComposer rank-concatenated parity, save/reload, and refusals (#675, #228)", .serialized)
struct SuperAdapterComposerTests {
    fileprivate struct SyntheticAdapter {
        let name: String
        let rank: Int
        let scale: Float
        let weight: Float
        let pairs: [PEFTAdapterTests.Pair]
        let directory: URL
    }

    @Test("Rank-concatenated super-adapter of 3 PEFT/mlx-lm adapters with weight 0.5 matches dense W + sum 0.5*s_i*B_i*A_i within 1e-4, keeps greedy tokens, and survives save/reload",
          arguments: [TinyModel.Family.qwen35, .lfm2Moe])
    func superAdapterMatchesMergedAndReloads(family: TinyModel.Family) throws {
        let adapters = try Self.makeThreeAdapters(for: family)
        defer { for adapter in adapters { try? FileManager.default.removeItem(at: adapter.directory) } }

        let spec = SuperAdapterSpec(sources: adapters.map {
            SuperAdapterSpec.Source(name: $0.name, directory: $0.directory, weight: $0.weight)
        })
        let baseModel = try TinyModel.make(family)
        let baseLogits = baseModel(PEFTAdapterTests.tokens, cache: nil)

        let composed = try SuperAdapterComposer.compose(spec, for: baseModel)
        #expect(composed.configuration.loraParameters.rank == 8)
        #expect(composed.configuration.loraParameters.scale == 1.0)
        #expect(composed.configuration.fineTuneType == .lora)

        let adaptedModel = try TinyModel.make(family)
        try adaptedModel.load(adapter: composed)
        let adaptedLogits = adaptedModel(PEFTAdapterTests.tokens, cache: nil)

        let mergedModel = try Self.denseMerged(family, adapters: adapters)
        let expectedLogits = mergedModel(PEFTAdapterTests.tokens, cache: nil)

        #expect(abs(adaptedLogits - expectedLogits).max().item(Float.self) < 1e-4)
        #expect(argMax(adaptedLogits, axis: -1).asArray(Int32.self) == argMax(expectedLogits, axis: -1).asArray(Int32.self))
        #expect(abs(adaptedLogits - baseLogits).max().item(Float.self) > 1e-4)

        let saveDir = FileManager.default.temporaryDirectory.appending(path: "super-saved-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: saveDir) }
        try SuperAdapterComposer.saveSuperAdapter(composed, to: saveDir)

        let savedConfig = try JSONDecoder().decode(
            LoRAConfiguration.self,
            from: Data(contentsOf: saveDir.appending(component: "adapter_config.json"))
        )
        #expect(savedConfig.loraParameters.rank == 8)
        #expect(savedConfig.loraParameters.scale == 1.0)
        #expect(FileManager.default.fileExists(atPath: saveDir.appending(component: "adapters.safetensors").path))

        let reloadedModel = try TinyModel.make(family)
        _ = try MLXInferenceEngine.attachAnyAdapterDirectory(saveDir, to: reloadedModel)
        let reloadedLogits = reloadedModel(PEFTAdapterTests.tokens, cache: nil)
        #expect(abs(reloadedLogits - adaptedLogits).max().item(Float.self) == 0)
    }

    @Test("Saved super-adapter serves through ModelOrchestratorActor as a single resident default adapter matching the dense merged model")
    func savedSuperAdapterServesThroughOrchestrator() async throws {
        let adapters = try Self.makeThreeAdapters(for: .qwen35)
        defer { for adapter in adapters { try? FileManager.default.removeItem(at: adapter.directory) } }

        let probe = try TinyModel.make(.qwen35)
        let composed = try SuperAdapterComposer.compose(
            sources: adapters.map { .init(name: $0.name, directory: $0.directory, weight: $0.weight) },
            for: probe
        )
        let superDir = FileManager.default.temporaryDirectory.appending(path: "super-served-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: superDir) }
        try SuperAdapterComposer.saveSuperAdapter(composed, to: superDir)

        let engine = try await TinyModel.engine(.qwen35)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadAdapters(["super": superDir])

        let mergedEngine = MLXInferenceEngine()
        await mergedEngine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-super-merged"),
            model: try Self.denseMerged(.qwen35, adapters: adapters),
            processor: ByteInputProcessor(),
            tokenizer: ByteTokenizer()
        )))

        let decision = HarnessRoute.resolve(
            FrozenHarnessFixtures.piCodingAgent,
            table: HarnessRoutingTable(defaultAdapter: "super")
        )
        #expect(decision.adapterName == "super")
        #expect(decision.defaultThinkingOff)

        let served = try await orchestrator.generate(request: greedy(Conversation.firstTurn, key: nil).withAdapter(decision.adapterName))
        let expected = try await mergedEngine.generate(request: greedy(Conversation.firstTurn, key: nil))
        #expect(served.generatedText == expected.generatedText)
        #expect(await engine.activeAdapter == "super")
    }

    @Test("SuperAdapterComposer refuses empty sources, non-finite weights, empty or duplicate names, unadaptable models, and empty saves")
    func refusals() throws {
        let model = try TinyModel.make(.qwen35)
        let dummyDir = FileManager.default.temporaryDirectory.appending(path: "super-dummy-\(UUID().uuidString)")

        #expect(throws: BasaltError.self) {
            _ = try SuperAdapterComposer.compose(sources: [], for: model)
        }
        for badWeight in [Float.nan, Float.infinity, -Float.infinity] {
            #expect(throws: BasaltError.self) {
                _ = try SuperAdapterComposer.compose(
                    sources: [.init(name: "bad", directory: dummyDir, weight: badWeight)],
                    for: model
                )
            }
        }
        #expect(throws: BasaltError.self) {
            _ = try SuperAdapterComposer.compose(
                sources: [.init(name: "  ", directory: dummyDir, weight: 0.5)],
                for: model
            )
        }
        #expect(throws: BasaltError.self) {
            _ = try SuperAdapterComposer.compose(
                sources: [
                    .init(name: "dup", directory: dummyDir, weight: 0.5),
                    .init(name: "dup", directory: dummyDir, weight: 0.5),
                ],
                for: model
            )
        }
        #expect(throws: BasaltError.self) {
            _ = try SuperAdapterComposer.compose(
                sources: [.init(name: "x", directory: dummyDir, weight: 0.5)],
                for: UnadaptableModel()
            )
        }
        let emptyContainer = LoRAContainer(
            configuration: LoRAConfiguration(numLayers: 4, fineTuneType: .lora, loraParameters: .init(rank: 4, scale: 1.0)),
            parameters: ModuleParameters()
        )
        #expect(throws: BasaltError.self) {
            try SuperAdapterComposer.saveSuperAdapter(emptyContainer, to: dummyDir)
        }
    }
}

extension SuperAdapterComposerTests {
    private static func makePairs(
        for model: any LanguageModel,
        suffixes: [String],
        rank: Int,
        seed: UInt64
    ) -> [PEFTAdapterTests.Pair] {
        let linears = model.namedModules()
            .compactMap { path, module in (module as? Linear).map { (path, $0) } }
            .filter { path, _ in suffixes.contains { path.hasSuffix(".\($0)") } }
        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            linears.map { path, linear in
                let (outputs, inputs) = linear.shape
                return PEFTAdapterTests.Pair(
                    path: path,
                    a: MLXRandom.normal([rank, inputs]) * 0.05,
                    b: MLXRandom.normal([outputs, rank]) * 0.05
                )
            }
        }
    }

    private static func writePEFTDirectory(
        pairs: [PEFTAdapterTests.Pair],
        rank: Int,
        alpha: Int
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(path: "super-peft-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let config = #"{"peft_type": "LORA", "r": \#(rank), "lora_alpha": \#(alpha), "use_rslora": false, "use_dora": false}"#
        try Data(config.utf8).write(to: dir.appending(component: "adapter_config.json"))
        var arrays: [String: MLXArray] = [:]
        for pair in pairs {
            arrays[PEFTAdapterTests.peftKey(pair.path) + ".lora_A.weight"] = pair.a
            arrays[PEFTAdapterTests.peftKey(pair.path) + ".lora_B.weight"] = pair.b
        }
        try save(arrays: arrays, url: dir.appending(component: PEFTLoRAReader.weightsFile))
        return dir
    }

    private static func writeMLXDirectory(
        pairs: [PEFTAdapterTests.Pair],
        rank: Int,
        scale: Float,
        numLayers: Int
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(path: "super-mlx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let keys = Set(pairs.compactMap { PEFTLoRAReader.layerRelative($0.path) }).sorted()
        let config = LoRAConfiguration(
            numLayers: numLayers,
            fineTuneType: .lora,
            loraParameters: .init(rank: rank, scale: scale, keys: keys)
        )
        try JSONEncoder().encode(config).write(to: dir.appending(component: "adapter_config.json"))
        var arrays: [String: MLXArray] = [:]
        for pair in pairs {
            arrays["\(pair.path).lora_a"] = pair.a.T
            arrays["\(pair.path).lora_b"] = pair.b.T
        }
        try save(arrays: arrays, url: dir.appending(component: "adapters.safetensors"))
        return dir
    }

    private static func makeThreeAdapters(for family: TinyModel.Family) throws -> [SyntheticAdapter] {
        let probe = try TinyModel.make(family)
        let allSuffixes = PEFTAdapterTests.targets[family] ?? []
        let subsetSuffixes = Array(allSuffixes.prefix(2))
        let numLayers = (probe as? LoRAModel)?.loraLayers.count ?? 4

        let piPairs = makePairs(for: probe, suffixes: allSuffixes, rank: 2, seed: 6751)
        let ocPairs = makePairs(for: probe, suffixes: allSuffixes, rank: 4, seed: 6752)
        let hmPairs = makePairs(for: probe, suffixes: subsetSuffixes, rank: 2, seed: 6753)

        let piDir = try writePEFTDirectory(pairs: piPairs, rank: 2, alpha: 4)
        let ocDir = try writeMLXDirectory(pairs: ocPairs, rank: 4, scale: 1.5, numLayers: numLayers)
        let hmDir = try writePEFTDirectory(pairs: hmPairs, rank: 2, alpha: 6)

        return [
            SyntheticAdapter(name: "pi", rank: 2, scale: 2.0, weight: 0.5, pairs: piPairs, directory: piDir),
            SyntheticAdapter(name: "opencode", rank: 4, scale: 1.5, weight: 0.5, pairs: ocPairs, directory: ocDir),
            SyntheticAdapter(name: "hermes", rank: 2, scale: 3.0, weight: 0.5, pairs: hmPairs, directory: hmDir),
        ]
    }

    private static func denseMerged(
        _ family: TinyModel.Family,
        adapters: [SyntheticAdapter]
    ) throws -> any LanguageModel {
        let model = try TinyModel.make(family)
        var deltas: [String: MLXArray] = [:]
        for adapter in adapters {
            let factor = adapter.weight * adapter.scale
            for pair in adapter.pairs {
                let delta = factor * matmul(pair.b, pair.a)
                deltas[pair.path] = deltas[pair.path].map { $0 + delta } ?? delta
            }
        }
        let linears = Dictionary(
            model.namedModules().compactMap { path, module in (module as? Linear).map { (path, $0) } },
            uniquingKeysWith: { first, _ in first }
        )
        let weights = deltas.compactMap { path, delta -> (String, MLXArray)? in
            guard let linear = linears[path] else { return nil }
            return ("\(path).weight", linear.weight + delta.asType(linear.weight.dtype))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        return model
    }
}
