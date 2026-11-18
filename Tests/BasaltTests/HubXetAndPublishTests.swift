@testable import BasaltCore
import Foundation
import Hummingbird
import HummingbirdTesting
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltServer

private final class CatalogURLProtocol: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let url = request.url ?? URL(fileURLWithPath: "/")
        let body = Data("[{\"id\":\"goodmai/Nemotron-3bit-MLX\",\"downloads\":500,\"likes\":25,\"tags\":[\"mlx\"]}]".utf8)
        if let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: ["Content-Type": "application/json"]) {
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
        }
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@Suite("HuggingFaceHub Xet/HubClient adapter, 10 Hz SSE telemetry, live Hub search and native model publishing (#585)")
struct HubXetAndPublishTests {
    @Test("HubProgressThrottle pre-seeds manifest totals, coalesces bursts below 100 ms, emits start and completion edges, and computes EWMA rate + ETA")
    func progressThrottleCoalescingAndRate() throws {
        var throttle = HubProgressThrottle(minIntervalNanoseconds: 100_000_000, ewmaAlpha: 0.5)
        throttle.seedTotals(["model-00001.safetensors": 1_000_000, "config.json": 1_000])

        let rawFirst = throttle.record(file: "config.json", done: 1_000, total: 1_000, at: 1_000_000_000)
        let first = try #require(rawFirst)
        #expect(first.done == 1_000)
        #expect(first.total == 1_001_000)

        let suppressed = throttle.record(file: "model-00001.safetensors", done: 250_000, total: 1_000_000, at: 1_040_000_000)
        #expect(suppressed == nil)

        let rawSampled = throttle.record(file: "model-00001.safetensors", done: 500_000, total: 1_000_000, at: 1_120_000_000)
        let sampled = try #require(rawSampled)
        #expect(sampled.done == 501_000)
        #expect(sampled.total == 1_001_000)
        #expect(sampled.bytesPerSecond > 0)
        #expect((sampled.etaSeconds ?? -1) > 0)

        let rawCompleted = throttle.record(file: "model-00001.safetensors", done: 1_000_000, total: 1_000_000, at: 1_150_000_000)
        let completed = try #require(rawCompleted)
        #expect(completed.done == 1_001_000)
        #expect(completed.etaSeconds == 0)
    }

    @Test("HuggingFaceHub.collectPublishableFiles skips dot-files and resolves symlinks deterministically")
    func collectPublishableFilesFiltersDotFiles() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "basalt-pub-files-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("{\"model_type\":\"qwen3\"}".utf8).write(to: root.appending(path: "config.json"))
        try Data(repeating: 0xAB, count: 64).write(to: root.appending(path: "model.safetensors"))
        try Data("secret".utf8).write(to: root.appending(path: ".config.json.sha256"))

        let collected = try HuggingFaceHub.collectPublishableFiles(in: root)
        #expect(collected.map(\.relativePath) == ["config.json", "model.safetensors"])
        #expect(collected.reduce(Int64(0)) { $0 + $1.size } == Int64(22 + 64))

        let resolved = try PublishSubcommand.resolveSourceDirectory(root.path)
        #expect(resolved.standardizedFileURL == root.standardizedFileURL)
    }

    @Test("HuggingFaceHub native client builder, repo parser, catalog search, and LibraryEffects Hub runners work end-to-end")
    func nativeHubClientSearchAndEffects() async throws {
        let repo = try HuggingFaceHub.parseNativeRepoID("goodmai/Flint-40B-A2.5B-PV-MACaQ-3bit-MLX")
        #expect(repo.namespace == "goodmai" && repo.name == "Flint-40B-A2.5B-PV-MACaQ-3bit-MLX")
        #expect(throws: HFError.self) { _ = try HuggingFaceHub.parseNativeRepoID("invalid-no-slash") }

        let opts = HubPublishOptions(repoId: "goodmai/test-model", isPrivate: true, commitMessage: "Init", token: "hf_unit", maxConcurrent: 2)
        #expect(opts.isPrivate && opts.maxConcurrent == 2)

        let cfg = URLSessionConfiguration.ephemeral
        cfg.protocolClasses = [CatalogURLProtocol.self]
        let hub = HuggingFaceHub(configuration: cfg)
        _ = try hub.makeNativeHubClient(token: "hf_unit")
        let catalog = try await hub.searchCatalog(query: "Nemotron", limit: 3)
        #expect(catalog.map(\HFModelInfo.id) == ["goodmai/Nemotron-3bit-MLX"])

        let actions = Mutex<[LibraryAction]>([])
        var mockClient = LibraryClient(
            models: { [] },
            fit: { _ in ModelFit(fits: true, level: "comfortable", requiredBytes: 1, availableBytes: 2) },
            load: { _ in },
            download: { _, _ in },
            delete: { _ in }
        )
        mockClient.searchHub = { q, _ in [HubModelSuggestion(id: "goodmai/\(q)", downloads: 10, likes: 2)] }
        mockClient.publish = { req, onProg in
            onProg(HubTransferTelemetry(done: 50, total: 100, bytesPerSecond: 1_000, etaSeconds: 1))
            return HubPublishResult(repoId: req.repoId, url: "https://huggingface.co/\(req.repoId)", uploadedFiles: 1, totalBytes: 100)
        }
        mockClient.downloadWithTelemetry = { _, onProg in
            onProg(HubTransferTelemetry(done: 100, total: 100, bytesPerSecond: 2_000, etaSeconds: 0))
        }
        let frozenClient = mockClient
        let handler = LibraryEffects.handler(LibraryEnvironment(client: { frozenClient }, savePending: { _ in }))
        for eff in [LibraryEffect.searchHub("Flint"), .download("goodmai/Flint"), .publish(HubPublishRequest(sourceId: "local/a", repoId: "goodmai/a"))] {
            if case .run(_, let work) = handler(eff) {
                await work { act in actions.withLock { $0.append(act) } }
            }
        }
        if case .cancel(let sId) = handler(.cancelHubSearch), case .cancel(let pId) = handler(.cancelPublish("goodmai/a")) {
            #expect(sId == "hub-search" && pId == "publish:goodmai/a")
        } else {
            Issue.record("Expected .cancel EffectRun")
        }
        #expect(actions.withLock { $0.count >= 5 })
    }

    @Test("LibraryReducer drives live Hub search, download speed/ETA updates, and publish lifecycle")
    func libraryReducerHubSearchAndPublish() {
        var state = LibraryState()
        let shortEffects = LibraryReducer.reduce(&state, .searchQueryChanged("q"))
        #expect(shortEffects == [.cancelHubSearch])

        let searchEffects = LibraryReducer.reduce(&state, .searchQueryChanged("Nemotron"))
        #expect(searchEffects == [.searchHub("Nemotron")])

        let suggestions = [HubModelSuggestion(id: "goodmai/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-PV-MACaQ-3bit-MLX", downloads: 420, likes: 19, quantization: "3bit", parameterSize: "30B")]
        _ = LibraryReducer.reduce(&state, .hubSuggestionsLoaded("Nemotron", suggestions))
        #expect(state.hubSuggestions == suggestions)

        let model = LibraryModel(id: "local/Nemotron-PV-3bit", sizeBytes: 14_000_000_000, isLoaded: false)
        _ = LibraryReducer.reduce(&state, .openPublish(model))
        #expect(state.publishDraft?.sourceId == model.id)
        #expect(state.publishDraft?.isValid == false)

        var draft = state.publishDraft ?? PublishDraft(sourceId: model.id, sourceBytes: model.sizeBytes)
        draft.repoId = "goodmai/Nemotron-PV-3bit"
        draft.isPrivate = true
        _ = LibraryReducer.reduce(&state, .updatePublishDraft(draft))
        #expect(state.publishDraft?.isValid == true)

        let pubEffects = LibraryReducer.reduce(&state, .startPublish)
        #expect(pubEffects.count == 1)
        #expect(state.publishes["goodmai/Nemotron-PV-3bit"]?.active == true)

        _ = LibraryReducer.reduce(
            &state,
            .publishProgressed(
                "goodmai/Nemotron-PV-3bit",
                HubTransferTelemetry(done: 7_000_000_000, total: 14_000_000_000, bytesPerSecond: 70_000_000, etaSeconds: 100)
            )
        )
        #expect(state.publishes["goodmai/Nemotron-PV-3bit"]?.fraction == 0.5)
        #expect(HubTransferFormatting.rateAndETA(bytesPerSecond: 70_000_000, etaSeconds: 100)?.contains("1m 40s") == true)

        _ = LibraryReducer.reduce(
            &state,
            .publishCompleted(
                "goodmai/Nemotron-PV-3bit",
                HubPublishResult(repoId: "goodmai/Nemotron-PV-3bit", url: "https://huggingface.co/goodmai/Nemotron-PV-3bit", uploadedFiles: 8, totalBytes: 14_000_000_000)
            )
        )
        #expect(state.publishes["goodmai/Nemotron-PV-3bit"]?.active == false)
        #expect(state.publishes["goodmai/Nemotron-PV-3bit"]?.publishedURL == "https://huggingface.co/goodmai/Nemotron-PV-3bit")
    }

    @Test("REST /v1/models/hub, /v1/models/download and /v1/models/publish stream throttled telemetry and reject path traversal")
    func restHubSearchDownloadAndPublish() async throws {
        let cache = FileManager.default.temporaryDirectory.appending(path: "basalt-hub-rest-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let snap = cache.appending(path: "models--local--converted/snapshots/main")
        try FileManager.default.createDirectory(at: snap, withIntermediateDirectories: true)
        try Data("{\"model_type\":\"qwen3\"}".utf8).write(to: snap.appending(path: "config.json"))
        try Data(repeating: 0x11, count: 128).write(to: snap.appending(path: "model.safetensors"))

        let hubOps = ModelHubOperations(cacheRoot: cache) { _, onFile in
            onFile("model.safetensors", 512, 1024)
            onFile("model.safetensors", 1024, 1024)
            return snap
        }
        .withCatalogAndPublish(
            search: { query, _ in
                let raw = Data("{\"id\":\"goodmai/\(query)-4bit\",\"downloads\":1200,\"likes\":30,\"tags\":[\"mlx\"]}".utf8)
                return [try JSONDecoder().decode(HFModelInfo.self, from: raw)]
            },
            publish: { _, req, onFile in
                let sz = Int64(150)
                onFile("config.json", 22, sz)
                onFile("model.safetensors", sz, sz)
                return HubPublishResult(repoId: req.repoId, url: "https://huggingface.co/\(req.repoId)", uploadedFiles: 2, totalBytes: sz)
            }
        )

        let chat = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        try await chat.loadModel(path: "/models/chat")
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model", apiKey: AppEngineClientTests.token))
            .withModelHub(hubOps)

        try await Application(router: server.makeRouter()).test(.live) { liveClient in
            let port = try #require(liveClient.port)
            let endpoint = EngineEndpoint(base: try #require(URL(string: "http://localhost:\(port)")), token: AppEngineClientTests.token)
            let client = LibraryClient.live(endpoint, session: .isolated)

            let found = try await client.searchHub("Flint", 5)
            #expect(found.map(\.id) == ["goodmai/Flint-4bit"])

            let dlFrames = Mutex<[HubTransferTelemetry]>([])
            try await client.downloadWithTelemetry?("goodmai/Flint-4bit") { item in
                dlFrames.withLock { $0.append(item) }
            }
            #expect(dlFrames.withLock { !$0.isEmpty && $0.last?.done == 1024 })

            let pubFrames = Mutex<[HubTransferTelemetry]>([])
            let pubRes = try await client.publish(
                HubPublishRequest(sourceId: "local/converted", repoId: "goodmai/published-model", isPrivate: true)
            ) { item in
                pubFrames.withLock { $0.append(item) }
            }
            #expect(pubRes.url == "https://huggingface.co/goodmai/published-model")
            #expect(pubRes.uploadedFiles == 2)
            #expect(pubFrames.withLock { !$0.isEmpty })

            await #expect(throws: EngineError.self) {
                _ = try await client.publish(
                    HubPublishRequest(sourceId: "../etc/passwd", repoId: "goodmai/bad")
                ) { _ in }
            }
        }
    }
}

@Suite("macOS PublishModelSheet and LibraryView Hub Search / Xet Telemetry visual rendering (#585)", .serialized)
struct HubPublishVisualTests {
    @MainActor
    @Test("PublishModelSheet and LibraryView Hub suggestions + Xet transfer rows render cleanly across Light, Dark, and Russian")
    func renderPublishSheetAndHubLibrary() throws {
        var sheetState = LibraryState()
        let models = [
            LibraryModel(id: "goodmai/Flint-40B-A2.5B-PV-MACaQ-3bit-MLX", sizeBytes: 17_524_055_746, isLoaded: true),
            LibraryModel(id: "local/Nemotron-3.5-Lightning-PV-MACaQ-4bit", sizeBytes: 16_830_000_000, isLoaded: false),
        ]
        sheetState.models = models
        sheetState.fits = [
            models[0].id: ModelFit(fits: true, level: "comfortable", requiredBytes: 17_600_000_000, availableBytes: 18_400_000_000),
            models[1].id: ModelFit(fits: true, level: "comfortable", requiredBytes: 16_900_000_000, availableBytes: 18_400_000_000),
        ]
        var draft = PublishDraft(sourceId: models[1].id, sourceBytes: models[1].sizeBytes)
        draft.repoId = "goodmai/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-PV-MACaQ-4bit-MXFP4-MLX"
        draft.isPrivate = false
        sheetState.publishDraft = draft

        let env = LibraryEnvironment(client: { nil }, savePending: { _ in })
        let sheetStore = LibraryStore(sheetState, reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(env))
        let libStore = makeHubActiveLibraryStore(base: sheetState, env: env)
        let outDir = ProcessInfo.processInfo.environment["BASALT_SAVE_GUI_585"].map { URL(fileURLWithPath: $0) }
        if let outDir { try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) }

        for variant in [TraitVariant.lightStandard, .darkStandard, .darkRussian] {
            let bmpSheet = try OffscreenSceneRenderer.render(
                PublishModelSheet(store: sheetStore), spec: .init(id: .libraryMixed, width: 500, height: 360), variant: variant)
            #expect(bmpSheet.width == 1_000 && bmpSheet.height == 720 && !bmpSheet.png.isEmpty)
            let bmpLib = try OffscreenSceneRenderer.render(
                LibraryView(store: libStore), spec: .init(id: .libraryMixed, width: 760, height: 540), variant: variant)
            #expect(bmpLib.width == 1_520 && bmpLib.height == 1_080 && !bmpLib.png.isEmpty)
            if let outDir {
                try bmpSheet.png.write(to: outDir.appending(component: "gui_585_publish_sheet__\(variant.slug).png"), options: .atomic)
                try bmpLib.png.write(to: outDir.appending(component: "gui_585_library_hub_xet__\(variant.slug).png"), options: .atomic)
            }
        }
    }

    @MainActor
    private func makeHubActiveLibraryStore(base: LibraryState, env: LibraryEnvironment) -> LibraryStore {
        var active = base
        active.publishDraft = nil
        active.searchQuery = "Nemotron"
        active.hubSuggestions = [
            HubModelSuggestion(
                id: "goodmai/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-PV-MACaQ-3bit-MLX",
                downloads: 1_840, likes: 42, quantization: "3bit", parameterSize: "30B"),
            HubModelSuggestion(
                id: "goodmai/Flint-40B-A2.5B-PV-MACaQ-3bit-MLX",
                downloads: 960, likes: 29, quantization: "3bit", parameterSize: "40B"),
        ]
        let dlId = "goodmai/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-PV-MACaQ-4bit-MXFP4-MLX"
        _ = LibraryReducer.reduce(&active, .download(dlId))
        _ = LibraryReducer.reduce(&active, .progressed(dlId, done: 11_200_000_000, total: 16_830_000_000, bytesPerSecond: 148_500_000, etaSeconds: 38))
        var pub = PublishJobState(sourceId: base.models[1].id, repoId: "goodmai/Nemotron-PV-4bit")
        pub.done = 9_400_000_000
        pub.total = 16_830_000_000
        pub.bytesPerSecond = 84_200_000
        pub.etaSeconds = 88
        active.publishes[pub.repoId] = pub
        return LibraryStore(active, reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(env))
    }
}
