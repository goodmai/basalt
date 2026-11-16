import AppKit
import BasaltCore
import Foundation
import Hummingbird
import HummingbirdTesting
import SwiftUI
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltServer

@Suite("Mac app closing slice 3: server-side model hub download and delete over HTTP (#434)", .timeLimit(.minutes(1)))
struct AppServerHubBoundaryTests {
    @Test("LibraryClient routes download progress and deletion strictly over HTTP without local cache access")
    func downloadAndDeleteOverHTTP() async throws {
        let cache = FileManager.default.temporaryDirectory.appending(path: "hub-http-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        try FileManager.default.createDirectory(
            at: cache.appending(path: "models--org--cached/snapshots/main"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: cache.appending(path: "models--org--loaded/snapshots/main"), withIntermediateDirectories: true)

        let downloaded = Mutex<[String]>([])
        let hubOps = ModelHubOperations(cacheRoot: cache) { repoId, onFile in
            downloaded.withLock { $0.append(repoId) }
            onFile("config.json", 128, 256)
            onFile("model.safetensors", 512, 1024)
            onFile("config.json", 256, 256)
            onFile("model.safetensors", 1024, 1024)
            return cache.appending(path: "models--\(repoId.replacingOccurrences(of: "/", with: "--"))")
        }
        let chat = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        try await chat.loadModel(path: "/models/chat")
        await chat.setModelId("org/loaded")
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/loaded", apiKey: "hub-token"))
            .withModelHub(hubOps)

        try await Application(router: server.makeRouter()).test(.live) { live in
            let port = try #require(live.port)
            let endpoint = EngineEndpoint(base: try #require(URL(string: "http://localhost:\(port)")), token: "hub-token")
            let client = LibraryClient.live(endpoint, session: .isolated)
            let progressEvents = Mutex<[(Int64, Int64)]>([])
            try await client.download("org/fresh") { done, total in
                progressEvents.withLock { $0.append((done, total)) }
            }
            #expect(downloaded.withLock { $0 } == ["org/fresh"])
            #expect(progressEvents.withLock { $0 }.last?.0 == 1280 && progressEvents.withLock { $0 }.last?.1 == 1280)

            try await client.delete("org/cached")
            #expect(!FileManager.default.fileExists(atPath: cache.appending(path: "models--org--cached").path))

            await #expect { try await client.delete("org/loaded") } throws: { error in
                if case EngineError.refused(409, _) = error { true } else { false }
            }
            await #expect { try await client.delete("org/missing") } throws: { error in
                if case EngineError.refused(404, _) = error { true } else { false }
            }
        }
    }
}

@Suite("Mac app closing slice 1: duplex voice push-to-talk, HTTP STT/TTS, and read-aloud (#434)", .timeLimit(.minutes(1)))
struct AppDuplexVoiceTests {
    @Test("AudioCaptureActor resamples 48 kHz samples into a valid 16 kHz mono PCM16 WAV")
    func mono16kEncoding() throws {
        let source = (0 ..< 4_800).map { i in Float(0.5 * sin(2 * .pi * 440 * Double(i) / 48_000)) }
        let wav = AudioCaptureActor.encodeMono16kWAV(samples: source, sourceRate: 48_000)
        let decoded = try WAVEncoder.decode(wav)
        #expect(decoded.sampleRate == 16_000 && decoded.channels == 1 && decoded.samples.count == 1_600)
    }

    @MainActor
    private static func verifyVoiceStore(client: EngineClient) async {
        let submitted = Mutex<[String]>([])
        let played = Mutex<[Int]>([])
        let env = VoiceEnvironment(
            engine: { client },
            startCapture: {},
            stopCapture: { AudioFixture.sine(seconds: 0.15, rate: 16_000, channels: 1) },
            playAudio: { wav in played.withLock { $0.append(wav.count) } },
            stopPlayback: {},
            submitTranscript: { text in submitted.withLock { $0.append(text) } }
        )
        let store = Store(VoiceSessionState(), reduce: VoiceReducer.reduce, handle: VoiceEffects.handler(env))
        store.send(.setHandsFree(true))
        store.send(.pushToTalkPressed)
        #expect(store.state.isRecording)
        store.send(.pushToTalkReleased)
        await store.settle()
        #expect(submitted.withLock { $0 } == ["What is basalt?"])
        #expect(store.state.capture == .idle)

        let replyID = MessageID()
        store.send(.readAloud(replyID, "Basalt is dark volcanic rock."))
        await store.settle()
        #expect(played.withLock { $0.first ?? 0 } > 44)
        #expect(store.state.speakingMessage == nil)
        #expect(store.state.capture == .recording)
        store.send(.setHandsFree(false))
        #expect(!store.state.isRecording)
    }

    @Test("Push-to-talk records, transcribes over HTTP, sends chat, and synthesizes read-aloud audio")
    func pushToTalkAndReadAloudCycle() async throws {
        let engine = MockInferenceEngine()
        await engine.setMockTranscription(Transcription(text: "What is basalt?"))
        await engine.setResponder { _ in "Basalt is dark volcanic rock." }
        let chat = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await chat.loadModel(path: "/models/chat")
        let tts = SineSpeechSynthesizer()
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock", apiKey: "voice-token"))
            .withSpeechSynthesizer(tts)

        try await Application(router: server.makeRouter()).test(.live) { live in
            let port = try #require(live.port)
            let endpoint = EngineEndpoint(base: try #require(URL(string: "http://localhost:\(port)")), token: "voice-token")
            let client = EngineClient.http(endpoint, session: .isolated)
            await Self.verifyVoiceStore(client: client)
        }
    }
}

@Suite("Mac app closing slice 2: readiness probe, compose SSE events, and zero-restart hot switching (#434)", .timeLimit(.minutes(1)))
struct AppHotComposeSwitchTests {
    @MainActor
    private static func verifyHotSwitchApp(client: EngineClient) async throws -> Bool {
        let app = try AppModel(
            launch: HelperLaunch(executable: URL(fileURLWithPath: "/usr/bin/false"), arguments: ["serve", "--compose", "/tmp/a.yaml"]),
            model: "org/chat", storeURL: nil, drafts: DraftStore(load: { _ in "" }, save: { _, _ in })
        )
        let hotApplied = try await app.applyComposeYAML("name: desk\n", filePath: "/tmp/b.yaml", client: client)
        let args = await app.supervisor.launch.arguments
        #expect(args == ["serve", "--compose", "/tmp/b.yaml"])
        return hotApplied
    }

    @Test("EngineClient decodes readiness, streams compose SSE events, and hot-switches without relaunching")
    func readinessEventsAndHotSwitch() async throws {
        let events = ComposeEvents()
        await events.emit("loaded", model: "chat", "chat ready")
        let applied = Mutex<[String]>([])
        var control = ComposeControl(
            status: {
                ComposeStatus(
                    name: "desk", pipelines: ["default"],
                    models: [ComposeModelStatus(name: "chat", source: "org/chat", keep: "resident", state: .loading)],
                    spentToday: 0
                )
            },
            apply: { req throws(BasaltError) in
                applied.withLock { $0.append(req.yaml) }
                var diff = ComposeDiff()
                diff.reloaded = ["chat"]
                return diff
            }
        )
        control.events = events
        let chat = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        try await chat.loadModel(path: "/models/chat")
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/chat", apiKey: "cmp-token"))
            .withPipelines(nil, compose: control)

        try await Application(router: server.makeRouter()).test(.live) { live in
            let port = try #require(live.port)
            let endpoint = EngineEndpoint(base: try #require(URL(string: "http://localhost:\(port)")), token: "cmp-token")
            let client = EngineClient.http(endpoint, session: .isolated)

            let readiness = try await client.readiness()
            #expect(!readiness.isReady && readiness.status == "loading" && readiness.waiting == ["chat (loading)"])

            var received: ComposeStreamEvent?
            for try await item in client.composeEvents() {
                received = item
                break
            }
            #expect(received?.kind == "loaded" && received?.model == "chat" && received?.message == "chat ready")

            let hotApplied = try await Self.verifyHotSwitchApp(client: client)
            #expect(hotApplied && applied.withLock { $0 } == ["name: desk\n"])
            #expect(HotSwitchPlan.plan(from: ["serve", "--model", "org/a"], to: ["serve", "--model", "org/b"], isReady: true) == .hotLoadModel("org/b"))
        }
    }
}

@Suite("Mac app closing slice 4: basalt app CLI target resolution and cancellation-shielded shutdown (#434)", .timeLimit(.minutes(1)))
struct AppCLIShieldTests {
    @Test("AppCommand resolves an enclosed or sibling Basalt.app bundle or sibling BasaltMac binary")
    func commandTargetResolution() {
        let bundle = AppCommand.resolveTarget(
            executablePath: "/Applications/Basalt.app/Contents/Helpers/basalt",
            currentDirectory: "/tmp",
            environment: [:],
            isDirectory: { $0 == "/Applications/Basalt.app" },
            isExecutable: { _ in true }
        )
        #expect(bundle == .appBundle(URL(fileURLWithPath: "/Applications/Basalt.app")))
        if let bundle {
            let proc = AppCommand.makeProcess(target: bundle, model: "org/m", compose: nil, wait: true)
            #expect(proc.executableURL?.path == "/usr/bin/open" && proc.arguments == ["-a", "/Applications/Basalt.app", "-W", "--args", "-model", "org/m"])
        }

        let sibling = AppCommand.resolveTarget(
            executablePath: "/work/.build/release/Basalt",
            currentDirectory: "/work",
            environment: [:],
            isDirectory: { _ in false },
            isExecutable: { $0 == "/work/.build/release/BasaltMac" || $0 == "/work/.build/release/Basalt" }
        )
        #expect(sibling == .binary(executable: URL(fileURLWithPath: "/work/.build/release/BasaltMac"), helper: URL(fileURLWithPath: "/work/.build/release/Basalt")))
    }

    @Test("EngineSupervisor.shutdown completes process cleanup even when the calling Task is already cancelled")
    func shieldedShutdown() async throws {
        let suite = "basalt-settings-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        #expect(!AppSettings.keepEngineOnQuit(in: defaults))
        AppSettings.setKeepEngineOnQuit(true, in: defaults)
        #expect(AppSettings.keepEngineOnQuit(in: defaults))

        let fixture = try EngineSupervisorTests.Fixture()
        let supervisor = EngineSupervisor(launch: try fixture.script(EngineSupervisorTests.ready), environment: fixture.environment())
        await supervisor.start()
        #expect(await EngineSupervisorTests.until(supervisor) { $0 == .ready(port: 4242) } == .ready(port: 4242))
        let record = try #require(EngineRecords.file(fixture.recordURL).load())

        let cancelledCaller = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            #expect(Task.isCancelled)
            await supervisor.shutdown()
        }
        await cancelledCaller.value
        #expect(await supervisor.state.phase == .stopped)
        #expect(EngineRecords.file(fixture.recordURL).load() == nil)
        #expect(kill(record.pid, 0) != 0)
    }
}

@Suite("Mac app closing slice 5: MarkdownBlocksView AST, sibling branching, sidebar search, and string parity (#434)")
struct AppMarkdownBranchSearchTests {
    @MainActor
    @Test("MarkdownBlockParser parses headings, Swift code blocks, and GFM tables, and renders cleanly")
    func markdownParsingAndCopy() {
        let source = """
        ## Decode Throughput

        Measured on **Apple M4 Pro**:

        ```swift
        let speed = 84.2
        ```

        | Model | Bits | tok/s |
        | :--- | :---: | ---: |
        | Gemma 4 | 4 | 84.2 |
        """
        #expect(MarkdownBlockParser.hasStructuredBlocks(source))
        #expect(!MarkdownBlockParser.hasStructuredBlocks("Inline **bold** text only."))
        let blocks = MarkdownBlockParser.parse(source)
        #expect(blocks.count == 4)
        #expect(blocks[0] == .heading(level: 2, text: "Decode Throughput"))
        #expect(blocks[1] == .paragraph("Measured on **Apple M4 Pro**:"))
        #expect(blocks[2] == .codeBlock(language: "swift", code: "let speed = 84.2"))
        #expect(blocks[3] == .table(.init(
            headers: ["Model", "Bits", "tok/s"],
            rows: [["Gemma 4", "4", "84.2"]],
            alignments: [.leading, .center, .trailing]
        )))
        let host = NSHostingView(rootView: MarkdownBlocksView(blocks: blocks).frame(width: 600))
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.height > 0)
    }

    @Test("Regenerating creates sibling branches (1/2 and 2/2) and switches trajectories cleanly")
    func siblingBranching() {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "org/m"))
        state.draft = "Explain MoE."
        let (q1, a1, r1) = (MessageID(), MessageID(), RunID())
        _ = ChatReducer.reduce(&state, .send(message: q1, reply: a1, run: r1))
        _ = ChatReducer.reduce(&state, .delta(r1, .text("First answer.")))
        _ = ChatReducer.reduce(&state, .finished(r1, RunTiming(timeToFirstToken: .milliseconds(50), streaming: .milliseconds(200))))
        #expect(state.branches.position(for: a1, in: state.conversation) == nil)

        let (a2, r2) = (MessageID(), RunID())
        _ = ChatReducer.reduce(&state, .regenerateFrom(message: a1, reply: a2, run: r2))
        _ = ChatReducer.reduce(&state, .delta(r2, .text("Second answer.")))
        _ = ChatReducer.reduce(&state, .finished(r2, RunTiming(timeToFirstToken: .milliseconds(40), streaming: .milliseconds(180))))
        let secondPos = state.branches.position(for: a2, in: state.conversation)
        #expect(secondPos == BranchPosition(index: 1, count: 2) && secondPos?.label == "‹ 2/2 ›")

        _ = ChatReducer.reduce(&state, .selectBranch(message: a2, index: 0))
        #expect(state.conversation.messages.last?.text == "First answer.")
        let firstPos = state.branches.position(for: a1, in: state.conversation)
        #expect(firstPos == BranchPosition(index: 0, count: 2) && firstPos?.label == "‹ 1/2 ›")

        _ = ChatReducer.reduce(&state, .selectBranch(message: a1, index: 1))
        #expect(state.conversation.messages.last?.text == "Second answer.")
    }

    @MainActor
    @Test("Sidebar search filters conversations by message content, and English/Russian strings have exact key parity")
    func sidebarSearchAndLocalizationParity() async throws {
        let model = try AppModel(
            launch: HelperLaunch(executable: URL(fileURLWithPath: "/usr/bin/false"), arguments: []),
            model: "org/m", storeURL: nil, drafts: DraftStore(load: { _ in "" }, save: { _, _ in })
        )
        var volcano = ConversationSnapshot(id: ConversationID(), model: "org/m")
        volcano.title = "Geology"
        volcano.messages = [MessageSnapshot(id: MessageID(), role: .user, parts: [.text("Columnar basalt formations")])]
        var math = ConversationSnapshot(id: ConversationID(), model: "org/m")
        math.title = "Algebra"
        math.messages = [MessageSnapshot(id: MessageID(), role: .user, parts: [.text("Eigenvalue spectrum")])]
        try await model.repository.save(volcano)
        try await model.repository.save(math)
        await model.reload()
        #expect(model.visibleSummaries.count == 2)
        await model.updateSearch("columnar")
        #expect(model.visibleSummaries.map(\.id) == [volcano.id])
        await model.updateSearch("")
        #expect(model.visibleSummaries.count == 2)

        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let en = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/en.lproj/Localizable.strings")) as? [String: String])
        let ru = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/ru.lproj/Localizable.strings")) as? [String: String])
        #expect(Set(en.keys) == Set(ru.keys))
    }

    @MainActor
    @Test("Renders full RootView + ChatView showcase with Markdown code/tables, 2/2 branching, search, and duplex voice pill at 2x sRGB")
    func rendersClosingSlicesVisualShowcase() async throws {
        let model = try await VisualSceneFixtures.makeAppModel(phase: .ready(port: 58_100), withSummaries: true)
        model.isSearching = true
        model.newConversation()
        let chatID = try #require(model.selection)
        let store = try #require(model.chats[chatID])
        let (q1, a1, r1) = (MessageID(), MessageID(), RunID())
        store.send(.draftChanged("Show Apple M4 Pro QTIP & CosyVoice 3 results in Swift and Markdown."))
        store.send(.send(message: q1, reply: a1, run: r1))
        store.send(.delta(r1, .text("Draft answer v1.")))
        store.send(.finished(r1, RunTiming(timeToFirstToken: .milliseconds(45), streaming: .milliseconds(160))))
        let (a2, r2) = (MessageID(), RunID())
        let richReply = """
        ## Sub-4-Bit & Duplex Voice Benchmarks

        Measured natively on **Apple M4 Pro (24 GB)**:

        ```swift
        let synth = try await SpeechSynthesizerLoader.load(identifier: "Fun-CosyVoice3-0.5B", directory: dir)
        let stream = try await synth.synthesize(SpeechRequest(model: "tts_cosy", input: "Привет, Basalt!"))
        ```

        | Subsystem | Format / Backend | Throughput / Latency |
        | :--- | :---: | ---: |
        | Qwen3.8-27B | QTIP 2.5 bpw | 17.8 tok/s |
        | Kokoro-82M | Core AI ANE | 42 ms TTFA |
        | Fun-CosyVoice 3 | 6561 FSQ + DiT + HiFT | 110 ms TTFA |
        """
        store.send(.regenerateFrom(message: a1, reply: a2, run: r2))
        store.send(.delta(r2, .text(richReply)))
        store.send(.finished(r2, RunTiming(timeToFirstToken: .milliseconds(38), streaming: .milliseconds(210))))
        model.voice.store.send(.setHandsFree(true))

        let spec = VisualSceneSpec(id: .chatConversationRich, width: 1120, height: 780)
        let dark = try OffscreenSceneRenderer.render(RootView(model: model), spec: spec, variant: .darkStandard)
        let russian = try OffscreenSceneRenderer.render(RootView(model: model), spec: spec, variant: .darkRussian)
        #expect(dark.width == 2240 && dark.height == 1560 && !dark.png.isEmpty)
        #expect(russian.width == 2240 && russian.height == 1560 && !russian.png.isEmpty)
        if let rawDir = ProcessInfo.processInfo.environment["BASALT_VISUAL_EXPORT_DIR"], !rawDir.isEmpty {
            let outDir = URL(fileURLWithPath: rawDir)
            try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
            try dark.png.write(to: outDir.appending(path: "gui_434_closing_slices__dark.png"), options: .atomic)
            try russian.png.write(to: outDir.appending(path: "gui_434_closing_slices__ru.png"), options: .atomic)
        }
    }
}
