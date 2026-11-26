import BasaltCore
import Foundation
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("App lifecycle, attachment I/O, image edit/regen, and localization defect regressions (A-A1, A-A3, A-A10, A-A11, A-A13)", .timeLimit(.minutes(1)))
struct AppLifecycleL10nDefectRegressionTests {
    typealias Fixture = EngineSupervisorTests.Fixture

    private static func makeStubEngine() -> EngineClient {
        EngineClient(
            chat: { _ in AsyncThrowingStream { $0.finish() } },
            health: { true },
            stats: { ServerTelemetrySnapshot() }
        )
    }

    @Test("refreshedEnvironment preserves BASALT_METALLIB while refreshing and dropping cloud keys (A-A1)")
    func refreshedEnvironmentPreservesMetallib() {
        let current = [
            "BASALT_METALLIB": "/Applications/Basalt.app/Contents/Resources/mlx.metallib",
            "OPENROUTER_API_KEY": "old-key",
        ]
        let updated = AppModel.refreshedEnvironment(
            preserving: current,
            cloud: ["GEMINI_API_KEY": "new-key"]
        )
        #expect(updated["BASALT_METALLIB"] == "/Applications/Basalt.app/Contents/Resources/mlx.metallib")
        #expect(updated["GEMINI_API_KEY"] == "new-key")
        #expect(updated["OPENROUTER_API_KEY"] == nil)

        let cleared = AppModel.refreshedEnvironment(preserving: current, cloud: [:])
        #expect(cleared == ["BASALT_METALLIB": "/Applications/Basalt.app/Contents/Resources/mlx.metallib"])
    }

    @MainActor
    @Test("restart and switchEngine preserve BASALT_METALLIB and relaunch when cloud keys change or forceRestart is set (A-A1)")
    func restartAndSwitchEngineRefreshEnvironment() async throws {
        let fixture = try Fixture()
        let baseLaunch = try fixture.script(EngineSupervisorTests.ready)
        let launch = HelperLaunch(
            executable: baseLaunch.executable,
            arguments: baseLaunch.arguments,
            environment: ["BASALT_METALLIB": "/app/mlx.metallib", "OLD_CLOUD_KEY": "stale"]
        )
        let cloudBox = Mutex<[String: String]>(["OLD_CLOUD_KEY": "stale"])
        let app = try AppModel(
            launch: launch,
            model: "org/m",
            storeURL: nil,
            environment: fixture.environment(),
            drafts: DraftStore(load: { _ in "" }, save: { _, _ in }),
            cloudEnvironment: { cloudBox.withLock { $0 } }
        )
        await app.start()
        #expect(await app.readyEndpoint(within: .seconds(20)) != nil)

        cloudBox.withLock { $0 = ["OPENROUTER_API_KEY": "sk-fresh"] }
        await app.switchEngine(launch.arguments)
        let afterCloudSwitch = await app.supervisor.launch
        #expect(afterCloudSwitch.environment["BASALT_METALLIB"] == "/app/mlx.metallib")
        #expect(afterCloudSwitch.environment["OPENROUTER_API_KEY"] == "sk-fresh")
        #expect(afterCloudSwitch.environment["OLD_CLOUD_KEY"] == nil)

        await app.restart()
        let afterRestart = await app.supervisor.launch
        #expect(afterRestart.environment["BASALT_METALLIB"] == "/app/mlx.metallib")
        #expect(afterRestart.environment["OPENROUTER_API_KEY"] == "sk-fresh")

        await app.switchEngine(launch.arguments, forceRestart: true)
        #expect(await app.readyEndpoint(within: .seconds(20)) != nil)
        await app.stop()
    }

    @MainActor
    @Test("recordLoadedModel and hotSwitchModel sync configuredModel, supervisor launch args, and loaded library state (A-A3)")
    func recordLoadedModelSyncsStateAndSupervisor() async throws {
        let fixture = try Fixture()
        let launch = try fixture.script(EngineSupervisorTests.ready)
        let previousDefault = UserDefaults.standard.string(forKey: EngineChoice.modelKey)
        defer {
            if let previousDefault {
                UserDefaults.standard.set(previousDefault, forKey: EngineChoice.modelKey)
            } else {
                UserDefaults.standard.removeObject(forKey: EngineChoice.modelKey)
            }
        }
        let app = try AppModel(
            launch: launch,
            model: "org/initial",
            storeURL: nil,
            environment: fixture.environment(),
            drafts: DraftStore(load: { _ in "" }, save: { _, _ in })
        )
        app.library.send(.listed([
            LibraryModel(id: "org/initial", sizeBytes: 100, isLoaded: true),
            LibraryModel(id: "org/next", sizeBytes: 100, isLoaded: false),
        ]))
        #expect(app.model == "org/initial")

        await app.recordLoadedModel("org/next")
        #expect(app.configuredModel == "org/next")
        #expect(app.model == "org/next")
        #expect(await app.supervisor.launch.arguments == ["serve", "--model", "org/next"])

        var stubClient = Self.makeStubEngine()
        stubClient.loadModel = { id in id }
        let switched = try await app.hotSwitchModel("org/third", client: stubClient)
        #expect(switched == "org/third")
        #expect(app.configuredModel == "org/third")
        #expect(await app.supervisor.launch.arguments == ["serve", "--model", "org/third"])
    }

    @Test("ChatAttachment.load rejects oversized, empty, unsupported, and directory URLs without reading and loads valid files async (A-A10)")
    func attachmentPreCheckAndAsyncLoad() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "attach-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(ChatAttachment.load(from: dir) == nil)

        let unsupported = dir.appending(path: "clip.mp4")
        try Data([0x00, 0x01]).write(to: unsupported)
        #expect(ChatAttachment.load(from: unsupported) == nil)

        let emptyPNG = dir.appending(path: "empty.png")
        try Data().write(to: emptyPNG)
        #expect(ChatAttachment.load(from: emptyPNG) == nil)

        let hugePNG = dir.appending(path: "huge.png")
        FileManager.default.createFile(atPath: hugePNG.path, contents: nil)
        let handle = try FileHandle(forWritingTo: hugePNG)
        try handle.truncate(atOffset: UInt64(ChatAttachment.byteLimit) + 1024)
        try handle.close()
        #expect(ChatAttachment.load(from: hugePNG) == nil)

        let validPNG = dir.appending(path: "ok.png")
        let payload = Data([0x89, 0x50, 0x4E, 0x47, 0x01, 0x02])
        try payload.write(to: validPNG)
        let loaded = await ChatAttachment.loadAsync(from: validPNG)
        #expect(loaded?.name == "ok.png" && loaded?.kind == "image" && loaded?.data == payload)
    }

    @Test("InlineVideoIO.copyFile copies atomically off MainActor and preserves source when destination matches (A-A10)")
    func inlineVideoAtomicCopy() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "video-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let src = dir.appending(path: "src.mp4")
        let dst = dir.appending(path: "dst.mp4")
        let bytes = Data(repeating: 0x5A, count: 4096)
        try bytes.write(to: src)

        try await InlineVideoIO.copyFile(from: src, to: dst)
        #expect(try Data(contentsOf: dst) == bytes)

        try Data([0x01]).write(to: dst)
        try await InlineVideoIO.copyFile(from: src, to: dst)
        #expect(try Data(contentsOf: dst) == bytes)

        try await InlineVideoIO.copyFile(from: src, to: src)
        #expect(try Data(contentsOf: src) == bytes)
    }

    @MainActor
    @Test("Missing reference image during edit or regeneration fails cleanly instead of uploading an 8-byte PNG header (A-A11)")
    func missingImageReferenceFailsCleanly() async throws {
        let missingPath = "/tmp/nonexistent-basalt-ref-\(UUID().uuidString).png"
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "Tongyi-MAI/Z-Image-Turbo"))
        var part = GeneratedImagePart(assetID: "gone.png", prompt: "Spire", width: 512, height: 512, totalSteps: 4)
        part.filePath = missingPath
        _ = ChatReducer.reduce(&state, .editGeneratedImage(part))
        #expect(state.attachments.count == 1 && state.attachments[0].data.isEmpty && state.attachments[0].sourcePath == missingPath)

        state.draft = "Make it purple"
        let run = RunID()
        let effects = ChatReducer.reduce(&state, .send(message: MessageID(), reply: MessageID(), run: run))
        guard effects.count == 2, case .generateImage(run, let call) = effects[1] else {
            Issue.record("Expected .generateImage effect")
            return
        }
        #expect(call.referenceImages.isEmpty && call.referenceImagePaths == [missingPath] && call.endpointPath == "v1/images/edits")

        let engineCalled = Mutex(false)
        var stubEngine = Self.makeStubEngine()
        stubEngine.streamImage = { _ in
            engineCalled.withLock { $0 = true }
            return AsyncThrowingStream { $0.finish() }
        }
        let frozenEngine = stubEngine
        let env = ChatEnvironment(engine: { frozenEngine }, persist: { _ in })
        let store = Store(
            ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "Tongyi-MAI/Z-Image-Turbo")),
            reduce: ChatReducer.reduce,
            handle: ChatEffects.handler(env)
        )
        store.send(.editGeneratedImage(part))
        store.send(.draftChanged("Make it purple"))
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        await store.settle()
        #expect(!engineCalled.withLock { $0 })
        if case .failed(.refused(let status, let message)) = store.state.run {
            #expect(status == 400 && message.contains("Reference image could not be read"))
        } else {
            Issue.record("Expected .failed(.refused(400, ...)), got \(store.state.run)")
        }
    }

    @Test("Image regeneration builds path-based references and resolves real bytes asynchronously (A-A11)")
    func regenerationResolvesRealReferenceBytes() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "regen-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let refFile = dir.appending(path: "before.png")
        let refBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0xDE, 0xAD])
        try refBytes.write(to: refFile)

        var img = GeneratedImagePart(assetID: "out.png", prompt: "Sunset", width: 512, height: 512, totalSteps: 4)
        img.beforeFilePath = refFile.path
        let replyID = MessageID()
        var conversation = ConversationSnapshot(id: ConversationID(), model: "Tongyi-MAI/Z-Image-Turbo")
        conversation.messages = [
            MessageSnapshot(id: MessageID(), role: .user, parts: [.text("Sunset")]),
            MessageSnapshot(id: replyID, role: .assistant, parts: [.image(img)]),
        ]
        let call = try #require(ChatReducer.imageRegenerationCall(in: conversation, target: replyID))
        #expect(call.referenceImages.isEmpty && call.referenceImagePaths == [refFile.path])
        let resolved = try await ChatEffects.resolveReferenceImages(call)
        #expect(resolved.referenceImages == [refBytes] && resolved.referenceImagePaths.isEmpty)
    }
}
