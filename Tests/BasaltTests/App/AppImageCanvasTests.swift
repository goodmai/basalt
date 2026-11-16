import AppKit
import BasaltCore
import CoreGraphics
import Foundation
import Hummingbird
import HummingbirdTesting
import SwiftUI
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

enum ImageCanvasTestFixtures {
    static func makeBasaltScenePNG(width: Int, height: Int, neonDusk: Bool) throws -> Data {
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: space, bitmapInfo: info
        ) else {
            throw VisualTestError.invalidImageData("failed to allocate basalt CGContext")
        }
        let w = CGFloat(width)
        let h = CGFloat(height)
        ctx.clear(CGRect(x: 0, y: 0, width: w, height: h))
        drawHorizonAndHalo(in: ctx, width: w, height: h, neonDusk: neonDusk)
        drawBasaltColumns(in: ctx, width: w, height: h, neonDusk: neonDusk)
        guard let cgImage = ctx.makeImage() else {
            throw VisualTestError.invalidImageData("failed to finalize basalt CGImage")
        }
        return try BitmapCodec.encodePNG(cgImage)
    }

    private static func drawHorizonAndHalo(in ctx: CGContext, width w: CGFloat, height h: CGFloat, neonDusk: Bool) {
        let seaColor = neonDusk
            ? CGColor(srgbRed: 0.06, green: 0.14, blue: 0.28, alpha: 0.96)
            : CGColor(srgbRed: 0.10, green: 0.38, blue: 0.48, alpha: 0.96)
        ctx.setFillColor(seaColor)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h * 0.34))

        let haloColor = neonDusk
            ? CGColor(srgbRed: 0.94, green: 0.18, blue: 0.68, alpha: 0.85)
            : CGColor(srgbRed: 0.98, green: 0.76, blue: 0.28, alpha: 0.85)
        ctx.setFillColor(haloColor)
        let radius = min(w, h) * 0.22
        ctx.fillEllipse(in: CGRect(x: w * 0.50 - radius, y: h * 0.56 - radius, width: radius * 2, height: radius * 2))
    }

    private static func drawBasaltColumns(in ctx: CGContext, width w: CGFloat, height h: CGFloat, neonDusk: Bool) {
        let heights: [CGFloat] = [0.42, 0.58, 0.76, 0.84, 0.68, 0.52, 0.38]
        let colW = w * 0.075
        let startX = w * 0.24
        for (index, ratio) in heights.enumerated() {
            let x = startX + CGFloat(index) * (colW * 0.96)
            let colH = h * ratio
            let shade = CGFloat(index % 3) * 0.04
            let body = neonDusk
                ? CGColor(srgbRed: 0.16 + shade, green: 0.12 + shade, blue: 0.28 + shade, alpha: 1.0)
                : CGColor(srgbRed: 0.22 + shade, green: 0.26 + shade, blue: 0.31 + shade, alpha: 1.0)
            ctx.setFillColor(body)
            ctx.fill(CGRect(x: x, y: h * 0.14, width: colW * 0.88, height: colH))

            let rim = neonDusk
                ? CGColor(srgbRed: 0.08, green: 0.92, blue: 0.96, alpha: 0.92)
                : CGColor(srgbRed: 0.98, green: 0.84, blue: 0.46, alpha: 0.88)
            ctx.setFillColor(rim)
            ctx.fill(CGRect(x: x, y: h * 0.14 + colH - 6, width: colW * 0.88, height: 6))
        }
        if neonDusk {
            ctx.setFillColor(CGColor(srgbRed: 0.0, green: 0.95, blue: 0.98, alpha: 0.90))
            ctx.fill(CGRect(x: w * 0.58, y: h * 0.20, width: 8, height: h * 0.26))
        }
    }
}

@Suite("Mac app image canvas: ChatReducer, I2I reference loop, sibling branching, AppMediaStore, and SQLite FTS (#517)")
struct AppImageCanvasReducerAndStoreTests {
    @Test("ChatReducer toggles Image Mode, parses /image prefix, streams partial previews, and clears previewPNG on completion")
    func reducerImageModeAndStreaming() throws {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "Tongyi-MAI/Z-Image-Turbo"))
        #expect(!state.imageMode && !state.usesImageComposer)
        _ = ChatReducer.reduce(&state, .toggleImageMode)
        #expect(state.imageMode && state.usesImageComposer)
        _ = ChatReducer.reduce(&state, .toggleImageMode)
        #expect(!state.imageMode)

        state.draft = "/image Hexagonal basalt spire at sunrise"
        #expect(state.usesImageComposer)
        let (q1, a1, r1) = (MessageID(), MessageID(), RunID())
        let effects1 = ChatReducer.reduce(&state, .send(message: q1, reply: a1, run: r1))
        guard effects1.count == 2, case .generateImage(let sentRun, let call1) = effects1[1] else {
            Issue.record("Expected .generateImage effect on /image prefix")
            return
        }
        #expect(sentRun == r1 && call1.prompt == "Hexagonal basalt spire at sunrise" && call1.referenceImages.isEmpty)
        #expect(state.conversation.title == "Hexagonal basalt spire at sunrise")

        let previewBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x01])
        _ = ChatReducer.reduce(&state, .imagePartial(r1, step: 2, totalSteps: 8, png: previewBytes, width: 256, height: 256))
        let partialImage = try #require(state.conversation.messages.last?.images.first)
        #expect(partialImage.currentStep == 2 && partialImage.previewPNG == previewBytes && partialImage.isStreamingPreview)

        var completed1 = GeneratedImagePart(
            assetID: "first.png", prompt: call1.prompt, model: "Tongyi-MAI/Z-Image-Turbo", width: 1024, height: 1024, totalSteps: 8
        )
        completed1.filePath = "/tmp/basalt-first.png"
        completed1.previewPNG = previewBytes
        completed1.hasAlpha = true
        _ = ChatReducer.reduce(&state, .imageCompleted(r1, completed1, StageReceipt(stage: "imagine", model: "z-image-turbo", ms: 4100)))
        let saved1 = try #require(state.conversation.messages.last?.images.first)
        #expect(saved1.previewPNG == nil && saved1.filePath == "/tmp/basalt-first.png" && state.run == .idle)
    }

    @Test("ChatReducer forks 1/2 and 2/2 sibling branches across generated images and attaches reference PNGs for conversational I2I edits")
    func reducerBranchingAndReferenceEdit() throws {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "Tongyi-MAI/Z-Image-Turbo"))
        state.imageMode = true
        state.draft = "Hexagonal basalt spire at sunrise"
        let (q1, a1, r1) = (MessageID(), MessageID(), RunID())
        _ = ChatReducer.reduce(&state, .send(message: q1, reply: a1, run: r1))
        var completed1 = GeneratedImagePart(
            assetID: "first.png", prompt: "Hexagonal basalt spire at sunrise",
            model: "Tongyi-MAI/Z-Image-Turbo", width: 1024, height: 1024, totalSteps: 8
        )
        completed1.filePath = "/tmp/basalt-first.png"
        _ = ChatReducer.reduce(&state, .imageCompleted(r1, completed1, StageReceipt(stage: "imagine", model: "z-image-turbo", ms: 4100)))

        let (a2, r2) = (MessageID(), RunID())
        let regenEffects = ChatReducer.reduce(&state, .regenerateFrom(message: a1, reply: a2, run: r2))
        #expect(regenEffects.contains { if case .generateImage(r2, _) = $0 { true } else { false } })
        var completed2 = completed1
        completed2.assetID = "second.png"
        completed2.filePath = "/tmp/basalt-second.png"
        _ = ChatReducer.reduce(&state, .imageCompleted(r2, completed2, StageReceipt(stage: "imagine", model: "z-image-turbo", ms: 3980)))
        let pos2 = state.branches.position(for: a2, in: state.conversation)
        #expect(pos2 == BranchPosition(index: 1, count: 2) && pos2?.label == "‹ 2/2 ›")

        _ = ChatReducer.reduce(&state, .selectBranch(message: a2, index: 0))
        #expect(state.conversation.messages.last?.images.first?.assetID == "first.png")
        _ = ChatReducer.reduce(&state, .selectBranch(message: a1, index: 1))
        #expect(state.conversation.messages.last?.images.first?.assetID == "second.png")

        _ = ChatReducer.reduce(&state, .editGeneratedImage(completed2))
        #expect(state.imageMode && state.beforeFilePath == "/tmp/basalt-second.png" && state.attachments.count == 1)
        state.draft = "Turn daylight into neon dusk"
        let (qEdit, aEdit, rEdit) = (MessageID(), MessageID(), RunID())
        let editEffects = ChatReducer.reduce(&state, .send(message: qEdit, reply: aEdit, run: rEdit))
        guard editEffects.count == 2, case .generateImage(rEdit, let editCall) = editEffects[1] else {
            Issue.record("Expected .generateImage effect for I2I edit")
            return
        }
        #expect(!editCall.referenceImages.isEmpty && editCall.beforeFilePath == "/tmp/basalt-second.png")
        #expect(editCall.endpointPath == "v1/images/edits")
    }

    @Test("AppMediaStore atomically stores SHA-256 PNGs with 0700 permissions, ConversationRepository strips previewPNG and indexes revisedPrompt, and en/ru strings match")
    func mediaStorePersistenceAndSearch() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appending(path: "media-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let store = AppMediaStore(directory: tempDir)
        let png = try ImageCanvasTestFixtures.makeBasaltScenePNG(width: 64, height: 64, neonDusk: false)
        let path1 = try store.store(png)
        let path2 = try store.store(png)
        #expect(path1 == path2)
        let expectedName = "\(AppMediaStore.sha256Hex(png)).png"
        #expect(URL(fileURLWithPath: path1).lastPathComponent == expectedName)
        #expect(store.read(assetID: expectedName) == png)
        #expect(store.read(assetID: "../outside.png") == nil)
        let attrs = try FileManager.default.attributesOfItem(atPath: tempDir.path)
        let perms = (attrs[.posixPermissions] as? Int ?? 0) & 0o777
        #expect(perms == 0o700)

        let repository = ConversationRepository(modelContainer: try ConversationRepository.container(at: nil))
        var conversation = ConversationSnapshot(id: ConversationID(), model: "Tongyi-MAI/Z-Image-Turbo")
        conversation.title = "Obsidian Study"
        var imagePart = GeneratedImagePart(
            assetID: expectedName, prompt: "Basalt arch", model: "Tongyi-MAI/Z-Image-Turbo", width: 2048, height: 2048, totalSteps: 8
        )
        imagePart.filePath = path1
        imagePart.beforeFilePath = "/tmp/prior.png"
        imagePart.hasAlpha = true
        imagePart.revisedPrompt = "Crystalline obsidian arch above bioluminescent caldera"
        imagePart.seed = 42
        imagePart.teacacheSkipped = 2
        imagePart.licenseTier = "permissive_commercial"
        imagePart.boxes = [ChatImageBox(label: "Arch", x: 0.2, y: 0.15, width: 0.6, height: 0.7)]
        imagePart.previewPNG = Data(repeating: 0xAA, count: 4096)
        conversation.messages = [
            MessageSnapshot(id: MessageID(), role: .user, parts: [.text("Draw a basalt arch")]),
            MessageSnapshot(id: MessageID(), role: .assistant, parts: [.image(imagePart)]),
        ]
        try await repository.save(conversation)
        let loaded = try #require(try await repository.load(conversation.id))
        let loadedImage = try #require(loaded.messages.last?.images.first)
        #expect(loadedImage.previewPNG == nil)
        #expect(loadedImage.revisedPrompt == imagePart.revisedPrompt && loadedImage.boxes == imagePart.boxes && loadedImage.seed == 42)
        #expect(try await repository.search("caldera") == [conversation.id])

        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let en = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/en.lproj/Localizable.strings")) as? [String: String])
        let ru = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/ru.lproj/Localizable.strings")) as? [String: String])
        #expect(Set(en.keys) == Set(ru.keys))
    }
}

@Suite("Mac app image canvas: live HTTP SSE streaming for generations and edits (#517)", .timeLimit(.minutes(1)))
struct AppImageEngineSSETests {
    private static func makeImageRouter(
        partialPNG: Data,
        finalPNG: Data,
        recordedPaths: Mutex<[String]>
    ) -> Router<BasicRequestContext> {
        let router = Router()
        let assetID = "\(AppMediaStore.sha256Hex(finalPNG)).png"
        router.post("/v1/images/generations") { request, _ in
            try await handleSSEImagePost(
                request: request, path: "/v1/images/generations", expectImages: false,
                partialPNG: partialPNG, finalPNG: finalPNG, recordedPaths: recordedPaths
            )
        }
        router.post("/v1/images/edits") { request, _ in
            try await handleSSEImagePost(
                request: request, path: "/v1/images/edits", expectImages: true,
                partialPNG: partialPNG, finalPNG: finalPNG, recordedPaths: recordedPaths
            )
        }
        router.get("/v1/images/content/\(assetID)") { _, _ in
            Response(status: .ok, headers: [.contentType: "image/png"], body: ResponseBody(byteBuffer: ByteBuffer(bytes: finalPNG)))
        }
        return router
    }

    private static func handleSSEImagePost(
        request: Request,
        path: String,
        expectImages: Bool,
        partialPNG: Data,
        finalPNG: Data,
        recordedPaths: Mutex<[String]>
    ) async throws -> Response {
        guard request.headers[.authorization] == "Bearer img-token" else {
            return Response(status: .unauthorized)
        }
        let bodyData = Data(try await request.body.collect(upTo: 4 << 20).readableBytesView)
        let json = (try? JSONSerialization.jsonObject(with: bodyData)) as? [String: Any]
        let images = json?["images"] as? [String] ?? []
        if expectImages && images.isEmpty {
            return Response(status: .badRequest)
        }
        recordedPaths.withLock { $0.append(path) }
        let assetID = "\(AppMediaStore.sha256Hex(finalPNG)).png"
        let partialFrame = """
        data: {"type":"image_generation.partial_image","step":2,"total_steps":8,"b64_json":"\(partialPNG.base64EncodedString())","width":256,"height":256,"format":"png"}

        """
        let completedFrame = """
        data: {"type":"image_generation.completed","asset_id":"\(assetID)","url":"/v1/images/content/\(assetID)","b64_json":"\(finalPNG.base64EncodedString())","revised_prompt":"Sunlit hexagonal basalt sea stack","width":1024,"height":1024,"format":"png","has_alpha":true,"seed":42,"steps":8,"teacache_skipped":2,"model":"Tongyi-MAI/Z-Image-Turbo","license_tier":"permissive_commercial","elapsed_ms":4820}

        data: [DONE]

        """
        let stream = AsyncStream<ByteBuffer> { continuation in
            continuation.yield(ByteBuffer(string: partialFrame + "\n"))
            continuation.yield(ByteBuffer(string: completedFrame + "\n"))
            continuation.finish()
        }
        return Response(status: .ok, headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"], body: ResponseBody(asyncSequence: stream))
    }

    @MainActor
    private static func verifyChatStoreImageFlow(client: EngineClient, mediaDir: URL, refPNG: Data) async throws {
        let env = ChatEnvironment(
            engine: { client },
            persist: { _ in },
            mediaStore: AppMediaStore(directory: mediaDir)
        )
        let store = Store(
            ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "Tongyi-MAI/Z-Image-Turbo")),
            reduce: ChatReducer.reduce,
            handle: ChatEffects.handler(env)
        )
        store.send(.toggleImageMode)
        store.send(.draftChanged("Hexagonal basalt spire"))
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        await store.settle()
        let firstGen = try #require(store.state.conversation.messages.last?.images.first)
        let savedPath = try #require(firstGen.filePath)
        #expect(FileManager.default.fileExists(atPath: savedPath) && firstGen.previewPNG == nil && firstGen.seed == 42)

        store.send(.editGeneratedImage(firstGen))
        store.send(.draftChanged("Add neon dusk reflections"))
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        await store.settle()
        let edited = try #require(store.state.conversation.messages.last?.images.first)
        #expect(edited.beforeFilePath == savedPath && edited.teacacheSkipped == 2)
    }

    @Test("EngineClient.streamImageGeneration streams partial and completed SSE frames against a live Hummingbird router")
    func liveSSEGenerationsAndEdits() async throws {
        let partialPNG = try ImageCanvasTestFixtures.makeBasaltScenePNG(width: 32, height: 32, neonDusk: false)
        let finalPNG = try ImageCanvasTestFixtures.makeBasaltScenePNG(width: 96, height: 96, neonDusk: true)
        let recordedPaths = Mutex<[String]>([])
        let mediaDir = FileManager.default.temporaryDirectory.appending(path: "sse-media-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: mediaDir) }
        let router = Self.makeImageRouter(partialPNG: partialPNG, finalPNG: finalPNG, recordedPaths: recordedPaths)

        try await Application(router: router).test(.live) { live in
            let port = try #require(live.port)
            let endpoint = EngineEndpoint(base: try #require(URL(string: "http://localhost:\(port)")), token: "img-token")
            let client = EngineClient.http(endpoint, session: URLSession(configuration: .ephemeral))

            var events: [ImageStreamEvent] = []
            for try await event in client.streamImageGeneration(prompt: "Basalt prism", model: "Tongyi-MAI/Z-Image-Turbo") {
                events.append(event)
            }
            #expect(events.count == 2)
            if case .partial(let step, let total, let png, _, _) = events.first {
                #expect(step == 2 && total == 8 && png == partialPNG)
            } else {
                Issue.record("Expected first event to be .partial")
            }
            let fetched = try await client.fetchImageContent("\(AppMediaStore.sha256Hex(finalPNG)).png")
            #expect(fetched == finalPNG)

            try await Self.verifyChatStoreImageFlow(client: client, mediaDir: mediaDir, refPNG: partialPNG)
            #expect(recordedPaths.withLock { $0 } == ["/v1/images/generations", "/v1/images/generations", "/v1/images/edits"])
        }
    }
}

@Suite("Mac app image canvas: 2x sRGB visual showcase of 2K RGBA checkerboard, Before/After slider, BBox overlay, and Russian UI (#517)", .serialized)
@MainActor
struct AppImageCanvasVisualShowcaseTests {
    private static func seedShowcaseStore(
        in model: AppModel,
        daylightPath: String,
        duskPath: String
    ) throws -> (ChatStore, MessageID) {
        model.newConversation()
        let chatID = try #require(model.selection)
        let store = try #require(model.chats[chatID])
        let (q1, a1, r1, a2, r2) = (
            MessageID(VisualSceneFixtures.fixedUUID(501)),
            MessageID(VisualSceneFixtures.fixedUUID(502)),
            RunID(VisualSceneFixtures.fixedUUID(503)),
            MessageID(VisualSceneFixtures.fixedUUID(504)),
            RunID(VisualSceneFixtures.fixedUUID(505))
        )
        let prompt = "Transform the daylight hexagonal basalt columns into neon dusk with RGBA sky."
        store.send(.toggleImageMode)
        store.send(.draftChanged(prompt))
        store.send(.send(message: q1, reply: a1, run: r1))
        var v1 = GeneratedImagePart(
            assetID: URL(fileURLWithPath: daylightPath).lastPathComponent,
            prompt: prompt, model: "Tongyi-MAI/Z-Image-Turbo", width: 2048, height: 1152, totalSteps: 8
        )
        (v1.filePath, v1.hasAlpha) = (daylightPath, true)
        store.send(.imageCompleted(r1, v1, StageReceipt(stage: "imagine", model: "z-image-turbo", ms: 4610)))

        store.send(.regenerateFrom(message: a1, reply: a2, run: r2))
        var v2 = v1
        (v2.assetID, v2.filePath, v2.beforeFilePath) = (URL(fileURLWithPath: duskPath).lastPathComponent, duskPath, daylightPath)
        v2.revisedPrompt = "2K RGBA geometric hexagonal basalt sea stack under neon magenta dusk with cyan lava seam"
        (v2.seed, v2.teacacheSkipped, v2.licenseTier) = (42, 2, "permissive_commercial")
        v2.boxes = [
            ChatImageBox(label: "Basalt Spire", x: 0.24, y: 0.14, width: 0.48, height: 0.72),
            ChatImageBox(label: "Neon Seam", x: 0.55, y: 0.52, width: 0.16, height: 0.30),
        ]
        store.send(.imageStage(r2, StageReceipt(stage: "upsample", model: "qwen3.5", ms: 140)))
        store.send(.imageCompleted(r2, v2, StageReceipt(stage: "imagine", model: "z-image-turbo", ms: 4820)))
        store.send(.editGeneratedImage(v2))
        store.send(.draftChanged("Add bioluminescent sea mist along the basalt base"))
        return (store, a2)
    }

    @Test("Renders full RootView + GeneratedImageCard showcase at 2240x1560 sRGB and exports dark & Russian PNGs")
    func rendersImageCanvasShowcase() async throws {
        let mediaDir = FileManager.default.temporaryDirectory.appending(path: "visual-media-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: mediaDir) }
        let mediaStore = AppMediaStore(directory: mediaDir)
        let daylightPNG = try ImageCanvasTestFixtures.makeBasaltScenePNG(width: 640, height: 360, neonDusk: false)
        let duskPNG = try ImageCanvasTestFixtures.makeBasaltScenePNG(width: 640, height: 360, neonDusk: true)
        let daylightPath = try mediaStore.store(daylightPNG)
        let duskPath = try mediaStore.store(duskPNG)

        let model = try await VisualSceneFixtures.makeAppModel(phase: .ready(port: 58_100), withSummaries: true)
        let (store, answerID) = try Self.seedShowcaseStore(in: model, daylightPath: daylightPath, duskPath: duskPath)
        #expect(store.state.conversation.messages.last?.receipt?.stages?.count == 2)

        let inspectorHost = NSHostingView(rootView: InspectorView(store: store, selected: answerID).frame(width: 320, height: 640))
        inspectorHost.layoutSubtreeIfNeeded()
        #expect(inspectorHost.fittingSize.width > 0)

        let spec = VisualSceneSpec(id: .chatConversationRich, width: 1120, height: 780)
        let dark = try OffscreenSceneRenderer.render(RootView(model: model), spec: spec, variant: .darkStandard)
        let russian = try OffscreenSceneRenderer.render(RootView(model: model), spec: spec, variant: .darkRussian)
        #expect(dark.width == 2240 && dark.height == 1560 && !dark.png.isEmpty)
        #expect(russian.width == 2240 && russian.height == 1560 && !russian.png.isEmpty)

        if let rawDir = ProcessInfo.processInfo.environment["BASALT_VISUAL_EXPORT_DIR"], !rawDir.isEmpty {
            let outDir = URL(fileURLWithPath: rawDir)
            try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
            try dark.png.write(to: outDir.appending(path: "gui_517_image_canvas__dark.png"), options: .atomic)
            try russian.png.write(to: outDir.appending(path: "gui_517_image_canvas__ru.png"), options: .atomic)
        }
    }
}
