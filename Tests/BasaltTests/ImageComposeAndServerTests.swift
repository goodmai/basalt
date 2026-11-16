import CryptoKit
import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import NIOCore
import Synchronization
import Testing
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltCore
@testable import BasaltServer

struct DeterministicTestImageSynthesizer: ImageSynthesizer {
    enum FailureMode: Sendable {
        case none
        case beforeFirstFrame
        case afterPartialFrame
    }

    let modelIdentifier: String
    let descriptor: ImageModelDescriptor
    var failureMode: FailureMode = .none
    var emitsPartials: Bool = false
    let recorded = Mutex<[ImageGenerationRequest]>([])

    init(
        id: String = "Tongyi-MAI/Z-Image-Turbo",
        descriptor: ImageModelDescriptor? = nil,
        failureMode: FailureMode = .none,
        emitsPartials: Bool = false
    ) {
        self.modelIdentifier = id
        self.descriptor = descriptor ?? ImageModelDescriptor.detect(identifier: id)
        self.failureMode = failureMode
        self.emitsPartials = emitsPartials
    }

    func generate(
        _ request: ImageGenerationRequest
    ) async throws -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
        try request.validate()
        recorded.withLock { $0.append(request) }
        if failureMode == .beforeFirstFrame {
            throw BasaltError.modelInferenceError(details: "simulated primary failure before frame")
        }
        let id = modelIdentifier
        let desc = descriptor
        let mode = failureMode
        let shouldEmitPartials = emitsPartials || request.stream || request.config.previewEverySteps > 0
        return AsyncThrowingStream { continuation in
            if shouldEmitPartials {
                let previewBytes = Data("preview:\(id):step2".utf8)
                let frame = ImagePartialFrame(
                    step: 2,
                    totalSteps: request.steps,
                    imageData: previewBytes,
                    format: request.outputFormat,
                    dimensions: ImageDimensions(width: min(256, request.width), height: min(256, request.height))
                )
                continuation.yield(.partial(frame))
            }
            if mode == .afterPartialFrame {
                continuation.finish(throwing: BasaltError.modelInferenceError(details: "simulated failure after partial"))
                return
            }
            let resolvedSeed = request.config.seed ?? 42
            let payload = Data(
                "img:\(id):seed=\(resolvedSeed):refs=\(request.config.referenceImages.count):prompt=\(request.effectivePrompt)".utf8
            )
            var artifact = ImageCompletedArtifact(
                imageData: payload,
                format: request.outputFormat,
                dimensions: ImageDimensions(width: request.width, height: request.height),
                hasAlpha: request.config.background == .transparent && desc.supportsNativeRGBA,
                seed: resolvedSeed,
                model: id
            )
            artifact.stepsExecuted = request.steps
            artifact.teacacheSkippedSteps = request.config.teacacheThreshold > 0 ? 1 : 0
            artifact.compute = desc.defaultCompute
            artifact.licenseTier = desc.licenseTier
            artifact.upsampledPrompt = request.config.upsampledPrompt
            artifact.elapsedMs = 37
            continuation.yield(.completed(artifact))
            continuation.finish()
        }
    }
}

@Suite("ComposeSpec imagine stage YAML parsing, validation and diffing (#517)")
struct ComposeImagineStageTests {
    private static let singleYAML = """
        version: 1
        name: studio-imagine
        models:
          chat: { source: mlx-community/Qwen3-4B-4bit }
          z_turbo: { source: Tongyi-MAI/Z-Image-Turbo }
        pipeline:
          answer: chat
          imagine: z_turbo
        """

    private static let fullYAML = """
        version: 1
        name: studio-imagine
        models:
          chat: { source: mlx-community/Qwen3-4B-4bit }
          vlm_judge: { source: mlx-community/Qwen3-VL-4B-4bit, keep: on-demand }
          z_turbo: { source: Tongyi-MAI/Z-Image-Turbo, keep: on-demand }
          sana_fast: { source: Efficient-Large-Model/SANA-1.5-1.6B }
        pipeline:
          answer: chat
          imagine:
            chain: [z_turbo, sana_fast]
            upsample_via: chat
            share_encoder: chat
            steps: 8
            size: 1024x768
            format: webp
            teacache_threshold: 0.15
            preview_every_steps: 2
            candidates: 3
            judge_via: vlm_judge
            on_gpu_busy: fallback
        pipelines:
          quick:
            answer: chat
            imagine: [sana_fast, z_turbo]
        """

    @Test("ComposeFile parses string shorthand, array chain, full mapping and ComposeDiff")
    func parsesShorthandChainAndFullMapping() throws {
        let singleSpec = try ComposeFile.parse(Self.singleYAML, file: "single.yaml")
        #expect(singleSpec.imagine?.chain == ["z_turbo"])

        let spec = try ComposeFile.parse(Self.fullYAML, file: "full.yaml")
        let imagine = try #require(spec.imagine)
        #expect(imagine.chain == ["z_turbo", "sana_fast"])
        #expect(imagine.upsampleVia == "chat" && imagine.shareEncoder == "chat")
        #expect(imagine.steps == 8 && imagine.size == "1024x768" && imagine.format == .webp)
        #expect(imagine.teacacheThreshold == 0.15 && imagine.previewEverySteps == 2)
        #expect(imagine.candidates == 3 && imagine.judgeVia == "vlm_judge" && imagine.onGPUBusy == .fallback)
        #expect(spec.view("quick")?.imagine?.chain == ["sana_fast", "z_turbo"])
        #expect(spec.imageGeneratorModels == ["z_turbo", "sana_fast"])
        #expect(spec.imagineAuxModels == ["chat", "vlm_judge"])

        let diff = ComposeDiff.between(singleSpec, spec)
        #expect(diff.imagine == "z_turbo → sana_fast")
        #expect(diff.added.contains("sana_fast") && diff.added.contains("vlm_judge"))
        #expect(diff.lines.contains { $0.contains("imagine: z_turbo → sana_fast") })
    }

    @Test("ComposeFile rejects undeclared imagine models, cloud shared encoders and invalid sizes or steps")
    func validatesImagineConstraints() {
        let badModel = """
            version: 1
            name: bad-model
            models:
              chat: { source: org/chat }
            pipeline:
              answer: chat
              imagine: [missing_gen]
            """
        #expect(throws: ServeConfigError.self) { try ComposeFile.parse(badModel, file: "bad1.yaml") }

        let badSize = """
            version: 1
            name: bad-size
            models:
              chat: { source: org/chat }
              z_turbo: { source: Tongyi-MAI/Z-Image-Turbo }
            pipeline:
              answer: chat
              imagine:
                chain: [z_turbo]
                size: 1000x1000
            """
        #expect(throws: ServeConfigError.self) { try ComposeFile.parse(badSize, file: "bad2.yaml") }

        let badAuxAndBounds = """
            version: 1
            name: bad-bounds
            models:
              chat: { source: org/chat }
              z_turbo: { source: Tongyi-MAI/Z-Image-Turbo }
            pipeline:
              answer: chat
              imagine:
                chain: [z_turbo]
                upsample_via: ghost_llm
                steps: 150
                candidates: 0
            """
        #expect(throws: ServeConfigError.self) { try ComposeFile.parse(badAuxAndBounds, file: "bad3.yaml") }
    }
}

@Suite("RoutedImageSynthesizer routing, upsample, best-of-N judge and ComposeHost wiring (#517)")
struct RoutedImageSynthesizerAndComposeHostTests {
    @Test("RoutedImageSynthesizer expands prompt, judges best-of-3 candidates and falls back on GPU busy")
    func routingUpsampleAndJudge() async throws {
        let primary = DeterministicTestImageSynthesizer(id: "Tongyi-MAI/Z-Image-Turbo")
        let sana = DeterministicTestImageSynthesizer(id: "Efficient-Large-Model/SANA-1.5-1.6B", descriptor: .sana15())
        let judgeCalls = Mutex<(String, [UInt64])?>(nil)
        var aux = RoutedImageSynthesizer.AuxStages(
            sharedEncoderModel: "chat",
            upsamplerModel: "chat",
            upsampler: { raw in "Vivid sunlit scene: \(raw)" },
            judgeModel: "vlm_judge",
            candidateJudge: { prompt, artifacts in
                judgeCalls.withLock { $0 = (prompt, artifacts.map(\.seed)) }
                return 1
            }
        )
        aux.defaultCandidates = 3
        let busyRouter = RoutedImageSynthesizer(
            entries: [.init(alias: "z_turbo", synthesizer: primary), .init(alias: "sana_fast", synthesizer: sana)],
            onGPUBusy: .fallback,
            isGPUBusy: { true },
            aux: aux
        )
        var req = try ImageGenerationRequest(model: "z_turbo", prompt: "alpine lake")
        req.config.seed = 100
        let busyDecision = await busyRouter.resolveRoute(req)
        #expect(busyDecision.selectedModel == sana.modelIdentifier && busyDecision.compute == .mlxANEHybrid)
        #expect(busyDecision.fallbackFrom == primary.modelIdentifier && busyDecision.sharedEncoderModel == "chat")

        var judgedArtifact: ImageCompletedArtifact?
        for try await event in try await busyRouter.generate(req) {
            if case .completed(let done) = event { judgedArtifact = done }
        }
        let winner = try #require(judgedArtifact)
        #expect(winner.seed == 101 && winner.upsampledPrompt == "Vivid sunlit scene: alpine lake")
        #expect(judgeCalls.withLock { $0?.1 == [100, 101, 102] } == true)
    }

    @Test("RoutedImageSynthesizer falls back to secondary entry when primary fails before emitting a frame")
    func primaryFailureFallback() async throws {
        let broken = DeterministicTestImageSynthesizer(id: "black-forest-labs/FLUX.2-klein-4B", failureMode: .beforeFirstFrame)
        let sana = DeterministicTestImageSynthesizer(id: "Efficient-Large-Model/SANA-1.5-1.6B", descriptor: .sana15())
        let router = RoutedImageSynthesizer(entries: [
            .init(alias: "flux_klein", synthesizer: broken),
            .init(alias: "sana_fast", synthesizer: sana),
        ])
        let req = try ImageGenerationRequest(model: "flux_klein", prompt: "alpine lake")
        var recovered: ImageCompletedArtifact?
        for try await event in try await router.generate(req) {
            if case .completed(let done) = event { recovered = done }
        }
        #expect(recovered?.model == sana.modelIdentifier)
    }

    private static let composedImagineYAML = """
        version: 1
        name: composed-imagine
        models:
          chat: { source: org/mock-chat }
          z_turbo: { source: Tongyi-MAI/Z-Image-Turbo }
          sana_fast: { source: Efficient-Large-Model/SANA-1.5-1.6B }
        pipeline:
          answer: chat
          imagine:
            chain: [z_turbo, sana_fast]
            upsample_via: chat
            share_encoder: chat
            candidates: 3
            judge_via: chat
            on_gpu_busy: fallback
        """

    @Test("ComposeHost wires imagine chain, shared encoder, upsample_via and judge_via into PipelineRegistry")
    func composeHostWiresImagineStage() async throws {
        let spec = try ComposeFile.parse(Self.composedImagineYAML, file: "imagine.yaml")
        let mockEngine = MockInferenceEngine()
        await Self.configureMockChat(mockEngine)
        let orchestrator = ModelOrchestratorActor(engine: mockEngine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "org/mock-chat")

        let sharedEncoderSeen = Mutex<[String]>([])
        let services = ComposeServices(
            resolve: { $0 }, load: { _, _, _, _ in nil }, decider: { _ in StubDecisionEngine() },
            decisions: { _, _, _, _, _ in nil }, key: { _ in "" }, cloudLimits: { _ in [:] },
            imageGenerator: { source, _, embedder in
                if embedder != nil { sharedEncoderSeen.withLock { $0.append(source) } }
                return DeterministicTestImageSynthesizer(id: source)
            },
            sharedTextEmbedder: { _ in [0.25, 0.5, 0.75] }
        )
        var start = ComposeStart(
            chat: orchestrator, decider: nil, decisions: nil,
            cloudGate: CloudGate(limits: [:], storage: nil), startup: PipelineStages(), besideDecode: true
        )
        start.admission = ComposeAdmission(
            sizes: ComposeSpec.Sizes(bytes: { _ in 1_000_000 }, onNeuralEngine: { $0.contains("SANA") })
        ) { (32_000_000_000, .desktop) }

        let host = ComposeHost(services: services, spec: spec, start: start)
        try await host.start(defaultDecisions: nil)
        #expect(sharedEncoderSeen.withLock { $0.count == 2 })

        let imageSynth = try #require(await host.registry.system(ComposeSpec.defaultPipeline)?.image)
        var req = try ImageGenerationRequest(model: "z_turbo", prompt: "neon street")
        req.config.seed = 10
        var finalArtifact: ImageCompletedArtifact?
        for try await event in try await imageSynth.generate(req) {
            if case .completed(let artifact) = event { finalArtifact = artifact }
        }
        let artifact = try #require(finalArtifact)
        #expect(artifact.seed == 12)
        #expect(artifact.upsampledPrompt == "Cinematic rain-slicked neon street with warm reflections.")
    }

    private static func configureMockChat(_ engine: MockInferenceEngine) async {
        await engine.setResponder { req in
            if req.prompt.contains("Pick the best candidate image") {
                return "2"
            }
            return "Cinematic rain-slicked neon street with warm reflections."
        }
    }
}

extension MockInferenceEngine {
    fileprivate func setResponder(_ fn: @escaping @Sendable (GenerationRequest) -> String) {
        responder = fn
    }
}

@Suite("OpenAI-compatible /v1/images/* REST, multipart edits, SSE streaming and asset store (#517)")
struct ImagesControllerRESTTests {
    private func makeTempStore() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(path: "basalt-images-\(UUID().uuidString)")
        return dir
    }

    private func makeApp(
        synthesizer: (any ImageSynthesizer)?,
        storeDirectory: URL
    ) throws -> Application<RouterResponder<BasicRequestContext>> {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        let server = RESTServer(
            orchestrator: orchestrator,
            config: try ServerConfig(modelPath: "org/mock-model"),
            imageSynthesizer: synthesizer,
            imageStoreDirectory: storeDirectory
        )
        return Application(router: server.makeRouter())
    }

    @Test("GET /v1/images/models and POST /v1/images/generations persist 0700 artifact and round-trip GET /v1/images/content/:id")
    func generationsModelsAndContentRoundTrip() async throws {
        let store = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: store) }
        let synth = DeterministicTestImageSynthesizer(id: "inclusionAI/Ming-Image-0.1-Design", descriptor: .mingImageDesign())
        try await makeApp(synthesizer: synth, storeDirectory: store).test(.router) { client in
            let modelsRes = try await client.execute(uri: "/v1/images/models", method: .get) { ($0.status, Data($0.body.readableBytesView)) }
            #expect(modelsRes.0 == .ok)
            let modelsEnv = try JSONDecoder().decode(ImagesController.ModelsEnvelope.self, from: modelsRes.1)
            #expect(modelsEnv.models.count == 1 && modelsEnv.models[0].supportsNativeRGBA)

            let genBody = #"{"prompt":"Translucent glass icon","size":"512x512","steps":6,"output_format":"png","background":"transparent","seed":77,"teacache_threshold":0.12}"#
            let genRes = try await client.execute(
                uri: "/v1/images/generations", method: .post,
                headers: [.contentType: "application/json"], body: ByteBuffer(string: genBody)
            ) { ($0.status, $0.headers[ImagesController.modelHeader], $0.headers[ImagesController.licenseHeader], Data($0.body.readableBytesView)) }
            #expect(genRes.0 == .ok && genRes.1 == "inclusionAI/Ming-Image-0.1-Design" && genRes.2 == "permissive_commercial")

            let decoded = try JSONDecoder().decode(ImagesController.ImagesResponseEnvelope.self, from: genRes.3)
            let item = try #require(decoded.data.first)
            #expect(item.width == 512 && item.height == 512 && item.hasAlpha && item.seed == 77 && item.teacacheSkipped == 1)

            let perms = (try FileManager.default.attributesOfItem(atPath: store.path)[.posixPermissions] as? NSNumber)?.intValue ?? 0
            #expect((perms & 0o777) == 0o700)

            let contentRes = try await client.execute(uri: item.url, method: .get) { ($0.status, $0.headers[.contentType], Data($0.body.readableBytesView)) }
            #expect(contentRes.0 == .ok && contentRes.1 == "image/png" && contentRes.2.base64EncodedString() == item.b64JSON)
        }
    }

    @Test("GET /v1/images/content/:id rejects path traversal and invalid asset identifiers")
    func contentPathTraversalGuard() async throws {
        let store = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: store) }
        let synth = DeterministicTestImageSynthesizer()
        try await makeApp(synthesizer: synth, storeDirectory: store).test(.router) { client in
            for badURI in [
                "/v1/images/content/..%2Fsecret.png",
                "/v1/images/content/../secret.png",
                "/v1/images/content/not-a-sha256.png",
                "/v1/images/content/nested/file.png",
            ] {
                let status = try await client.execute(uri: badURI, method: .get) { $0.status }
                #expect(status == .badRequest || status == .notFound)
            }
            let unknownSHA = String(repeating: "a", count: 64) + ".png"
            let missingStatus = try await client.execute(uri: "/v1/images/content/\(unknownSHA)", method: .get) { $0.status }
            #expect(missingStatus == .notFound)
        }
    }

    @Test("POST /v1/images/generations with stream true emits partial_image, completed and [DONE] SSE events")
    func streamingGenerationsSSE() async throws {
        let store = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: store) }
        let synth = DeterministicTestImageSynthesizer(emitsPartials: true)
        try await makeApp(synthesizer: synth, storeDirectory: store).test(.router) { client in
            let body = #"{"prompt":"Sunrise over basalt cliffs","size":"512x512","steps":8,"stream":true}"#
            let sse = try await client.execute(
                uri: "/v1/images/generations",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: body)
            ) { res in
                (res.status, res.headers[.contentType], String(buffer: res.body))
            }
            #expect(sse.0 == .ok && sse.1 == "text/event-stream")
            #expect(sse.2.contains(#""type":"image_generation.partial_image""#))
            #expect(sse.2.contains(#""type":"image_generation.completed""#))
            #expect(sse.2.contains("data: [DONE]\n\n"))
        }
    }

    @Test("POST /v1/images/edits accepts both JSON and multipart/form-data with multiple reference images, mask and audio")
    func jsonAndMultipartEdits() async throws {
        let store = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: store) }
        let synth = DeterministicTestImageSynthesizer(
            id: "black-forest-labs/FLUX.3-Image-Klein",
            descriptor: .flux3Klein()
        )
        let refA = Data("ref-a".utf8).base64EncodedString()
        let refB = Data("ref-b".utf8).base64EncodedString()
        let maskB64 = Data("mask-bytes".utf8).base64EncodedString()
        let audioB64 = Data("audio-bytes".utf8).base64EncodedString()

        try await makeApp(synthesizer: synth, storeDirectory: store).test(.router) { client in
            let jsonEdit = """
                {
                  "prompt": "Merge subjects on studio backdrop",
                  "images": ["\(refA)", "\(refB)"],
                  "mask": "\(maskB64)",
                  "audio": "\(audioB64)",
                  "size": "768x768",
                  "steps": 12,
                  "output_format": "webp"
                }
                """
            let jsonRes = try await client.execute(
                uri: "/v1/images/edits",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: jsonEdit)
            ) { $0.status }
            #expect(jsonRes == .ok)

            let boundary = "BasaltEditBoundary77"
            let multipart = Self.makeEditMultipart(
                boundary: boundary,
                images: [Data("part-1".utf8), Data("part-2".utf8), Data("part-3".utf8)],
                mask: Data("mask-part".utf8),
                audio: Data("audio-part".utf8)
            )
            let mpRes = try await client.execute(
                uri: "/v1/images/edits",
                method: .post,
                headers: [.contentType: "multipart/form-data; boundary=\(boundary)"],
                body: ByteBuffer(bytes: multipart)
            ) { $0.status }
            #expect(mpRes == .ok)
        }

        let recorded = synth.recorded.withLock { $0 }
        #expect(recorded.count == 2)
        #expect(recorded[0].config.referenceImages.count == 2 && recorded[0].config.maskImage != nil && recorded[0].isAudioConditioned)
        #expect(recorded[1].config.referenceImages.count == 3 && recorded[1].outputFormat == .jpeg)
    }

    private static func makeEditMultipart(
        boundary: String,
        images: [Data],
        mask: Data,
        audio: Data
    ) -> Data {
        var out = Data()
        func appendPart(name: String, filename: String? = nil, contentType: String? = nil, value: Data) {
            out.append(Data("--\(boundary)\r\n".utf8))
            var disposition = "Content-Disposition: form-data; name=\"\(name)\""
            if let filename { disposition += "; filename=\"\(filename)\"" }
            out.append(Data("\(disposition)\r\n".utf8))
            if let contentType { out.append(Data("Content-Type: \(contentType)\r\n".utf8)) }
            out.append(Data("\r\n".utf8))
            out.append(value)
            out.append(Data("\r\n".utf8))
        }
        for (idx, img) in images.enumerated() {
            appendPart(name: "image[]", filename: "ref\(idx).png", contentType: "image/png", value: img)
        }
        appendPart(name: "mask", filename: "mask.png", contentType: "image/png", value: mask)
        appendPart(name: "audio", filename: "clip.wav", contentType: "audio/wav", value: audio)
        appendPart(name: "prompt", value: Data("Multipart edit prompt".utf8))
        appendPart(name: "size", value: Data("512x512".utf8))
        appendPart(name: "steps", value: Data("4".utf8))
        appendPart(name: "background", value: Data("opaque".utf8))
        appendPart(name: "output_format", value: Data("jpeg".utf8))
        out.append(Data("--\(boundary)--\r\n".utf8))
        return out
    }
}

@Suite("MCP basalt_generate_image and basalt_edit_image tools (#517)")
struct MCPImageToolsTests {
    @Test("MCPServer advertises and executes basalt_generate_image and basalt_edit_image with persisted outputs")
    func mcpGenerateAndEditImage() async throws {
        let store = FileManager.default.temporaryDirectory.appending(path: "mcp-images-\(UUID().uuidString)")
        let allowedDir = FileManager.default.temporaryDirectory.appending(path: "mcp-allowed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: allowedDir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: store)
            try? FileManager.default.removeItem(at: allowedDir)
        }
        let refFile = allowedDir.appending(component: "input.png")
        try Data("allowed-ref-bytes".utf8).write(to: refFile)

        let synth = DeterministicTestImageSynthesizer(
            id: "Tongyi-MAI/Z-Image-Turbo",
            descriptor: .zImageTurbo()
        )
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine())) {
            capturer.capture($0)
        }
        .withImageDirectories([allowedDir])
        .withImageSynthesizer(synth, storeDirectory: store)

        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"basalt_generate_image","arguments":{"prompt":"Basalt hex columns","size":"512x512","steps":8,"seed":99}}}"#)
        let editCall = """
            {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"basalt_edit_image","arguments":{"prompt":"Add mist","path":"\(refFile.path)","size":"512x512","steps":6}}}
            """
        await server.dispatch(editCall)

        let lines = capturer.lines
        #expect(lines.count == 3)
        #expect(lines[0].contains("basalt_generate_image") && lines[0].contains("basalt_edit_image"))
        #expect(!lines[1].contains("\"error\"") && lines[1].contains("permissive_commercial"))
        #expect(!lines[2].contains("\"error\"") && lines[2].contains("/v1/images/content/"))

        let recorded = synth.recorded.withLock { $0 }
        #expect(recorded.count == 2)
        #expect(recorded[0].config.seed == 99 && !recorded[0].isEditRequest)
        #expect(recorded[1].isEditRequest && recorded[1].config.referenceImages.count == 1)
    }
}
