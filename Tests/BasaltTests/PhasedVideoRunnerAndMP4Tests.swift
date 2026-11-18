import AVFoundation
import ArgumentParser
import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import MLX
import MLXNN
import NIOCore
import SwiftUI
import Testing
@testable import BasaltApp
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Phased 24 GB UMA VideoMemoryPlanner, TeaCacheStepSchedule and request validation (#583)")
struct PhasedVideoPlanCoreTests {
    @Test("VideoMemoryPlanner selects strict phased teardown on 16-24 GB UMA and resident mode on 128-256 GB")
    func memoryPlannerAcrossUMATiers() {
        let plan16 = VideoMemoryPlanner.plan(tier: .gb16, footprint: .standard24GB)
        #expect(plan16.mode == .strictPhasedTeardown)
        #expect(plan16.streamMoEExpertsFromSSD)
        #expect(plan16.unloadTextEncoderAfterPhase1 && plan16.unloadDiTAfterPhase2 && plan16.unloadDecodersAfterPhase3)
        #expect(plan16.clearCacheBetweenPhases && plan16.useTiledVAE && plan16.fitsWithoutSwap)

        let plan24 = VideoMemoryPlanner.plan(tier: .gb24, footprint: .standard24GB)
        #expect(plan24.mode == .strictPhasedTeardown)
        #expect(plan24.residentTotalGB > ComposeMemoryTier.gb24.workingSetGB)
        #expect(plan24.peakPhaseGB <= ComposeMemoryTier.gb24.workingSetGB)
        #expect(plan24.fitsWithoutSwap && plan24.clearCacheBetweenPhases)

        let plan128 = VideoMemoryPlanner.plan(tier: .gb128, footprint: .standard24GB)
        #expect(plan128.mode == .resident)
        #expect(!plan128.unloadTextEncoderAfterPhase1 && !plan128.unloadDiTAfterPhase2 && !plan128.useTiledVAE)

        let tight128 = VideoMemoryPlanner.plan(tier: .gb128, footprint: .standard24GB, directive: .tight)
        #expect(tight128.mode == .strictPhasedTeardown && tight128.clearCacheBetweenPhases)

        for tier in ComposeMemoryTier.allCases {
            let planned = VideoMemoryPlanner.plan(tier: tier, footprint: .standard24GB)
            #expect(planned.tier == tier && planned.peakPhaseGB > 0)
        }
    }

    @Test("TeaCacheStepSchedule preserves boundary steps, skips low-delta interior steps, and resets accumulated drift")
    func teaCacheScheduleInvariants() {
        let deltas = [0.50, 0.002, 0.002, 0.22, 0.002, 0.40]
        let schedule = TeaCacheStepSchedule.simulate(totalSteps: 6, threshold: 0.15, deltas: deltas)
        #expect(schedule.executedSteps.first == 0)
        #expect(schedule.executedSteps.last == 5)
        #expect(!schedule.skippedSteps.isEmpty)
        #expect(!schedule.skippedSteps.contains(0) && !schedule.skippedSteps.contains(5))
        #expect(schedule.executedSteps.count + schedule.skippedSteps.count == 6)

        let disabled = TeaCacheStepSchedule.simulate(totalSteps: 6, threshold: 0.0, deltas: deltas)
        #expect(disabled.skippedSteps.isEmpty && disabled.executedSteps == Array(0 ..< 6))
    }

    @Test("VideoGenerationRequest validates dimensions, frame count, fps, steps, and audio sample rates")
    func requestValidationContracts() throws {
        let valid = try VideoGenerationRequest(
            model: "goodmai/Wan2.2-14B-MACaQ-MXFP4",
            prompt: "Ocean swell crashing against hexagonal basalt cliffs at golden hour",
            dimensions: try VideoGenerationRequest.parseSize("832x480"),
            temporal: VideoTemporalSpec(frames: 25, fps: 24, steps: 16)
        )
        #expect(valid.width == 832 && valid.height == 480 && abs(valid.durationSeconds - (25.0 / 24.0)) < 1e-6)

        #expect(throws: BasaltError.self) {
            _ = try VideoGenerationRequest(model: "m", prompt: "   ")
        }
        #expect(throws: BasaltError.self) {
            _ = try VideoGenerationRequest(model: "m", prompt: "ok", dimensions: VideoDimensions(width: 500, height: 512))
        }
        #expect(throws: BasaltError.self) {
            _ = try VideoGenerationRequest(model: "m", prompt: "ok", temporal: VideoTemporalSpec(frames: 0, fps: 24, steps: 8))
        }
        var badAudioCfg = VideoGenerationConfig()
        badAudioCfg.audioSampleRate = 11_025
        #expect(throws: BasaltError.self) {
            _ = try VideoGenerationRequest(model: "m", prompt: "ok", config: badAudioCfg)
        }
    }
}

actor VideoTestCancellationBarrier {
    private var taskToCancel: Task<Void, Never>?
    private var cancelledOnce = false
    private var sessionCleaned = false
    private var cleanupWaiter: CheckedContinuation<Void, Never>?

    func register(_ task: Task<Void, Never>) {
        taskToCancel = task
    }

    func onPhaseEvent(_ event: VideoPhaseLifecycleEvent) {
        if event == .loaded(.denoising), !cancelledOnce {
            cancelledOnce = true
            taskToCancel?.cancel()
        }
        if event == .sessionCleanedUp {
            sessionCleaned = true
            cleanupWaiter?.resume()
            cleanupWaiter = nil
        }
    }

    func waitForSessionCleanup() async {
        if sessionCleaned { return }
        await withCheckedContinuation { continuation in
            if sessionCleaned {
                continuation.resume()
            } else {
                cleanupWaiter = continuation
            }
        }
    }
}

@Suite("AVAssetWriterMP4Muxer and PhasedVideoRunnerActor deterministic phased teardown (#583)", .serialized)
struct PhasedVideoRunnerExecutionTests {
    private static func makeSyntheticFrames(width: Int, height: Int, count: Int) throws -> [AVVideoFrameBuffer] {
        var frames: [AVVideoFrameBuffer] = []
        for f in 0 ..< count {
            var rgb = [Float](repeating: 0, count: width * height * 3)
            for idx in 0 ..< (width * height) {
                rgb[idx * 3] = Float((idx + f * 7) % 64) / 63.0
                rgb[idx * 3 + 1] = Float((idx / width + f * 3) % 64) / 63.0
                rgb[idx * 3 + 2] = 0.55
            }
            frames.append(try AVVideoFrameBuffer.fromRGBFloats(rgb.span, width: width, height: height))
        }
        return frames
    }

    @Test("AVAssetWriterMP4Muxer writes playable H.264+AAC .mp4 and HEVC+PCM .mov with verified AVURLAsset video and audio tracks")
    func nativeAVAssetWriterMP4AndMOVMuxing() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "mux-583-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let frames = try Self.makeSyntheticFrames(width: 64, height: 64, count: 12)
        let pcmSamples = [Float](repeating: 0.15, count: 22_050 * 2)
        let audio = try AVAudioTrackPayload.fromFloatSamples(pcmSamples.span, sampleRate: 44_100, channels: 2)

        let mp4URL = root.appending(component: "clip.mp4")
        let mp4Spec = AVMuxingSpec(containerFormat: .mp4, videoCodec: .h264, audioCodec: .aac, dimensions: .init(width: 64, height: 64), fps: 24)
        try await AVAssetWriterMP4Muxer.muxToFile(frames: frames, audio: audio, spec: mp4Spec, outputURL: mp4URL)
        let mp4Asset = AVURLAsset(url: mp4URL)
        let mp4VideoTracks = try await mp4Asset.loadTracks(withMediaType: .video)
        let mp4AudioTracks = try await mp4Asset.loadTracks(withMediaType: .audio)
        let mp4Playable = try await mp4Asset.load(.isPlayable)
        let mp4Duration = try await mp4Asset.load(.duration)
        #expect(mp4VideoTracks.count == 1 && mp4AudioTracks.count == 1 && mp4Playable && mp4Duration.seconds > 0.3)
        let naturalSize = try await mp4VideoTracks[0].load(.naturalSize)
        #expect(Int(naturalSize.width) == 64 && Int(naturalSize.height) == 64)

        let movURL = root.appending(component: "clip.mov")
        let movSpec = AVMuxingSpec(containerFormat: .mov, videoCodec: .hevc, audioCodec: .linearPCM, dimensions: .init(width: 64, height: 64), fps: 24)
        try await AVAssetWriterMP4Muxer.muxToFile(frames: frames, audio: audio, spec: movSpec, outputURL: movURL)
        let movAsset = AVURLAsset(url: movURL)
        #expect(try await movAsset.loadTracks(withMediaType: .video).count == 1)
        #expect(try await movAsset.loadTracks(withMediaType: .audio).count == 1)
    }

    @Test("PhasedVideoRunnerActor executes Phases 1-3 with strict inter-phase teardown and produces a playable A/V .mp4")
    func phasedRunnerFullExecutionAndTrackVerification() async throws {
        let runner = try PhasedVideoRunnerActor(identifier: "goodmai/Wan2.2-14B-MACaQ-4bit", options: .init(memoryTier: .gb24))
        var config = VideoGenerationConfig(guidanceScale: 4.5, seed: 77, codec: .h264, includeAudio: true, teacacheThreshold: 0.15)
        config.previewEverySteps = 2
        let request = try VideoGenerationRequest(
            model: "goodmai/Wan2.2-14B-MACaQ-4bit",
            prompt: "Bioluminescent tide flowing through basalt sea cave",
            dimensions: VideoDimensions(width: 64, height: 64),
            temporal: VideoTemporalSpec(frames: 8, fps: 16, steps: 6),
            containerFormat: .mp4,
            config: config
        )
        let (result, seenPhases, sawPreview) = try await Self.collectRunnerOutput(runner: runner, request: request)
        #expect(sawPreview && result.executionMode == .strictPhasedTeardown)
        #expect(Set(seenPhases) == Set(VideoExecutionPhase.allCases))
        #expect(!(await runner.hasResidentStageModules))
        #expect(await runner.recordedPhaseEvents == Self.expectedPhasedEvents)

        let tmpMP4 = FileManager.default.temporaryDirectory.appending(path: "runner-out-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: tmpMP4) }
        try result.videoData.write(to: tmpMP4)
        let asset = AVURLAsset(url: tmpMP4)
        #expect(try await asset.loadTracks(withMediaType: .video).count == 1)
        #expect(try await asset.loadTracks(withMediaType: .audio).count == 1)
    }

    private static let expectedPhasedEvents: [VideoPhaseLifecycleEvent] = [
        .loaded(.textConditioning), .completed(.textConditioning), .unloaded(.textConditioning), .cacheCleared(after: .textConditioning),
        .loaded(.denoising), .completed(.denoising), .unloaded(.denoising), .cacheCleared(after: .denoising),
        .loaded(.vaeAndAudioMuxing), .completed(.vaeAndAudioMuxing), .unloaded(.vaeAndAudioMuxing), .cacheCleared(after: .vaeAndAudioMuxing),
        .sessionCleanedUp,
    ]

    private static func collectRunnerOutput(
        runner: PhasedVideoRunnerActor,
        request: VideoGenerationRequest
    ) async throws -> (VideoGenerationResult, [VideoExecutionPhase], Bool) {
        var seenPhases: [VideoExecutionPhase] = []
        var completed: VideoGenerationResult?
        var sawPreview = false
        for try await event in try await runner.generate(request) {
            switch event {
            case .progress(let prog):
                seenPhases.append(prog.phase)
                if prog.previewPNG != nil { sawPreview = true }
            case .completed(let done):
                completed = done
            }
        }
        return (try #require(completed), seenPhases, sawPreview)
    }

    @Test("Cancelling PhasedVideoRunnerActor mid-Phase-2 runs shielded teardown and clears MLX cache without leaking modules")
    func shieldedTeardownOnMidPhaseCancellation() async throws {
        let barrier = VideoTestCancellationBarrier()
        let runner = try PhasedVideoRunnerActor(
            identifier: "goodmai/LTX-2-MACaQ-4bit",
            options: .init(memoryTier: .gb24, phaseObserver: { await barrier.onPhaseEvent($0) })
        )
        let request = try VideoGenerationRequest(
            model: "goodmai/LTX-2-MACaQ-4bit",
            prompt: "Storm lightning over volcanic caldera",
            dimensions: VideoDimensions(width: 64, height: 64),
            temporal: VideoTemporalSpec(frames: 8, fps: 16, steps: 12)
        )
        let runTask = Task {
            do {
                for try await _ in try await runner.generate(request) {}
            } catch {}
        }
        await barrier.register(runTask)
        await runTask.value
        await barrier.waitForSessionCleanup()

        let events = await runner.recordedPhaseEvents
        #expect(events.contains(.unloaded(.textConditioning)))
        #expect(events.contains(.unloaded(.denoising)))
        #expect(events.contains(.cacheCleared(after: .denoising)))
        #expect(events.last == .sessionCleanedUp)
        #expect(!(await runner.hasResidentStageModules))
    }
}

@Suite("Universal Converter Video DiT checkpoint compatibility: Dense -> MACaQ MXFP4 & QTIP -> PV-Tuning (#583)", .serialized)
struct VideoDiTUniversalConverterTests {
    private static func writeTinyVideoDiTCheckpoint(to denseDir: URL) throws {
        try FileManager.default.createDirectory(at: denseDir, withIntermediateDirectories: true)
        let cfg = VideoDiTConfig(modelID: "local/tiny-vdit", modelType: "wan_video_dit", hiddenSize: 64, numHeads: 4, numLayers: 2, latentChannels: 8)
        let dit = CausalVideoDiTModule(config: cfg)
        eval(dit)
        let flat = Dictionary(uniqueKeysWithValues: dit.parameters().flattened())
        try save(arrays: flat, url: denseDir.appending(component: "model.safetensors"))
        let cfgJSON: [String: Any] = [
            "model_type": "wan_video_dit", "hidden_size": 64,
            "num_attention_heads": 4, "num_hidden_layers": 2,
            "latent_channels": 8, "torch_dtype": "bfloat16",
        ]
        try JSONSerialization.data(withJSONObject: cfgJSON, options: [.sortedKeys])
            .write(to: denseDir.appending(component: "config.json"))
    }

    @Test("UniversalModelConverter converts a tiny Video DiT checkpoint through MACaQ MXFP4 + PV-Tuning, QTIP 2.0 bpw + PV-Tuning, and back to Dense BF16")
    func roundTripVideoDiTQuantizationAndPVTune() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "vdit-conv-\(UUID().uuidString)")
        let denseDir = root.appending(component: "dense-vdit")
        let macaqDir = root.appending(component: "macaq-mxfp4")
        let qtipDir = root.appending(component: "qtip-2bpw")
        let recoveredDir = root.appending(component: "recovered-dense")
        defer { try? FileManager.default.removeItem(at: root) }
        try Self.writeTinyVideoDiTCheckpoint(to: denseDir)

        let reqMACaQ = ModelConversionRequest(source: denseDir.path, outputId: "local/vdit-macaq", target: .macaqMXFP4, pvTune: true, stream: false)
        let resMACaQ = try await UniversalModelConverter.convertToDirectory(request: reqMACaQ, sourceURL: denseDir, outputURL: macaqDir) { _ in }
        #expect(try UniversalSourceReader.detect(at: macaqDir).format == "macaq:mxfp4")
        let pvMACaQ = try #require(resMACaQ.pvTune)
        #expect(pvMACaQ.valKLAfter <= pvMACaQ.valKLBefore + 1e-6)

        let reqQTIP = ModelConversionRequest(source: denseDir.path, outputId: "local/vdit-qtip", target: .qtip(bpw: 2.0), pvTune: true, stream: false)
        let resQTIP = try await UniversalModelConverter.convertToDirectory(request: reqQTIP, sourceURL: denseDir, outputURL: qtipDir) { _ in }
        #expect(try UniversalSourceReader.detect(at: qtipDir).format == "qtip:2.00" && resQTIP.pvTune != nil)

        let reqRecover = ModelConversionRequest(source: macaqDir.path, outputId: "local/vdit-bf16", target: .denseBF16, stream: false)
        _ = try await UniversalModelConverter.convertToDirectory(request: reqRecover, sourceURL: macaqDir, outputURL: recoveredDir) { _ in }
        #expect(try UniversalSourceReader.detect(at: recoveredDir).format == "dense:bf16")

        let loadedRunner = try PhasedVideoRunnerActor(identifier: "local/vdit-macaq", options: .init(directory: macaqDir, memoryTier: .gb24))
        let genReq = try VideoGenerationRequest(
            model: "local/vdit-macaq", prompt: "Crystalline basalt columns",
            dimensions: .init(width: 64, height: 64), temporal: .init(frames: 4, fps: 12, steps: 2)
        )
        var generatedBytes = 0
        for try await event in try await loadedRunner.generate(genReq) {
            if case .completed(let r) = event { generatedBytes = r.videoData.count }
        }
        #expect(generatedBytes > 256)
    }
}

actor StubVideoSynthesizer: VideoSynthesizer {
    nonisolated let modelIdentifier = "goodmai/Wan2.2-14B-MACaQ-MXFP4"
    nonisolated let descriptor = VideoModelDescriptor(identifier: "goodmai/Wan2.2-14B-MACaQ-MXFP4")

    func generate(_ request: VideoGenerationRequest) async throws -> AsyncThrowingStream<VideoGenerationEvent, any Error> {
        try request.validate()
        let model = modelIdentifier
        return AsyncThrowingStream { continuation in
            continuation.yield(.progress(VideoGenerationProgress(
                phase: .textConditioning, step: 1, totalSteps: 1, message: "Conditioned prompt"
            )))
            continuation.yield(.progress(VideoGenerationProgress(
                phase: .denoising, step: request.steps, totalSteps: request.steps, teacacheSkipped: true, message: "Denoised"
            )))
            let payload = Data("ftypisom-basalt-video-\(request.prompt)-\(request.frames)".utf8)
            var result = VideoGenerationResult(
                videoData: payload, request: request, seed: request.config.seed ?? 42, executionMode: .strictPhasedTeardown
            )
            result.model = model
            result.teacacheSkippedSteps = 1
            result.elapsedMs = 45
            continuation.yield(.completed(result))
            continuation.finish()
        }
    }
}

@Suite("REST POST /v1/videos/generations JSON + SSE, MCP video_generate, and CLI VideoCommand (#583)", .serialized)
struct VideoServerMCPAndCLITests {
    @Test("POST /v1/videos/generations persists 0700 .mp4, serves GET /v1/videos/content/:id, rejects traversal, and streams SSE progress")
    func restVideoJSONAndSSEEndpoints() async throws {
        let storeDir = FileManager.default.temporaryDirectory.appending(path: "vid-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: storeDir) }
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        try await orch.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock"))
            .withVideoSynthesizer(StubVideoSynthesizer(), storeDirectory: storeDir)
        let app = Application(router: server.makeRouter())

        try await app.test(.router) { client in
            var persistedURL = ""
            let jsonBody = #"{"prompt":"Volcanic lightning over basalt bay","size":"512x512","frames":25,"fps":24,"steps":8,"seed":99}"#
            try await client.execute(
                uri: "/v1/videos/generations", method: .post,
                headers: [.contentType: "application/json"], body: ByteBuffer(string: jsonBody)
            ) { res in
                #expect(res.status == .ok)
                #expect(res.headers[VideoGenerationController.modeHeader] == "strict_phased")
                let env = try JSONDecoder().decode(VideosResponseEnvelope.self, from: Data(res.body.readableBytesView))
                let item = try #require(env.data.first)
                #expect(item.width == 512 && item.frames == 25 && item.fps == 24 && item.hasAudio)
                persistedURL = item.url
            }
            try await client.execute(uri: persistedURL, method: .get) { res in
                #expect(res.status == .ok && res.headers[.contentType] == "video/mp4")
                #expect(res.body.readableBytes > 16)
            }
            try await client.execute(uri: "/v1/videos/content/..%2Fsecret.mp4", method: .get) { res in
                #expect(res.status == .badRequest)
            }

            let sseBody = #"{"prompt":"Basalt arch","stream":true,"steps":4}"#
            try await client.execute(
                uri: "/v1/videos/generations", method: .post,
                headers: [.contentType: "application/json"], body: ByteBuffer(string: sseBody)
            ) { res in
                #expect(res.status == .ok && res.headers[.contentType] == "text/event-stream")
                let raw = String(buffer: res.body)
                #expect(raw.contains("video_generation.progress"))
                #expect(raw.contains("video_generation.completed"))
                #expect(raw.contains("data: [DONE]"))
            }
        }
    }

    @Test("MCPServer advertises and executes video_generate and CLI VideoCommand parses basalt video generate flags")
    func mcpVideoGenerateAndCLIParsing() async throws {
        let storeDir = FileManager.default.temporaryDirectory.appending(path: "mcp-vid-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: storeDir) }
        let capturer = OutputCapturer()
        let mcp = MCPServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine())) { capturer.capture($0) }
            .withVideoSynthesizer(StubVideoSynthesizer(), storeDirectory: storeDir)

        await mcp.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
        await mcp.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"video_generate","arguments":{"prompt":"Tide pool reflections","size":"512x512","frames":17,"fps":24,"seed":7}}}"#)
        let lines = capturer.lines
        #expect(lines.count == 2 && lines[0].contains("\"video_generate\""))
        let envelope = try #require(JSONSerialization.jsonObject(with: Data(lines[1].utf8)) as? [String: Any])
        let resultObj = try #require(envelope["result"] as? [String: Any])
        let contentArr = try #require(resultObj["content"] as? [[String: Any]])
        let innerJSON = try #require(contentArr.first?["text"] as? String)
        let decodedOut = try JSONDecoder().decode(MCPVideoToolOutput.self, from: Data(innerJSON.utf8))
        #expect(decodedOut.url.hasPrefix("/v1/videos/content/") && decodedOut.executionMode == "strict_phased")

        let parsedRoot = try BasaltCLI.parseAsRoot([
            "video", "generate",
            "--model", "goodmai/Wan2.2-14B-MACaQ-MXFP4",
            "--prompt", "Aurora over obsidian glacier",
            "--output", "/tmp/aurora.mp4",
            "--frames", "33",
            "--fps", "24",
            "--size", "832x480",
        ])
        let genCmd = try #require(parsedRoot as? VideoGenerateSubcommand)
        let built = try genCmd.buildRequest()
        #expect(built.frames == 33 && built.fps == 24 && built.width == 832 && built.height == 480 && built.containerFormat == .mp4)
        #expect(throws: (any Error).self) {
            let bad = try VideoGenerateSubcommand.parse(["--model", "m", "--prompt", "p", "--output", "/tmp/bad.avi"])
            _ = try bad.buildRequest()
        }
    }
}

@Suite("macOS InlineVideoPlayerCard 2x Light, Dark, and Russian rendering & Localizable.strings parity (#583)", .serialized)
@MainActor
struct InlineVideoPlayerVisualTests {
    @Test("InlineVideoPlayerCard renders completed scrubber and active phased generation states at 2x sRGB across Light, Dark, and Russian")
    func renderInlineVideoPlayerCardVariants() throws {
        let posterPNG = try ImageCanvasTestFixtures.makeBasaltScenePNG(width: 640, height: 360, neonDusk: true)
        var completedPart = GeneratedVideoPart(
            assetID: "sample-video.mp4",
            prompt: "Bioluminescent surf against hexagonal basalt columns",
            model: "goodmai/Wan2.2-14B-MACaQ-MXFP4",
            dimensions: VideoDimensions(width: 832, height: 480),
            temporal: VideoTemporalSpec(frames: 49, fps: 24, steps: 20),
            executionMode: .strictPhasedTeardown
        )
        completedPart.posterPNG = posterPNG

        var generatingPart = completedPart
        generatingPart.isGenerating = true
        generatingPart.currentPhase = .denoising
        generatingPart.currentStep = 9

        var exportedID: String?
        let spec = VisualSceneSpec(id: .libraryMixed, width: 620, height: 460)
        for variant in [TraitVariant.lightStandard, .darkStandard, .darkRussian] {
            let readyCard = InlineVideoPlayerCard(part: completedPart, initialPlayheadSeconds: 0.8) { exportedID = $0.assetID }
                .padding(16)
            let bmpReady = try OffscreenSceneRenderer.render(readyCard, spec: spec, variant: variant)
            #expect(bmpReady.width == 1_240 && bmpReady.height == 920 && !bmpReady.png.isEmpty)

            let activeCard = InlineVideoPlayerCard(part: generatingPart).padding(16)
            let bmpActive = try OffscreenSceneRenderer.render(activeCard, spec: spec, variant: variant)
            #expect(bmpActive.width == 1_240 && bmpActive.height == 920 && !bmpActive.png.isEmpty)
        }
        #expect(exportedID == nil)

        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let en = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/en.lproj/Localizable.strings")) as? [String: String])
        let ru = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/ru.lproj/Localizable.strings")) as? [String: String])
        #expect(Set(en.keys) == Set(ru.keys))
        #expect(ru["Export MP4…"] == "Экспорт MP4…" && ru["Phased UMA"] == "Фазовая UMA")
    }
}
