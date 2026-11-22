import BasaltCore
import Foundation
import MLX
import MLXNN

public enum VideoPhaseLifecycleEvent: Sendable, Equatable {
    case loaded(VideoExecutionPhase)
    case completed(VideoExecutionPhase)
    case unloaded(VideoExecutionPhase)
    case cacheCleared(after: VideoExecutionPhase)
    case sessionCleanedUp
}

public actor PhasedVideoRunnerActor: VideoSynthesizer {
    public struct Options: Sendable {
        public var directory: URL?
        public var memoryTier: ComposeMemoryTier
        public var memoryCoordinator: MemoryCoordinator?
        public var sharedTextEmbedder: (@Sendable (String) async throws -> [Float])?
        public var phaseObserver: (@Sendable (VideoPhaseLifecycleEvent) async -> Void)?

        public init(
            directory: URL? = nil,
            memoryTier: ComposeMemoryTier = ComposeMemoryTier.detect(),
            memoryCoordinator: MemoryCoordinator? = nil,
            sharedTextEmbedder: (@Sendable (String) async throws -> [Float])? = nil,
            phaseObserver: (@Sendable (VideoPhaseLifecycleEvent) async -> Void)? = nil
        ) {
            self.directory = directory
            self.memoryTier = memoryTier
            self.memoryCoordinator = memoryCoordinator
            self.sharedTextEmbedder = sharedTextEmbedder
            self.phaseObserver = phaseObserver
        }
    }

    public nonisolated let modelIdentifier: String
    public nonisolated let descriptor: VideoModelDescriptor
    package let directory: URL?
    package let ditConfig: VideoDiTConfig
    private let memoryTier: ComposeMemoryTier
    private let memoryCoordinator: MemoryCoordinator?
    private let sharedTextEmbedder: (@Sendable (String) async throws -> [Float])?
    private let phaseObserver: (@Sendable (VideoPhaseLifecycleEvent) async -> Void)?
    private var epoch: Int = 0
    private var isGenerating: Bool = false
    private var residentTextEncoder: TextConditioningEncoder?
    private var residentDiT: CausalVideoDiTModule?
    private var residentVAE: Causal3DVAEDecoder?
    public private(set) var recordedPhaseEvents: [VideoPhaseLifecycleEvent] = []

    public init(identifier: String, options: Options = Options()) throws(BasaltError) {
        let (desc, cfg) = try VideoDiTConfig.load(from: options.directory, identifier: identifier)
        self.modelIdentifier = identifier
        self.descriptor = desc
        self.directory = options.directory
        self.ditConfig = cfg
        self.memoryTier = options.memoryTier
        self.memoryCoordinator = options.memoryCoordinator
        self.sharedTextEmbedder = options.sharedTextEmbedder
        self.phaseObserver = options.phaseObserver
    }

    public func unload() async {
        epoch &+= 1
        residentTextEncoder = nil
        residentDiT = nil
        residentVAE = nil
        Memory.clearCache()
    }

    public var hasResidentStageModules: Bool {
        residentTextEncoder != nil || residentDiT != nil || residentVAE != nil
    }

    public func generate(
        _ request: VideoGenerationRequest
    ) async throws -> AsyncThrowingStream<VideoGenerationEvent, any Error> {
        try request.validate()
        guard !isGenerating else {
            throw BasaltError.serverBusy(retryAfterSeconds: 1)
        }
        isGenerating = true
        var handedOff = false
        defer { if !handedOff { isGenerating = false } }
        let sessionEpoch = epoch
        let plan = await resolveExecutionPlan(for: request)
        recordedPhaseEvents.removeAll(keepingCapacity: true)
        handedOff = true
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runPhasedPipeline(
                        request: request,
                        plan: plan,
                        expectedEpoch: sessionEpoch,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error is CancellationError ? BasaltError.cancelled : error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func resolveExecutionPlan(for request: VideoGenerationRequest) async -> VideoExecutionPlan {
        let directive = if let memoryCoordinator {
            await memoryCoordinator.sample().directive
        } else {
            MemoryDirective.relaxed
        }
        return VideoMemoryPlanner.plan(
            tier: memoryTier,
            footprint: ditConfig.footprint,
            directive: directive,
            forcePhased: request.config.forcePhasedTeardown
        )
    }

    fileprivate func recordEvent(_ event: VideoPhaseLifecycleEvent) async {
        recordedPhaseEvents.append(event)
        if let phaseObserver {
            await phaseObserver(event)
        }
    }

    fileprivate func checkEpoch(_ expected: Int) throws(BasaltError) {
        guard epoch == expected else {
            throw .cancelled
        }
    }
}

extension PhasedVideoRunnerActor {
    fileprivate func runPhasedPipeline(
        request: VideoGenerationRequest,
        plan: VideoExecutionPlan,
        expectedEpoch: Int,
        continuation: AsyncThrowingStream<VideoGenerationEvent, any Error>.Continuation
    ) async throws {
        let started = ContinuousClock.now
        defer {
            await withTaskCancellationShield {
                await self.finishSessionCleanup()
            }
        }
        let conditioning = try await executePhase1TextConditioning(
            request: request,
            plan: plan,
            expectedEpoch: expectedEpoch,
            continuation: continuation
        )
        let latents = try await executePhase2Denoising(
            request: request,
            conditioning: conditioning,
            plan: plan,
            expectedEpoch: expectedEpoch,
            continuation: continuation
        )
        let muxedData = try await executePhase3Muxing(
            request: request,
            latents: latents,
            plan: plan,
            expectedEpoch: expectedEpoch,
            continuation: continuation
        )
        let elapsed = started.duration(to: .now)
        let elapsedMs = max(1, Int(elapsed.components.seconds * 1_000 + elapsed.components.attoseconds / 1_000_000_000_000_000))
        let seed = Self.resolveSeed(request)
        var result = VideoGenerationResult(
            videoData: muxedData,
            request: request,
            seed: seed,
            executionMode: plan.mode
        )
        result.stepsExecuted = latents.stepsExecuted
        result.teacacheSkippedSteps = latents.teacacheSkipped
        result.model = modelIdentifier
        result.elapsedMs = elapsedMs
        continuation.yield(.completed(result))
    }

    private func executePhase1TextConditioning(
        request: VideoGenerationRequest,
        plan: VideoExecutionPlan,
        expectedEpoch: Int,
        continuation: AsyncThrowingStream<VideoGenerationEvent, any Error>.Continuation
    ) async throws -> MaterializedTextConditioning {
        await recordEvent(.loaded(.textConditioning))
        defer {
            await withTaskCancellationShield {
                await self.teardownPhase1(plan: plan)
            }
        }
        try Task.checkCancellation()
        try checkEpoch(expectedEpoch)
        let sharedVec = try await computeSharedVector(request.prompt)
        let trimmedNeg = request.negativePrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let negShared: [Float]? = if let trimmedNeg, !trimmedNeg.isEmpty {
            try await computeSharedVector(trimmedNeg)
        } else {
            nil
        }
        try checkEpoch(expectedEpoch)
        let encoder = residentTextEncoder ?? TextConditioningEncoder(hiddenDim: ditConfig.hiddenSize)
        if !plan.unloadTextEncoderAfterPhase1 {
            residentTextEncoder = encoder
        }
        let encoded = encoder.encode(TextConditioningInput(prompt: request.prompt, sharedVector: sharedVec))
        eval(encoded.tokens, encoded.pooled)
        let posPooled = encoded.pooled.asType(.float32).asArray(Float.self)
        let posTokens = encoded.tokens.asType(.float32).asArray(Float.self)
        let negPooled: [Float]?
        if let trimmedNeg, let negShared {
            let negEnc = encoder.encode(TextConditioningInput(prompt: trimmedNeg, sharedVector: negShared))
            eval(negEnc.pooled)
            negPooled = negEnc.pooled.asType(.float32).asArray(Float.self)
        } else {
            negPooled = nil
        }
        continuation.yield(.progress(VideoGenerationProgress(
            phase: .textConditioning, step: 1, totalSteps: 1, message: "Materialized text conditioning"
        )))
        await recordEvent(.completed(.textConditioning))
        return MaterializedTextConditioning(
            positivePooled: posPooled,
            negativePooled: negPooled,
            tokenEmbeddings: posTokens,
            tokenCount: encoded.tokens.dim(1),
            hiddenDim: ditConfig.hiddenSize
        )
    }

    private func computeSharedVector(_ text: String) async throws -> [Float] {
        if let sharedTextEmbedder {
            return try await sharedTextEmbedder(text)
        }
        return FourierTextEmbedder.embed(text, dimension: 256)
    }

    private func teardownPhase1(plan: VideoExecutionPlan) async {
        if plan.unloadTextEncoderAfterPhase1 {
            residentTextEncoder = nil
            await recordEvent(.unloaded(.textConditioning))
        }
        if plan.clearCacheBetweenPhases {
            Memory.clearCache()
            await recordEvent(.cacheCleared(after: .textConditioning))
        }
    }
}

extension PhasedVideoRunnerActor {
    struct LatentGridShape: Sendable {
        let t: Int
        let h: Int
        let w: Int
        let c: Int
    }

    struct DenoisingLoopState {
        var latents: MLXArray
        var schedule: TeaCacheStepSchedule
        var prevMod: MLXArray?
        var cachedVel: MLXArray?
        var latestAudio: MLXArray
    }

    fileprivate func executePhase2Denoising(
        request: VideoGenerationRequest,
        conditioning: MaterializedTextConditioning,
        plan: VideoExecutionPlan,
        expectedEpoch: Int,
        continuation: AsyncThrowingStream<VideoGenerationEvent, any Error>.Continuation
    ) async throws -> MaterializedVideoLatents {
        await recordEvent(.loaded(.denoising))
        defer {
            await withTaskCancellationShield {
                await self.teardownPhase2(plan: plan)
            }
        }
        let dit = try loadOrReuseDiT(keepResident: !plan.unloadDiTAfterPhase2)
        let shape = LatentGridShape(
            t: max(1, (request.frames + ditConfig.temporalScale - 1) / ditConfig.temporalScale),
            h: max(2, request.height / ditConfig.spatialScale),
            w: max(2, request.width / ditConfig.spatialScale),
            c: ditConfig.latentChannels
        )
        let seed = Self.resolveSeed(request)
        var state = DenoisingLoopState(
            latents: Self.makeInitialNoise(seed: seed, seqLen: shape.t * shape.h * shape.w, channels: shape.c),
            schedule: TeaCacheStepSchedule(totalSteps: request.steps, threshold: request.config.teacacheThreshold),
            prevMod: nil,
            cachedVel: nil,
            latestAudio: MLXArray.zeros([1, 32], dtype: .float32)
        )
        let posCond = MLXArray(conditioning.positivePooled, [1, conditioning.hiddenDim])
        for step in 0 ..< request.steps {
            try Task.checkCancellation()
            try checkEpoch(expectedEpoch)
            let progress = try Self.advanceDenoisingStep(
                step: step, dit: dit, posCond: posCond, request: request, shape: shape, state: &state
            )
            continuation.yield(.progress(progress))
            await Task.yield()
        }
        await recordEvent(.completed(.denoising))
        return MaterializedVideoLatents(
            values: state.latents.asType(.float32).asArray(Float.self),
            latentFrames: shape.t,
            latentHeight: shape.h,
            latentWidth: shape.w,
            channels: shape.c,
            audioEnvelope: state.latestAudio.asType(.float32).asArray(Float.self),
            stepsExecuted: state.schedule.executedSteps.count,
            teacacheSkipped: state.schedule.skippedSteps.count
        )
    }

    private static func advanceDenoisingStep(
        step: Int,
        dit: CausalVideoDiTModule,
        posCond: MLXArray,
        request: VideoGenerationRequest,
        shape: LatentGridShape,
        state: inout DenoisingLoopState
    ) throws(BasaltError) -> VideoGenerationProgress {
        let t = 1.0 - Float(step) / Float(max(1, request.steps))
        let mod = dit.modulatedInput(state.latents, timestep: t, pooledCond: posCond)
        let relDelta = state.prevMod.map { Double(TeaCacheState.relativeL1Distance(current: mod, previous: $0)) } ?? 1.0
        let shouldRun: Bool
        let velocity: MLXArray
        if !state.schedule.shouldExecuteStep(step, relativeL1Delta: relDelta), let cached = state.cachedVel {
            shouldRun = false
            velocity = cached
        } else {
            shouldRun = true
            let out = dit.predictVelocity(mod, pooledCond: posCond, staWindow: request.config.staWindow)
            velocity = out.velocity
            state.latestAudio = out.audio
            eval(velocity, state.latestAudio)
            state.prevMod = mod
            state.cachedVel = velocity
        }
        let dt = 1.0 / Float(max(1, request.steps))
        state.latents = state.latents + dt * velocity
        eval(state.latents)
        let preview = try makeStepPreviewIfNeeded(latents: state.latents, step: step, request: request, shape: shape)
        return VideoGenerationProgress(
            phase: .denoising,
            step: step + 1,
            totalSteps: request.steps,
            teacacheSkipped: !shouldRun,
            previewPNG: preview,
            message: "Denoising step \(step + 1)/\(request.steps)"
        )
    }

    private func loadOrReuseDiT(keepResident: Bool) throws(BasaltError) -> CausalVideoDiTModule {
        if let residentDiT { return residentDiT }
        let module = CausalVideoDiTModule(config: ditConfig)
        if let directory {
            try VideoDiTCheckpointCompat.loadWeightsIfPresent(in: directory, into: module)
        }
        if keepResident { residentDiT = module }
        return module
    }

    private func teardownPhase2(plan: VideoExecutionPlan) async {
        if plan.unloadDiTAfterPhase2 {
            residentDiT = nil
            await recordEvent(.unloaded(.denoising))
        }
        if plan.clearCacheBetweenPhases {
            Memory.clearCache()
            await recordEvent(.cacheCleared(after: .denoising))
        }
    }
}

extension PhasedVideoRunnerActor {
    fileprivate func executePhase3Muxing(
        request: VideoGenerationRequest,
        latents: MaterializedVideoLatents,
        plan: VideoExecutionPlan,
        expectedEpoch: Int,
        continuation: AsyncThrowingStream<VideoGenerationEvent, any Error>.Continuation
    ) async throws -> Data {
        await recordEvent(.loaded(.vaeAndAudioMuxing))
        defer {
            await withTaskCancellationShield {
                await self.teardownPhase3(plan: plan)
            }
        }
        try Task.checkCancellation()
        try checkEpoch(expectedEpoch)
        let vae = residentVAE ?? Causal3DVAEDecoder(latentChannels: latents.channels)
        if !plan.unloadDecodersAfterPhase3 {
            residentVAE = vae
        }
        let frames = try decodeVideoFramesTiled(vae: vae, latents: latents, request: request, plan: plan)
        let audio: AVAudioTrackPayload? = if request.config.includeAudio {
            try Self.synthesizeSynchronizedAudio(request: request, envelope: latents.audioEnvelope)
        } else {
            nil
        }
        continuation.yield(.progress(VideoGenerationProgress(
            phase: .vaeAndAudioMuxing, step: 1, totalSteps: 2, message: "Decoded \(frames.count) frames; muxing container"
        )))
        let muxSpec = AVMuxingSpec(
            containerFormat: request.containerFormat,
            videoCodec: request.config.codec,
            audioCodec: request.config.audioCodec,
            dimensions: VideoDimensions(width: request.width, height: request.height),
            fps: request.fps
        )
        let muxed = try await AVAssetWriterMP4Muxer.muxToData(frames: frames, audio: audio, spec: muxSpec)
        try checkEpoch(expectedEpoch)
        continuation.yield(.progress(VideoGenerationProgress(
            phase: .vaeAndAudioMuxing, step: 2, totalSteps: 2, message: "Completed native AVAssetWriter muxing"
        )))
        await recordEvent(.completed(.vaeAndAudioMuxing))
        return muxed
    }

    private func decodeVideoFramesTiled(
        vae: Causal3DVAEDecoder,
        latents: MaterializedVideoLatents,
        request: VideoGenerationRequest,
        plan: VideoExecutionPlan
    ) throws(BasaltError) -> [AVVideoFrameBuffer] {
        let fullGrid = MLXArray(
            latents.values,
            [latents.latentFrames, latents.latentHeight, latents.latentWidth, latents.channels]
        )
        let tileLen = plan.useTiledVAE ? max(1, plan.vaeTemporalTileFrames) : latents.latentFrames
        var decodedLatentFrames: [MLXArray] = []
        var start = 0
        while start < latents.latentFrames {
            let end = min(latents.latentFrames, start + tileLen)
            let tile = fullGrid[start ..< end]
            decodedLatentFrames.append(contentsOf: vae.decodeTile(
                latents: tile, targetHeight: request.height, targetWidth: request.width
            ))
            start = end
        }
        var output: [AVVideoFrameBuffer] = []
        output.reserveCapacity(request.frames)
        for frameIdx in 0 ..< request.frames {
            let srcIdx = min(decodedLatentFrames.count - 1, (frameIdx * decodedLatentFrames.count) / max(1, request.frames))
            let rgbTensor = decodedLatentFrames[max(0, srcIdx)].asType(.float32)
            eval(rgbTensor)
            let floats = rgbTensor.asArray(Float.self)
            output.append(try AVVideoFrameBuffer.fromRGBFloats(floats.span, width: request.width, height: request.height))
        }
        return output
    }

    private func teardownPhase3(plan: VideoExecutionPlan) async {
        if plan.unloadDecodersAfterPhase3 {
            residentVAE = nil
            await recordEvent(.unloaded(.vaeAndAudioMuxing))
        }
        if plan.clearCacheBetweenPhases {
            Memory.clearCache()
            await recordEvent(.cacheCleared(after: .vaeAndAudioMuxing))
        }
    }

    fileprivate func finishSessionCleanup() async {
        isGenerating = false
        Memory.clearCache()
        await recordEvent(.sessionCleanedUp)
    }
}

extension PhasedVideoRunnerActor {
    static func resolveSeed(_ request: VideoGenerationRequest) -> UInt64 {
        if let explicit = request.config.seed { return explicit }
        return request.prompt.unicodeScalars.reduce(UInt64(2_166_136_261)) {
            ($0 &* 16_777_619) ^ UInt64($1.value)
        }
    }

    static func makeInitialNoise(seed: UInt64, seqLen: Int, channels: Int) -> MLXArray {
        var rng = SplitMix64(seed: seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed)
        let total = seqLen * channels
        var values = [Float](repeating: 0, count: total)
        for idx in 0 ..< total {
            let u = Double(rng.next() & 0x1F_FFFF) / Double(0x1F_FFFF)
            values[idx] = Float((u * 2.0 - 1.0) * 0.5)
        }
        return MLXArray(values, [1, seqLen, channels])
    }

    static func makeStepPreviewIfNeeded(
        latents: MLXArray,
        step: Int,
        request: VideoGenerationRequest,
        shape: LatentGridShape
    ) throws(BasaltError) -> Data? {
        guard TinyAutoencoderPreview.shouldEmitPreview(
            step: step,
            totalSteps: request.steps,
            previewEverySteps: request.config.previewEverySteps
        ) else { return nil }
        let firstFrame = latents.reshaped(shape.t, shape.h, shape.w, shape.c)[0]
        let rgb = sigmoid(firstFrame[0..., 0..., ..<min(3, shape.c)])
        let upsampled = TinyAutoencoderPreview.upsampleBilinearGrid(
            rgb, height: min(256, request.height), width: min(256, request.width)
        )
        return try ImagePNGCodec.encode(upsampled, format: .png, forceOpaque: true).data
    }

    static func synthesizeSynchronizedAudio(
        request: VideoGenerationRequest,
        envelope: [Float]
    ) throws(BasaltError) -> AVAudioTrackPayload {
        let rate = request.config.audioSampleRate
        let ch = request.config.audioChannels
        let frameSamples = max(rate / 10, Int((request.durationSeconds * Double(rate)).rounded()))
        let baseFreq = 220.0 + Double(abs(envelope.first ?? 0.2)) * 110.0
        var pcm = [Float](repeating: 0, count: frameSamples * ch)
        for s in 0 ..< frameSamples {
            let time = Double(s) / Double(rate)
            let envBucket = envelope.isEmpty ? 0 : min(envelope.count - 1, (s * envelope.count) / frameSamples)
            let envGain = 0.18 + 0.12 * Double(min(1.0, abs(envelope.isEmpty ? 0.3 : envelope[envBucket])))
            let fade = min(1.0, min(time * 24.0, (Double(frameSamples - s) / Double(rate)) * 24.0))
            let carrier = sin(2.0 * .pi * baseFreq * time) + 0.35 * sin(2.0 * .pi * baseFreq * 1.5 * time)
            let sample = Float(carrier * envGain * max(0.0, fade))
            for c in 0 ..< ch {
                pcm[s * ch + c] = c == 0 ? sample : sample * 0.94
            }
        }
        return try AVAudioTrackPayload.fromFloatSamples(pcm.span, sampleRate: rate, channels: ch)
    }
}
