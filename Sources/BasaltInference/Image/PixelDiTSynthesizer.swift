import Foundation
import MLX
import MLXNN
import BasaltCore

final class PixelDiTPipelineModule: Module {
    let config: ImageDiTConfig
    let family: ImageArchitectureFamily
    @ModuleInfo(key: "text_enc") var textEncoder: TextConditioningEncoder
    @ModuleInfo(key: "pixel_patch_in") var pixelPatchIn: Conv2d
    @ModuleInfo(key: "pixel_head") var pixelHead: PixelHead16x16
    @ModuleInfo(key: "latent_patch_in") var latentPatchIn: Conv2d
    @ModuleInfo(key: "ref_patch_in") var refPatchIn: Conv2d
    @ModuleInfo(key: "sana_in") var sanaIn: Linear
    @ModuleInfo(key: "sana_out") var sanaOut: Linear
    @ModuleInfo(key: "probe_adaln") var probeAdaLN: AdaLNZeroModulation
    @ModuleInfo(key: "probe_norm") var probeNorm: LayerNorm
    @ModuleInfo(key: "dual_blocks") var dualBlocks: [JointDiTBlock]
    @ModuleInfo(key: "single_blocks") var singleBlocks: [SingleStreamS3DiTBlock]
    @ModuleInfo(key: "sana_blocks") var sanaBlocks: [LinearAttentionSanaBlock]
    @ModuleInfo(key: "ternary_blocks") var ternaryBlocks: [TernaryBitNetDiTBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm
    @ModuleInfo(key: "latent_vel_head") var latentVelHead: Linear
    @ModuleInfo(key: "ternary_vel_head") var ternaryVelHead: TernaryBitNetLinear
    @ModuleInfo(key: "vae") var vae: PatchVAECodec
    @ModuleInfo(key: "dcae") var dcae: DCAE32xCodec
    @ModuleInfo(key: "mot_decoder") var motDecoder: UnifiedMoTPixelShuffleDecoder
    @ModuleInfo(key: "audio_proj") var audioProjector: AudioFlowEditProjector

    init(config: ImageDiTConfig, family: ImageArchitectureFamily) {
        self.config = config
        self.family = family
        let d = config.hiddenSize
        let latCh = config.latentChannels
        self._textEncoder.wrappedValue = TextConditioningEncoder(hiddenDim: d, sharedInputDim: config.textEmbedDim)
        self._pixelPatchIn.wrappedValue = Conv2d(inputChannels: 4, outputChannels: d, kernelSize: 16, stride: 16)
        self._pixelHead.wrappedValue = PixelHead16x16(dimensions: d, patchSize: 16)
        self._latentPatchIn.wrappedValue = Conv2d(inputChannels: latCh, outputChannels: d, kernelSize: 2, stride: 2)
        self._refPatchIn.wrappedValue = Conv2d(inputChannels: latCh, outputChannels: d, kernelSize: 2, stride: 2)
        self._sanaIn.wrappedValue = Linear(latCh, d)
        self._sanaOut.wrappedValue = Linear(d, latCh)
        self._probeAdaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._probeNorm.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._dualBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in JointDiTBlock(dimensions: d, heads: config.numHeads) }
        self._singleBlocks.wrappedValue = [SingleStreamS3DiTBlock(dimensions: d, heads: config.numHeads)]
        self._sanaBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in LinearAttentionSanaBlock(dimensions: d, heads: config.numHeads) }
        self._ternaryBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in TernaryBitNetDiTBlock(dimensions: d, heads: config.numHeads) }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: d)
        self._latentVelHead.wrappedValue = Linear(d, 4 * latCh)
        self._ternaryVelHead.wrappedValue = TernaryBitNetLinear(d, 4 * latCh)
        self._vae.wrappedValue = PatchVAECodec(latentChannels: latCh)
        self._dcae.wrappedValue = DCAE32xCodec(latentChannels: latCh)
        self._motDecoder.wrappedValue = UnifiedMoTPixelShuffleDecoder(inChannels: latCh)
        self._audioProjector.wrappedValue = AudioFlowEditProjector(featureDim: config.textEmbedDim, hiddenDim: d, outChannels: latCh)
    }

    func patchifyTokens(_ state: MLXArray) -> (tokens: MLXArray, tokH: Int, tokW: Int) {
        switch family {
        case .pixelDiT:
            let p = pixelPatchIn(state)
            return (p.reshaped(1, p.dim(1) * p.dim(2), config.hiddenSize), p.dim(1), p.dim(2))
        case .linearSana:
            let p = sanaIn(state)
            return (p.reshaped(1, p.dim(1) * p.dim(2), config.hiddenSize), p.dim(1), p.dim(2))
        default:
            let p = latentPatchIn(state)
            return (p.reshaped(1, p.dim(1) * p.dim(2), config.hiddenSize), p.dim(1), p.dim(2))
        }
    }

    func modulatedProbe(state: MLXArray, condVec: MLXArray, timestep: Float, guidance: Float) -> MLXArray {
        let (tokens, _, _) = patchifyTokens(state)
        let gates = probeAdaLN.gates(timestep: timestep, guidance: guidance, condVec: condVec)
        return probeNorm(tokens) * (1.0 + gates.scaleMSA) + gates.shiftMSA
    }

    func predictVelocity(_ input: FluxForwardInput) -> MLXArray {
        var (img, tokH, tokW) = patchifyTokens(input.latent)
        if let bbox = input.bboxGrid {
            img = img + bbox.bias
        }
        var cond = input.condTokens
        let rope = RoPE2D.frequencies(gridH: tokH, gridW: tokW, headDim: config.headDim)
        let stepCtx = DiTStepContext(condVec: input.condVec, rope: rope, timestep: input.timestep, guidance: input.guidance)
        switch family {
        case .pixelDiT:
            for block in dualBlocks { (img, cond) = block(img: img, txt: cond, stepCtx: stepCtx) }
            for block in singleBlocks { (cond, img) = block(condTokens: cond, imgTokens: img, stepCtx: stepCtx) }
            return pixelHead(img, tokH: tokH, tokW: tokW)
        case .linearSana:
            for block in sanaBlocks { img = block(img: img, condTokens: cond, stepCtx: stepCtx) }
            return sanaOut(finalNorm(img)).reshaped(1, tokH, tokW, config.latentChannels)
        case .ternaryBitNet:
            for block in ternaryBlocks { (cond, img) = block(condTokens: cond, imgTokens: img, stepCtx: stepCtx) }
            return unpatchLatent2x2(ternaryVelHead(finalNorm(img)), tokH: tokH, tokW: tokW)
        default:
            for block in dualBlocks { (img, cond) = block(img: img, txt: cond, stepCtx: stepCtx) }
            return unpatchLatent2x2(latentVelHead(finalNorm(img)), tokH: tokH, tokW: tokW)
        }
    }

    private func unpatchLatent2x2(_ raw: MLXArray, tokH: Int, tokW: Int) -> MLXArray {
        raw.reshaped(1, tokH, tokW, 2, 2, config.latentChannels)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(1, tokH * 2, tokW * 2, config.latentChannels)
    }

    func decodeFinalPixels(state: MLXArray, targetHeight: Int, targetWidth: Int, includeAlpha: Bool) -> MLXArray {
        switch family {
        case .pixelDiT:
            let rgb = sigmoid(state[0..., 0..., 0..., ..<3])
            guard includeAlpha else { return rgb }
            let prior = PatchVAECodec.subjectMattePrior(height: targetHeight, width: targetWidth)
            let alpha = sigmoid(state[0..., 0..., 0..., 3...] + prior)
            return concatenated([rgb, alpha], axis: 3)
        case .linearSana:
            return dcae.decode(state, targetHeight: targetHeight, targetWidth: targetWidth, includeAlpha: includeAlpha)
        case .unifiedMoT:
            return motDecoder.decode(state, includeAlpha: includeAlpha)
        default:
            return vae.decode(state, includeAlpha: includeAlpha)
        }
    }
}

public actor PixelDiTSynthesizer: ImageStepRunner {
    struct ActiveSession {
        let request: ImageGenerationRequest
        let promptVector: [Float]
        let audioVector: [Float]?
        let audioStrength: Float
        let seed: UInt64
        let epoch: Int
        let wantsAlpha: Bool
        let isLatentSpace: Bool
        var state: MLXArray
        let initialNoise: MLXArray
        let primaryRefState: MLXArray?
        let primaryRefImage: MLXArray?
        let pixelMask: MLXArray?
        let stateMask: MLXArray?
        let condTokens: MLXArray
        let condVec: MLXArray
        let uncondTokens: MLXArray?
        let uncondVec: MLXArray?
        let bboxGrid: (mask: MLXArray, bias: MLXArray)?
        var teaCache: TeaCacheState
    }

    public nonisolated let modelIdentifier: String
    public nonisolated let descriptor: ImageModelDescriptor
    nonisolated let config: ImageDiTConfig
    private let sharedTextEmbedder: (@Sendable (String) async throws -> [Float])?
    private var module: PixelDiTPipelineModule?
    private var sessions: [Int: ActiveSession] = [:]
    private var nextSessionID = 0
    private var epoch = 0

    public init(
        descriptor: ImageModelDescriptor = .iris3BTurbo(),
        config: ImageDiTConfig = .tiny(modelID: "Speridlabs/Iris-3B-Turbo"),
        sharedTextEmbedder: (@Sendable (String) async throws -> [Float])? = nil
    ) {
        self.modelIdentifier = descriptor.id
        self.descriptor = descriptor
        self.config = config
        self.sharedTextEmbedder = sharedTextEmbedder
        let built = withRandomState(MLXRandom.RandomState(seed: config.initSeed)) {
            PixelDiTPipelineModule(config: config, family: descriptor.family)
        }
        eval(built)
        self.module = built
    }

    public init(
        directory: URL,
        identifier: String,
        sharedTextEmbedder: (@Sendable (String) async throws -> [Float])? = nil
    ) throws(BasaltError) {
        let (resolvedDesc, resolvedCfg) = try ImageRunnerCore.parseDirectoryConfig(
            directory: directory,
            identifier: identifier
        )
        self.modelIdentifier = identifier
        self.descriptor = resolvedDesc
        self.config = resolvedCfg
        self.sharedTextEmbedder = sharedTextEmbedder
        let built = withRandomState(MLXRandom.RandomState(seed: resolvedCfg.initSeed)) {
            PixelDiTPipelineModule(config: resolvedCfg, family: resolvedDesc.family)
        }
        try ImageRunnerCore.loadSafetensorsIfPresent(in: directory, into: built)
        eval(built)
        self.module = built
    }

    public func unload() async {
        epoch &+= 1
        sessions.removeAll()
        module = nil
        Memory.clearCache()
    }

    public func generate(_ request: ImageGenerationRequest) async throws -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
        try request.validate()
        guard module != nil else { throw BasaltError.modelNotLoaded }
        let capturedEpoch = epoch
        let vectors = try await ImageRunnerCore.resolvePromptVectors(
            request: request,
            embedDim: config.textEmbedDim,
            sharedEmbedder: sharedTextEmbedder
        )
        guard module != nil, epoch == capturedEpoch else { throw BasaltError.modelNotLoaded }
        return ImageRunnerCore.makeGenerationStream(
            runner: self,
            request: request,
            promptVectors: vectors,
            expectedEpoch: capturedEpoch
        )
    }
}

extension PixelDiTSynthesizer {
    func beginCandidateSession(
        request: ImageGenerationRequest,
        promptVectors: PromptVectorPair,
        seed: UInt64,
        expectedEpoch: Int
    ) async throws(BasaltError) -> Int {
        guard let module, epoch == expectedEpoch else { throw .modelNotLoaded }
        let (stateShape, tokH, tokW) = resolveStateGeometry(width: request.width, height: request.height)
        let noise = MLXRandom.normal(stateShape, dtype: .float32, key: MLXRandom.key(seed))
        let (refs, pixelMask) = try ImageRunnerCore.decodeReferencesAndMask(request: request)
        let refStates = refs.map { encodeReferenceState(module: module, image: $0) }
        let startState = refStates.first.map { 0.55 * $0 + 0.45 * noise } ?? noise
        let audioVec = ImageRunnerCore.extractAudioConditioning(from: request.config.audioConditioning, dimension: config.textEmbedDim)
        let audioStrength = Float(max(0.0, request.config.audioStrength ?? (audioVec == nil ? 0.0 : 1.0)))
        let (condTokens, condVec) = buildPositiveConditioning(
            module: module, request: request, positiveVector: promptVectors.positive,
            audioVector: audioVec, audioStrength: audioStrength, refStates: refStates
        )
        let (uncondTokens, uncondVec) = buildNegativeConditioning(module: module, prompt: request.negativePrompt, vector: promptVectors.negative)
        let stateMask = pixelMask.map { TinyAutoencoderPreview.upsampleBilinearGrid($0[0], height: stateShape[1], width: stateShape[2])[.newAxis] }
        let bboxGrid = ImageRunnerCore.spatialBBoxModulation(boxes: request.config.bboxLayout, gridH: tokH, gridW: tokW, hiddenDim: config.hiddenSize)
        eval(startState, condTokens, condVec)
        nextSessionID += 1
        sessions[nextSessionID] = ActiveSession(
            request: request, promptVector: promptVectors.positive, audioVector: audioVec, audioStrength: audioStrength,
            seed: seed, epoch: expectedEpoch, wantsAlpha: ImageRunnerCore.wantsNativeAlpha(request: request, descriptor: descriptor),
            isLatentSpace: descriptor.family != .pixelDiT, state: startState, initialNoise: noise,
            primaryRefState: refStates.first, primaryRefImage: refs.first, pixelMask: pixelMask, stateMask: stateMask,
            condTokens: condTokens, condVec: condVec, uncondTokens: uncondTokens, uncondVec: uncondVec, bboxGrid: bboxGrid,
            teaCache: TeaCacheState(totalSteps: request.steps, threshold: request.config.teacacheThreshold)
        )
        return nextSessionID
    }

    private func buildNegativeConditioning(module: PixelDiTPipelineModule, prompt: String?, vector: [Float]?) -> (MLXArray?, MLXArray?) {
        guard let prompt, let vector else { return (nil, nil) }
        let (tok, vec) = module.textEncoder.encode(TextConditioningInput(prompt: prompt, sharedVector: vector))
        return (tok, vec)
    }

    private func resolveStateGeometry(width: Int, height: Int) -> (shape: [Int], tokH: Int, tokW: Int) {
        switch descriptor.family {
        case .pixelDiT:
            return ([1, height, width, 4], height / 16, width / 16)
        case .linearSana:
            let (_, _, latH, latW) = DCAE32xCodec.paddedDimensions(height: height, width: width)
            return ([1, latH, latW, config.latentChannels], latH, latW)
        default:
            let hLat = max(2, height / 8)
            let wLat = max(2, width / 8)
            return ([1, hLat, wLat, config.latentChannels], hLat / 2, wLat / 2)
        }
    }

    private func encodeReferenceState(module: PixelDiTPipelineModule, image: MLXArray) -> MLXArray {
        switch descriptor.family {
        case .pixelDiT:
            return (image.ndim == 3 ? image[.newAxis] : image) * 2.0 - 1.0
        case .linearSana:
            return module.dcae.encode(image)
        default:
            return module.vae.encode(image)
        }
    }

    private func buildPositiveConditioning(
        module: PixelDiTPipelineModule,
        request: ImageGenerationRequest,
        positiveVector: [Float],
        audioVector: [Float]?,
        audioStrength: Float,
        refStates: [MLXArray]
    ) -> (MLXArray, MLXArray) {
        let input = TextConditioningInput(
            prompt: request.effectivePrompt,
            sharedVector: positiveVector,
            bboxLayout: request.config.bboxLayout,
            vglSceneJSON: request.config.vglSceneJSON
        )
        var (condTokens, _) = module.textEncoder.encode(input)
        if let audioVector, audioStrength > 0 {
            let aTok = module.audioProjector.conditioningTokens(audioVector) * audioStrength
            condTokens = concatenated([condTokens, aTok], axis: 1)
        }
        if !refStates.isEmpty && descriptor.family != .pixelDiT && descriptor.family != .linearSana {
            let refTok = concatenated(
                refStates.map { r in
                    let p = module.refPatchIn(r)
                    return p.reshaped(1, p.dim(1) * p.dim(2), config.hiddenSize)
                },
                axis: 1
            )
            condTokens = concatenated([condTokens, refTok], axis: 1)
        }
        return (condTokens, mean(condTokens, axis: 1))
    }
}

extension PixelDiTSynthesizer {
    func stepCandidateSession(
        _ sessionID: Int,
        step: Int,
        emitPreview: Bool
    ) async throws(BasaltError) -> ImagePartialFrame? {
        guard let module, var session = sessions[sessionID], epoch == session.epoch else { throw .modelNotLoaded }
        let total = Float(max(1, session.request.steps))
        let t0 = Float(step) / total
        let t1 = Float(step + 1) / total
        let guidance = Float(max(0.0, session.request.config.guidanceScale))
        let probe = module.modulatedProbe(state: session.state, condVec: session.condVec, timestep: t0, guidance: guidance)
        var velocity = computeGuidedVelocity(module: module, session: &session, probe: probe, step: step, timestep: t0, guidance: guidance)
        if let audioVec = session.audioVector, session.audioStrength > 0, session.isLatentSpace {
            let flowAudio = module.audioProjector.flowEditVelocity(
                audioVector: audioVec, height: session.state.dim(1), width: session.state.dim(2), timestep: t0
            )
            velocity = velocity + session.audioStrength * flowAudio
        }
        var nextState = session.state + (t1 - t0) * velocity
        if let refState = session.primaryRefState, let sMask = session.stateMask {
            nextState = sMask * nextState + (1.0 - sMask) * ((1.0 - t1) * session.initialNoise + t1 * refState)
        }
        eval(nextState)
        session.state = nextState
        sessions[sessionID] = session
        guard emitPreview else { return nil }
        return try TinyAutoencoderPreview.makePartialFrame(
            state: nextState, step: step, request: session.request, isLatent: session.isLatentSpace, includeAlpha: session.wantsAlpha
        )
    }

    private func computeGuidedVelocity(
        module: PixelDiTPipelineModule,
        session: inout ActiveSession,
        probe: MLXArray,
        step: Int,
        timestep: Float,
        guidance: Float
    ) -> MLXArray {
        let curState = session.state
        let (cTok, cVec, uTok, uVec, bGrid) = (session.condTokens, session.condVec, session.uncondTokens, session.uncondVec, session.bboxGrid)
        return session.teaCache.resolveVelocity(step: step, modulatedInput: probe) {
            let vPos = module.predictVelocity(
                FluxForwardInput(latent: curState, condTokens: cTok, condVec: cVec, timestep: timestep, guidance: guidance, bboxGrid: bGrid)
            )
            guard let uTok, let uVec, guidance > 1.0 else { return vPos }
            let vNeg = module.predictVelocity(
                FluxForwardInput(latent: curState, condTokens: uTok, condVec: uVec, timestep: timestep, guidance: 1.0, bboxGrid: nil)
            )
            return vNeg + guidance * (vPos - vNeg)
        }
    }

    func finalizeCandidateSession(_ sessionID: Int) async throws(BasaltError) -> CandidateOutcome {
        guard let module, let session = sessions[sessionID], epoch == session.epoch else {
            throw .modelNotLoaded
        }
        let decoded = module.decodeFinalPixels(
            state: session.state,
            targetHeight: session.request.height,
            targetWidth: session.request.width,
            includeAlpha: session.wantsAlpha
        )
        let composited = ImageRunnerCore.compositeMaskedPixels(
            generated: decoded,
            reference: session.primaryRefImage,
            mask: session.pixelMask,
            includeAlpha: session.wantsAlpha
        )
        eval(composited)
        let pixels = composited[0].asType(.float32).asArray(Float.self)
        let channels = composited.dim(3)
        let dims = ImageDimensions(width: session.request.width, height: session.request.height)
        let encoded = try ImagePNGCodec.encodePixels(
            pixels,
            dimensions: dims,
            channels: channels,
            format: session.request.outputFormat,
            forceOpaque: !session.wantsAlpha
        )
        let score = FourierTextEmbedder.scoreImageAlignment(
            pixels: pixels,
            dimensions: dims,
            channels: channels,
            promptVector: session.promptVector
        )
        return CandidateOutcome(
            encoded: encoded,
            seed: session.seed,
            stepsExecuted: session.teaCache.executedSteps,
            teacacheSkippedSteps: session.teaCache.skippedSteps,
            alignmentScore: score
        )
    }

    func releaseCandidateSession(_ sessionID: Int) async {
        sessions.removeValue(forKey: sessionID)
    }
}

extension PixelDiTSynthesizer {
    package func debugPixelDiTParity(seed: UInt64 = 101) throws(BasaltError) -> (mlxVelocity: [Float], referenceVelocity: [Float]) {
        guard let module else { throw .modelNotLoaded }
        let pixels = MLXRandom.normal([1, 64, 64, 4], dtype: .float32, key: MLXRandom.key(seed))
        let pVec = FourierTextEmbedder.embed("Iris-3B pixel parity", dimension: config.textEmbedDim)
        let (condTokens, condVec) = module.textEncoder.encode(
            TextConditioningInput(prompt: "Iris-3B pixel parity", sharedVector: pVec)
        )
        let input = FluxForwardInput(latent: pixels, condTokens: condTokens, condVec: condVec, timestep: 0.25, guidance: 2.5, bboxGrid: nil)
        let fast = module.predictVelocity(input)
        let patched = module.pixelPatchIn(pixels)
        let (tokH, tokW) = (patched.dim(1), patched.dim(2))
        var img = patched.reshaped(1, tokH * tokW, config.hiddenSize)
        var txt = condTokens
        let rope = RoPE2D.frequencies(gridH: tokH, gridW: tokW, headDim: config.headDim)
        let stepCtx = DiTStepContext(condVec: condVec, rope: rope, timestep: 0.25, guidance: 2.5)
        for b in module.dualBlocks { (img, txt) = b(img: img, txt: txt, stepCtx: stepCtx) }
        for b in module.singleBlocks { (txt, img) = b(condTokens: txt, imgTokens: img, stepCtx: stepCtx) }
        var h = img
        for u in module.pixelHead.blocks { h = u(h) }
        let unpatched = module.pixelHead.pixelProj(module.pixelHead.norm(h))
            .reshaped(1, tokH, tokW, 16, 16, 4)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(1, tokH * 16, tokW * 16, 4)
        let ref = unpatched + 0.25 * module.pixelHead.smoothConv(unpatched)
        eval(fast, ref)
        return (fast.asType(.float32).reshaped(-1).asArray(Float.self), ref.asType(.float32).reshaped(-1).asArray(Float.self))
    }

    package func debugLinearSanaParity(seed: UInt64 = 202) throws(BasaltError) -> (linearOutput: [Float], quadraticReference: [Float]) {
        guard let module, let block = module.sanaBlocks.first else { throw .modelNotLoaded }
        let tokens = MLXRandom.normal([1, 16, config.hiddenSize], dtype: .float32, key: MLXRandom.key(seed))
        let linearOut = block.reluLinearAttention(tokens)
        let headDim = block.dimensions / block.heads
        let parts = block.qkv(tokens).reshaped(1, 16, 3, block.heads, headDim).transposed(2, 0, 3, 1, 4)
        let phiQ = MLXNN.relu(parts[0]) + 1e-4
        let phiK = MLXNN.relu(parts[1]) + 1e-4
        let attnWeights = matmul(phiQ, phiK.transposed(0, 1, 3, 2))
        let denom = maximum(sum(attnWeights, axis: -1, keepDims: true), MLXArray(Float(1e-4)))
        let quadAgg = matmul(attnWeights / denom, parts[2]).transposed(0, 2, 1, 3).reshaped(1, 16, block.dimensions)
        let quadOut = block.outProj(quadAgg)
        eval(linearOut, quadOut)
        return (
            linearOut.asType(.float32).reshaped(-1).asArray(Float.self),
            quadOut.asType(.float32).reshaped(-1).asArray(Float.self)
        )
    }

    package func debugTernaryBitNetParity(seed: UInt64 = 303) throws(BasaltError) -> (codes: [Float], mlxOutput: [Float], cpuReference: [Float]) {
        guard let module else { throw .modelNotLoaded }
        let layer = module.ternaryVelHead
        let x = MLXRandom.normal([2, layer.inFeatures], dtype: .float32, key: MLXRandom.key(seed))
        let (codesArr, gammaArr) = layer.quantizedTernaryWeights()
        let mlxOut = layer(x)
        let normed = layer.actNorm(x)
        eval(codesArr, gammaArr, mlxOut, normed)
        let codes = codesArr.asType(.float32).reshaped(-1).asArray(Float.self)
        let gamma = gammaArr.item(Float.self)
        let bias = layer.bias?.asType(.float32).asArray(Float.self) ?? [Float](repeating: 0, count: layer.outFeatures)
        let normedFlat = normed.asType(.float32).reshaped(-1).asArray(Float.self)
        let cpuOut = Self.cpuTernaryForward(
            normedInput: normedFlat,
            inDim: layer.inFeatures,
            outDim: layer.outFeatures,
            codes: codes,
            gamma: gamma,
            bias: bias
        )
        return (codes, mlxOut.asType(.float32).reshaped(-1).asArray(Float.self), cpuOut)
    }

    private static func cpuTernaryForward(
        normedInput: [Float],
        inDim: Int,
        outDim: Int,
        codes: [Float],
        gamma: Float,
        bias: [Float]
    ) -> [Float] {
        let rows = max(1, normedInput.count / max(1, inDim))
        var out = [Float](repeating: 0, count: rows * outDim)
        for r in 0 ..< rows {
            let rowSlice = normedInput[(r * inDim) ..< ((r + 1) * inDim)]
            let absMax = max(Float(1e-6), rowSlice.reduce(Float(0)) { max($0, abs($1)) })
            let qScale = 127.0 / absMax
            let xQuant = rowSlice.map { min(127.0, max(-128.0, ($0 * qScale).rounded(.toNearestOrEven))) / qScale }
            for o in 0 ..< outDim {
                var acc: Float = 0
                let wBase = o * inDim
                for i in 0 ..< inDim {
                    acc += xQuant[i] * codes[wBase + i]
                }
                out[r * outDim + o] = acc * gamma + bias[o]
            }
        }
        return out
    }
}
