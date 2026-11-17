import Foundation
import MLX
import MLXNN
import BasaltCore

final class FiboPerceiverResampler: Module {
    let dimensions: Int
    let gistTokens: Int
    let heads: Int
    let scale: Float
    let latentQueries: MLXArray
    @ModuleInfo(key: "q1") var q1: Linear
    @ModuleInfo(key: "kv1") var kv1: Linear
    @ModuleInfo(key: "out1") var out1: Linear
    @ModuleInfo(key: "ffn1_in") var ffn1In: Linear
    @ModuleInfo(key: "ffn1_out") var ffn1Out: Linear
    @ModuleInfo(key: "q2") var q2: Linear
    @ModuleInfo(key: "kv2") var kv2: Linear
    @ModuleInfo(key: "out2") var out2: Linear
    @ModuleInfo(key: "ffn2_in") var ffn2In: Linear
    @ModuleInfo(key: "ffn2_out") var ffn2Out: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(dimensions: Int, heads: Int, gistTokens: Int = 8) {
        let d = max(16, dimensions)
        let h = max(1, heads)
        let g = max(2, gistTokens)
        self.dimensions = d
        self.heads = h
        self.gistTokens = g
        self.scale = 1.0 / Float(d / h).squareRoot()
        self.latentQueries = MLXRandom.normal([1, g, d], dtype: .float32) * 0.02
        self._q1.wrappedValue = Linear(d, d, bias: false)
        self._kv1.wrappedValue = Linear(d, d * 2, bias: false)
        self._out1.wrappedValue = Linear(d, d)
        self._ffn1In.wrappedValue = Linear(d, d * 2)
        self._ffn1Out.wrappedValue = Linear(d * 2, d)
        self._q2.wrappedValue = Linear(d, d, bias: false)
        self._kv2.wrappedValue = Linear(d, d * 2, bias: false)
        self._out2.wrappedValue = Linear(d, d)
        self._ffn2In.wrappedValue = Linear(d, d * 2)
        self._ffn2Out.wrappedValue = Linear(d * 2, d)
        self._norm.wrappedValue = RMSNorm(dimensions: d)
    }

    func callAsFunction(_ contextTokens: MLXArray) -> MLXArray {
        let stage1 = crossStage(queries: latentQueries, context: contextTokens, qProj: q1, kvProj: kv1, outProj: out1)
        let mid = stage1 + ffn1Out(geluApproximate(ffn1In(norm(stage1))))
        let stage2 = crossStage(queries: mid, context: contextTokens, qProj: q2, kvProj: kv2, outProj: out2)
        return norm(stage2 + ffn2Out(geluApproximate(ffn2In(norm(stage2)))))
    }

    private func crossStage(
        queries: MLXArray,
        context: MLXArray,
        qProj: Linear,
        kvProj: Linear,
        outProj: Linear
    ) -> MLXArray {
        let b = context.dim(0)
        let nQ = queries.dim(1)
        let nCtx = context.dim(1)
        let headDim = dimensions / heads
        let q = qProj(norm(queries)).reshaped(b, nQ, heads, headDim).transposed(0, 2, 1, 3)
        let kv = kvProj(norm(context)).reshaped(b, nCtx, 2, heads, headDim).transposed(2, 0, 3, 1, 4)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: kv[0], values: kv[1], scale: scale, mask: .none
        ).transposed(0, 2, 1, 3).reshaped(b, nQ, dimensions)
        return queries + outProj(attended)
    }
}

struct FluxForwardInput {
    let latent: MLXArray
    let condTokens: MLXArray
    let condVec: MLXArray
    let timestep: Float
    let guidance: Float
    let bboxGrid: (mask: MLXArray, bias: MLXArray)?
}

final class FluxKleinDiTModule: Module {
    let config: ImageDiTConfig
    let usePerceiver: Bool
    @ModuleInfo(key: "text_enc") var textEncoder: TextConditioningEncoder
    @ModuleInfo(key: "perceiver") var perceiver: FiboPerceiverResampler
    @ModuleInfo(key: "vae") var vae: PatchVAECodec
    @ModuleInfo(key: "patch_in") var patchIn: Conv2d
    @ModuleInfo(key: "ref_patch_in") var refPatchIn: Conv2d
    @ModuleInfo(key: "probe_adaln") var probeAdaLN: AdaLNZeroModulation
    @ModuleInfo(key: "probe_norm") var probeNorm: LayerNorm
    @ModuleInfo(key: "bbox_adaln") var bboxAdaLN: AdaLNZeroModulation
    @ModuleInfo(key: "bbox_norm") var bboxNorm: LayerNorm
    @ModuleInfo(key: "dual_blocks") var dualBlocks: [JointDiTBlock]
    @ModuleInfo(key: "single_blocks") var singleBlocks: [SingleStreamS3DiTBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm
    @ModuleInfo(key: "vel_head") var velocityHead: Linear

    init(config: ImageDiTConfig, usePerceiver: Bool) {
        self.config = config
        self.usePerceiver = usePerceiver
        let d = config.hiddenSize
        let p = config.patchSize
        let latCh = config.latentChannels
        self._textEncoder.wrappedValue = TextConditioningEncoder(hiddenDim: d, sharedInputDim: config.textEmbedDim)
        self._perceiver.wrappedValue = FiboPerceiverResampler(dimensions: d, heads: config.numHeads)
        self._vae.wrappedValue = PatchVAECodec(latentChannels: latCh, decoderChannels: config.decoderBaseChannels)
        self._patchIn.wrappedValue = Conv2d(inputChannels: latCh, outputChannels: d, kernelSize: IntOrPair(p), stride: IntOrPair(p))
        self._refPatchIn.wrappedValue = Conv2d(inputChannels: latCh, outputChannels: d, kernelSize: IntOrPair(p), stride: IntOrPair(p))
        self._probeAdaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._probeNorm.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._bboxAdaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._bboxNorm.wrappedValue = LayerNorm(dimensions: d, affine: false)
        let dualCount = max(1, config.numLayers / 2)
        let singleCount = max(1, config.numLayers - dualCount + 1)
        self._dualBlocks.wrappedValue = (0 ..< dualCount).map { _ in JointDiTBlock(dimensions: d, heads: config.numHeads) }
        self._singleBlocks.wrappedValue = (0 ..< singleCount).map { _ in SingleStreamS3DiTBlock(dimensions: d, heads: config.numHeads) }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: d)
        self._velocityHead.wrappedValue = Linear(d, p * p * latCh)
    }

    func modulatedProbe(latent: MLXArray, condVec: MLXArray, timestep: Float, guidance: Float) -> MLXArray {
        let patched = patchIn(latent)
        let flat = patched.reshaped(1, patched.dim(1) * patched.dim(2), config.hiddenSize)
        let gates = probeAdaLN.gates(timestep: timestep, guidance: guidance, condVec: condVec)
        return probeNorm(flat) * (1.0 + gates.scaleMSA) + gates.shiftMSA
    }

    func predictVelocity(_ input: FluxForwardInput) -> MLXArray {
        let patched = patchIn(input.latent)
        let tokH = patched.dim(1)
        let tokW = patched.dim(2)
        var img = patched.reshaped(1, tokH * tokW, config.hiddenSize)
        if let bbox = input.bboxGrid {
            let localT = input.timestep * (1.0 - 0.35 * bbox.mask) + 0.35 * bbox.mask
            let modulated = bboxAdaLN.modulatePerToken(
                bboxNorm(img),
                tokenTimesteps: localT,
                guidance: input.guidance,
                condVec: input.condVec
            )
            img = img + 0.45 * modulated + bbox.bias
        }
        var txt = input.condTokens
        let rope = RoPE2D.frequencies(gridH: tokH, gridW: tokW, headDim: config.headDim)
        let stepCtx = DiTStepContext(condVec: input.condVec, rope: rope, timestep: input.timestep, guidance: input.guidance)
        for block in dualBlocks {
            (img, txt) = block(img: img, txt: txt, stepCtx: stepCtx)
        }
        for block in singleBlocks {
            (txt, img) = block(condTokens: txt, imgTokens: img, stepCtx: stepCtx)
        }
        var normedOut = finalNorm(img)
        if let bbox = input.bboxGrid {
            normedOut = normedOut + 0.65 * bbox.bias
        }
        return unpatchify(velocityHead(normedOut), tokH: tokH, tokW: tokW)
    }

    func unpatchify(_ patchOut: MLXArray, tokH: Int, tokW: Int) -> MLXArray {
        let p = config.patchSize
        let ch = config.latentChannels
        return patchOut
            .reshaped(1, tokH, tokW, p, p, ch)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(1, tokH * p, tokW * p, ch)
    }
}

public actor FluxKleinSynthesizer: ImageStepRunner {
    struct ActiveSession {
        let request: ImageGenerationRequest
        let promptVector: [Float]
        let seed: UInt64
        let epoch: Int
        let wantsAlpha: Bool
        var state: MLXArray
        let initialNoise: MLXArray
        let primaryRefLatent: MLXArray?
        let primaryRefImage: MLXArray?
        let pixelMask: MLXArray?
        let latentMask: MLXArray?
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
    private var module: FluxKleinDiTModule?
    private var sessions: [Int: ActiveSession] = [:]
    private var nextSessionID = 0
    private var epoch = 0

    package init(
        descriptor: ImageModelDescriptor = .flux2Klein4B(),
        config: ImageDiTConfig = .tiny(modelID: "black-forest-labs/FLUX.2-klein-4B"),
        sharedTextEmbedder: (@Sendable (String) async throws -> [Float])? = nil
    ) {
        self.modelIdentifier = descriptor.id
        self.descriptor = descriptor
        self.config = config
        self.sharedTextEmbedder = sharedTextEmbedder
        let usePerceiver = descriptor.family == .fiboPerceiver
        let built = withRandomState(MLXRandom.RandomState(seed: config.initSeed)) {
            FluxKleinDiTModule(config: config, usePerceiver: usePerceiver)
        }
        eval(built)
        self.module = built
    }

    public init(
        directory: URL,
        identifier: String,
        sharedTextEmbedder: (@Sendable (String) async throws -> [Float])? = nil,
        externalDecoderDirectory: URL? = nil
    ) throws(BasaltError) {
        let (resolvedDesc, resolvedCfg) = try ImageRunnerCore.parseDirectoryConfig(
            directory: directory,
            identifier: identifier,
            externalDecoderDirectory: externalDecoderDirectory
        )
        self.modelIdentifier = identifier
        self.descriptor = resolvedDesc
        self.config = resolvedCfg
        self.sharedTextEmbedder = sharedTextEmbedder
        let usePerceiver = resolvedDesc.family == .fiboPerceiver
        let built = withRandomState(MLXRandom.RandomState(seed: resolvedCfg.initSeed)) {
            FluxKleinDiTModule(config: resolvedCfg, usePerceiver: usePerceiver)
        }
        try ImageRunnerCore.loadSafetensorsIfPresent(in: directory, into: built)
        if let externalDecoderDirectory {
            try ImageRunnerCore.loadSafetensorsIfPresent(in: externalDecoderDirectory, into: built.vae)
        }
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

extension FluxKleinSynthesizer {
    func beginCandidateSession(
        request: ImageGenerationRequest,
        promptVectors: PromptVectorPair,
        seed: UInt64,
        expectedEpoch: Int
    ) async throws(BasaltError) -> Int {
        guard let module, epoch == expectedEpoch else { throw .modelNotLoaded }
        let hLat = max(config.patchSize, request.height / 8)
        let wLat = max(config.patchSize, request.width / 8)
        let noise = MLXRandom.normal([1, hLat, wLat, config.latentChannels], dtype: .float32, key: MLXRandom.key(seed))
        let (refs, pixelMask) = try ImageRunnerCore.decodeReferencesAndMask(request: request)
        let refLatents = refs.map { module.vae.encode($0) }
        let startState = refLatents.first.map { 0.55 * $0 + 0.45 * noise } ?? noise
        let (condTokens, condVec) = buildPositiveConditioning(
            module: module,
            request: request,
            positiveVector: promptVectors.positive,
            refLatents: refLatents
        )
        let (uncondTokens, uncondVec) = buildNegativeConditioning(
            module: module,
            negativePrompt: request.negativePrompt,
            negativeVector: promptVectors.negative
        )
        let latentMask = pixelMask.map { TinyAutoencoderPreview.upsampleBilinearGrid($0[0], height: hLat, width: wLat)[.newAxis] }
        let bboxGrid = ImageRunnerCore.spatialBBoxModulation(
            boxes: request.config.bboxLayout,
            gridH: hLat / config.patchSize,
            gridW: wLat / config.patchSize,
            hiddenDim: config.hiddenSize
        )
        eval(startState, condTokens, condVec)
        nextSessionID += 1
        sessions[nextSessionID] = ActiveSession(
            request: request,
            promptVector: promptVectors.positive,
            seed: seed,
            epoch: expectedEpoch,
            wantsAlpha: ImageRunnerCore.wantsNativeAlpha(request: request, descriptor: descriptor),
            state: startState,
            initialNoise: noise,
            primaryRefLatent: refLatents.first,
            primaryRefImage: refs.first,
            pixelMask: pixelMask,
            latentMask: latentMask,
            condTokens: condTokens,
            condVec: condVec,
            uncondTokens: uncondTokens,
            uncondVec: uncondVec,
            bboxGrid: bboxGrid,
            teaCache: TeaCacheState(totalSteps: request.steps, threshold: request.config.teacacheThreshold)
        )
        return nextSessionID
    }

    private func buildPositiveConditioning(
        module: FluxKleinDiTModule,
        request: ImageGenerationRequest,
        positiveVector: [Float],
        refLatents: [MLXArray]
    ) -> (MLXArray, MLXArray) {
        let posInput = TextConditioningInput(
            prompt: request.effectivePrompt,
            sharedVector: positiveVector,
            bboxLayout: request.config.bboxLayout,
            vglSceneJSON: request.config.vglSceneJSON
        )
        var (condTokens, condVec) = module.textEncoder.encode(posInput)
        if module.usePerceiver {
            condTokens = module.perceiver(condTokens)
            condVec = mean(condTokens, axis: 1)
        }
        if !refLatents.isEmpty {
            let refTokens = concatenated(
                refLatents.map { rLat in
                    let p = module.refPatchIn(rLat)
                    return p.reshaped(1, p.dim(1) * p.dim(2), config.hiddenSize)
                },
                axis: 1
            )
            condTokens = concatenated([condTokens, refTokens], axis: 1)
        }
        return (condTokens, condVec)
    }

    private func buildNegativeConditioning(
        module: FluxKleinDiTModule,
        negativePrompt: String?,
        negativeVector: [Float]?
    ) -> (MLXArray?, MLXArray?) {
        guard let negativePrompt, let negativeVector else { return (nil, nil) }
        var (uTok, uVec) = module.textEncoder.encode(
            TextConditioningInput(prompt: negativePrompt, sharedVector: negativeVector)
        )
        if module.usePerceiver {
            uTok = module.perceiver(uTok)
            uVec = mean(uTok, axis: 1)
        }
        return (uTok, uVec)
    }
}

extension FluxKleinSynthesizer {
    func stepCandidateSession(
        _ sessionID: Int,
        step: Int,
        emitPreview: Bool
    ) async throws(BasaltError) -> ImagePartialFrame? {
        guard let module, var session = sessions[sessionID], epoch == session.epoch else {
            throw .modelNotLoaded
        }
        let total = Float(max(1, session.request.steps))
        let t0 = Float(step) / total
        let t1 = Float(step + 1) / total
        let dt = t1 - t0
        let guidance = Float(max(0.0, session.request.config.guidanceScale))
        let probe = module.modulatedProbe(
            latent: session.state,
            condVec: session.condVec,
            timestep: t0,
            guidance: guidance
        )
        let curState = session.state
        let cTok = session.condTokens
        let cVec = session.condVec
        let uTok = session.uncondTokens
        let uVec = session.uncondVec
        let bGrid = session.bboxGrid
        let velocity = session.teaCache.resolveVelocity(step: step, modulatedInput: probe) {
            let vPos = module.predictVelocity(
                FluxForwardInput(latent: curState, condTokens: cTok, condVec: cVec, timestep: t0, guidance: guidance, bboxGrid: bGrid)
            )
            guard let uTok, let uVec, guidance > 1.0 else { return vPos }
            let vNeg = module.predictVelocity(
                FluxForwardInput(latent: curState, condTokens: uTok, condVec: uVec, timestep: t0, guidance: 1.0, bboxGrid: nil)
            )
            return vNeg + guidance * (vPos - vNeg)
        }
        var nextLatent = session.state + dt * velocity
        if let refLat = session.primaryRefLatent, let lMask = session.latentMask {
            let refAtT = (1.0 - t1) * session.initialNoise + t1 * refLat
            nextLatent = lMask * nextLatent + (1.0 - lMask) * refAtT
        }
        eval(nextLatent)
        session.state = nextLatent
        sessions[sessionID] = session
        guard emitPreview else { return nil }
        return try TinyAutoencoderPreview.makePartialFrame(
            state: nextLatent,
            step: step,
            request: session.request,
            isLatent: true,
            includeAlpha: session.wantsAlpha
        )
    }

    func finalizeCandidateSession(_ sessionID: Int) async throws(BasaltError) -> CandidateOutcome {
        guard let module, let session = sessions[sessionID], epoch == session.epoch else {
            throw .modelNotLoaded
        }
        let decoded = module.vae.decode(session.state, includeAlpha: session.wantsAlpha)
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

extension FluxKleinSynthesizer {
    package func debugVelocityParity(seed: UInt64 = 42) throws(BasaltError) -> (mlxVelocity: [Float], referenceVelocity: [Float]) {
        guard let module else { throw .modelNotLoaded }
        let latent = MLXRandom.normal([1, 8, 8, config.latentChannels], dtype: .float32, key: MLXRandom.key(seed))
        let pVec = FourierTextEmbedder.embed("FLUX parity verification fixture", dimension: config.textEmbedDim)
        let (condTokens, condVec) = module.textEncoder.encode(
            TextConditioningInput(prompt: "FLUX parity verification fixture", sharedVector: pVec)
        )
        let input = FluxForwardInput(
            latent: latent,
            condTokens: condTokens,
            condVec: condVec,
            timestep: 0.35,
            guidance: 3.5,
            bboxGrid: nil
        )
        let fastVel = module.predictVelocity(input)
        let refVel = Self.referenceExplicitVelocity(module: module, input: input)
        eval(fastVel, refVel)
        return (
            fastVel.asType(.float32).reshaped(-1).asArray(Float.self),
            refVel.asType(.float32).reshaped(-1).asArray(Float.self)
        )
    }

    private static func referenceExplicitVelocity(
        module: FluxKleinDiTModule,
        input: FluxForwardInput
    ) -> MLXArray {
        let cfg = module.config
        let patched = module.patchIn(input.latent)
        let tokH = patched.dim(1)
        let tokW = patched.dim(2)
        var img = patched.reshaped(1, tokH * tokW, cfg.hiddenSize)
        var txt = input.condTokens
        let rope = RoPE2D.frequencies(gridH: tokH, gridW: tokW, headDim: cfg.headDim)
        let stepCtx = DiTStepContext(condVec: input.condVec, rope: rope, timestep: input.timestep, guidance: input.guidance)
        for block in module.dualBlocks {
            (img, txt) = explicitDualBlock(block, img: img, txt: txt, stepCtx: stepCtx)
        }
        for block in module.singleBlocks {
            (txt, img) = explicitSingleBlock(block, cond: txt, img: img, stepCtx: stepCtx)
        }
        var normedOut = module.finalNorm(img)
        if let bbox = input.bboxGrid {
            normedOut = normedOut + 0.65 * bbox.bias
        }
        return module.unpatchify(module.velocityHead(normedOut), tokH: tokH, tokW: tokW)
    }

    private static func explicitDualBlock(
        _ block: JointDiTBlock,
        img: MLXArray,
        txt: MLXArray,
        stepCtx: DiTStepContext
    ) -> (MLXArray, MLXArray) {
        let gI = block.imgAdaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let gT = block.txtAdaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let mImg = block.imgNorm1(img) * (1.0 + gI.scaleMSA) + gI.shiftMSA
        let mTxt = block.txtNorm1(txt) * (1.0 + gT.scaleMSA) + gT.shiftMSA
        let nImg = mImg.dim(1)
        let nTxt = mTxt.dim(1)
        let headDim = block.dimensions / block.heads
        let iP = block.imgQKV(mImg).reshaped(1, nImg, 3, block.heads, headDim).transposed(2, 0, 3, 1, 4)
        let tP = block.txtQKV(mTxt).reshaped(1, nTxt, 3, block.heads, headDim).transposed(2, 0, 3, 1, 4)
        let q = concatenated([tP[0], RoPE2D.apply(iP[0], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)], axis: 2)
        let k = concatenated([tP[1], RoPE2D.apply(iP[1], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)], axis: 2)
        let v = concatenated([tP[2], iP[2]], axis: 2)
        let weights = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * block.scale, axis: -1)
        let attn = matmul(weights, v).transposed(0, 2, 1, 3).reshaped(1, nTxt + nImg, block.dimensions)
        let hI = img + gI.gateMSA * block.imgOut(attn[0..., nTxt..., 0...])
        let hT = txt + gT.gateMSA * block.txtOut(attn[0..., ..<nTxt, 0...])
        let fI = block.imgNorm2(hI) * (1.0 + gI.scaleMLP) + gI.shiftMLP
        let fT = block.txtNorm2(hT) * (1.0 + gT.scaleMLP) + gT.shiftMLP
        return (
            hI + gI.gateMLP * block.imgMLP2(geluApproximate(block.imgMLP1(fI))),
            hT + gT.gateMLP * block.txtMLP2(geluApproximate(block.txtMLP1(fT)))
        )
    }

    private static func explicitSingleBlock(
        _ block: SingleStreamS3DiTBlock,
        cond: MLXArray,
        img: MLXArray,
        stepCtx: DiTStepContext
    ) -> (MLXArray, MLXArray) {
        let nC = cond.dim(1)
        let nI = img.dim(1)
        let unified = concatenated([cond, img], axis: 1)
        let gates = block.adaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let mod = block.norm(unified) * (1.0 + gates.scaleMSA) + gates.shiftMSA
        let headDim = block.dimensions / block.heads
        let p = block.qkv(mod).reshaped(1, nC + nI, 3, block.heads, headDim).transposed(2, 0, 3, 1, 4)
        let q = concatenated([p[0][0..., 0..., ..<nC, 0...], RoPE2D.apply(p[0][0..., 0..., nC..., 0...], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)], axis: 2)
        let k = concatenated([p[1][0..., 0..., ..<nC, 0...], RoPE2D.apply(p[1][0..., 0..., nC..., 0...], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)], axis: 2)
        let weights = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * block.scale, axis: -1)
        let attn = matmul(weights, p[2]).transposed(0, 2, 1, 3).reshaped(1, nC + nI, block.dimensions)
        let swiglu = MLXNN.silu(block.mlpGate(mod)) * block.mlpUp(mod)
        let out = unified + gates.gateMSA * block.fusedOut(concatenated([attn, swiglu], axis: -1))
        return (out[0..., ..<nC, 0...], out[0..., nC..., 0...])
    }
}
