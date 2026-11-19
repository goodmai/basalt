import Foundation
import MLX
import MLXNN
import BasaltCore

final class Step1XReasonEditAdapter: Module {
    let dimensions: Int
    @ModuleInfo(key: "reason_in") var reasonIn: Linear
    @ModuleInfo(key: "reason_gate") var reasonGate: Linear
    @ModuleInfo(key: "reason_out") var reasonOut: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(dimensions: Int) {
        let d = max(16, dimensions)
        self.dimensions = d
        self._reasonIn.wrappedValue = Linear(d, d * 2)
        self._reasonGate.wrappedValue = Linear(d, d * 2)
        self._reasonOut.wrappedValue = Linear(d * 2, d)
        self._norm.wrappedValue = RMSNorm(dimensions: d)
    }

    func callAsFunction(textTokens: MLXArray, refTokens: MLXArray?) -> MLXArray {
        let combined = if let refTokens {
            concatenated([textTokens, refTokens], axis: 1)
        } else {
            textTokens
        }
        let n = norm(combined)
        let reasoned = reasonOut(MLXNN.silu(reasonGate(n)) * reasonIn(n))
        return combined + 0.5 * reasoned
    }
}

final class S3DiTModule: Module {
    let config: ImageDiTConfig
    let useReasonAdapter: Bool
    @ModuleInfo(key: "text_enc") var textEncoder: TextConditioningEncoder
    @ModuleInfo(key: "reason_adapter") var reasonAdapter: Step1XReasonEditAdapter
    @ModuleInfo(key: "vae") var vae: PatchVAECodec
    @ModuleInfo(key: "patch_in") var patchIn: Conv2d
    @ModuleInfo(key: "ref_patch_in") var refPatchIn: Conv2d
    @ModuleInfo(key: "probe_adaln") var probeAdaLN: AdaLNZeroModulation
    @ModuleInfo(key: "probe_norm") var probeNorm: LayerNorm
    @ModuleInfo(key: "blocks") var blocks: [SingleStreamS3DiTBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm
    @ModuleInfo(key: "vel_head") var velocityHead: Linear
    @ModuleInfo(key: "alpha_refiner") var alphaRefiner: Conv2d

    init(config: ImageDiTConfig, useReasonAdapter: Bool) {
        self.config = config
        self.useReasonAdapter = useReasonAdapter
        let d = config.hiddenSize
        let p = config.patchSize
        let latCh = config.latentChannels
        self._textEncoder.wrappedValue = TextConditioningEncoder(hiddenDim: d, sharedInputDim: config.textEmbedDim)
        self._reasonAdapter.wrappedValue = Step1XReasonEditAdapter(dimensions: d)
        self._vae.wrappedValue = PatchVAECodec(latentChannels: latCh, decoderChannels: config.decoderBaseChannels)
        self._patchIn.wrappedValue = Conv2d(inputChannels: latCh, outputChannels: d, kernelSize: IntOrPair(p), stride: IntOrPair(p))
        self._refPatchIn.wrappedValue = Conv2d(inputChannels: latCh, outputChannels: d, kernelSize: IntOrPair(p), stride: IntOrPair(p))
        self._probeAdaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._probeNorm.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._blocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            SingleStreamS3DiTBlock(dimensions: d, heads: config.numHeads)
        }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: d)
        self._velocityHead.wrappedValue = Linear(d, p * p * latCh)
        self._alphaRefiner.wrappedValue = Conv2d(inputChannels: 4, outputChannels: 1, kernelSize: 3, padding: 1)
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
            img = img + bbox.bias
        }
        var cond = input.condTokens
        let rope = RoPE2D.frequencies(gridH: tokH, gridW: tokW, headDim: config.headDim)
        let stepCtx = DiTStepContext(condVec: input.condVec, rope: rope, timestep: input.timestep, guidance: input.guidance)
        for (idx, block) in blocks.enumerated() {
            let refKV = idx < input.singleRefKV.count ? input.singleRefKV[idx] : nil
            (cond, img) = block(condTokens: cond, imgTokens: img, stepCtx: stepCtx, refKV: refKV)
        }
        let p = config.patchSize
        let ch = config.latentChannels
        return velocityHead(finalNorm(img))
            .reshaped(1, tokH, tokW, p, p, ch)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(1, tokH * p, tokW * p, ch)
    }

    func decodeNativeRGBA(_ latent: MLXArray, includeAlpha: Bool) -> MLXArray {
        let baseDecoded = vae.decode(latent, includeAlpha: includeAlpha)
        guard includeAlpha else { return baseDecoded }
        let rgb = baseDecoded[0..., 0..., 0..., ..<3]
        let prior = PatchVAECodec.subjectMattePrior(height: baseDecoded.dim(1), width: baseDecoded.dim(2))
        let refinedLogit = 0.35 * alphaRefiner(baseDecoded) + prior
        let alpha = sigmoid(refinedLogit)
        return concatenated([rgb, alpha], axis: 3)
    }
}

public actor S3DiTSynthesizer: ImageStepRunner {
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
        let singleRefKV: [CachedReferenceLayerKV]
        let referenceKVReused: Bool
        var teaCache: TeaCacheState
    }

    public nonisolated let modelIdentifier: String
    public nonisolated let descriptor: ImageModelDescriptor
    nonisolated let config: ImageDiTConfig
    private let sharedTextEmbedder: (@Sendable (String) async throws -> [Float])?
    private var module: S3DiTModule?
    private var referenceKVCache = ReferenceImageKVCache(capacity: 8)
    private var sessions: [Int: ActiveSession] = [:]
    private var nextSessionID = 0
    private var epoch = 0

    package init(
        descriptor: ImageModelDescriptor = .zImageTurbo(),
        config: ImageDiTConfig = .tiny(modelID: "Tongyi-MAI/Z-Image-Turbo"),
        sharedTextEmbedder: (@Sendable (String) async throws -> [Float])? = nil
    ) {
        self.modelIdentifier = descriptor.id
        self.descriptor = descriptor
        self.config = config
        self.sharedTextEmbedder = sharedTextEmbedder
        let useReason = descriptor.family == .step1XReasonEdit
        let built = withRandomState(MLXRandom.RandomState(seed: config.initSeed)) {
            S3DiTModule(config: config, useReasonAdapter: useReason)
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
        let useReason = resolvedDesc.family == .step1XReasonEdit
        let built = withRandomState(MLXRandom.RandomState(seed: resolvedCfg.initSeed)) {
            S3DiTModule(config: resolvedCfg, useReasonAdapter: useReason)
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
        referenceKVCache.clear()
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

extension S3DiTSynthesizer {
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
        let (refSnapshot, uncachedRefLatents, primaryImg, pixelMask, cacheHit) = try resolveS3References(module: module, request: request)
        let primaryLatent = refSnapshot?.primaryLatent ?? uncachedRefLatents.first
        let startState = primaryLatent.map { 0.55 * $0 + 0.45 * noise } ?? noise
        let inlineBoxes = FourierTextEmbedder.parseInlineJSONBoundingBoxes(request.effectivePrompt)
        let mergedBoxes = request.config.bboxLayout + inlineBoxes
        let (condTokens, condVec) = buildS3Conditioning(
            module: module,
            request: request,
            boxes: mergedBoxes,
            positiveVector: promptVectors.positive,
            refLatents: refSnapshot == nil ? uncachedRefLatents : []
        )
        let (uncondTokens, uncondVec): (MLXArray?, MLXArray?) = if let neg = request.negativePrompt, let nVec = promptVectors.negative {
            module.textEncoder.encode(TextConditioningInput(prompt: neg, sharedVector: nVec))
        } else {
            (nil, nil)
        }
        let latentMask = pixelMask.map { TinyAutoencoderPreview.upsampleBilinearGrid($0[0], height: hLat, width: wLat)[.newAxis] }
        let bboxGrid = ImageRunnerCore.spatialBBoxModulation(
            boxes: mergedBoxes,
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
            primaryRefLatent: primaryLatent,
            primaryRefImage: refSnapshot?.primaryImage ?? primaryImg,
            pixelMask: pixelMask,
            latentMask: latentMask,
            condTokens: condTokens,
            condVec: condVec,
            uncondTokens: uncondTokens,
            uncondVec: uncondVec,
            bboxGrid: bboxGrid,
            singleRefKV: refSnapshot?.singleLayerKV ?? [],
            referenceKVReused: cacheHit,
            teaCache: TeaCacheState(totalSteps: request.steps, threshold: request.config.teacacheThreshold)
        )
        return nextSessionID
    }

    private func resolveS3References(
        module: S3DiTModule,
        request: ImageGenerationRequest
    ) throws(BasaltError) -> (CachedReferenceSnapshot?, [MLXArray], MLXArray?, MLXArray?, Bool) {
        guard !request.config.referenceImages.isEmpty, request.config.reuseReferenceKV, !module.useReasonAdapter else {
            let (refs, mask) = try ImageRunnerCore.decodeReferencesAndMask(request: request)
            return (nil, refs.map { module.vae.encode($0) }, refs.first, mask, false)
        }
        let key = ReferenceKVCacheKey(
            modelID: modelIdentifier,
            width: request.width,
            height: request.height,
            hiddenSize: config.hiddenSize,
            patchSize: config.patchSize,
            referenceImages: request.config.referenceImages
        )
        let maskOnly: MLXArray? = if request.config.maskImage != nil {
            try ImageRunnerCore.decodeReferencesAndMask(request: request).mask
        } else {
            nil
        }
        if let hit = referenceKVCache.lookup(key) {
            return (hit, [], hit.primaryImage, maskOnly, true)
        }
        let (refs, decodedMask) = try ImageRunnerCore.decodeReferencesAndMask(request: request)
        let built = ReferenceKVProjector.buildS3Snapshot(key: key, refImages: refs, module: module)
        referenceKVCache.insert(built)
        return (built, [], built.primaryImage, decodedMask, false)
    }

    private func buildS3Conditioning(
        module: S3DiTModule,
        request: ImageGenerationRequest,
        boxes: [ImageBoundingBox],
        positiveVector: [Float],
        refLatents: [MLXArray]
    ) -> (MLXArray, MLXArray) {
        let input = TextConditioningInput(
            prompt: request.effectivePrompt,
            sharedVector: positiveVector,
            bboxLayout: boxes,
            vglSceneJSON: request.config.vglSceneJSON
        )
        let (txtTokens, _) = module.textEncoder.encode(input)
        let refTok: MLXArray? = if refLatents.isEmpty {
            nil
        } else {
            concatenated(
                refLatents.map { rLat in
                    let p = module.refPatchIn(rLat)
                    return p.reshaped(1, p.dim(1) * p.dim(2), config.hiddenSize)
                },
                axis: 1
            )
        }
        let unifiedCond: MLXArray
        if module.useReasonAdapter {
            unifiedCond = module.reasonAdapter(textTokens: txtTokens, refTokens: refTok)
        } else if let refTok {
            unifiedCond = concatenated([txtTokens, refTok], axis: 1)
        } else {
            unifiedCond = txtTokens
        }
        return (unifiedCond, mean(unifiedCond, axis: 1))
    }
}

extension S3DiTSynthesizer {
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
        let refKVs = session.singleRefKV
        let velocity = session.teaCache.resolveVelocity(step: step, modulatedInput: probe) {
            let vPos = module.predictVelocity(
                FluxForwardInput(latent: curState, condTokens: cTok, condVec: cVec, timestep: t0, guidance: guidance, bboxGrid: bGrid, singleRefKV: refKVs)
            )
            guard let uTok, let uVec, guidance > 1.0 else { return vPos }
            let vNeg = module.predictVelocity(
                FluxForwardInput(latent: curState, condTokens: uTok, condVec: uVec, timestep: t0, guidance: 1.0, bboxGrid: nil, singleRefKV: refKVs)
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
        let decoded = module.decodeNativeRGBA(session.state, includeAlpha: session.wantsAlpha)
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
        var outcome = CandidateOutcome(
            encoded: encoded,
            seed: session.seed,
            stepsExecuted: session.teaCache.executedSteps,
            teacacheSkippedSteps: session.teaCache.skippedSteps,
            alignmentScore: score
        )
        outcome.dimensions = dims
        outcome.referenceKVReused = session.referenceKVReused
        return outcome
    }

    func releaseCandidateSession(_ sessionID: Int) async {
        sessions.removeValue(forKey: sessionID)
    }
}

extension S3DiTSynthesizer {
    package func debugVelocityParity(seed: UInt64 = 77) throws(BasaltError) -> (mlxVelocity: [Float], referenceVelocity: [Float]) {
        guard let module else { throw .modelNotLoaded }
        let latent = MLXRandom.normal([1, 8, 8, config.latentChannels], dtype: .float32, key: MLXRandom.key(seed))
        let pVec = FourierTextEmbedder.embed("S3-DiT parity check", dimension: config.textEmbedDim)
        let (condTokens, condVec) = module.textEncoder.encode(
            TextConditioningInput(prompt: "S3-DiT parity check", sharedVector: pVec)
        )
        let input = FluxForwardInput(latent: latent, condTokens: condTokens, condVec: condVec, timestep: 0.4, guidance: 2.0, bboxGrid: nil)
        let fastVel = module.predictVelocity(input)
        let refVel = Self.referenceS3Velocity(module: module, input: input)
        eval(fastVel, refVel)
        return (
            fastVel.asType(.float32).reshaped(-1).asArray(Float.self),
            refVel.asType(.float32).reshaped(-1).asArray(Float.self)
        )
    }

    private static func referenceS3Velocity(module: S3DiTModule, input: FluxForwardInput) -> MLXArray {
        let cfg = module.config
        let patched = module.patchIn(input.latent)
        let tokH = patched.dim(1)
        let tokW = patched.dim(2)
        var img = patched.reshaped(1, tokH * tokW, cfg.hiddenSize)
        var cond = input.condTokens
        let rope = RoPE2D.frequencies(gridH: tokH, gridW: tokW, headDim: cfg.headDim)
        for block in module.blocks {
            let nC = cond.dim(1)
            let nI = img.dim(1)
            let unified = concatenated([cond, img], axis: 1)
            let gates = block.adaLN.gates(timestep: input.timestep, guidance: input.guidance, condVec: input.condVec)
            let mod = block.norm(unified) * (1.0 + gates.scaleMSA) + gates.shiftMSA
            let headDim = block.dimensions / block.heads
            let p = block.qkv(mod).reshaped(1, nC + nI, 3, block.heads, headDim).transposed(2, 0, 3, 1, 4)
            let q = concatenated([p[0][0..., 0..., ..<nC, 0...], RoPE2D.apply(p[0][0..., 0..., nC..., 0...], cos: rope.cos, sin: rope.sin)], axis: 2)
            let k = concatenated([p[1][0..., 0..., ..<nC, 0...], RoPE2D.apply(p[1][0..., 0..., nC..., 0...], cos: rope.cos, sin: rope.sin)], axis: 2)
            let weights = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * block.scale, axis: -1)
            let attn = matmul(weights, p[2]).transposed(0, 2, 1, 3).reshaped(1, nC + nI, block.dimensions)
            let swiglu = MLXNN.silu(block.mlpGate(mod)) * block.mlpUp(mod)
            let updated = unified + gates.gateMSA * block.fusedOut(concatenated([attn, swiglu], axis: -1))
            cond = updated[0..., ..<nC, 0...]
            img = updated[0..., nC..., 0...]
        }
        let p = cfg.patchSize
        return module.velocityHead(module.finalNorm(img))
            .reshaped(1, tokH, tokW, p, p, cfg.latentChannels)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(1, tokH * p, tokW * p, cfg.latentChannels)
    }
}
