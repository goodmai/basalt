import Foundation
import MLX
import MLXNN
import BasaltCore

extension FluxKleinSynthesizer {
    package func debugVelocityParity(seed: UInt64 = 42) throws(BasaltError) -> (mlxVelocity: [Float], referenceVelocity: [Float]) {
        guard let module else { throw .modelNotLoaded }
        let latent = MLXRandom.normal([1, 8, 8, config.latentChannels], dtype: .float32, key: MLXRandom.key(seed))
        let pVec = FourierTextEmbedder.embed("FLUX parity verification fixture", dimension: config.textEmbedDim)
        let (condTokens, condVec) = module.textEncoder.encode(
            TextConditioningInput(prompt: "FLUX parity verification fixture", sharedVector: pVec)
        )
        let input = FluxForwardInput(
            latent: latent, condTokens: condTokens, condVec: condVec,
            timestep: 0.35, guidance: 3.5, bboxGrid: nil
        )
        let fastVel = module.predictVelocity(input)
        let refVel = FluxExplicitReference.velocity(module: module, input: input)
        eval(fastVel, refVel)
        return (
            fastVel.asType(.float32).reshaped(-1).asArray(Float.self),
            refVel.asType(.float32).reshaped(-1).asArray(Float.self)
        )
    }

    package func debugReferenceKVCacheParity(
        referenceImages: [Data],
        prompt: String = "Add bioluminescent basalt reflections",
        seed: UInt64 = 109
    ) throws -> (cachedVelocity: [Float], uncachedVelocity: [Float]) {
        guard let module else { throw BasaltError.modelNotLoaded }
        let refs = try referenceImages.map {
            try ImagePNGCodec.decode($0, targetWidth: 64, targetHeight: 64)
        }
        let cacheKey = ReferenceKVCacheKey(
            modelID: modelIdentifier, width: 64, height: 64,
            hiddenSize: config.hiddenSize, patchSize: config.patchSize,
            referenceImages: referenceImages
        )
        let snapshot = ReferenceKVProjector.buildFluxSnapshot(key: cacheKey, refImages: refs, module: module)
        let refLatents = refs.map { module.vae.encode($0) }
        let rawRef = ReferenceKVProjector.encodeRefPatchTokens(
            refLatents: refLatents, refPatchIn: module.refPatchIn, hiddenSize: config.hiddenSize
        )
        let (midRef, uncachedDual) = ReferenceKVProjector.projectDualBlocks(
            refTokens: rawRef, blocks: module.dualBlocks, headDim: config.headDim
        )
        let uncachedSingle = ReferenceKVProjector.projectSingleBlocks(
            refTokens: midRef, blocks: module.singleBlocks, headDim: config.headDim
        )
        let latent = MLXRandom.normal([1, 8, 8, config.latentChannels], dtype: .float32, key: MLXRandom.key(seed))
        let pVec = FourierTextEmbedder.embed(prompt, dimension: config.textEmbedDim)
        let (condTokens, condVec) = module.textEncoder.encode(
            TextConditioningInput(prompt: prompt, sharedVector: pVec)
        )
        let cachedVel = module.predictVelocity(
            FluxForwardInput(
                latent: latent, condTokens: condTokens, condVec: condVec,
                timestep: 0.5, guidance: 3.0, bboxGrid: nil,
                dualRefKV: snapshot.dualLayerKV, singleRefKV: snapshot.singleLayerKV
            )
        )
        let uncachedVel = module.predictVelocity(
            FluxForwardInput(
                latent: latent, condTokens: condTokens, condVec: condVec,
                timestep: 0.5, guidance: 3.0, bboxGrid: nil,
                dualRefKV: uncachedDual, singleRefKV: uncachedSingle
            )
        )
        eval(cachedVel, uncachedVel)
        return (
            cachedVel.asType(.float32).reshaped(-1).asArray(Float.self),
            uncachedVel.asType(.float32).reshaped(-1).asArray(Float.self)
        )
    }
}

private enum FluxExplicitReference {
    static func velocity(module: FluxKleinDiTModule, input: FluxForwardInput) -> MLXArray {
        let cfg = module.config
        let patched = module.patchIn(input.latent)
        let tokH = patched.dim(1)
        let tokW = patched.dim(2)
        var img = patched.reshaped(1, tokH * tokW, cfg.hiddenSize)
        var txt = input.condTokens
        let rope = RoPE2D.frequencies(gridH: tokH, gridW: tokW, headDim: cfg.headDim)
        let stepCtx = DiTStepContext(condVec: input.condVec, rope: rope, timestep: input.timestep, guidance: input.guidance)
        for block in module.dualBlocks {
            (img, txt) = dualBlock(block, img: img, txt: txt, stepCtx: stepCtx)
        }
        for block in module.singleBlocks {
            (txt, img) = singleBlock(block, cond: txt, img: img, stepCtx: stepCtx)
        }
        var normedOut = module.finalNorm(img)
        if let bbox = input.bboxGrid {
            normedOut = normedOut + 0.65 * bbox.bias
        }
        return module.unpatchify(module.velocityHead(normedOut), tokH: tokH, tokW: tokW)
    }

    private static func dualBlock(
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

    private static func singleBlock(
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
