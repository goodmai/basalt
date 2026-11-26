import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct DualSegmentShape {
    let batch: Int
    let qHeads: Int
    let kvHeads: Int
    let qLen: Int
    let prefixLen: Int
    let activeLen: Int
    let suffixLen: Int
    let headDim: Int
}

private enum DualBlockTestHelpers {
    static func makeInputs(_ s: DualSegmentShape, dtype: DType, seed: UInt64) -> (MLXArray, DualBlockFlashAttentionKernel.Segments) {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let q = MLXRandom.normal([s.batch, s.qHeads, s.qLen, s.headDim]).asType(dtype)
            let kPre = s.prefixLen > 0 ? MLXRandom.normal([s.batch, s.kvHeads, s.prefixLen, s.headDim]).asType(dtype) : nil
            let vPre = s.prefixLen > 0 ? MLXRandom.normal([s.batch, s.kvHeads, s.prefixLen, s.headDim]).asType(dtype) : nil
            let kAct = MLXRandom.normal([s.batch, s.kvHeads, s.activeLen, s.headDim]).asType(dtype)
            let vAct = MLXRandom.normal([s.batch, s.kvHeads, s.activeLen, s.headDim]).asType(dtype)
            let kSuf = s.suffixLen > 0 ? MLXRandom.normal([s.batch, s.kvHeads, s.suffixLen, s.headDim]).asType(dtype) : nil
            let vSuf = s.suffixLen > 0 ? MLXRandom.normal([s.batch, s.kvHeads, s.suffixLen, s.headDim]).asType(dtype) : nil
            let arrays = [q, kPre, vPre, kAct, vAct, kSuf, vSuf].compactMap { $0 }
            eval(arrays)
            let seg = DualBlockFlashAttentionKernel.Segments(
                prefixKeys: kPre, prefixValues: vPre,
                activeKeys: kAct, activeValues: vAct,
                suffixKeys: kSuf, suffixValues: vSuf
            )
            return (q, seg)
        }
    }

    static func stockAttention(queries: MLXArray, segments: DualBlockFlashAttentionKernel.Segments, scale: Float) -> MLXArray {
        let kParts = [segments.prefixKeys, segments.activeKeys, segments.suffixKeys].compactMap { $0 }
        let vParts = [segments.prefixValues, segments.activeValues, segments.suffixValues].compactMap { $0 }
        let keys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let values = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)
        return MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: .none)
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }

    static func makeLLaDAD128Model(seed: UInt64 = 8190) throws -> LLaDAModel {
        let json = """
            {"model_type": "llada", "d_model": 256, "n_layers": 2, "n_heads": 2, "n_kv_heads": 2,
             "mlp_hidden_size": 384, "vocab_size": 256, "embedding_size": 256,
             "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "mask_token_id": 255}
            """
        return try withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = try LLaDAModel(configurationData: Data(json.utf8))
            eval(model)
            return model
        }
    }
}

private struct DenseForwardOnlyAdapter: BidirectionalBlockModel {
    let model: any LanguageModel
    let layerCount: Int

    init(_ model: any LanguageModel) throws {
        self.model = model
        self.layerCount = try model.newCache(parameters: nil).count
    }

    func forward(tokens: MLXArray, caches: [any KVCache]?) -> MLXArray {
        model(LMInput.Text(tokens: tokens), cache: caches, state: nil).logits
    }
}

@Suite("Zero-copy 3-segment DualBlockFlashAttention Metal kernel parity (#819, #228)", .serialized)
struct DualBlockFlashAttentionParityTests {
    @Test(
        "Kernel matches stock concatenated SDPA across head_dim 64 and 128 and all three floating dtypes",
        arguments: [(64, DType.float32), (64, .bfloat16), (64, .float16), (128, .float32), (128, .bfloat16), (128, .float16)]
    )
    func matchesStockAcrossHeadDimsAndDTypes(headDim: Int, dtype: DType) {
        let tol: Float = dtype == .float32 ? 1e-4 : (dtype == .float16 ? 3e-3 : 2e-3)
        let shape = DualSegmentShape(batch: 1, qHeads: 8, kvHeads: 2, qLen: 19, prefixLen: 37, activeLen: 19, suffixLen: 45, headDim: headDim)
        let (q, seg) = DualBlockTestHelpers.makeInputs(shape, dtype: dtype, seed: UInt64(headDim * 100 + (dtype == .float32 ? 1 : dtype == .float16 ? 2 : 3)))
        let scale = 1.0 / Float(headDim).squareRoot()
        let fused = DualBlockFlashAttentionKernel.attention(queries: q, segments: seg, scale: scale)
        let stock = DualBlockTestHelpers.stockAttention(queries: q, segments: seg, scale: scale)
        #expect(fused.shape == stock.shape && fused.dtype == dtype)
        #expect(DualBlockTestHelpers.maxAbsDiff(fused, stock) < tol)
    }

    @Test(
        "LLaDA-8B (32:32) and Dream-7B (28:4) GQA shapes plus prefix-only and empty-prefix boundary segments match stock SDPA",
        arguments: [
            (32, 32, 32, 64, 32, 96),
            (32, 32, 16, 51, 16, 0),
            (28, 4, 17, 0, 17, 33),
            (16, 4, 7, 13, 7, 19),
        ]
    )
    func gqaAndBoundarySegmentsParity(qHeads: Int, kvHeads: Int, qLen: Int, prefixLen: Int, activeLen: Int, suffixLen: Int) {
        let shape = DualSegmentShape(
            batch: 1, qHeads: qHeads, kvHeads: kvHeads, qLen: qLen,
            prefixLen: prefixLen, activeLen: activeLen, suffixLen: suffixLen, headDim: 128
        )
        let (q, seg) = DualBlockTestHelpers.makeInputs(shape, dtype: .bfloat16, seed: UInt64(qHeads * 1000 + prefixLen * 10 + suffixLen))
        let scale = 1.0 / Float(128).squareRoot()
        let fused = DualBlockFlashAttentionKernel.attention(queries: q, segments: seg, scale: scale)
        let stock = DualBlockTestHelpers.stockAttention(queries: q, segments: seg, scale: scale)
        #expect(DualBlockTestHelpers.maxAbsDiff(fused, stock) < 2e-3)
    }

    @Test("Capacity-padded strided prefix slices and threadgroup budget + eligibility predicates hold")
    func stridedPrefixAndEligibilityPredicates() {
        let (q, preBuf, vBuf, kAct, vAct) = withRandomState(MLXRandom.RandomState(seed: 8199)) {
            let q = MLXRandom.normal([1, 4, 16, 128]).asType(.float32)
            let preK = MLXRandom.normal([1, 2, 256, 128]).asType(.float32)
            let preV = MLXRandom.normal([1, 2, 256, 128]).asType(.float32)
            let actK = MLXRandom.normal([1, 2, 16, 128]).asType(.float32)
            let actV = MLXRandom.normal([1, 2, 16, 128]).asType(.float32)
            eval(q, preK, preV, actK, actV)
            return (q, preK, preV, actK, actV)
        }
        let seg = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: preBuf[.ellipsis, ..<39, 0...],
            prefixValues: vBuf[.ellipsis, ..<39, 0...],
            activeKeys: kAct, activeValues: vAct,
            suffixKeys: nil, suffixValues: nil
        )
        let scale = 1.0 / Float(128).squareRoot()
        let fused = DualBlockFlashAttentionKernel.attention(queries: q, segments: seg, scale: scale)
        let stock = DualBlockTestHelpers.stockAttention(queries: q, segments: seg, scale: scale)
        #expect(DualBlockTestHelpers.maxAbsDiff(fused, stock) < 1e-4)

        for headDim in [64, 128] {
            for dtype in [DType.bfloat16, .float16, .float32] {
                let bytes = DualBlockFlashAttentionKernel.threadgroupMemoryBytes(headDim: headDim, dtype: dtype)
                #expect(bytes > 0 && bytes <= DualBlockFlashAttentionKernel.maxThreadgroupMemoryBytes)
            }
        }
        #expect(DualBlockFlashAttentionKernel.isEligible(queries: q, segments: seg, mask: .none, disabled: false))
        #expect(!DualBlockFlashAttentionKernel.isEligible(queries: q, segments: seg, mask: .none, disabled: true))
        #expect(!DualBlockFlashAttentionKernel.isEligible(queries: q, segments: seg, mask: .causal, disabled: false))
    }
}

@Suite("Fast-dLLM masked-only lm_head, zero-copy kernel dispatch, suffix slicing, and block-streaming iterator (#819)", .serialized)
struct FastDLLMBoostIntegrationTests {
    @Test("D=128 LLaDA invokes DualBlockFlashAttentionKernel, skips wasted lm_head projections, and matches dense forward bit-identically")
    func maskedOnlyLMHeadAndKernelDispatchParity() throws {
        let model = try DualBlockTestHelpers.makeLLaDAD128Model()
        let fastAdapter = try LanguageModelBlockAdapter(model)
        let denseAdapter = try DenseForwardOnlyAdapter(model)
        let request = FastDLLMRequest(
            promptTokens: [5, 19, 42, 7, 88, 13, 64, 21],
            maxNewTokens: 32,
            includeTrailingMaskSuffix: true
        )
        let config = BlockDiffusionConfig(
            blockSize: 16,
            confidenceThreshold: 0.95,
            maxStepsPerBlock: 16,
            maskTokenId: model.maskTokenId,
            dualCacheEnabled: true
        )
        var fastSession = FastDLLMSession(model: fastAdapter, request: request, config: config)
        var streamed: [Int] = []
        var blockCount = 0
        while let block = fastSession.nextBlock() {
            streamed.append(contentsOf: block)
            blockCount += 1
        }
        let fast = fastSession.buildResult()
        let dense = FastDLLMRunner.generate(model: denseAdapter, request: request, config: config)

        #expect(blockCount == 2)
        #expect(streamed == fast.generatedTokens)
        #expect(fast.generatedTokens == dense.generatedTokens)
        #expect(fast.tokens == dense.tokens)
        #expect(fastSession.totalKernelInvocations > 0)
        #expect(fast.projectedLogitPositions > 0)
        #expect(fast.projectedLogitPositions < dense.projectedLogitPositions)
    }

    @Test("DiffusionTokenIterator streams block-by-block and strips trailing stop tokens cleanly")
    func iteratorStreamsBlocksAndStripsStopToken() throws {
        let model = try DualBlockTestHelpers.makeLLaDAD128Model(seed: 8191)
        let prompt: [Int32] = [3, 11, 29, 47]
        let rawAdapter = try LanguageModelBlockAdapter(model)
        let baseReq = FastDLLMRequest(promptTokens: prompt.map(Int.init), maxNewTokens: 12, includeTrailingMaskSuffix: true)
        var cfg = BlockDiffusionConfig(blockSize: 4, confidenceThreshold: 0.9, maxStepsPerBlock: 4, maskTokenId: model.maskTokenId)
        cfg.bidirectionalPrompt = true
        let full = FastDLLMRunner.generate(model: rawAdapter, request: baseReq, config: cfg).generatedTokens
        #expect(full.count == 12)

        let stopToken = full[5]
        var iterator = try DiffusionTokenIterator(
            model: model,
            input: LMInput(text: LMInput.Text(tokens: MLXArray(prompt)[.newAxis])),
            maxTokens: 12,
            config: cfg,
            stopTokenIds: [stopToken]
        )
        var emitted: [Int] = []
        while let token = iterator.next() {
            emitted.append(token)
        }
        #expect(!emitted.contains(stopToken))
        #expect(emitted == Array(full.prefix(while: { $0 != stopToken })))
        #expect(iterator.tokenCount == emitted.count)
    }

    @Test("Live defaults, env overrides, rolling maxSuffixLength, and LLaDA <|eot_id|> stop token configure cleanly")
    func cacheModesAndLiveEnvConfig() throws {
        let liveDefault = BlockDiffusionConfig.live(environment: [:])
        #expect(liveDefault.blockSize == 8 && liveDefault.maxStepsPerBlock == 8)
        #expect(liveDefault.reuseSuffixCache && !liveDefault.prefixCacheOnly)
        #expect(liveDefault.maxSuffixLength == 128)
        #expect(liveDefault.factorParallelUnmasking == 1.25)

        let thresholdOverride = BlockDiffusionConfig.live(environment: ["BASALT_DLLM_THRESHOLD": "0.85", "BASALT_DLLM_REUSE_SUFFIX": "0"])
        #expect(thresholdOverride.confidenceThreshold == 0.85)
        #expect(thresholdOverride.factorParallelUnmasking == nil)
        #expect(!thresholdOverride.reuseSuffixCache)

        let env = [
            "BASALT_DLLM_BLOCK_SIZE": "16",
            "BASALT_DLLM_THRESHOLD": "0.8",
            "BASALT_DLLM_FACTOR": "1.5",
            "BASALT_DLLM_PREFIX_ONLY": "1",
            "BASALT_DLLM_REUSE_SUFFIX": "1",
            "BASALT_DLLM_MAX_SUFFIX": "64",
        ]
        let parsed = BlockDiffusionConfig.live(environment: env)
        #expect(parsed.blockSize == 16 && parsed.maxStepsPerBlock == 16)
        #expect(parsed.confidenceThreshold == 0.8)
        #expect(parsed.factorParallelUnmasking == 1.5)
        #expect(parsed.prefixCacheOnly && parsed.reuseSuffixCache)
        #expect(parsed.maxSuffixLength == 64)

        let model = try DualBlockTestHelpers.makeLLaDAD128Model(seed: 8192)
        #expect(model.additionalStopTokens.contains("<|eot_id|>"))
        let adapter = try LanguageModelBlockAdapter(model)
        let prompt = [4, 12, 22, 34, 55, 67]

        var reuseCfg = BlockDiffusionConfig(blockSize: 4, confidenceThreshold: 0.9, maxStepsPerBlock: 4, maskTokenId: model.maskTokenId)
        reuseCfg.bidirectionalPrompt = true
        reuseCfg.reuseSuffixCache = true
        var cappedCfg = reuseCfg
        cappedCfg.maxSuffixLength = 4
        let exactCfg = BlockDiffusionConfig(blockSize: 4, confidenceThreshold: 0.9, maxStepsPerBlock: 4, maskTokenId: model.maskTokenId)
        let reqDual = FastDLLMRequest(promptTokens: prompt, maxNewTokens: 12, includeTrailingMaskSuffix: true)
        let reqPrefix = FastDLLMRequest(promptTokens: prompt, maxNewTokens: 12, includeTrailingMaskSuffix: false)

        let exactRes = FastDLLMRunner.generate(model: adapter, request: reqDual, config: exactCfg)
        let reuseRes = FastDLLMRunner.generate(model: adapter, request: reqDual, config: reuseCfg)
        let cappedRes = FastDLLMRunner.generate(model: adapter, request: reqDual, config: cappedCfg)
        let prefixRes = FastDLLMRunner.generate(model: adapter, request: reqPrefix, config: exactCfg)

        #expect(reuseRes.generatedTokens.count == 12 && !reuseRes.generatedTokens.contains(model.maskTokenId))
        #expect(cappedRes.generatedTokens.count == 12 && !cappedRes.generatedTokens.contains(model.maskTokenId))
        #expect(prefixRes.generatedTokens.count == 12 && !prefixRes.generatedTokens.contains(model.maskTokenId))
        #expect(reuseRes.evaluatedTokenPositions < exactRes.evaluatedTokenPositions)
        #expect(prefixRes.evaluatedTokenPositions < reuseRes.evaluatedTokenPositions)
    }
}
