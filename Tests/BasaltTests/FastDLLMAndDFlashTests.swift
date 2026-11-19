import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltInference

@Suite("Fast-dLLM unmasker: monotonic termination in <= B steps, factor-parallel threshold math, and LLaDA 2.1 editable remasking (#574)")
struct FastDLLMUnmaskerTests {
    @Test("Confidence unmasking guarantees at least one token unmasks per step so any block terminates in at most B steps",
          arguments: [1, 4, 8, 16])
    func monotonicTerminationWithinBlockSize(blockSize: Int) {
        let config = BlockDiffusionConfig(
            blockSize: blockSize,
            confidenceThreshold: 0.95,
            maxStepsPerBlock: blockSize,
            maskTokenId: 255
        )
        var masked = [Bool](repeating: true, count: blockSize)
        var steps = 0

        while masked.contains(true) {
            let before = masked.filter { $0 }.count
            let lowConfidences = (0 ..< blockSize).map { Float($0 + 1) / Float(blockSize * 4) }
            let selected = FastDLLMUnmasker.selectPositions(
                confidences: lowConfidences,
                maskedMask: masked,
                config: config
            )
            #expect(!selected.isEmpty)
            for pos in selected {
                #expect(masked[pos])
                masked[pos] = false
            }
            let after = masked.filter { $0 }.count
            #expect(after < before)
            steps += 1
            #expect(steps <= blockSize)
        }
        #expect(!masked.contains(true))
    }

    @Test("Threshold 1.0 unmasks one highest-confidence position per step while threshold 0.0 unmasks all positions in one step")
    func extremeConfidenceThresholds() {
        let confidences: [Float] = [0.42, 0.99, 0.87, 0.65]
        let masked = [true, true, true, true]

        let greedySingle = BlockDiffusionConfig(blockSize: 4, confidenceThreshold: 1.0, maskTokenId: 255)
        #expect(FastDLLMUnmasker.selectPositions(confidences: confidences, maskedMask: masked, config: greedySingle) == [1])

        let parallelAll = BlockDiffusionConfig(blockSize: 4, confidenceThreshold: 0.0, maskTokenId: 255)
        #expect(FastDLLMUnmasker.selectPositions(confidences: confidences, maskedMask: masked, config: parallelAll) == [0, 1, 2, 3])
    }

    @Test("Factor-based parallel unmasking selects ranks satisfying (k + 1) * (1 - p_(k)) < f and falls back to top-1 when none qualify")
    func factorParallelUnmaskingMath() {
        let confidences: [Float] = [0.60, 0.92, 0.80, 0.50]
        let masked = [true, true, true, true]
        let base = BlockDiffusionConfig(blockSize: 4, confidenceThreshold: 0.99, maskTokenId: 255)

        let moderate = base.withFactorParallelUnmasking(0.50)
        #expect(FastDLLMUnmasker.selectPositions(confidences: confidences, maskedMask: masked, config: moderate) == [1, 2])

        let wide = base.withFactorParallelUnmasking(1.50)
        #expect(FastDLLMUnmasker.selectPositions(confidences: confidences, maskedMask: masked, config: wide) == [0, 1, 2])

        let strict = base.withFactorParallelUnmasking(0.02)
        #expect(FastDLLMUnmasker.selectPositions(confidences: confidences, maskedMask: masked, config: strict) == [1])
    }

    @Test("Top-2 margin threshold unmasks positions meeting either top-1 probability or probability gap criteria")
    func top2MarginThresholdSelection() {
        let confidences: [Float] = [0.55, 0.93, 0.48, 0.30]
        let margins: [Float] = [0.40, 0.10, 0.05, 0.02]
        let masked = [true, true, true, true]
        let config = BlockDiffusionConfig(blockSize: 4, confidenceThreshold: 0.90, maskTokenId: 255)
            .withTop2MarginThreshold(0.35)

        let selected = FastDLLMUnmasker.selectPositions(
            confidences: confidences,
            maskedMask: masked,
            config: config,
            top2Margins: margins
        )
        #expect(selected == [0, 1])
    }

    @Test("LLaDA 2.1 editable remasking selects low-confidence already-unmasked non-pinned positions ordered by lowest confidence")
    func editableRemaskingSelectionAndConfigValidation() throws {
        let config = try BlockDiffusionConfig(
            blockSize: 5,
            confidenceThreshold: 0.85,
            maxStepsPerBlock: 8,
            maskTokenId: 255,
            remaskThreshold: 0.40
        ).validated()
        let confidences: [Float] = [0.15, 0.91, 0.32, 0.10, 0.25]
        let masked = [false, false, false, true, false]
        let pinned = [false, false, false, false, true]

        let allEligible = FastDLLMUnmasker.selectRemaskPositions(
            confidences: confidences,
            maskedMask: masked,
            config: config,
            pinnedMask: pinned
        )
        #expect(allEligible == [0, 2])

        let capped = FastDLLMUnmasker.selectRemaskPositions(
            confidences: confidences,
            maskedMask: masked,
            config: config,
            pinnedMask: pinned,
            maxRemaskCount: 1
        )
        #expect(capped == [0])

        var invalid = config
        invalid.confidenceThreshold = 1.5
        #expect(throws: BlockDiffusionConfigError.invalidConfidenceThreshold(1.5)) {
            _ = try invalid.validated()
        }
    }
}

private final class SelfCorrectingToyBlockModel: BidirectionalBlockModel {
    let layerCount = 1
    let maskTokenId: Int
    let vocabSize = 32

    init(maskTokenId: Int) {
        self.maskTokenId = maskTokenId
    }

    func forward(tokens: MLXArray, caches _: [any KVCache]?) -> MLXArray {
        let ids = tokens[0].asArray(Int32.self).map { Int($0) }
        let length = ids.count
        var flat = [Float](repeating: -8.0, count: length * vocabSize)
        let revealedContext = ids.count >= 2 && ids[1] == 12

        for pos in 0 ..< length {
            let base = pos * vocabSize
            if pos == 0 {
                if ids[0] == maskTokenId && !revealedContext {
                    flat[base + 7] = 6.0
                } else if ids[0] == 7 && revealedContext {
                    flat[base + 7] = 0.2
                    flat[base + 8] = 0.2
                    flat[base + 9] = 0.2
                    flat[base + 10] = 0.2
                } else {
                    flat[base + 19] = 8.0
                }
            } else if pos == 1 {
                flat[base + 12] = 7.0
            } else {
                flat[base + 15] = 7.0
            }
        }
        return MLXArray(flat).reshaped(1, length, vocabSize)
    }
}

@Suite("Fast-dLLM DualBlockKVCache and runner parity against full-sequence uncached evaluation (#574)", .serialized)
struct FastDLLMRunnerParityTests {
    @Test("DualCache produces exact identical greedy unmasked tokens to full-sequence uncached evaluation at tau_conf = 1.0 on TinyModel")
    func dualCacheMatchesUncachedAtThresholdOne() throws {
        let tiny = try TinyModel.make(.qwen3Small)
        let adapter = try LanguageModelBlockAdapter(tiny)
        let request = FastDLLMRequest(
            promptTokens: [5, 19, 42, 77, 103, 12],
            maxNewTokens: 8,
            includeTrailingMaskSuffix: true
        )
        let cachedConfig = BlockDiffusionConfig(
            blockSize: 4,
            confidenceThreshold: 1.0,
            maxStepsPerBlock: 4,
            maskTokenId: 255,
            dualCacheEnabled: true
        )
        let uncachedConfig = BlockDiffusionConfig(
            blockSize: 4,
            confidenceThreshold: 1.0,
            maxStepsPerBlock: 4,
            maskTokenId: 255,
            dualCacheEnabled: false
        )

        let cached = FastDLLMRunner.generate(model: adapter, request: request, config: cachedConfig)
        let uncached = FastDLLMRunner.generate(model: adapter, request: request, config: uncachedConfig)

        #expect(cached.tokens == uncached.tokens)
        #expect(cached.generatedTokens == uncached.generatedTokens)
        #expect(cached.generatedTokens.count == 8)
        #expect(!cached.generatedTokens.contains(255))
        #expect(cached.denoisingSteps == uncached.denoisingSteps)
        #expect(cached.evaluatedTokenPositions < uncached.evaluatedTokenPositions)
    }

    @Test("DualCache maintains exact parity under middle-of-file FIM suffix infilling and parallel confidence unmasking")
    func dualCacheMatchesUncachedForInfillingAndParallelThresholds() throws {
        let tiny = try TinyModel.make(.qwen3Small)
        let qwen = try #require(tiny as? Qwen3Model)
        let moduleAdapter = MLXModuleBlockAdapter(
            module: qwen,
            layerCount: try qwen.newCache(parameters: nil).count
        ) { module, tokens, caches in
            module(LMInput.Text(tokens: tokens), cache: caches, state: nil).logits
        }
        let request = FastDLLMRequest(
            promptTokens: [7, 14, 21, 28],
            maxNewTokens: 6,
            suffixTokens: [90, 91, 92],
            includeTrailingMaskSuffix: true
        )
        let cachedParallel = BlockDiffusionConfig(
            blockSize: 3,
            confidenceThreshold: 0.25,
            maxStepsPerBlock: 3,
            maskTokenId: 255,
            dualCacheEnabled: true
        )
        var uncachedParallel = cachedParallel
        uncachedParallel.dualCacheEnabled = false

        let cached = FastDLLMRunner.generate(model: moduleAdapter, request: request, config: cachedParallel)
        let uncached = FastDLLMRunner.generate(model: moduleAdapter, request: request, config: uncachedParallel)

        #expect(cached.tokens == uncached.tokens)
        #expect(Array(cached.tokens.suffix(3)) == [90, 91, 92])
        #expect(!cached.generatedTokens.contains(255))
    }

    @Test("LLaDA 2.1 editable remasking re-masks low-confidence early predictions and converges to the self-corrected block")
    func runnerSelfCorrectsViaEditableRemasking() {
        let model = SelfCorrectingToyBlockModel(maskTokenId: 31)
        let request = FastDLLMRequest(promptTokens: [], maxNewTokens: 3, includeTrailingMaskSuffix: false)
        let config = BlockDiffusionConfig(
            blockSize: 3,
            confidenceThreshold: 0.80,
            maxStepsPerBlock: 6,
            maskTokenId: 31,
            remaskThreshold: 0.35,
            dualCacheEnabled: false
        )

        let result = FastDLLMRunner.generate(model: model, request: request, config: config)
        #expect(result.remaskedTokenCount >= 1)
        #expect(result.generatedTokens == [19, 12, 15])
    }

    @Test("DualBlockLayerKVCache copy, trim, and reset maintain prefix and suffix tensor lengths")
    func layerCacheLifecycleOperations() {
        let layer = DualBlockLayerKVCache()
        layer.phase = .promptPrefill
        let k1 = MLXArray.zeros([1, 2, 4, 8], type: Float.self)
        let v1 = MLXArray.zeros([1, 2, 4, 8], type: Float.self)
        _ = layer.update(keys: k1, values: v1)
        #expect(layer.prefixLength == 4)

        layer.phase = .suffixPrefill(offset: 8)
        let ks = MLXArray.zeros([1, 2, 3, 8], type: Float.self)
        let vs = MLXArray.zeros([1, 2, 3, 8], type: Float.self)
        let (mergedK, _) = layer.update(keys: ks, values: vs)
        #expect(layer.suffixLength == 3)
        #expect(mergedK.dim(2) == 7)

        let copy = layer.copy() as? DualBlockLayerKVCache
        #expect(copy?.prefixLength == 4 && copy?.suffixLength == 3)

        #expect(layer.trim(2) == 2)
        #expect(layer.prefixLength == 2)
        layer.clearSuffix()
        #expect(layer.suffixLength == 0)
        layer.reset()
        #expect(layer.prefixLength == 0 && layer.offset == 0)
    }
}

private enum DFlashTestCalibration {
    static func invertSmallMatrix(_ matrix: [[Double]]) -> [[Double]] {
        let n = matrix.count
        var aug = (0 ..< n).map { row in
            matrix[row] + (0 ..< n).map { col in row == col ? 1.0 : 0.0 }
        }
        for pivot in 0 ..< n {
            var best = pivot
            for candidate in (pivot + 1) ..< n where abs(aug[candidate][pivot]) > abs(aug[best][pivot]) {
                best = candidate
            }
            aug.swapAt(pivot, best)
            let diag = abs(aug[pivot][pivot]) > 1e-12 ? aug[pivot][pivot] : 1e-12
            for col in 0 ..< (2 * n) { aug[pivot][col] /= diag }
            for row in 0 ..< n where row != pivot {
                let factor = aug[row][pivot]
                for col in 0 ..< (2 * n) {
                    aug[row][col] -= factor * aug[pivot][col]
                }
            }
        }
        return aug.map { Array($0[n ..< (2 * n)]) }
    }

    static func alignHead(
        drafter: DFlashDraftModel,
        anchorToken: Int,
        contextHidden: MLXArray,
        queryOffset: Int,
        expectedTokens: [Int]
    ) {
        let blockLen = expectedTokens.count
        let hidden = drafter.encodeDraftHidden(
            anchorTokens: MLXArray([Int32(anchorToken)]).reshaped(1, 1),
            targetHidden: contextHidden,
            queryOffset: queryOffset,
            draftCount: blockLen
        )[0].asType(.float32)
        eval(hidden)

        let dim = hidden.dim(-1)
        let vocab = drafter.config.vocabularySize
        let hFlat = hidden.asArray(Float.self).map { Double($0) }

        var gram = Array(repeating: Array(repeating: 0.0, count: blockLen), count: blockLen)
        for r1 in 0 ..< blockLen {
            for r2 in 0 ..< blockLen {
                var dot = r1 == r2 ? 1e-6 : 0.0
                for d in 0 ..< dim { dot += hFlat[r1 * dim + d] * hFlat[r2 * dim + d] }
                gram[r1][r2] = dot
            }
        }
        let invGram = invertSmallMatrix(gram)
        var dualRows = Array(repeating: 0.0, count: blockLen * dim)
        for r in 0 ..< blockLen {
            for d in 0 ..< dim {
                var acc = 0.0
                for c in 0 ..< blockLen { acc += invGram[r][c] * hFlat[c * dim + d] }
                dualRows[r * dim + d] = acc
            }
        }
        var weights = [Float](repeating: 0, count: vocab * dim)
        for (slot, token) in expectedTokens.enumerated() {
            for d in 0 ..< dim {
                weights[token * dim + d] += Float(12.0 * dualRows[slot * dim + d])
            }
        }
        let updated = MLXArray(weights).reshaped(vocab, dim)
        drafter.update(parameters: ModuleParameters.unflattened(["lm_head.weight": updated]))
        eval(drafter)
    }
}

@Suite("DFlash single-forward block-diffusion drafter and lossless verifier parity (#574)", .serialized)
struct DFlashSpeculativeTests {
    @Test("DFlashBlockDrafter + DFlashVerifier produce bit-identical T = 0 greedy token sequences to solo TinyModel across multi-block rollouts")
    func unalignedDrafterMatchesSoloGreedyExactly() throws {
        let tiny = try TinyModel.make(.qwen3Small)
        let target = LanguageModelDFlashTarget(tiny, hiddenSize: 64)
        let drafter = withRandomState(MLXRandom.RandomState(seed: 574)) {
            DFlashDraftModel(DFlashConfig(
                hiddenSize: 64,
                vocabularySize: 256,
                numHiddenLayers: 2,
                numAttentionHeads: 4,
                blockSize: 6,
                maskTokenId: 255
            ))
        }
        eval(drafter)

        let prompt = [11, 23, 37, 53]
        let solo = try DFlashBlockDrafter.soloGreedy(target: target, promptTokens: prompt, maxNewTokens: 15)
        let dflash = try DFlashBlockDrafter.generateGreedy(
            target: target,
            drafter: drafter,
            promptTokens: prompt,
            maxNewTokens: 15
        )

        #expect(dflash.tokens == solo)
        #expect(dflash.generatedTokens.count == 15)
        #expect(dflash.telemetry.draftModelCallCount == dflash.telemetry.roundCount)
    }

    @Test("Aligned DFlash drafter achieves > 1 accepted tokens per step while preserving bit-identical T = 0 output to solo TinyModel")
    func alignedDrafterAcceleratesRolloutWithExactGreedyParity() throws {
        let tiny = try TinyModel.make(.qwen3Small)
        let target = LanguageModelDFlashTarget(tiny, hiddenSize: 64)
        let drafter = withRandomState(MLXRandom.RandomState(seed: 1574)) {
            DFlashDraftModel(DFlashConfig(
                hiddenSize: 64,
                vocabularySize: 256,
                numHiddenLayers: 2,
                numAttentionHeads: 4,
                blockSize: 6,
                maskTokenId: 255
            ))
        }
        eval(drafter)

        let prompt = [9, 18, 27, 36]
        let solo = try DFlashBlockDrafter.soloGreedy(target: target, promptTokens: prompt, maxNewTokens: 12)

        let prefillCache = try target.newCache()
        let (_, prefillHidden) = target.forwardWithHidden(
            tokens: MLXArray(prompt.map { Int32($0) })[.newAxis, 0...],
            cache: prefillCache
        )
        eval(prefillHidden)

        DFlashTestCalibration.alignHead(
            drafter: drafter,
            anchorToken: solo[prompt.count],
            contextHidden: prefillHidden,
            queryOffset: prompt.count,
            expectedTokens: Array(solo[(prompt.count + 1) ..< (prompt.count + 7)])
        )

        let dflash = try DFlashBlockDrafter.generateGreedy(
            target: target,
            drafter: drafter,
            promptTokens: prompt,
            maxNewTokens: 12
        )

        #expect(dflash.tokens == solo)
        #expect(dflash.telemetry.acceptedDraftTokenCount >= 6)
        #expect(dflash.telemetry.meanEmittedTokensPerTargetCall > 1.5)
    }

    @Test("DFlashVerifier.verifyRejection matches greedy at T = 0, emits bonus on full accept, and resamples residual (p - q)+ on rejection")
    func rejectionSamplingExactnessAndResidualSupport() {
        let draftTokens = [1, 2]
        let targetLogits = MLXArray([
            Float(0.0), 10.0, 0.0, 0.0,
            0.0, 0.0, -20.0, 12.0,
            5.0, 0.0, 0.0, 0.0,
        ]).reshaped(3, 4)
        let draftLogits = MLXArray([
            Float(0.0), 10.0, 0.0, 0.0,
            0.0, 0.0, 12.0, -20.0,
        ]).reshaped(2, 4)

        let greedyAtZero = DFlashVerifier.verifyRejection(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: 0.0,
            acceptanceUniforms: [0.5, 0.5],
            resampleUniforms: [0.5, 0.5, 0.5]
        )
        #expect(greedyAtZero == DFlashVerifier.verifyGreedy(draftTokens: draftTokens, targetLogits: targetLogits))
        #expect(greedyAtZero.acceptedDraftCount == 1)
        #expect(greedyAtZero.emittedTokens == [1, 3])
        #expect(greedyAtZero.cacheTrimCount == 1)

        let stochasticReject = DFlashVerifier.verifyRejection(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: 1.0,
            acceptanceUniforms: [0.2, 0.2],
            resampleUniforms: [0.4, 0.4, 0.4]
        )
        #expect(stochasticReject.acceptedDraftCount == 1)
        #expect(stochasticReject.emittedTokens == [1, 3])
        #expect(stochasticReject.cacheTrimCount == 1)

        let matchingTarget = MLXArray([
            Float(0.0), 10.0, 0.0, 0.0,
            0.0, 0.0, 12.0, 0.0,
            0.0, 0.0, 0.0, 15.0,
        ]).reshaped(3, 4)
        let fullAccept = DFlashVerifier.verifyRejection(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: matchingTarget,
            temperature: 1.0,
            acceptanceUniforms: [0.1, 0.1],
            resampleUniforms: [0.2, 0.2, 0.2]
        )
        #expect(fullAccept.acceptedDraftCount == 2)
        #expect(fullAccept.emittedTokens == [1, 2, 3])
        #expect(fullAccept.cacheTrimCount == 0)
    }

    @Test("ServeCommand parses --experimental-dllm and --experimental-dflash off by default and rejects enabling both simultaneously")
    func experimentalFlagsParseAndValidate() throws {
        let defaults = try ServeCommand.parse([])
        #expect(!defaults.experimental.dllm && !defaults.experimental.dflash)
        #expect(defaults.experimental.blockDiffusionConfig == nil)
        #expect(defaults.experimental.dflashConfig == nil)

        let dllmOnly = try ServeCommand.parse(["--experimental-dllm"])
        #expect(dllmOnly.experimental.dllm && !dllmOnly.experimental.dflash)
        #expect(dllmOnly.experimental.blockDiffusionConfig != nil)

        let dflashOnly = try ServeCommand.parse(["--experimental-dflash"])
        #expect(!dflashOnly.experimental.dllm && dflashOnly.experimental.dflash)
        #expect(dflashOnly.experimental.dflashConfig != nil)

        #expect(throws: (any Error).self) {
            _ = try ServeCommand.parse(["--experimental-dllm", "--experimental-dflash"])
        }
    }
}
