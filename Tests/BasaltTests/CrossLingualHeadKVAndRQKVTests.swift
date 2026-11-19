import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct UTF8ProbeTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map(Int.init)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(clamping: $0) }, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { decode(tokenIds: [id], skipSpecialTokens: false) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { encode(text: $0["content"] as? String ?? "", addSpecialTokens: false) }
    }
}

@Suite("Rank-Biased Overlap and Reciprocal Rank Fusion satisfy exact rank-similarity identities (#572)")
struct CrossLingualRBOTests {
    @Test("Identical rankings yield exact RBO 1.0 and disjoint rankings yield exact RBO 0.0 across persistence and depth")
    func exactBoundaryIdentities() {
        let primary = (0 ..< 8).map { KVHeadCoordinate(layer: $0 / 2, head: $0 % 2) }
        let disjoint = (8 ..< 16).map { KVHeadCoordinate(layer: $0 / 2, head: $0 % 2) }
        for p in [0.0, 0.5, 0.9, 0.95, 1.0] {
            for depth in [1, 3, 8, nil] {
                #expect(CrossLingualHeadKV.rankBiasedOverlap(primary, primary, persistence: p, depth: depth) == 1.0)
                #expect(CrossLingualHeadKV.rankBiasedOverlap(primary, disjoint, persistence: p, depth: depth) == 0.0)
            }
        }
        let empty: [KVHeadCoordinate] = []
        #expect(CrossLingualHeadKV.rankBiasedOverlap(empty, empty) == 1.0)
        #expect(CrossLingualHeadKV.rankBiasedOverlap(primary, empty) == 0.0)
        #expect(CrossLingualHeadKV.rankBiasedOverlap(empty, primary) == 0.0)
    }

    @Test("RBO is symmetric, matches the closed-form truncated series, and weights top ranks strictly above tail ranks")
    func symmetryClosedFormAndTopWeighting() {
        let base = [0, 1, 2, 3, 4, 5]
        let topSwap = [1, 0, 2, 3, 4, 5]
        let midSwap = [0, 2, 1, 3, 4, 5]
        let tailSwap = [0, 1, 2, 3, 5, 4]
        for candidate in [topSwap, midSwap, tailSwap, [0, 1, 2]] {
            let forward = CrossLingualHeadKV.rankBiasedOverlap(base, candidate, persistence: 0.9)
            let reverse = CrossLingualHeadKV.rankBiasedOverlap(candidate, base, persistence: 0.9)
            #expect(forward == reverse)
        }
        let topScore = CrossLingualHeadKV.rankBiasedOverlap(base, topSwap, persistence: 0.9)
        let midScore = CrossLingualHeadKV.rankBiasedOverlap(base, midSwap, persistence: 0.9)
        let tailScore = CrossLingualHeadKV.rankBiasedOverlap(base, tailSwap, persistence: 0.9)
        #expect(topScore < midScore)
        #expect(midScore < tailScore)
        #expect(tailScore < 1.0)

        let fourA = [0, 1, 2, 3]
        let fourB = [1, 0, 2, 3]
        let analytical = ((1.0 - 0.9) / (1.0 - pow(0.9, 4.0))) * (0.0 + 0.9 * 1.0 + 0.81 * 1.0 + 0.729 * 1.0)
        let computed = CrossLingualHeadKV.rankBiasedOverlap(fourA, fourB, persistence: 0.9)
        #expect(abs(computed - analytical) < 1e-12)
    }

    @Test("Cross-lingual mask resolution reuses cached masks zero-shot at RBO >= 0.75 and fuses via RRF below 0.75")
    func zeroShotReuseVersusReciprocalRankFusion() {
        let heads = [2, 2, 2, 2]
        let coords = (0 ..< 8).map { KVHeadCoordinate(layer: $0 / 2, head: $0 % 2) }
        let cachedMask = KVHeadMask(retrieval: [[0, 1], [0], [1], []])
        let alignedTarget = Array(coords.prefix(5)) + Array(coords.suffix(3).reversed())
        let zeroShot = CrossLingualHeadKV.resolveMask(
            cachedMask: cachedMask, sourceRanking: coords, targetRanking: alignedTarget, heads: heads)
        #expect(zeroShot.rbo >= CrossLingualHeadKV.zeroShotThreshold)
        #expect(zeroShot.mode == .zeroShot)
        #expect(zeroShot.mask == cachedMask)

        let divergentTarget = Array(coords.reversed())
        let fused = CrossLingualHeadKV.resolveMask(
            cachedMask: cachedMask, sourceRanking: coords, targetRanking: divergentTarget, heads: heads,
            policy: CrossLingualHeadKV.TransferPolicy(rrfConstant: 1.0))
        #expect(fused.rbo < CrossLingualHeadKV.zeroShotThreshold)
        #expect(fused.mode == .reciprocalRankFused)
        let expectedFused = CrossLingualHeadKV.reciprocalRankFusion(
            [coords, divergentTarget], heads: heads, keep: 4, k: 1.0)
        #expect(fused.mask == expectedFused)
        #expect(KVHeadSelection(mask: fused.mask, meanKL: 0, answersKept: true).keptHeads == 4)
        #expect(CrossLingualHeadKV.reciprocalRankFusion([coords], heads: heads, keep: 0) == KVHeadMask(retrieval: [[], [], [], []]))
        #expect(CrossLingualHeadKV.reciprocalRankFusion([coords], heads: heads, keep: 8) == KVHeadMask(retrieval: [[0, 1], [0, 1], [0, 1], [0, 1]]))
    }
}

@Suite("Multilingual EN, RU, KO, and JA needle probes tokenize and transfer or fuse KVHeadMask on TinyModel (#572)", .serialized)
struct CrossLingualHeadKVTests {
    @Test("Compact EN, RU, KO, and JA probes embed every passkey and align answer rows to the 5-digit passkey tokens",
          arguments: KVProbeLanguage.allCases)
    func multilingualProbeTokenization(language: KVProbeLanguage) {
        let tokenizer = UTF8ProbeTokenizer()
        let depths = [0.2, 0.5, 0.8]
        let probe = KVHeadClassifier.probe(
            tokenizer: tokenizer, length: 512, depths: depths, seed: 1, language: language)
        #expect(probe.context.count >= 512)
        #expect(probe.context.allSatisfy { (0 ..< 256).contains($0) })
        #expect(probe.tail.allSatisfy { (0 ..< 256).contains($0) })

        let decodedContext = tokenizer.decode(tokenIds: probe.context, skipSpecialTokens: false)
        for index in depths.indices {
            let key = 41_000 + (1 * depths.count + index) * 7_919
            #expect(decodedContext.components(separatedBy: "\(key)").count == 3)
        }
        switch language {
        case .en:
            #expect(decodedContext.unicodeScalars.allSatisfy { $0.isASCII })
        case .ru:
            #expect(decodedContext.unicodeScalars.contains { (0x0400 ... 0x04FF).contains($0.value) })
        case .ko:
            #expect(decodedContext.unicodeScalars.contains { (0xAC00 ... 0xD7A3).contains($0.value) })
        case .ja:
            #expect(decodedContext.unicodeScalars.contains { (0x3040 ... 0x9FFF).contains($0.value) })
        }
        let fullTail = probe.tail + tokenizer.encode(text: ".", addSpecialTokens: false)
        #expect(probe.answerRows.allSatisfy { $0 >= 0 && $0 + 1 < fullTail.count })
        let predicted = tokenizer.decode(tokenIds: probe.answerRows.map { fullTail[$0 + 1] }, skipSpecialTokens: false)
        #expect(predicted.filter(\.isNumber).count == depths.count * 5)
    }

    @Test("TinyModel ranks KV heads across EN, RU, KO, and JA and executes both zero-shot reuse and RRF fusion")
    func tinyModelCrossLingualSelection() throws {
        let model = try TinyModel.make(.qwen3)
        let tokenizer = UTF8ProbeTokenizer()
        let enSelection = try KVHeadClassifier.select(
            model: model, tokenizer: tokenizer, ratio: 0.5, length: 256, language: .en)
        let (enHeads, enRanking) = try CrossLingualHeadKV.rankHeads(
            model: model, tokenizer: tokenizer, language: .en, length: 256)
        #expect(enHeads == [2, 2, 2, 2])
        #expect(Set(enRanking).count == 8)

        let cached = CrossLingualHeadKV.SourceSelection(mask: enSelection.mask, ranking: enRanking)
        let selfTransfer = try CrossLingualHeadKV.transferSelection(
            model: model, tokenizer: tokenizer, cached: cached, targetLanguage: .en, length: 256)
        #expect(selfTransfer.transfer.rbo == 1.0)
        #expect(selfTransfer.transfer.mode == .zeroShot)
        #expect(selfTransfer.transfer.mask == enSelection.mask)

        for target in [KVProbeLanguage.ru, .ko, .ja] {
            let natural = try CrossLingualHeadKV.transferSelection(
                model: model, tokenizer: tokenizer, cached: cached, targetLanguage: target, length: 256)
            #expect(natural.selection.fits(enHeads))
            #expect(natural.selection.keptHeads == enSelection.keptHeads)
            #expect(natural.selection.meanKL.isFinite)
            let expectedMode: CrossLingualHeadKV.TransferMode = natural.transfer.rbo >= CrossLingualHeadKV.zeroShotThreshold
                ? .zeroShot : .reciprocalRankFused
            #expect(natural.transfer.mode == expectedMode)
        }

        let reversed = CrossLingualHeadKV.SourceSelection(mask: enSelection.mask, ranking: Array(enRanking.reversed()))
        let forcedRRF = try CrossLingualHeadKV.transferSelection(
            model: model, tokenizer: tokenizer, cached: reversed, targetLanguage: .ko, length: 256,
            policy: CrossLingualHeadKV.TransferPolicy(threshold: 1.01))
        #expect(forcedRRF.transfer.mode == .reciprocalRankFused)

        let singleLang = try CrossLingualHeadKV.fuseMultilingual(
            model: model, tokenizer: tokenizer, languages: [.en], ratio: 0.5, length: 256)
        #expect(singleLang.mode == .zeroShot && singleLang.minRBO == 1.0)
        let multiLang = try CrossLingualHeadKV.fuseMultilingual(
            model: model, tokenizer: tokenizer, languages: [.en, .ru, .ko, .ja], ratio: 0.5, length: 256)
        #expect(KVHeadSelection(mask: multiLang.mask, meanKL: 0, answersKept: true).fits(enHeads))
        let caches = multiLang.mask.replacingFullAttention(try model.newCache(parameters: nil))
        let out = model(MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4), cache: caches)
        eval(out)
        #expect(out.shape == [1, 4, 256])
    }
}

@Suite("Per-head centroid residual quantization RQ-KV strictly reduces reconstruction MSE on anisotropic keys (#572)", .serialized)
struct CentroidRQKVTests {
    static func anisotropicKeys(clusters: Int, tokens: Int, heads: Int = 4, dim: Int = 64, seed: UInt64 = 17) -> (keys: MLXArray, noiseless: MLXArray) {
        let centers = MLXRandom.normal([1, heads, clusters, dim], key: MLXRandom.key(seed))
        let anisotropy = MLXArray((0 ..< dim).map { idx in Float(1.0 + 7.0 * Double(idx % 8) / 7.0) }).reshaped(1, 1, 1, dim)
        let scaledCenters = centers * anisotropy * Float(3.5)
        var rng = SplitMix64(seed: seed ^ 0xA5A5_5A5A)
        let ids = (0 ..< heads * tokens).map { _ in Int32(rng.next() % UInt64(clusters)) }
        let gatherIdx = MLXArray(ids).reshaped(1, heads, tokens, 1)
        let noiseless = takeAlong(scaledCenters, gatherIdx, axis: 2)
        let noise = MLXRandom.normal([1, heads, tokens, dim], key: MLXRandom.key(seed + 1)) * Float(0.18)
        let keys = noiseless + noise
        eval(keys, noiseless)
        return (keys, noiseless)
    }

    static func mse(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        (lhs.asType(.float32) - rhs.asType(.float32)).square().mean().item(Float.self)
    }

    @Test("RQ-KV with K in 4, 8, and 16 centroids strictly lowers 2-bit and 4-bit key MSE below direct affine quantization",
          arguments: [4, 8, 16])
    func strictMSEReductionAndExactResidualIdentity(clusters: Int) throws {
        let (keys, noiseless) = Self.anisotropicKeys(clusters: clusters, tokens: 256)
        for bits in [2, 4] {
            var direct = ZonedKVCache.QuantizedRows(bits: bits, groupSize: 32)
            direct.append(keys[.ellipsis, ..<128, 0...])
            direct.append(keys[.ellipsis, 128..., 0...])
            let directDecoded = try #require(direct.dequantizedRows())

            var rq = CentroidRQKVRows(clusters: clusters, bits: bits, groupSize: 32, iterations: 6, seed: 42)
            rq.append(keys[.ellipsis, ..<128, 0...])
            rq.append(keys[.ellipsis, 128..., 0...])
            let rqDecoded = try #require(rq.dequantizedRows(dtype: .float32))
            let cHat = try #require(rq.dequantizedCentroids(dtype: .float32))
            let rHat = try #require(rq.dequantizedResiduals(dtype: .float32))
            #expect((abs(rqDecoded - (cHat + rHat)).max()).item(Float.self) < 1e-6)
            #expect(!rq.arrays.isEmpty)

            let directMSE = Self.mse(directDecoded, keys)
            let rqMSE = Self.mse(rqDecoded, keys)
            if bits == 2 {
                #expect(rqMSE < directMSE * 0.35, "2-bit RQ-KV MSE \(rqMSE) did not beat 0.35x direct MSE \(directMSE)")
            } else {
                #expect(rqMSE < directMSE, "4-bit RQ-KV MSE \(rqMSE) did not beat direct MSE \(directMSE)")
            }
        }

        let exactBook = CentroidCodebook.fitKMeansPlusPlus(keys: noiseless, clusters: clusters, iterations: 8, seed: 7)
        var exactRQ = CentroidRQKVRows(clusters: clusters, bits: 2, groupSize: 32, codebook: exactBook)
        exactRQ.append(noiseless)
        let exactDecoded = try #require(exactRQ.dequantizedRows(dtype: .float32))
        #expect(Self.mse(exactDecoded, noiseless) < 1e-10)
    }

    @Test("Fused RQ-KV decode scores, trim, online centroid updates, polar residuals, and ZonedKVCache preserve accuracy")
    func fusedScoresOnlineUpdatesPolarAndZonedCache() throws {
        let (keys, _) = Self.anisotropicKeys(clusters: 8, tokens: 192, heads: 2, dim: 64, seed: 99)
        var rq = CentroidRQKVRows(clusters: 8, bits: 2, groupSize: 32, iterations: 5, seed: 11)
        rq.append(keys)
        let decoded = try #require(rq.dequantizedRows(dtype: .float32))
        let grouped = MLXRandom.normal([1, 2, 2, 1, 64], key: MLXRandom.key(23))
        let expectedScores = matmul(grouped, expandedDimensions(decoded, axis: 2).swappedAxes(-1, -2))
        #expect((abs(try #require(rq.scores(grouped)) - expectedScores).max()).item(Float.self) < 1e-4)
        #expect(rq.trim(64) == 64 && rq.count == 128)
        #expect(rq.trim(128) == 128 && rq.count == 0 && rq.dequantizedRows() == nil)

        var book = CentroidCodebook.fitKMeansPlusPlus(keys: keys[.ellipsis, ..<64, 0...], clusters: 8, iterations: 0, seed: 3)
        let shifted = keys[.ellipsis, 64..., 0...] + Float(2.5)
        let beforeMSE = Self.mse(book.gather(CentroidCodebook.nearestIndices(keys: shifted, centroids: book.centroids)), shifted)
        book.updateOnline(shifted, decay: 0.25)
        #expect(Self.mse(book.gather(CentroidCodebook.nearestIndices(keys: shifted, centroids: book.centroids)), shifted) < beforeMSE)

        var polarRQ = CentroidRQKVRows(clusters: 8, polar: PolarKeyRows(rotated: 32, restBits: 2, groupSize: 32, headDim: 64), seed: 5)
        polarRQ.append(keys)
        let polarDecoded = try #require(polarRQ.dequantizedRows(dtype: .float32))
        let expectedPolarScores = matmul(grouped, expandedDimensions(polarDecoded, axis: 2).swappedAxes(-1, -2))
        #expect((abs(try #require(polarRQ.scores(grouped)) - expectedPolarScores).max()).item(Float.self) < 1e-4)
        var polarDirect = PolarKeyRows(rotated: 32, restBits: 2, groupSize: 32, headDim: 64)
        polarDirect.append(keys)
        #expect(Self.mse(polarDecoded, keys) < Self.mse(try #require(polarDirect.dequantizedRows(dtype: .float32)), keys))
        #expect(polarRQ.trim(32) == 32 && !polarRQ.arrays.isEmpty)

        let basePrec = KVPrecision(keyBits: 2, valueBits: 8)
        let rqPrecision = basePrec.withCentroids(8)
        #expect(rqPrecision.description == "k2v8+rq8" && !rqPrecision.isUniformAffine)
        #expect(KVPrecision(bits: 2).withCentroids(8).description == "2-bit+rq8")
        #expect(KVPrecision.polar4.withCentroids(4).resolvingPolar(rotated: 32).description == "polar4+rq4")
        #expect(KVPrecision.flagConflict(kvBitsSet: true, fixed: rqPrecision, zones: false) != nil)
        let query = MLXRandom.normal([1, 2, 16, 64], key: MLXRandom.key(31))
        let exactAttn = MLXFast.scaledDotProductAttention(
            queries: query, keys: concatenated([keys, keys[.ellipsis, 191..., 0...]], axis: 2),
            values: concatenated([keys, keys[.ellipsis, 191..., 0...]], axis: 2), scale: 0.125, mask: .none)
        let directAttn = PolarKeysTests.decode(basePrec, keys: keys, query: query)
        let rqAttn = PolarKeysTests.decode(rqPrecision, keys: keys, query: query)
        #expect(Self.mse(rqAttn, exactAttn) < Self.mse(directAttn, exactAttn))
    }
}

@Suite("DynamicTreeBudget synchronously and monotonically throttles speculative depth and width (#572)")
struct DynamicTreeBudgetTests {
    @Test("Tree depth stays within 1...6 and width within 1...4 across extreme and non-finite inputs")
    func boundsAndLiveProcessInfo() {
        let states: [ProcessInfo.ThermalState] = [.nominal, .fair, .serious, .critical]
        for state in states {
            for lowPower in [false, true] {
                for alpha in [-1.0, 0.0, 0.5, 1.0, 2.5, .nan, .infinity, -.infinity] {
                    for batch in [-4, 0, 1, 4, 16] {
                        let budget = DynamicTreeBudget.compute(
                            thermalState: state, isLowPowerModeEnabled: lowPower,
                            acceptanceEMA: alpha, activeBatchSize: batch)
                        #expect(DynamicTreeBudget.depthRange.contains(budget.depth))
                        #expect(DynamicTreeBudget.widthRange.contains(budget.width))
                        #expect(budget.verifyBudget == budget.depth * budget.width)
                        #expect(budget.maxDraftNodes >= budget.depth)
                    }
                }
            }
        }
        let live = DynamicTreeBudget.current(acceptanceEMA: 0.8, activeBatchSize: 1)
        #expect(DynamicTreeBudget.depthRange.contains(live.depth))
        #expect(DynamicTreeBudget.widthRange.contains(live.width))
    }

    @Test("Escalating thermal states monotonically throttle speculative depth and width and collapse to 1x1 at critical")
    func monotonicThermalEscalation() {
        let ordered: [ProcessInfo.ThermalState] = [.nominal, .fair, .serious, .critical]
        let budgets = ordered.map {
            DynamicTreeBudget.compute(thermalState: $0, isLowPowerModeEnabled: false, acceptanceEMA: 0.9, activeBatchSize: 1)
        }
        #expect(budgets[0] == DynamicTreeBudget(depth: 6, width: 4))
        #expect(budgets[1].depth < budgets[0].depth && budgets[1].width < budgets[0].width)
        #expect(budgets[2].depth < budgets[1].depth && budgets[2].width < budgets[1].width)
        #expect(budgets[3] == DynamicTreeBudget(depth: 1, width: 1))

        for alpha in [0.0, 0.4, 0.7, 1.0] {
            for lowPower in [false, true] {
                let crit = DynamicTreeBudget.compute(
                    thermalState: .critical, isLowPowerModeEnabled: lowPower, acceptanceEMA: alpha, activeBatchSize: 1)
                #expect(crit == DynamicTreeBudget(depth: 1, width: 1))
            }
        }
    }

    @Test("Low power mode, lower acceptance EMA, and larger active batch sizes monotonically throttle across the state grid")
    func fullGridMonotonicity() {
        let states: [ProcessInfo.ThermalState] = [.nominal, .fair, .serious, .critical]
        let alphas = [0.05, 0.22, 0.35, 0.55, 0.70, 0.90]
        let batches = [1, 2, 4, 6, 8]
        for sIdx in states.indices {
            for aIdx in alphas.indices {
                for bIdx in batches.indices {
                    let ac = DynamicTreeBudget.compute(
                        thermalState: states[sIdx], isLowPowerModeEnabled: false,
                        acceptanceEMA: alphas[aIdx], activeBatchSize: batches[bIdx])
                    let lowPower = DynamicTreeBudget.compute(
                        thermalState: states[sIdx], isLowPowerModeEnabled: true,
                        acceptanceEMA: alphas[aIdx], activeBatchSize: batches[bIdx])
                    #expect(lowPower.depth <= ac.depth && lowPower.width <= ac.width)
                    if sIdx + 1 < states.count {
                        let hotter = DynamicTreeBudget.compute(
                            thermalState: states[sIdx + 1], isLowPowerModeEnabled: false,
                            acceptanceEMA: alphas[aIdx], activeBatchSize: batches[bIdx])
                        #expect(hotter.depth <= ac.depth && hotter.width <= ac.width)
                    }
                    if aIdx > 0 {
                        let lowerAlpha = DynamicTreeBudget.compute(
                            thermalState: states[sIdx], isLowPowerModeEnabled: false,
                            acceptanceEMA: alphas[aIdx - 1], activeBatchSize: batches[bIdx])
                        #expect(lowerAlpha.depth <= ac.depth && lowerAlpha.width <= ac.width)
                    }
                    if bIdx + 1 < batches.count {
                        let busier = DynamicTreeBudget.compute(
                            thermalState: states[sIdx], isLowPowerModeEnabled: false,
                            acceptanceEMA: alphas[aIdx], activeBatchSize: batches[bIdx + 1])
                        #expect(busier.depth <= ac.depth && busier.width <= ac.width)
                    }
                }
            }
        }
    }

    @Test("Individual battery, acceptance, and batch stressors strictly shrink the nominal budget and acceptance EMA updates cleanly")
    func individualStressorsAndEMAUpdates() {
        let favorable = DynamicTreeBudget.compute(thermalState: .nominal, isLowPowerModeEnabled: false, acceptanceEMA: 0.9, activeBatchSize: 1)
        let lowPowerOnly = DynamicTreeBudget.compute(thermalState: .nominal, isLowPowerModeEnabled: true, acceptanceEMA: 0.9, activeBatchSize: 1)
        let lowAlphaOnly = DynamicTreeBudget.compute(thermalState: .nominal, isLowPowerModeEnabled: false, acceptanceEMA: 0.1, activeBatchSize: 1)
        let highBatchOnly = DynamicTreeBudget.compute(thermalState: .nominal, isLowPowerModeEnabled: false, acceptanceEMA: 0.9, activeBatchSize: 8)
        #expect(lowPowerOnly.depth < favorable.depth && lowPowerOnly.width < favorable.width)
        #expect(lowAlphaOnly.depth < favorable.depth && lowAlphaOnly.width < favorable.width)
        #expect(highBatchOnly.depth < favorable.depth && highBatchOnly.width < favorable.width)

        var ema = SpeculativeAcceptanceEMA(initial: 0.5, smoothing: 0.25)
        let before = ema.value
        ema.observe(accepted: 0, drafted: 0)
        #expect(ema.value == before)
        ema.observe(accepted: 4, drafted: 4)
        #expect(ema.value > before)
        let dropped = ema.observing(accepted: 0, drafted: 4)
        #expect(dropped.value < ema.value)
    }
}
