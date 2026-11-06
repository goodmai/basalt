import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("KV heads for --kv-head-split are chosen by output fidelity against the dense cache, and a load-time gate refuses a set that loses answers (#282)", .serialized)
struct KVHeadSelectionTests {
    static func redundantPairLoss(_ retrieval: [[Int]]) -> Float {
        let layer0 = Set(retrieval[0]), layer1 = Set(retrieval[1])
        let pair: Float = layer0.isDisjoint(with: [0, 1]) ? 5 : 0
        return pair + (layer1.contains(1) ? 0 : 3) + (layer1.contains(0) ? 0 : 0.5)
    }

    @Test("Greedy selection keeps one head of a redundant pair that single-head ablation ranks last, and so keeps the output that ablation's top heads lose")
    func greedyHandlesInteractingHeads() {
        let full: [[Int]] = [[0, 1], [0, 1]]
        let ablation = full.enumerated().map { layer, heads in
            heads.map { head in
                var without = full
                without[layer].removeAll { $0 == head }
                return Self.redundantPairLoss(without) - Self.redundantPairLoss(full)
            }
        }
        let ablated = KVHeadMask.classify(scores: ablation, ratio: 0.5).retrieval
        let greedy = KVHeadClassifier.greedy(heads: [2, 2], keep: 2, loss: Self.redundantPairLoss)
        #expect(ablated == [[], [0, 1]])
        #expect(greedy == [[0], [1]])
        #expect(Self.redundantPairLoss(greedy) < Self.redundantPairLoss(ablated))
    }

    @Test("Keeping every head needs no evaluation, keeping none returns empty layers")
    func greedyBounds() {
        var calls = 0
        #expect(KVHeadClassifier.greedy(heads: [2, 3], keep: 5) { _ in calls += 1; return 0 } == [[0, 1], [0, 1, 2]])
        #expect(calls == 0)
        #expect(KVHeadClassifier.greedy(heads: [2, 3], keep: 0) { _ in 0 } == [[], []])
    }

    @Test("The gate passes only when every calibration answer keeps the dense argmax and the mean KL stays within the limit")
    func gate() {
        let mask = KVHeadMask(retrieval: [[0]])
        #expect(KVHeadSelection(mask: mask, meanKL: 0.02, answersKept: true).keepsFidelity)
        #expect(!KVHeadSelection(mask: mask, meanKL: 0.02, answersKept: false).keepsFidelity)
        #expect(!KVHeadSelection(mask: mask, meanKL: KVHeadSelection.maxMeanKL * 2, answersKept: true).keepsFidelity)
        #expect(!KVHeadSelection(mask: mask, meanKL: .nan, answersKept: true).keepsFidelity)
    }

    @Test("Calibration probes ask for every needle, and each answer row predicts an answer digit",
          arguments: [[0.2, 0.5, 0.8], [0.35, 0.65, 0.9]])
    func probeLayout(depths: [Double]) {
        let tokenizer = ByteTokenizer()
        let probe = KVHeadClassifier.probe(tokenizer: tokenizer, length: 1024, depths: depths, seed: 1)
        let context = String(decoding: probe.context.map(UInt8.init), as: UTF8.self)
        let full = probe.tail + [Int(Character(".").asciiValue ?? 0)]
        #expect(probe.context.count >= 1024)
        #expect(context.components(separatedBy: "The pass key for the").count == depths.count + 1)
        #expect(probe.answerRows.allSatisfy { $0 + 1 < full.count })
        let predicted = String(decoding: probe.answerRows.map { UInt8(full[$0 + 1]) }, as: UTF8.self)
        #expect(predicted.filter(\.isNumber).count == depths.count * 5)
    }

    @Test("On tiny dense and hybrid models, keeping every head matches the dense output and passes the gate; half the heads keeps exactly half",
          arguments: [TinyModel.Family.qwen3, .qwen35, .lfm2])
    func selectsOnTinyModels(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let total = try model.newCache(parameters: nil).filter { type(of: $0) == KVCacheSimple.self }.count
        let all = try KVHeadClassifier.select(model: model, tokenizer: ByteTokenizer(), ratio: 1, length: 1024)
        let heads = all.mask.retrieval.map(\.count)
        #expect(heads.count == total)
        #expect(all.meanKL <= 1e-4)
        #expect(all.keepsFidelity)
        let half = try KVHeadClassifier.select(model: model, tokenizer: ByteTokenizer(), ratio: 0.5, length: 1024)
        #expect(half.keptHeads == Int((Double(heads.reduce(0, +)) * 0.5).rounded()))
        #expect(zip(half.mask.retrieval, heads).allSatisfy { kept, count in kept.allSatisfy { (0 ..< count).contains($0) } })
        #expect(half.meanKL.isFinite && half.meanKL >= -1e-4)
    }

    @Test("At load the split applies only when the stored selection passes the gate", arguments: [true, false])
    func loadGate(passes: Bool) async throws {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: FeatureMatrixTests.promptCacheBytes, kvHeadSplit: 0.5))
        await engine.attach(try TinyModel.container(.qwen3HeadDim32))
        let directory = FileManager.default.temporaryDirectory.appending(path: "head-gate-\(UUID().uuidString)")
        let stored = KVHeadClassifier.storedSelection(for: directory, ratio: 0.5)
        defer { try? FileManager.default.removeItem(at: stored) }
        let selection = KVHeadSelection(mask: KVHeadMask(retrieval: Array(repeating: [0], count: 4)), meanKL: passes ? 0 : 1, answersKept: true)
        try FileManager.default.createDirectory(at: stored.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(selection).write(to: stored)
        await engine.probeKVHeads(directory: directory)
        #expect(await engine.kvHeadMask == (passes ? selection.mask : nil))
    }

    @Test("A cached selection whose mask does not fit the loaded model is ignored and chosen again")
    func staleShapeIsReselected() async throws {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: FeatureMatrixTests.promptCacheBytes, kvHeadSplit: 0.5))
        await engine.attach(try TinyModel.container(.qwen3HeadDim32))
        let directory = FileManager.default.temporaryDirectory.appending(path: "head-stale-\(UUID().uuidString)")
        let stored = KVHeadClassifier.storedSelection(for: directory, ratio: 0.5)
        defer { try? FileManager.default.removeItem(at: stored) }
        let stale = KVHeadSelection(mask: KVHeadMask(retrieval: Array(repeating: [7], count: 6)), meanKL: 0, answersKept: true)
        try FileManager.default.createDirectory(at: stored.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(stale).write(to: stored)
        await engine.probeKVHeads(directory: directory)
        #expect(await engine.kvHeadMask != stale.mask)
        let rewritten = try JSONDecoder().decode(KVHeadSelection.self, from: Data(contentsOf: stored))
        #expect(rewritten.fits([2, 2, 2, 2]))
        #expect(!stale.fits([2, 2, 2, 2]))
    }

    @Test("Cached selections are keyed by the model's fingerprint, so changed files choose again")
    func cacheKeyFollowsFingerprint() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "head-key-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"model_type\": \"qwen3\"}".utf8).write(to: directory.appending(path: "config.json"))
        let before = KVHeadClassifier.storedSelection(for: directory, ratio: 0.25)
        try Data("{\"model_type\": \"qwen3_5\"}".utf8).write(to: directory.appending(path: "config.json"))
        #expect(KVHeadClassifier.storedSelection(for: directory, ratio: 0.25) != before)
        #expect(KVHeadClassifier.storedSelection(for: directory, ratio: 0.25) != KVHeadClassifier.storedSelection(for: directory, ratio: 0.5))
    }

    @Test("The planned trial count is exactly the number of loss evaluations the greedy search makes",
          arguments: [([2, 2], 2), ([4, 4], 4), ([64], 16), ([64], 32), ([8, 8, 8], 3), ([4], 4)])
    func plannedTrialsMatchGreedy(heads: [Int], keep: Int) {
        var calls = 0
        _ = KVHeadClassifier.greedy(heads: heads, keep: keep) { trial in
            calls += 1
            return Float(trial.joined().reduce(0, +))
        }
        #expect(calls == KVHeadClassifier.plannedTrials(heads: heads.reduce(0, +), keep: keep))
    }

    @Test("After the first step the search keeps only the best candidates: a head that restores nothing alone is not tried again once the pool is smaller than the heads")
    func greedyPrunesToThePool() {
        var tried: [Int: Int] = [:]
        let picked = KVHeadClassifier.greedy(heads: [8], keep: 2) { trial in
            let added = trial[0].last ?? -1
            tried[added, default: 0] += 1
            return Float(added) - (trial[0].count == 2 ? 0.5 : 0)
        }
        #expect(picked == [[0, 1]])
        #expect(tried[7] == 1)
        #expect(tried[1] == 2)
    }

    @Test("Full-attention models with many KV heads are refused before calibration; the measured hybrids stay within the budget")
    func trialBudget() {
        #expect(KVHeadClassifier.plannedTrials(heads: 64, keep: 16) == 424)
        #expect(KVHeadClassifier.plannedTrials(heads: 64, keep: 32) <= KVHeadClassifier.maxTrials)
        #expect(KVHeadClassifier.plannedTrials(heads: 512, keep: 128) > KVHeadClassifier.maxTrials)
        #expect(KVHeadClassifier.plannedTrials(heads: 512, keep: 512) == 0)
    }

    @Test("A non-finite KL never passes the gate, in any order, and a NaN loss is never picked by the search")
    func nonFiniteKLFails() {
        #expect(KVHeadClassifier.worst([0.01, .nan]) == .greatestFiniteMagnitude)
        #expect(KVHeadClassifier.worst([.nan, 0.01]) == .greatestFiniteMagnitude)
        #expect(KVHeadClassifier.worst([0.01, .infinity]) == .greatestFiniteMagnitude)
        #expect(KVHeadClassifier.worst([0.01, 0.03]) == 0.03)
        let mask = KVHeadMask(retrieval: [[0]])
        #expect(!KVHeadSelection(mask: mask, meanKL: KVHeadClassifier.worst([.nan, 0.01]), answersKept: true).keepsFidelity)
        #expect(!KVHeadSelection(mask: mask, meanKL: .infinity, answersKept: true).keepsFidelity)
        let picked = KVHeadClassifier.greedy(heads: [3], keep: 1) { trial in trial[0] == [0] ? .nan : Float(trial[0][0]) }
        #expect(picked == [[1]])
    }

    @Test("The Qwen3.5-family and LFM2 / Nemotron-H hybrids reach the fidelity gate again instead of a blanket refusal")
    func familiesBehindTheGate() {
        for type in ["qwen3_5", "qwen3_5_moe", "lfm2", "lfm2_moe", "nemotron_h"] {
            #expect(ModelCapabilities.of(modelType: type)?.allows(.kvHeadSplit) == true)
        }
    }
}
