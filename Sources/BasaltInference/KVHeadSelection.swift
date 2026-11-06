import Foundation
import MLX
import MLXLMCommon

struct KVHeadSelection: Codable, Sendable, Equatable {
    let mask: KVHeadMask
    let meanKL: Float
    let answersKept: Bool

    static let maxMeanKL: Float = 0.1

    var keepsFidelity: Bool { answersKept && meanKL.isFinite && meanKL <= Self.maxMeanKL }

    func fits(_ heads: [Int]) -> Bool {
        mask.retrieval.count == heads.count && zip(mask.retrieval, heads).allSatisfy { kept, count in kept.allSatisfy { (0 ..< count).contains($0) } }
    }
    var keptHeads: Int { mask.retrieval.map(\.count).reduce(0, +) }
}

enum KVHeadClassifier {
    static let filler = [
        "The river ran past the old mill, where the wheel had stopped turning years ago.",
        "Each morning the baker opened the shutters and set the first loaves on the sill.",
        "A committee met on Tuesdays to argue about the budget for the new library roof.",
        "Trains to the coast left every hour, and the last one waited for the ferry.",
        "In the archive, ledgers from three centuries filled the shelves along the north wall.",
        "The orchestra tuned for a long time before the conductor finally raised his baton.",
    ]
    static let places = ["north gate", "east tower", "west bridge", "south harbour"]
    static let calibrationDepths: [[Double]] = [[0.2, 0.5, 0.8], [0.35, 0.65, 0.9]]

    struct Probe {
        let context: [Int]
        let tail: [Int]
        let answerRows: [Int]
    }

    struct Reference {
        let probe: Probe
        let dense: [any KVCache]
        let logp: MLXArray
    }

    static let candidatePool = 2
    static let maxTrials = 2_048

    static func plannedTrials(heads: Int, keep: Int) -> Int {
        guard keep > 0, keep < heads else { return 0 }
        let pool = min(heads, candidatePool * keep)
        return heads + (keep - 1) * pool - keep * (keep - 1) / 2
    }

    static func kept(_ heads: [Int], ratio: Double) -> Int { Int((Double(heads.reduce(0, +)) * ratio).rounded()) }

    static func greedy(heads: [Int], keep: Int, loss: ([[Int]]) throws -> Float) rethrows -> [[Int]] {
        var candidates = heads.enumerated().flatMap { layer, count in (0 ..< count).map { (layer, $0) } }
        guard keep < candidates.count else { return heads.map { Array(0 ..< $0) } }
        let pool = min(candidates.count, candidatePool * keep)
        var chosen = heads.map { _ in [Int]() }
        for step in 0 ..< max(0, keep) {
            let scored = try candidates.map { layer, head in
                var trial = chosen
                trial[layer].append(head)
                let value = try loss(trial)
                return value.isFinite ? value : .infinity
            }
            guard let best = scored.indices.min(by: { scored[$0] < scored[$1] }) else { break }
            chosen[candidates[best].0].append(candidates[best].1)
            if step == 0, pool < candidates.count {
                candidates = scored.indices.filter { $0 != best }.sorted { (scored[$0], $0) < (scored[$1], $1) }
                    .prefix(pool - 1).map { candidates[$0] }
            } else {
                candidates.remove(at: best)
            }
        }
        return chosen.map { $0.sorted() }
    }

    static func worst(_ means: [Float]) -> Float {
        means.isEmpty || means.contains { !$0.isFinite } ? .greatestFiniteMagnitude : means.max() ?? .greatestFiniteMagnitude
    }

    static func attentionHeads(_ caches: [any KVCache]) -> [Int] {
        caches.filter { type(of: $0) == KVCacheSimple.self }.map { $0.state.first?.dim(1) ?? 0 }
    }

    static func attentionHeads(model: any LanguageModel) throws -> [Int] {
        let caches = try model.newCache(parameters: nil)
        eval(model(MLXArray([Int32(0)]).reshaped(1, 1), cache: caches))
        return attentionHeads(caches)
    }
}

extension KVHeadClassifier {
    static func select(model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer, ratio: Double, length: Int = 3072) throws -> KVHeadSelection {
        let references = try calibrationDepths.enumerated().map { seed, depths in
            try reference(model, probe(tokenizer: tokenizer, length: length, depths: depths, seed: seed))
        }
        let heads = attentionHeads(references[0].dense)
        let retrieval = greedy(heads: heads, keep: kept(heads, ratio: ratio)) { divergence(model, references[0], $0).total }
        let gates = references.map { divergence(model, $0, retrieval) }
        return KVHeadSelection(
            mask: KVHeadMask(retrieval: retrieval), meanKL: worst(gates.map(\.mean)),
            answersKept: gates.allSatisfy(\.answersKept))
    }

    static func probe(tokenizer: any MLXLMCommon.Tokenizer, length: Int, depths: [Double], seed: Int) -> Probe {
        let sentences = filler.map { tokenizer.encode(text: $0 + " ", addSpecialTokens: false) }
        var context: [Int] = [], sentence = 0
        while context.count < length {
            context += sentences[sentence % sentences.count]
            sentence += 1
        }
        let facts = depths.enumerated().map { index, depth in
            (place: places[index % places.count], key: 41_000 + (seed * depths.count + index) * 7_919, at: Int(Double(context.count) * depth))
        }
        for fact in facts.sorted(by: { $0.at > $1.at }) {
            let needle = " The pass key for the \(fact.place) is \(fact.key). Remember it: \(fact.key). "
            context.insert(contentsOf: tokenizer.encode(text: needle, addSpecialTokens: false), at: fact.at)
        }
        var tail: [Int] = [], answerRows: [Int] = []
        for fact in facts.reversed() {
            tail += tokenizer.encode(text: " What is the pass key for the \(fact.place)? The pass key for the \(fact.place) is", addSpecialTokens: false)
            let answer = tokenizer.encode(text: " \(fact.key).", addSpecialTokens: false)
            answerRows += (tail.count - 1) ..< (tail.count + answer.count - 1)
            tail += answer
        }
        return Probe(context: context, tail: Array(tail.dropLast()), answerRows: answerRows)
    }

    static func reference(_ model: any LanguageModel, _ probe: Probe) throws -> Reference {
        let dense = try model.newCache(parameters: nil)
        eval(model(MLXArray(probe.context.map(Int32.init)).reshaped(1, -1), cache: dense))
        eval(dense.flatMap { $0.innerState() })
        let logp = logProbabilities(model, probe, dense.map { $0.copy() })
        eval(logp)
        return Reference(probe: probe, dense: dense, logp: logp)
    }

    static func logProbabilities(_ model: any LanguageModel, _ probe: Probe, _ caches: [any KVCache]) -> MLXArray {
        let logits = model(MLXArray(probe.tail.map(Int32.init)).reshaped(1, -1), cache: caches)[0].asType(.float32)
        return logits - logSumExp(logits, axis: -1, keepDims: true)
    }

    static func splitting(_ dense: [any KVCache], retrieval: [[Int]]) -> [any KVCache] {
        var layer = 0
        return dense.map { cache in
            guard type(of: cache) == KVCacheSimple.self, let simple = cache as? KVCacheSimple, layer < retrieval.count else { return cache.copy() }
            defer { layer += 1 }
            return HeadSplitKVCache(splitting: simple, retrieval: retrieval[layer])
        }
    }

    static func divergence(_ model: any LanguageModel, _ reference: Reference, _ retrieval: [[Int]]) -> (total: Float, mean: Float, answersKept: Bool) {
        let logp = logProbabilities(model, reference.probe, splitting(reference.dense, retrieval: retrieval))
        let perToken = (exp(reference.logp) * (reference.logp - logp)).sum(axis: -1)
        let rows = MLXArray(reference.probe.answerRows.map(Int32.init))
        let kept = (argMax(logp[rows], axis: -1) .== argMax(reference.logp[rows], axis: -1)).all()
        eval(perToken, kept)
        return (perToken.sum().item(Float.self), perToken.mean().item(Float.self), kept.item(Bool.self))
    }
}

struct KVHeadSelectionOverBudget: Error, CustomStringConvertible {
    let trials: Int

    var description: String {
        "choosing the heads would take \(trials) forward passes over the calibration tail, above the budget of \(KVHeadClassifier.maxTrials)"
    }
}

extension KVHeadClassifier {
    static func storedSelection(for directory: URL, ratio: Double) -> URL {
        let name = directory.standardizedFileURL.path.replacingOccurrences(of: "/", with: "_")
        let fingerprint = PromptCacheDiskStore.fingerprint(modelDirectory: directory)
        return FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/head-scores/\(name)-\(fingerprint)-fidelity-\(ratio).json")
    }

    static func loadOrSelect(
        directory: URL, container: ModelContainer, ratio: Double, starting: @Sendable (Int) -> Void = { _ in }
    ) async throws -> KVHeadSelection {
        let url = storedSelection(for: directory, ratio: ratio)
        let heads = try await container.perform { try attentionHeads(model: $0.model) }
        if let data = try? Data(contentsOf: url), let stored = try? JSONDecoder().decode(KVHeadSelection.self, from: data), stored.fits(heads) {
            return stored
        }
        let trials = plannedTrials(heads: heads.reduce(0, +), keep: kept(heads, ratio: ratio))
        guard trials <= maxTrials else { throw KVHeadSelectionOverBudget(trials: trials) }
        starting(trials)
        let selection = try await container.perform { try select(model: $0.model, tokenizer: $0.tokenizer, ratio: ratio) }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(selection).write(to: url, options: .atomic)
        return selection
    }
}
