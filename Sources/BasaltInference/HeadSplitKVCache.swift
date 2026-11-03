import Foundation
import MLX
import MLXLMCommon

struct KVHeadMask: Codable, Sendable, Equatable {
    let retrieval: [[Int]]

    static let concentration: Float = 0.25

    static func isConcentrated(_ scores: [[Float]]) -> Bool {
        let ranked = scores.joined().sorted(by: >)
        guard let top = ranked.first, top > 0 else { return false }
        return ranked[ranked.count / 2] <= top * concentration
    }

    static func classify(scores: [[Float]], ratio: Double) -> KVHeadMask {
        let ranked = scores.enumerated().flatMap { layer, heads in heads.enumerated().map { (layer, $0.offset, $0.element) } }
            .sorted { $0.2 > $1.2 }
        let kept = Int((Double(ranked.count) * ratio).rounded())
        var retrieval = scores.map { _ in [Int]() }
        for (layer, head, _) in ranked.prefix(kept) { retrieval[layer].append(head) }
        return KVHeadMask(retrieval: retrieval.map { $0.sorted() })
    }

    func replacingFullAttention(_ caches: [any KVCache]) -> [any KVCache] {
        var layer = 0
        return caches.map { cache in
            guard type(of: cache) == KVCacheSimple.self, layer < retrieval.count else { return cache }
            defer { layer += 1 }
            return HeadSplitKVCache(retrieval: retrieval[layer])
        }
    }
}

final class HeadSplitKVCache: KVCacheAttentionProtocol {
    static let sinkTokens = 128
    static let windowTokens = 256
    static let trimStep = 64

    let retrieval: [Int]
    private let full = KVCacheSimple()
    private let stream = KVCacheSimple()
    private(set) var offset = 0
    private(set) var materializedForCaller = false
    private var layout: Layout?

    struct Layout {
        let retrieval: MLXArray?
        let streaming: MLXArray?
        let retrievalQueries: MLXArray?
        let streamingQueries: MLXArray?
        let inverse: MLXArray?
        let kvInverse: MLXArray?
        let kvHeads: Int
    }

    init(retrieval: [Int]) {
        self.retrieval = retrieval
    }

    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }
    var retrievalTokens: Int { full.offset }
    var streamingTokens: Int { stream.offset }

    func innerState() -> [MLXArray] { full.innerState() + stream.innerState() }

    var state: [MLXArray] {
        get { full.state + stream.state }
        set { preconditionFailure("HeadSplitKVCache is never restored from a serialized prompt cache") }
    }

    var metaState: [String] {
        get { [retrieval.map(String.init).joined(separator: ","), String(offset)] }
        set { preconditionFailure("HeadSplitKVCache is never restored from a serialized prompt cache") }
    }

    func layout(kvHeads: Int, queryHeads: Int) -> Layout {
        if let layout, layout.kvHeads == kvHeads { return layout }
        let groups = queryHeads / kvHeads
        let kept = Set(retrieval.filter { $0 < kvHeads })
        let retrievalHeads = (0 ..< kvHeads).filter { kept.contains($0) }
        let streamingHeads = (0 ..< kvHeads).filter { !kept.contains($0) }
        let queryOrder = (retrievalHeads + streamingHeads).flatMap { head in (head * groups) ..< ((head + 1) * groups) }
        var inverse = [Int32](repeating: 0, count: queryOrder.count)
        for (position, head) in queryOrder.enumerated() { inverse[head] = Int32(position) }
        var kvInverse = [Int32](repeating: 0, count: kvHeads)
        for (position, head) in (retrievalHeads + streamingHeads).enumerated() { kvInverse[head] = Int32(position) }
        func indices(_ heads: [Int]) -> MLXArray? { heads.isEmpty ? nil : MLXArray(heads.map(Int32.init)) }
        let made = Layout(
            retrieval: indices(retrievalHeads), streaming: indices(streamingHeads),
            retrievalQueries: indices(retrievalHeads.flatMap { ($0 * groups) ..< (($0 + 1) * groups) }),
            streamingQueries: indices(streamingHeads.flatMap { ($0 * groups) ..< (($0 + 1) * groups) }),
            inverse: retrievalHeads.isEmpty || streamingHeads.isEmpty ? nil : MLXArray(inverse),
            kvInverse: retrievalHeads.isEmpty || streamingHeads.isEmpty ? nil : MLXArray(kvInverse), kvHeads: kvHeads)
        layout = made
        return made
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let heads = layout(kvHeads: keys.dim(1), queryHeads: queries.dim(1))
        offset += keys.dim(2)
        var parts: [MLXArray] = []
        if let index = heads.retrieval, let queryIndex = heads.retrievalQueries {
            let (k, v) = full.update(keys: keys[0..., index], values: values[0..., index])
            parts.append(MLXFast.scaledDotProductAttention(
                queries: queries[0..., queryIndex], keys: k, values: v, scale: scale,
                mask: ZonedKVCache.trailingColumns(of: mask, keys: k.dim(2))))
        }
        if let index = heads.streaming, let queryIndex = heads.streamingQueries {
            let (k, v) = stream.update(keys: keys[0..., index], values: values[0..., index])
            parts.append(MLXFast.scaledDotProductAttention(
                queries: queries[0..., queryIndex], keys: k, values: v, scale: scale,
                mask: queries.dim(2) == 1 ? .none : .causal))
            dropMiddle()
        }
        let output = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
        return heads.inverse.map { output[0..., $0] } ?? output
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        materializedForCaller = true
        let heads = layout(kvHeads: keys.dim(1), queryHeads: keys.dim(1))
        offset += keys.dim(2)
        let retrieved = heads.retrieval.map { full.update(keys: keys[0..., $0], values: values[0..., $0]) }
        let streamed = heads.streaming.map { stream.update(keys: keys[0..., $0], values: values[0..., $0]) }
        guard let retrieved, let streamed else { return retrieved ?? streamed ?? (keys, values) }
        func padded(_ part: MLXArray) -> MLXArray {
            let missing = retrieved.0.dim(2) - part.dim(2)
            guard missing > 0 else { return part }
            var shape = part.shape
            shape[2] = missing
            return concatenated([MLXArray.zeros(shape, dtype: part.dtype), part], axis: 2)
        }
        let kvInverse = heads.kvInverse ?? MLXArray(Array(0 ..< Int32(keys.dim(1))))
        let k = concatenated([retrieved.0, padded(streamed.0)], axis: 1)[0..., kvInverse]
        let v = concatenated([retrieved.1, padded(streamed.1)], axis: 1)[0..., kvInverse]
        return (k, v)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let removed = min(max(0, n), offset)
        _ = full.trim(removed)
        _ = stream.trim(min(removed, max(0, stream.offset - Self.sinkTokens)))
        offset -= removed
        return removed
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || windowSize.map({ n > $0 }) == true {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func copy() -> any KVCache {
        guard let fullCopy = full.copy() as? KVCacheSimple, let streamCopy = stream.copy() as? KVCacheSimple else {
            preconditionFailure("KVCacheSimple copies to its own type")
        }
        let copied = HeadSplitKVCache(retrieval: retrieval)
        copied.adopt(offset: offset, materialized: materializedForCaller, layout: layout, full: fullCopy, stream: streamCopy)
        return copied
    }

    private func adopt(offset: Int, materialized: Bool, layout: Layout?, full: KVCacheSimple, stream: KVCacheSimple) {
        self.offset = offset
        materializedForCaller = materialized
        self.layout = layout
        if !full.state.isEmpty { self.full.state = full.state }
        if !stream.state.isEmpty { self.stream.state = stream.state }
    }

    private func dropMiddle() {
        let count = stream.offset
        guard count > Self.sinkTokens + Self.windowTokens + Self.trimStep else { return }
        stream.state = stream.state.map {
            concatenated([$0[.ellipsis, ..<Self.sinkTokens, 0...], $0[.ellipsis, (count - Self.windowTokens)..., 0...]], axis: 2)
        }
    }
}

final class HeadScoringKVCache: KVCacheAttentionProtocol {
    private let base = KVCacheSimple()
    var target: Range<Int>?
    private(set) var scores: [Float] = []
    private(set) var attended = false

    var offset: Int { base.offset }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }
    func innerState() -> [MLXArray] { base.innerState() }
    var state: [MLXArray] {
        get { base.state }
        set { base.state = newValue }
    }
    var metaState: [String] {
        get { base.metaState }
        set { base.metaState = newValue }
    }
    @discardableResult
    func trim(_ n: Int) -> Int { base.trim(n) }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
    func copy() -> any KVCache { base.copy() }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { base.update(keys: keys, values: values) }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        attended = true
        let (k, v) = base.update(keys: keys, values: values)
        if let target, target.upperBound <= k.dim(2) {
            let kvHeads = k.dim(1)
            let groups = queries.dim(1) / kvHeads
            let grouped = (queries.asType(.float32) * scale)
                .reshaped(queries.dim(0), kvHeads, groups, queries.dim(2), queries.dim(3))
            let logits = matmul(grouped, expandedDimensions(k.asType(.float32), axis: 2).swappedAxes(-1, -2))
            let weights = softmax(logits, axis: -1, precise: true)
            let mass = weights[.ellipsis, target].sum(axis: -1).mean(axes: [0, 2, 3])
            let values = mass.asArray(Float.self)
            scores = scores.isEmpty ? values : zip(scores, values).map { $0 + $1 }
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: k, values: v, scale: scale, mask: ZonedKVCache.trailingColumns(of: mask, keys: k.dim(2)))
    }
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

    static func scores(model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer, length: Int = 3072) throws -> [[Float]] {
        var total: [[Float]] = []
        for (index, depth) in [0.2, 0.5, 0.8].enumerated() {
            let key = 41_000 + index * 7_919
            var text = ""
            var sentence = 0
            while tokenizer.encode(text: text, addSpecialTokens: false).count < length {
                text += filler[sentence % filler.count] + " "
                sentence += 1
            }
            let body = tokenizer.encode(text: text, addSpecialTokens: false)
            let needle = tokenizer.encode(text: " The pass key is \(key). Remember it: \(key). ", addSpecialTokens: false)
            let question = tokenizer.encode(text: " What is the pass key? The pass key is", addSpecialTokens: false)
            let at = Int(Double(body.count) * depth)
            let prompt = Array(body[..<at]) + needle + Array(body[at...]) + question
            let caches = try model.newCache(parameters: nil).map { cache -> any KVCache in
                type(of: cache) == KVCacheSimple.self ? HeadScoringKVCache() : cache
            }
            let scoring = caches.compactMap { $0 as? HeadScoringKVCache }
            let head = prompt.count - question.count
            _ = model(MLXArray(prompt[..<head].map(Int32.init)).reshaped(1, -1), cache: caches)
            eval(caches.flatMap { $0.innerState() })
            for cache in scoring { cache.target = at ..< (at + needle.count) }
            eval(model(MLXArray(prompt[head...].map(Int32.init)).reshaped(1, -1), cache: caches))
            let layers = scoring.map(\.scores)
            total = total.isEmpty ? layers : zip(total, layers).map { zip($0, $1).map { $0 + $1 } }
        }
        return total
    }
}

extension KVHeadClassifier {
    static func storedScores(for directory: URL) -> URL {
        let name = directory.standardizedFileURL.path.replacingOccurrences(of: "/", with: "_")
        return FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/head-scores/\(name).json")
    }

    static func loadOrScore(directory: URL, container: ModelContainer) async throws -> [[Float]] {
        let url = storedScores(for: directory)
        if let data = try? Data(contentsOf: url), let scores = try? JSONDecoder().decode([[Float]].self, from: data) { return scores }
        let scores = try await container.perform { try scores(model: $0.model, tokenizer: $0.tokenizer) }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(scores).write(to: url, options: .atomic)
        return scores
    }
}
