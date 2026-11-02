import Foundation
import MLX
import MLXLMCommon

final class BatchedDecoder: @unchecked Sendable {
    let slots: Int
    private let model: any LanguageModel
    private let signal = NSCondition()
    private var seated = 0
    private var joining: [BatchRow] = []
    private var looping = false
    private var admitted = 0
    private var rows: [BatchRow] = []
    private var layers: [BatchKVCache] = []
    private var pending: MLXArray?
    private var steps = 0

    init(model: any LanguageModel, slots: Int) {
        self.model = model
        self.slots = slots
    }

    var admittedRows: Int { signal.withLock { admitted } }

    func seat() -> BatchSeat? {
        signal.withLock {
            guard seated < slots else { return nil }
            seated += 1
            return BatchSeat(decoder: self)
        }
    }

    fileprivate func join(_ row: BatchRow) {
        let starts = signal.withLock {
            joining.append(row)
            admitted += 1
            defer { looping = true }
            return !looping
        }
        if starts { Thread { self.loop() }.start() }
    }

    fileprivate func leave(_ row: BatchRow?) {
        signal.withLock {
            seated -= 1
            guard let row else { return }
            row.abandoned = true
            joining.removeAll { $0 === row }
        }
    }

    fileprivate func take(from row: BatchRow) -> Int? {
        signal.lock()
        defer { signal.unlock() }
        while row.delivered.isEmpty, !row.finished { signal.wait() }
        return row.delivered.isEmpty ? nil : row.delivered.removeFirst()
    }

    private func loop() {
        while let arrivals = nextRound() {
            autoreleasepool { step(admitting: arrivals) }
        }
    }

    private func nextRound() -> [BatchRow]? {
        signal.withLock {
            let kept = rows.indices.filter { !rows[$0].finished && !rows[$0].abandoned }
            if kept.count < rows.count { retain(kept) }
            let arrivals = joining
            joining = []
            guard !rows.isEmpty || !arrivals.isEmpty else {
                looping = false
                return nil
            }
            return arrivals
        }
    }

    private func retain(_ kept: [Int]) {
        rows = kept.map { rows[$0] }
        guard !kept.isEmpty else {
            layers = []
            pending = nil
            return
        }
        for layer in layers { layer.filter(rows: kept) }
        pending = pending?[MLXArray(kept.map(Int32.init))]
    }

    private func step(admitting arrivals: [BatchRow]) {
        if !arrivals.isEmpty {
            if layers.isEmpty { layers = arrivals[0].layers.map { _ in BatchKVCache() } }
            for (index, layer) in layers.enumerated() { layer.append(arrivals.map { $0.layers[index] }) }
            for row in arrivals { row.layers = [] }
            pending = concatenated((pending.map { [$0] } ?? []) + arrivals.map(\.first))
            rows += arrivals
        }
        guard let feed = pending else { return }
        let logits = model(feed.reshaped(rows.count, 1), cache: layers)[0..., -1, 0...]
        let next = concatenated(rows.indices.map { rows[$0].sample(logits[$0 ..< $0 + 1]) })
        asyncEval([next] + layers.flatMap(\.state))
        let tokens = feed.asArray(Int.self)
        pending = next
        steps += 1
        if steps % 256 == 0 { Memory.clearCache() }
        signal.withLock {
            for (row, token) in zip(rows, tokens) { row.deliver(token) }
            signal.broadcast()
        }
    }
}

final class BatchSeat {
    let decoder: BatchedDecoder
    fileprivate var row: BatchRow?

    fileprivate init(decoder: BatchedDecoder) {
        self.decoder = decoder
    }

    deinit { decoder.leave(row) }
}

private final class BatchRow {
    let first: MLXArray
    var layers: [KVCacheSimple]
    var processor: (any LogitProcessor)?
    let sampler: any LogitSampler
    let maxTokens: Int?
    let stopTokens: Set<Int>
    var produced = 0
    var delivered: [Int] = []
    var finished = false
    var abandoned = false

    init(
        first: MLXArray, layers: [KVCacheSimple], processor: (any LogitProcessor)?, sampler: any LogitSampler,
        maxTokens: Int?, stopTokens: Set<Int>
    ) {
        self.first = first
        self.layers = layers
        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens
        self.stopTokens = stopTokens
    }

    func sample(_ logits: MLXArray) -> MLXArray {
        let token = sampler.sample(logits: processor?.process(logits: logits) ?? logits)
        processor?.didSample(token: token)
        return token
    }

    func deliver(_ token: Int) {
        guard !finished else { return }
        delivered.append(token)
        produced += 1
        finished = stopTokens.contains(token) || maxTokens.map { produced >= $0 } == true
    }
}

struct BatchedTokenIterator: TokenIteratorProtocol {
    private let seat: BatchSeat
    private let row: BatchRow
    private var joined = false
    private(set) var tokenCount = 0
    let maxTokens: Int?
    let promptPrefillTime: TimeInterval

    static func plainLayers(_ cache: [KVCache]) -> [KVCacheSimple]? {
        let layers = cache.compactMap { type(of: $0) == KVCacheSimple.self ? $0 as? KVCacheSimple : nil }
        return layers.isEmpty || layers.count < cache.count ? nil : layers
    }

    init(
        input: LMInput, model: any LanguageModel, layers: [KVCacheSimple], processor: (any LogitProcessor)?,
        sampler: any LogitSampler, prefill: PrefillParameters, maxTokens: Int?, stopTokens: Set<Int>, seat: BatchSeat
    ) throws {
        let started = ContinuousClock.now
        var processor = processor
        processor?.prompt(input.text.tokens)
        let logits = switch try model.prepare(input, cache: layers, state: nil, prefill: prefill) {
        case .tokens(let rest): model(rest[text: .newAxis], cache: layers, state: nil).logits
        case .logits(let output): output.logits
        }
        let last = logits[0..., -1, 0...]
        let first = sampler.sample(logits: processor?.process(logits: last) ?? last)
        processor?.didSample(token: first)
        asyncEval(first)
        row = BatchRow(
            first: first, layers: layers, processor: processor, sampler: sampler, maxTokens: maxTokens, stopTokens: stopTokens)
        seat.row = row
        self.seat = seat
        self.maxTokens = maxTokens
        promptPrefillTime = (ContinuousClock.now - started) / .seconds(1)
    }

    mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        if !joined {
            seat.decoder.join(row)
            joined = true
        }
        guard let token = seat.decoder.take(from: row) else { return nil }
        tokenCount += 1
        return token
    }
}
