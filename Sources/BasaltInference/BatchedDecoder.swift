import Foundation
import MLX
import MLXLMCommon
import Synchronization

final class BatchedDecoder: Sendable {
    let slots: Int
    private let mailbox = Mutex(Mailbox())

    init(slots: Int) {
        self.slots = slots
    }

    var admittedRows: Int { mailbox.withLock { $0.admitted } }

    func seat(in context: ModelContext) -> BatchSeat? {
        (context.batch as? BatchCore).flatMap { seat(on: $0) }
    }

    func seat(on core: BatchCore) -> BatchSeat? {
        guard core.decoder === self else { return nil }
        return mailbox.withLock { $0.seat(within: slots) }.map { BatchSeat(core: core, id: $0) }
    }

    fileprivate func leave(_ id: Int, from core: BatchCore) {
        guard mailbox.withLock({ $0.leave(id) }) else { return }
        core.evaluation.withLock {
            if mailbox.withLock({ $0.seated == 0 }) { core.reset() }
        }
    }

    fileprivate func take(_ id: Int, from core: BatchCore, joining arrival: BatchRow?) -> Int? {
        if arrival != nil { mailbox.withLock { $0.open(id) } }
        if mailbox.withLock({ $0.waits(id) }) {
            core.evaluation.withLock {
                var arrivals = arrival.map { [$0] } ?? []
                while mailbox.withLock({ $0.waits(id) }), advance(core, admitting: arrivals) { arrivals = [] }
            }
        }
        return mailbox.withLock { $0.take(id) }
    }

    private func advance(_ core: BatchCore, admitting arrivals: [BatchRow]) -> Bool {
        let ids = core.ids
        let seated = mailbox.withLock { box in ids.map { box.rows[$0] != nil } }
        let delivered = core.step(keeping: seated, admitting: arrivals)
        mailbox.withLock { box in for row in delivered { box.deliver(row) } }
        return !delivered.isEmpty
    }
}

private struct Mailbox {
    struct Mail {
        var tokens: [Int] = []
        var finished = false
    }

    var seated = 0
    var admitted = 0
    var issued = 0
    var rows: [Int: Mail] = [:]

    mutating func seat(within slots: Int) -> Int? {
        guard seated < slots else { return nil }
        seated += 1
        issued += 1
        return issued
    }

    mutating func leave(_ id: Int) -> Bool {
        seated -= 1
        rows[id] = nil
        return seated == 0
    }

    mutating func open(_ id: Int) {
        rows[id] = Mail()
        admitted += 1
    }

    func waits(_ id: Int) -> Bool {
        rows[id].map { $0.tokens.isEmpty && !$0.finished } ?? false
    }

    mutating func take(_ id: Int) -> Int? {
        guard rows[id]?.tokens.isEmpty == false else { return nil }
        return rows[id]?.tokens.removeFirst()
    }

    mutating func deliver(_ row: BatchDelivery) {
        rows[row.id]?.tokens.append(row.token)
        if row.finished { rows[row.id]?.finished = true }
    }
}

private typealias BatchDelivery = (id: Int, token: Int, finished: Bool)

final class BatchCore {
    let decoder: BatchedDecoder
    fileprivate let evaluation: FairRecursiveLock
    private let model: any LanguageModel
    private var rows: [BatchRow] = []
    private var layers: [KVCache] = []
    private var pending: MLXArray?
    private var steps = 0

    init(model: any LanguageModel, slots: Int) {
        self.model = model
        decoder = BatchedDecoder(slots: slots)
        evaluation = ModelEvaluationLock.lock(for: model)
    }

    fileprivate var ids: [Int] { rows.map(\.id) }

    fileprivate func reset() {
        rows = []
        layers = []
        pending = nil
    }

    fileprivate func step(keeping seated: [Bool], admitting arrivals: [BatchRow]) -> [BatchDelivery] {
        let kept = rows.indices.filter { seated[$0] && !rows[$0].finished }
        if kept.count < rows.count { retain(kept) }
        if !arrivals.isEmpty { admit(arrivals) }
        guard let feed = pending else { return [] }
        let logits = model(feed.reshaped(rows.count, 1), cache: layers)[0..., -1, 0...]
        let next = rows.allSatisfy(\.isPlainGreedy)
            ? argMax(logits, axis: -1)
            : concatenated(rows.indices.map { rows[$0].sample(logits[$0 ..< $0 + 1]) })
        asyncEval([next] + layers.flatMap { $0.innerState() })
        let tokens = feed.asArray(Int.self)
        pending = next
        steps += 1
        if steps % 256 == 0 { Memory.clearCache() }
        return zip(rows, tokens).map { ($0.id, $1, $0.deliver($1)) }
    }

    private func retain(_ kept: [Int]) {
        rows = kept.map { rows[$0] }
        guard !kept.isEmpty else {
            layers = []
            pending = nil
            return
        }
        let index = MLXArray(kept.map(Int32.init))
        for layer in layers {
            (layer as? BatchKVCache)?.filter(rows: kept)
            (layer as? ArraysCache)?.filter(batchIndices: index)
        }
        pending = pending?[MLXArray(kept.map(Int32.init))]
    }

    private func admit(_ arrivals: [BatchRow]) {
        if layers.isEmpty { layers = arrivals[0].layers.map { $0 is ArraysCache ? $0 : BatchKVCache() } }
        for (index, layer) in layers.enumerated() {
            let joining = arrivals.map { $0.layers[index] }
            if let recurrent = layer as? ArraysCache {
                for case let row as ArraysCache in joining where row !== recurrent { recurrent.extend(other: row) }
            } else {
                (layer as? BatchKVCache)?.append(joining.compactMap { $0 as? KVCacheSimple })
            }
        }
        for row in arrivals { row.layers = [] }
        pending = concatenated((pending.map { [$0] } ?? []) + arrivals.map(\.first))
        rows += arrivals
    }
}

final class BatchSeat {
    fileprivate let core: BatchCore
    fileprivate let id: Int

    fileprivate init(core: BatchCore, id: Int) {
        self.core = core
        self.id = id
    }

    deinit { core.decoder.leave(id, from: core) }
}

private final class BatchRow {
    let id: Int
    let first: MLXArray
    var layers: [KVCache]
    var processor: (any LogitProcessor)?
    let sampler: any LogitSampler
    let maxTokens: Int?
    let stopTokens: Set<Int>
    var produced = 0
    var finished = false

    init(id: Int, first: MLXArray, layers: [KVCache], decoding: DecodingSettings, stopTokens: Set<Int>) {
        self.id = id
        self.first = first
        self.layers = layers
        processor = decoding.processor
        sampler = decoding.sampler
        maxTokens = decoding.maxTokens
        self.stopTokens = stopTokens
    }

    var isPlainGreedy: Bool { processor == nil && sampler is ArgMaxSampler }

    func sample(_ logits: MLXArray) -> MLXArray {
        let token = sampler.sample(logits: processor?.process(logits: logits) ?? logits)
        processor?.didSample(token: token)
        return token
    }

    func deliver(_ token: Int) -> Bool {
        produced += 1
        finished = stopTokens.contains(token) || maxTokens.map { produced >= $0 } == true
        return finished
    }
}

struct DecodingSettings {
    var processor: (any LogitProcessor)?
    let sampler: any LogitSampler
    let prefill: PrefillParameters
    let maxTokens: Int?
}

extension DecodingSettings {
    init(processor: (any LogitProcessor)?, parameters: GenerateParameters) {
        self.init(processor: processor, sampler: parameters.sampler(), prefill: parameters.prefill, maxTokens: parameters.maxTokens)
    }
}

struct BatchedTokenIterator: SelfSerializingTokenIterator {
    private let seat: BatchSeat
    private var arrival: BatchRow?
    private(set) var tokenCount = 0
    let maxTokens: Int?
    let promptPrefillTime: TimeInterval

    static func batchableLayers(_ cache: [KVCache]) -> [KVCache]? {
        let batchable = !cache.isEmpty && cache.contains { type(of: $0) == KVCacheSimple.self }
            && cache.allSatisfy { type(of: $0) == KVCacheSimple.self || $0 is ArraysCache }
        return batchable ? cache : nil
    }

    init(
        input: LMInput, model: any LanguageModel, layers: [KVCache], decoding: DecodingSettings, stopTokens: Set<Int>,
        seat: BatchSeat
    ) throws {
        let started = ContinuousClock.now
        var decoding = decoding
        decoding.processor?.prompt(input.text.tokens)
        let first = try seat.core.evaluation.withLock {
            let logits = switch try model.prepare(input, cache: layers, state: nil, prefill: decoding.prefill) {
            case .tokens(let rest): model(rest[text: .newAxis], cache: layers, state: nil).logits
            case .logits(let output): output.logits
            }
            let last = logits[0..., -1, 0...]
            let first = decoding.sampler.sample(logits: decoding.processor?.process(logits: last) ?? last)
            eval([first] + layers.flatMap { $0.innerState() })
            return first
        }
        decoding.processor?.didSample(token: first)
        arrival = BatchRow(id: seat.id, first: first, layers: layers, decoding: decoding, stopTokens: stopTokens)
        self.seat = seat
        self.maxTokens = decoding.maxTokens
        promptPrefillTime = (ContinuousClock.now - started) / .seconds(1)
    }

    mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        guard let token = seat.core.decoder.take(seat.id, from: seat.core, joining: arrival.take()) else { return nil }
        tokenCount += 1
        return token
    }
}
