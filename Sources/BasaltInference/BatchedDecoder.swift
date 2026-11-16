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

    func checkInvariants() -> [String] {
        mailbox.withLock { $0.checkInvariants(slots: slots) }
    }

    func seat(in context: ModelContext, kvBits: Int? = nil) -> BatchSeat? {
        (context.batch as? BatchCore).flatMap { seat(on: $0, kvBits: kvBits) }
    }

    func seat(on core: BatchCore, kvBits: Int? = nil) -> BatchSeat? {
        guard core.decoder === self else { return nil }
        return mailbox.withLock { $0.seat(within: slots, kvBits: kvBits) }.map { BatchSeat(core: core, id: $0) }
    }

    fileprivate func leave(_ id: Int, from core: BatchCore) {
        guard mailbox.withLock({ $0.leave(id) }) else { return }
        core.model.evaluating {
            if mailbox.withLock({ $0.seated == 0 }) { core.reset() }
        }
    }

    fileprivate func take(_ id: Int, from core: BatchCore, joining arrival: BatchRow?) -> Int? {
        if arrival != nil { mailbox.withLock { $0.open(id) } }
        if mailbox.withLock({ $0.waits(id) }) {
            core.model.evaluating {
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
    var activeKVBits: Int??
    var rows: [Int: Mail] = [:]

    mutating func seat(within slots: Int, kvBits: Int?) -> Int? {
        guard seated < slots else { return nil }
        if seated == 0 {
            activeKVBits = .some(kvBits)
        } else if activeKVBits != .some(kvBits) {
            return nil
        }
        seated += 1
        issued += 1
        return issued
    }

    mutating func leave(_ id: Int) -> Bool {
        seated -= 1
        rows[id] = nil
        if seated == 0 { activeKVBits = nil }
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

    func checkInvariants(slots: Int) -> [String] {
        var problems: [String] = []
        if !(0...slots).contains(seated) { problems.append("\(seated) rows seated in \(slots) slots") }
        if rows.count > seated { problems.append("\(rows.count) rows queued for \(seated) seats") }
        if !rows.keys.allSatisfy({ $0 >= 1 && $0 <= issued }) { problems.append("a queued row has no issued seat") }
        if admitted > issued { problems.append("\(admitted) rows admitted from \(issued) seats") }
        return problems
    }
}

private typealias BatchDelivery = (id: Int, token: Int, finished: Bool)

final class BatchCore {
    let decoder: BatchedDecoder
    fileprivate let model: any LanguageModel
    private var rows: [BatchRow] = []
    private var typedLayers: [BatchedLayer] = []
    private var layers: [KVCache] = []
    private var pending: MLXArray?
    private var steps = 0

    init(model: any LanguageModel, slots: Int) {
        self.model = model
        decoder = BatchedDecoder(slots: slots)
    }

    fileprivate var ids: [Int] { rows.map(\.id) }

    fileprivate func reset() {
        rows = []
        typedLayers = []
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
        asyncEval([next] + typedLayers.flatMap { $0.innerState() })
        let tokens = feed.asArray(Int.self)
        pending = next
        steps += 1
        if steps % 256 == 0 { Memory.clearCache() }
        return zip(rows, tokens).map { ($0.id, $1, $0.deliver($1)) }
    }

    private func retain(_ kept: [Int]) {
        rows = kept.map { rows[$0] }
        guard !kept.isEmpty else {
            typedLayers = []
            layers = []
            pending = nil
            return
        }
        let index = MLXArray(kept.map(Int32.init))
        for layer in typedLayers {
            layer.filter(rows: kept, batchIndices: index)
        }
        pending = pending?[index]
    }

    private func admit(_ arrivals: [BatchRow]) {
        if typedLayers.isEmpty {
            typedLayers = arrivals[0].layers.map(BatchedLayer.make)
            layers = typedLayers.map(\.raw)
        }
        for (index, layer) in typedLayers.enumerated() { layer.join(arrivals.map { $0.layers[index] }) }
        for row in arrivals { row.layers = [] }
        pending = concatenated((pending.map { [$0] } ?? []) + arrivals.map(\.first))
        rows += arrivals
    }

    static func batchLayer(_ row: KVCache) -> KVCache {
        BatchedLayer.make(row).raw
    }

    static func join(_ joining: [KVCache], into layer: KVCache) {
        switch layer {
        case let recurrent as ArraysCache:
            for case let row as ArraysCache in joining where row !== recurrent { recurrent.extend(other: row) }
        case let rotating as BatchRotatingKVCache: rotating.append(joining.compactMap { $0 as? RotatingKVCache })
        case let quantized as BatchQuantizedKVCache: quantized.append(joining)
        case let batch as BatchKVCache: batch.append(joining.compactMap { $0 as? KVCacheSimple })
        default: break
        }
    }
}

private enum BatchedLayer {
    case kv(BatchKVCache)
    case quantized(BatchQuantizedKVCache)
    case rotating(BatchRotatingKVCache)
    case recurrent(ArraysCache)

    static func make(_ row: KVCache) -> BatchedLayer {
        switch row {
        case let recurrent as ArraysCache: .recurrent(recurrent)
        case let rotating as RotatingKVCache: .rotating(BatchRotatingKVCache(window: rotating.maxSize ?? 0))
        case let quantized as QuantizedKVCache:
            .quantized(BatchQuantizedKVCache(groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode))
        default: .kv(BatchKVCache())
        }
    }

    var raw: any KVCache {
        switch self {
        case .kv(let cache): cache
        case .quantized(let cache): cache
        case .rotating(let cache): cache
        case .recurrent(let cache): cache
        }
    }

    func innerState() -> [MLXArray] {
        switch self {
        case .kv(let cache): cache.innerState()
        case .quantized(let cache): cache.innerState()
        case .rotating(let cache): cache.innerState()
        case .recurrent(let cache): cache.innerState()
        }
    }

    func filter(rows kept: [Int], batchIndices: MLXArray) {
        switch self {
        case .kv(let cache): cache.filter(rows: kept)
        case .quantized(let cache): cache.filter(rows: kept)
        case .rotating(let cache): cache.filter(rows: kept)
        case .recurrent(let cache): cache.filter(batchIndices: batchIndices)
        }
    }

    func join(_ joining: [KVCache]) {
        switch self {
        case .kv(let cache): cache.append(joining.compactMap { $0 as? KVCacheSimple })
        case .quantized(let cache): cache.append(joining)
        case .rotating(let cache): cache.append(joining.compactMap { $0 as? RotatingKVCache })
        case .recurrent(let cache):
            for case let row as ArraysCache in joining where row !== cache { cache.extend(other: row) }
        }
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
    var kvBits: Int? = nil
    var kvGroupSize: Int = 64
}

extension DecodingSettings {
    init(processor: (any LogitProcessor)?, parameters: GenerateParameters) {
        self.init(
            processor: processor, sampler: parameters.sampler(), prefill: parameters.prefill,
            maxTokens: parameters.maxTokens, kvBits: parameters.kvBits, kvGroupSize: parameters.kvGroupSize)
    }
}

struct BatchedTokenIterator: SelfSerializingTokenIterator {
    private let seat: BatchSeat
    private var arrival: BatchRow?
    private(set) var tokenCount = 0
    let maxTokens: Int?
    let promptPrefillTime: TimeInterval

    static func batchableLayers(_ cache: [KVCache]) -> [KVCache]? {
        let hasAttention = cache.contains { type(of: $0) == KVCacheSimple.self || type(of: $0) == QuantizedKVCache.self }
        let batchable = !cache.isEmpty && hasAttention
            && cache.allSatisfy {
                type(of: $0) == KVCacheSimple.self || type(of: $0) == QuantizedKVCache.self
                    || $0 is ArraysCache || BatchRotatingKVCache.admits($0)
            }
        return batchable ? cache : nil
    }

    init(
        input: LMInput, model: any LanguageModel, layers: [KVCache], decoding: DecodingSettings, stopTokens: Set<Int>,
        seat: BatchSeat
    ) throws {
        let started = ContinuousClock.now
        var decoding = decoding
        var layers = layers
        decoding.processor?.prompt(input.text.tokens)
        let first = try seat.core.model.evaluating {
            let logits = switch try model.prepare(input, cache: layers, state: nil, prefill: decoding.prefill) {
            case .tokens(let rest): model(rest[text: .newAxis], cache: layers, state: nil).logits
            case .logits(let output): output.logits
            }
            let last = logits[0..., -1, 0...]
            let first = decoding.sampler.sample(logits: decoding.processor?.process(logits: last) ?? last)
            if let kvBits = decoding.kvBits {
                maybeQuantizeKVCache(cache: &layers, kvBits: kvBits, kvGroupSize: decoding.kvGroupSize, quantizedKVStart: 0)
            }
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
