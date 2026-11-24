import Foundation

public struct ExpertRoutingProfile: Sendable, Codable, Equatable {
    public static let currentVersion = 1

    public struct RankedExpert: Sendable, Codable, Equatable {
        public var expert: Int
        public var count: Int
        public var decayedScore: Double

        public init(expert: Int, count: Int, decayedScore: Double = 0) {
            self.expert = expert
            self.count = count
            self.decayedScore = decayedScore
        }

        enum CodingKeys: String, CodingKey {
            case expert
            case count
            case decayedScore = "decayed_score"
        }
    }

    public struct LayerProfile: Sendable, Codable, Equatable {
        public var layer: Int
        public var rankedExperts: [RankedExpert]

        public init(layer: Int, rankedExperts: [RankedExpert]) {
            self.layer = layer
            self.rankedExperts = rankedExperts
        }

        enum CodingKeys: String, CodingKey {
            case layer
            case rankedExperts = "ranked_experts"
        }
    }

    public var version: Int
    public var modelId: String
    public var totalTokens: Int
    public var layers: [LayerProfile]

    public init(
        version: Int = Self.currentVersion,
        modelId: String,
        totalTokens: Int,
        layers: [LayerProfile]
    ) {
        self.version = version
        self.modelId = modelId
        self.totalTokens = max(0, totalTokens)
        self.layers = layers
    }

    enum CodingKeys: String, CodingKey {
        case version
        case modelId = "model_id"
        case totalTokens = "total_tokens"
        case layers
    }

    public static func from(
        routing: ExpertRouting,
        modelId: String,
        totalTokens: Int = 0
    ) -> ExpertRoutingProfile {
        let built = routing.counts.keys.sorted().compactMap { layer -> LayerProfile? in
            guard let table = routing.counts[layer] else { return nil }
            let ranked = table
                .sorted { ($0.value, -$0.key) > ($1.value, -$1.key) }
                .map { RankedExpert(expert: $0.key, count: $0.value, decayedScore: Double($0.value)) }
            return LayerProfile(layer: layer, rankedExperts: ranked)
        }
        return ExpertRoutingProfile(modelId: modelId, totalTokens: totalTokens, layers: built)
    }

    public func rankedExpertIDs(forLayer layer: Int, limit: Int? = nil) -> [Int] {
        guard let found = layers.first(where: { $0.layer == layer }) else { return [] }
        let ids = found.rankedExperts.map(\.expert)
        guard let limit else { return ids }
        return Array(ids.prefix(max(0, limit)))
    }

    public func encodeJSON() throws(BasaltError) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        do {
            return try encoder.encode(self)
        } catch {
            throw .invalidRequestStructure(details: "expert routing profile encode failed: \(error.localizedDescription)")
        }
    }

    public static func decodeJSON(_ data: Data) throws(BasaltError) -> ExpertRoutingProfile {
        let profile: ExpertRoutingProfile
        do {
            profile = try JSONDecoder().decode(ExpertRoutingProfile.self, from: data)
        } catch {
            throw .invalidRequestStructure(details: "expert routing profile decode failed: \(error.localizedDescription)")
        }
        try profile.validate()
        return profile
    }

    private func validate() throws(BasaltError) {
        guard version > 0, version <= Self.currentVersion else {
            throw .invalidRequestStructure(details: "expert routing profile version \(version) is not supported")
        }
        guard totalTokens >= 0 else {
            throw .invalidRequestStructure(details: "expert routing profile total_tokens must be non-negative")
        }
        var seenLayers: Set<Int> = []
        for layer in layers {
            guard layer.layer >= 0, seenLayers.insert(layer.layer).inserted else {
                throw .invalidRequestStructure(details: "expert routing profile layer \(layer.layer) is invalid or duplicated")
            }
            for entry in layer.rankedExperts {
                guard entry.expert >= 0, entry.count >= 0, entry.decayedScore.isFinite, entry.decayedScore >= 0 else {
                    throw .invalidRequestStructure(details: "expert routing profile has invalid entry for layer \(layer.layer)")
                }
            }
        }
    }
}

public struct ExpertTraceReplayResult: Sendable, Equatable, Codable {
    public var hits: Int
    public var misses: Int
    public var bypassed: Int
    public var admissions: Int
    public var evictions: Int

    public init(hits: Int = 0, misses: Int = 0, bypassed: Int = 0, admissions: Int = 0, evictions: Int = 0) {
        self.hits = hits
        self.misses = misses
        self.bypassed = bypassed
        self.admissions = admissions
        self.evictions = evictions
    }

    public var totalLookups: Int { hits + misses }
    public var hitRate: Double { totalLookups > 0 ? Double(hits) / Double(totalLookups) : 0 }
}

public struct ExpertResidencyPolicy: Sendable, Equatable {
    public struct Config: Sendable, Equatable, Codable {
        public static let maxTrackedScore = 1e9

        public var decayFactor: Double
        public var decayWindow: Int
        public var admitThreshold: Double
        public var victimThreshold: Double

        public init(
            decayFactor: Double = 0.5,
            decayWindow: Int = 64,
            admitThreshold: Double = 2.0,
            victimThreshold: Double = 0.5
        ) {
            self.decayFactor = decayFactor.isFinite ? min(1.0, max(0.01, decayFactor)) : 0.5
            self.decayWindow = max(1, decayWindow)
            self.admitThreshold = admitThreshold.isFinite ? max(0.0, admitThreshold) : 2.0
            self.victimThreshold = victimThreshold.isFinite ? max(0.0, victimThreshold) : 0.5
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let factor = try container.decodeIfPresent(Double.self, forKey: .decayFactor) ?? 0.5
            let window = try container.decodeIfPresent(Int.self, forKey: .decayWindow) ?? 64
            let admit = try container.decodeIfPresent(Double.self, forKey: .admitThreshold) ?? 2.0
            let victim = try container.decodeIfPresent(Double.self, forKey: .victimThreshold) ?? 0.5
            self.init(decayFactor: factor, decayWindow: window, admitThreshold: admit, victimThreshold: victim)
        }
    }

    public struct Decision: Sendable, Equatable {
        public var loads: [ExpertSlotTable.Load]
        public var bypassed: [Int]
        public var hits: Int
        public var misses: Int
        public var admissions: Int
        public var evictions: Int

        public init(
            loads: [ExpertSlotTable.Load] = [],
            bypassed: [Int] = [],
            hits: Int = 0,
            misses: Int = 0,
            admissions: Int = 0,
            evictions: Int = 0
        ) {
            self.loads = loads
            self.bypassed = bypassed
            self.hits = hits
            self.misses = misses
            self.admissions = admissions
            self.evictions = evictions
        }
    }

    private struct ScoreEntry: Sendable, Equatable {
        var score: Double
        var epoch: Int
    }

    public let capacity: Int
    public let config: Config
    private var slotOf: [Int: Int] = [:]
    private var expertIn: [Int?]
    private var lastUse: [Int]
    private var scores: [Int: ScoreEntry] = [:]
    private var stepCount = 0
    private var clock = 0

    public init(capacity: Int, config: Config = Config()) {
        self.capacity = max(0, capacity)
        self.config = config
        self.expertIn = Array(repeating: nil, count: self.capacity)
        self.lastUse = Array(repeating: -1, count: self.capacity)
    }

    public var currentEpoch: Int { stepCount > 0 ? (stepCount - 1) / config.decayWindow : 0 }
    public var residentCount: Int { slotOf.count }
    public var residentExperts: [Int] { expertIn.compactMap { $0 } }

    public func slot(of expert: Int) -> Int? { slotOf[expert] }

    public func decayedScore(of expert: Int) -> Double {
        guard let entry = scores[expert] else { return 0 }
        return Self.applyDecay(entry, toEpoch: currentEpoch, factor: config.decayFactor)
    }

    public mutating func step(_ experts: [Int]) -> Decision {
        let unique = Self.deduplicated(experts)
        guard !unique.isEmpty else { return Decision() }
        stepCount += 1
        clock += 1
        let epoch = currentEpoch

        var decision = Decision()
        let pinned = Set(unique)
        var misses: [(expert: Int, score: Double)] = []
        misses.reserveCapacity(unique.count)
        for expert in unique {
            let score = recordHit(expert: expert, epoch: epoch)
            if let slot = slotOf[expert] {
                lastUse[slot] = clock
                decision.hits += 1
            } else {
                decision.misses += 1
                misses.append((expert, score))
            }
        }
        for miss in misses {
            admitOrBypass(
                expert: miss.expert, candidateScore: miss.score, epoch: epoch,
                pinned: pinned, decision: &decision
            )
        }
        return decision
    }

    public mutating func warmStart(from ranked: [Int], initialScore: Double? = nil) -> Decision {
        let target = Array(Self.deduplicated(ranked).prefix(capacity))
        guard !target.isEmpty else { return Decision() }
        let epoch = currentEpoch
        let seedScore = sanitizeScore(initialScore ?? config.admitThreshold)
        let baseClock = clock
        let protected = Set(target)
        var decision = Decision()

        for (rank, expert) in target.enumerated() {
            let orderClock = baseClock + target.count - rank
            let existing = scores[expert].map { Self.applyDecay($0, toEpoch: epoch, factor: config.decayFactor) } ?? 0
            scores[expert] = ScoreEntry(score: min(Config.maxTrackedScore, max(existing, seedScore)), epoch: epoch)
            if let slot = slotOf[expert] {
                lastUse[slot] = orderClock
            }
        }
        for (rank, expert) in target.enumerated() where slotOf[expert] == nil {
            guard let slot = freeOrColdestLRUSlot(excluding: protected) else { continue }
            if let evicted = expertIn[slot] {
                slotOf[evicted] = nil
                decision.evictions += 1
            }
            expertIn[slot] = expert
            slotOf[expert] = slot
            lastUse[slot] = baseClock + target.count - rank
            decision.loads.append(ExpertSlotTable.Load(expert: expert, slot: slot))
            decision.admissions += 1
        }
        clock = baseClock + target.count
        return decision
    }

    public mutating func warmStart(from profile: ExpertRoutingProfile, layer: Int) -> Decision {
        guard let found = profile.layers.first(where: { $0.layer == layer }) else { return Decision() }
        let ids = found.rankedExperts.map(\.expert)
        return warmStart(from: ids)
    }
}

extension ExpertResidencyPolicy {
    public static func replay(
        trace: [[Int]],
        capacity: Int,
        config: Config = Config(),
        warmStart ranked: [Int] = []
    ) -> ExpertTraceReplayResult {
        var policy = ExpertResidencyPolicy(capacity: capacity, config: config)
        if !ranked.isEmpty {
            _ = policy.warmStart(from: ranked)
        }
        var result = ExpertTraceReplayResult()
        for stepExperts in trace {
            let decision = policy.step(stepExperts)
            result.hits += decision.hits
            result.misses += decision.misses
            result.bypassed += decision.bypassed.count
            result.admissions += decision.admissions
            result.evictions += decision.evictions
        }
        return result
    }

    public static func replayLRU(
        trace: [[Int]],
        capacity: Int,
        warmStart ranked: [Int] = []
    ) -> ExpertTraceReplayResult {
        var table = ExpertSlotTable(slots: capacity)
        if !ranked.isEmpty {
            _ = table.warm(ranked)
        }
        var result = ExpertTraceReplayResult()
        for stepExperts in trace {
            let unique = Self.deduplicated(stepExperts)
            guard !unique.isEmpty else { continue }
            if let placement = table.place(unique) {
                result.hits += unique.count - placement.misses
                result.misses += placement.misses
                result.admissions += placement.loads.count
                result.evictions += placement.evictions
            } else {
                result.misses += unique.count
                result.bypassed += unique.count
            }
        }
        return result
    }
}

extension ExpertResidencyPolicy {
    private mutating func recordHit(expert: Int, epoch: Int) -> Double {
        let prior = scores[expert].map { Self.applyDecay($0, toEpoch: epoch, factor: config.decayFactor) } ?? 0
        let updated = min(Config.maxTrackedScore, prior + 1.0)
        scores[expert] = ScoreEntry(score: updated, epoch: epoch)
        return updated
    }

    private mutating func admitOrBypass(
        expert: Int,
        candidateScore: Double,
        epoch: Int,
        pinned: Set<Int>,
        decision: inout Decision
    ) {
        guard capacity > 0 else {
            decision.bypassed.append(expert)
            return
        }
        if let emptySlot = firstFreeSlot() {
            install(expert: expert, into: emptySlot, decision: &decision)
            return
        }
        guard candidateScore >= config.admitThreshold else {
            decision.bypassed.append(expert)
            return
        }
        guard let victim = selectVictim(candidateScore: candidateScore, epoch: epoch, pinned: pinned) else {
            decision.bypassed.append(expert)
            return
        }
        install(expert: expert, into: victim, decision: &decision)
    }

    private mutating func install(expert: Int, into slot: Int, decision: inout Decision) {
        if let old = expertIn[slot] {
            slotOf[old] = nil
            decision.evictions += 1
        }
        expertIn[slot] = expert
        slotOf[expert] = slot
        lastUse[slot] = clock
        decision.loads.append(ExpertSlotTable.Load(expert: expert, slot: slot))
        decision.admissions += 1
    }

    private func firstFreeSlot() -> Int? {
        for slot in 0 ..< capacity where expertIn[slot] == nil {
            return slot
        }
        return nil
    }

    private func selectVictim(candidateScore: Double, epoch: Int, pinned: Set<Int>) -> Int? {
        var coldVictim: (score: Double, lastUse: Int, slot: Int)?
        var fallbackLRU: (lastUse: Int, score: Double, slot: Int)?

        for slot in 0 ..< capacity {
            guard let resident = expertIn[slot], !pinned.contains(resident) else { continue }
            let score = scores[resident].map { Self.applyDecay($0, toEpoch: epoch, factor: config.decayFactor) } ?? 0
            let used = lastUse[slot]
            if score < config.victimThreshold {
                let candidate = (score, used, slot)
                if coldVictim.map({ candidate < $0 }) ?? true {
                    coldVictim = candidate
                }
            } else if score <= candidateScore {
                let candidate = (used, score, slot)
                if fallbackLRU.map({ candidate < $0 }) ?? true {
                    fallbackLRU = candidate
                }
            }
        }
        return coldVictim?.slot ?? fallbackLRU?.slot
    }

    private func freeOrColdestLRUSlot(excluding pinned: Set<Int>) -> Int? {
        if let empty = firstFreeSlot() { return empty }
        var best: (slot: Int, lastUse: Int)?
        for slot in 0 ..< capacity {
            if let resident = expertIn[slot], pinned.contains(resident) { continue }
            if best == nil || lastUse[slot] < best!.lastUse {
                best = (slot, lastUse[slot])
            }
        }
        return best?.slot
    }

    private static func deduplicated(_ experts: [Int]) -> [Int] {
        var seen: Set<Int> = []
        var unique: [Int] = []
        unique.reserveCapacity(experts.count)
        for expert in experts where seen.insert(expert).inserted {
            unique.append(expert)
        }
        return unique
    }

    private func sanitizeScore(_ value: Double) -> Double {
        guard value.isFinite, value > 0 else { return config.admitThreshold }
        return min(Config.maxTrackedScore, value)
    }

    private static func applyDecay(_ entry: ScoreEntry, toEpoch epoch: Int, factor: Double) -> Double {
        guard entry.score.isFinite, entry.score > 0 else { return 0 }
        let delta = epoch - entry.epoch
        guard delta > 0 else { return min(Config.maxTrackedScore, entry.score) }
        if factor >= 1.0 { return min(Config.maxTrackedScore, entry.score) }
        let decayed = entry.score * Foundation.pow(factor, Double(delta))
        guard decayed.isFinite, decayed >= 1e-12 else { return 0 }
        return min(Config.maxTrackedScore, decayed)
    }
}
