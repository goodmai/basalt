import Foundation

public struct SemanticCacheConfig: Sendable, Codable, Equatable {
    public var minSimilarity: Float
    public var maxEntriesPerTenant: Int
    public var ttlSeconds: Double

    public init(
        minSimilarity: Float = 0.94,
        maxEntriesPerTenant: Int = 512,
        ttlSeconds: Double = 3600
    ) {
        self.minSimilarity = minSimilarity
        self.maxEntriesPerTenant = maxEntriesPerTenant
        self.ttlSeconds = ttlSeconds
    }

    public func validate() throws(BasaltError) {
        guard minSimilarity.isFinite, (-1.0 ... 1.0).contains(minSimilarity) else {
            throw .invalidRequestStructure(details: "semantic cache: minSimilarity must lie in -1...1")
        }
        guard maxEntriesPerTenant >= 1 else {
            throw .invalidRequestStructure(details: "semantic cache: maxEntriesPerTenant must be >= 1")
        }
        guard ttlSeconds.isFinite, ttlSeconds > 0 else {
            throw .invalidRequestStructure(details: "semantic cache: ttlSeconds must be positive")
        }
    }

    enum CodingKeys: String, CodingKey {
        case minSimilarity = "min_similarity"
        case maxEntriesPerTenant = "max_entries_per_tenant"
        case ttlSeconds = "ttl_seconds"
    }
}

public struct SemanticCacheEntry: Sendable, Codable, Equatable {
    public var key: String
    public var tenantId: String
    public var embedding: [Float]
    public var response: String
    public var createdAt: Double
    public var lastAccessedAt: Double

    public init(
        key: String,
        tenantId: String,
        embedding: [Float],
        response: String,
        createdAt: Double,
        lastAccessedAt: Double
    ) {
        self.key = key
        self.tenantId = tenantId
        self.embedding = embedding
        self.response = response
        self.createdAt = createdAt
        self.lastAccessedAt = lastAccessedAt
    }
}

public struct SemanticCacheMetrics: Sendable, Codable, Equatable {
    public var hits: Int
    public var misses: Int
    public var evictions: Int
    public var entryCount: Int

    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0.0
    }

    public init(hits: Int = 0, misses: Int = 0, evictions: Int = 0, entryCount: Int = 0) {
        self.hits = hits
        self.misses = misses
        self.evictions = evictions
        self.entryCount = entryCount
    }
}

public struct SemanticCacheHit: Sendable, Equatable {
    public var key: String
    public var response: String
    public var similarity: Float

    public init(key: String, response: String, similarity: Float) {
        self.key = key
        self.response = response
        self.similarity = similarity
    }
}

public struct SemanticCacheStore: Sendable, Equatable {
    public var config: SemanticCacheConfig
    private var entriesByTenant: [String: [SemanticCacheEntry]]
    public private(set) var hits: Int
    public private(set) var misses: Int
    public private(set) var evictions: Int

    public init(config: SemanticCacheConfig = .init()) {
        self.config = config
        self.entriesByTenant = [:]
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    }

    public var metrics: SemanticCacheMetrics {
        let count = entriesByTenant.values.reduce(0) { $0 + $1.count }
        return SemanticCacheMetrics(hits: hits, misses: misses, evictions: evictions, entryCount: count)
    }

    public static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        validCosineSimilarity(lhs, rhs) ?? -1.0
    }

    private static func validCosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float? {
        guard !lhs.isEmpty, lhs.count == rhs.count else { return nil }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for idx in 0 ..< lhs.count {
            let a = lhs[idx]
            let b = rhs[idx]
            dot += a * b
            normA += a * a
            normB += b * b
        }
        let denom = Foundation::sqrt(normA) * Foundation::sqrt(normB)
        guard denom.isFinite, denom > 1e-8, dot.isFinite else { return nil }
        return min(1.0, max(-1.0, dot / denom))
    }

    public mutating func lookup(tenantId: String, embedding: [Float], now: Double) -> SemanticCacheHit? {
        pruneExpired(tenantId: tenantId, now: now)
        guard var list = entriesByTenant[tenantId], !list.isEmpty else {
            misses += 1
            return nil
        }
        var bestIdx: Int?
        var bestSim: Float = config.minSimilarity
        for idx in list.indices {
            guard let sim = Self.validCosineSimilarity(list[idx].embedding, embedding) else { continue }
            if sim >= bestSim {
                bestSim = sim
                bestIdx = idx
            }
        }
        guard let idx = bestIdx else {
            misses += 1
            return nil
        }
        list[idx].lastAccessedAt = now
        let chosen = list[idx]
        entriesByTenant[tenantId] = list
        hits += 1
        return SemanticCacheHit(key: chosen.key, response: chosen.response, similarity: bestSim)
    }

    public mutating func store(
        key: String,
        tenantId: String,
        embedding: [Float],
        response: String,
        now: Double
    ) {
        guard !embedding.isEmpty else { return }
        pruneExpired(tenantId: tenantId, now: now)
        var list = entriesByTenant[tenantId] ?? []
        list.removeAll { $0.key == key }
        if list.count >= config.maxEntriesPerTenant {
            if let lruIdx = list.indices.min(by: { list[$0].lastAccessedAt < list[$1].lastAccessedAt }) {
                list.remove(at: lruIdx)
                evictions += 1
            }
        }
        list.append(SemanticCacheEntry(
            key: key, tenantId: tenantId, embedding: embedding, response: response, createdAt: now, lastAccessedAt: now
        ))
        entriesByTenant[tenantId] = list
    }

    public mutating func invalidate(tenantId: String) {
        if let removed = entriesByTenant.removeValue(forKey: tenantId) {
            evictions += removed.count
        }
    }

    private mutating func pruneExpired(tenantId: String, now: Double) {
        guard var list = entriesByTenant[tenantId] else { return }
        let before = list.count
        list.removeAll { now - $0.createdAt > config.ttlSeconds }
        evictions += before - list.count
        entriesByTenant[tenantId] = list.isEmpty ? nil : list
    }
}

public actor SemanticCacheActor {
    private var store: SemanticCacheStore
    private let embedder: (any EmbeddingEngine)?

    public init(config: SemanticCacheConfig = .init(), embedder: (any EmbeddingEngine)? = nil) {
        self.store = SemanticCacheStore(config: config)
        self.embedder = embedder
    }

    public func metrics() -> SemanticCacheMetrics {
        store.metrics
    }

    public func lookup(tenantId: String, embedding: [Float], now: Double) -> SemanticCacheHit? {
        store.lookup(tenantId: tenantId, embedding: embedding, now: now)
    }

    public func storeEntry(key: String, tenantId: String, embedding: [Float], response: String, now: Double) {
        store.store(key: key, tenantId: tenantId, embedding: embedding, response: response, now: now)
    }

    public func lookupText(tenantId: String, query: String, now: Double) async throws(BasaltError) -> SemanticCacheHit? {
        guard let embedder else { return nil }
        let emb = try await embedder.embed([.text(query)], dimensions: nil, yieldingTo: nil)
        guard let vec = emb.vectors.first else { return nil }
        return store.lookup(tenantId: tenantId, embedding: vec, now: now)
    }

    public func storeText(key: String, tenantId: String, query: String, response: String, now: Double) async throws(BasaltError) {
        guard let embedder else { return }
        let emb = try await embedder.embed([.text(query)], dimensions: nil, yieldingTo: nil)
        guard let vec = emb.vectors.first else { return }
        store.store(key: key, tenantId: tenantId, embedding: vec, response: response, now: now)
    }

    public func invalidate(tenantId: String) {
        store.invalidate(tenantId: tenantId)
    }
}
