import Foundation
import Testing
@testable import BasaltCore

private struct StubSemanticEmbedder: EmbeddingEngine {
    let table: [String: [Float]]

    var models: [EmbeddingModelInfo] {
        [EmbeddingModelInfo(id: "stub-embed", task: .embedding)]
    }

    func embed(_ inputs: [EmbeddingInput], dimensions _: Int?, yieldingTo _: GenerationGate?) async throws(BasaltError) -> Embeddings {
        let vecs: [[Float]] = inputs.map { input in
            switch input {
            case .text(let s): table[s] ?? [1.0, 0.0, 0.0]
            case .tokens: [1.0, 0.0, 0.0]
            }
        }
        return Embeddings(vectors: vecs, promptTokens: inputs.count * 4)
    }

    func rerank(query _: String, documents: [String], yieldingTo _: GenerationGate?) async throws(BasaltError) -> [Double] {
        Array(repeating: 0.5, count: documents.count)
    }
}

@Suite("5-Component Business Orchestrator (TH Deggendorf 2026) and LANTERN Semantic Cache (#618)")
struct BusinessRouterAndSemanticCacheTests {
    @Test("SemanticCacheStore enforces tenant isolation, cosine threshold, TTL expiry, and LRU eviction")
    func semanticCacheStoreBehavior() throws {
        try SemanticCacheConfig().validate()
        #expect(throws: BasaltError.self) { try SemanticCacheConfig(minSimilarity: 1.5).validate() }
        #expect(throws: BasaltError.self) { try SemanticCacheConfig(maxEntriesPerTenant: 0).validate() }
        #expect(throws: BasaltError.self) { try SemanticCacheConfig(ttlSeconds: 0).validate() }

        var cache = SemanticCacheStore(config: SemanticCacheConfig(minSimilarity: 0.94, maxEntriesPerTenant: 2, ttlSeconds: 100))
        #expect(cache.metrics.hitRate == 0.0)
        cache.store(key: "empty", tenantId: "tA", embedding: [], response: "ignored", now: 5)
        #expect(cache.metrics.entryCount == 0)
        cache.store(key: "k1", tenantId: "tA", embedding: [1.0, 0.0, 0.0], response: "Paris", now: 10)
        cache.store(key: "k2", tenantId: "tA", embedding: [0.0, 1.0, 0.0], response: "Tokyo", now: 12)

        let rawHit = cache.lookup(tenantId: "tA", embedding: [0.99, 0.1, 0.0], now: 20)
        let hit = try #require(rawHit)
        #expect(hit.response == "Paris")
        #expect(hit.similarity >= 0.94)

        let crossTenantMiss = cache.lookup(tenantId: "tB", embedding: [1.0, 0.0, 0.0], now: 20)
        #expect(crossTenantMiss == nil)

        let belowThreshold = cache.lookup(tenantId: "tA", embedding: [0.0, 0.0, 1.0], now: 21)
        #expect(belowThreshold == nil)

        cache.store(key: "k3", tenantId: "tA", embedding: [0.0, 0.0, 1.0], response: "Berlin", now: 25)
        #expect(cache.metrics.evictions == 1)
        #expect(cache.lookup(tenantId: "tA", embedding: [0.0, 1.0, 0.0], now: 26) == nil)
        #expect(cache.lookup(tenantId: "tA", embedding: [0.0, 0.0, 1.0], now: 26)?.response == "Berlin")

        #expect(cache.lookup(tenantId: "tA", embedding: [0.0, 0.0, 1.0], now: 200) == nil)
        cache.invalidate(tenantId: "tA")
        #expect(cache.metrics.entryCount == 0)
        #expect(cache.metrics.hitRate > 0)
        #expect(SemanticCacheStore.cosineSimilarity([], []) == -1.0)
        #expect(SemanticCacheStore.cosineSimilarity([0, 0], [1, 0]) == -1.0)
    }

    @Test("SemanticCacheActor serves LANTERN cache hits via EmbeddingEngine without calling an LLM")
    func semanticCacheActorWithEmbedder() async throws {
        let embedder = StubSemanticEmbedder(table: [
            "capital of france": [1.0, 0.0, 0.0],
            "what is the capital of france": [0.995, 0.08, 0.0],
            "unrelated query": [0.0, 1.0, 0.0],
        ])
        let actor = SemanticCacheActor(config: SemanticCacheConfig(minSimilarity: 0.94), embedder: embedder)
        try await actor.storeText(key: "q1", tenantId: "acme", query: "capital of france", response: "Paris", now: 100)

        let hit = try await actor.lookupText(tenantId: "acme", query: "what is the capital of france", now: 110)
        #expect(hit?.response == "Paris")
        let miss = try await actor.lookupText(tenantId: "acme", query: "unrelated query", now: 111)
        #expect(miss == nil)
        let direct = await actor.lookup(tenantId: "acme", embedding: [1.0, 0.0, 0.0], now: 112)
        #expect(direct?.response == "Paris")
        await actor.storeEntry(key: "q2", tenantId: "acme", embedding: [0.0, 1.0, 0.0], response: "Other", now: 113)
        let metrics = await actor.metrics()
        #expect(metrics.hits == 2 && metrics.misses == 1 && metrics.entryCount == 2)
        var snapshot = ServerTelemetrySnapshot()
        snapshot.semanticCache = metrics
        let exported = TelemetryExport.openMetrics(snapshot)
        #expect(exported.contains("basalt_semantic_cache_hits_total 2"))
        #expect(exported.contains("basalt_semantic_cache_hit_ratio"))
        var compose = ComposeSpec(name: "lantern", models: [:], answer: "fast")
        compose.semanticCache = SemanticCacheConfig(minSimilarity: 0.95)
        #expect(compose.semanticCache?.minSimilarity == 0.95)
        await actor.invalidate(tenantId: "acme")
        #expect(await actor.metrics().entryCount == 0)

        let noEmbedder = SemanticCacheActor()
        try await noEmbedder.storeText(key: "x", tenantId: "t", query: "q", response: "r", now: 1)
        #expect(try await noEmbedder.lookupText(tenantId: "t", query: "q", now: 2) == nil)
    }

    @Test("BusinessOrchestratorCore detects complexity, decomposes DAG waves, routes regulated vs fast tasks, and aggregates outputs")
    func fiveComponentBusinessOrchestration() throws {
        let simple = BusinessOrchestratorCore.detectComplexity(query: "Write a Swift function to reverse a string")
        #expect(!simple.isComplex && simple.estimatedSubtasks == 1)

        let longWords = Array(repeating: "token", count: 30).joined(separator: " ")
        #expect(BusinessOrchestratorCore.detectComplexity(query: longWords).signals.contains("long_prompt"))

        let complexQuery = "Review the patient ICD-10 diagnosis and then write a Python function to format the output summary"
        let assessment = BusinessOrchestratorCore.detectComplexity(query: complexQuery)
        #expect(assessment.isComplex && assessment.estimatedSubtasks == 2)

        let subtasks = BusinessOrchestratorCore.decompose(query: complexQuery)
        #expect(subtasks.count == 2)
        #expect(subtasks[0].domainHint == .medical)
        #expect(subtasks[1].domainHint == .coding)
        #expect(subtasks[1].dependsOn == ["t1"])

        let dag = try BusinessOrchestratorCore.buildDAG(subtasks)
        #expect(dag.waves.count == 2 && dag.totalTasks == 2)

        let models = BusinessTierModels(fastLocalModel: "gemma4-e4b", reasoningModel: "gemma4-12b", symbolicVerifierModel: "laas-kg")
        #expect(models.model(for: .reasoning) == "gemma4-12b")
        let gate = CalibratedRouterGate(config: CalibratedRouterConfig(temperature: 1.2, fastConfidenceThreshold: 0.80, regulatedConfidenceThreshold: 0.92))
        let routed = try BusinessOrchestratorCore.routePlan(
            dag, gate: gate, models: models,
            estimatedFastTTFTSeconds: 0.25
        ) { task in
            RouterDomainClass.allCases.map { $0 == task.domainHint ? 4.0 : 0.0 }
        }
        #expect(routed[0][0].decision.tier == .symbolicVerified)
        #expect(routed[0][0].selectedModel == "laas-kg")
        #expect(routed[1][0].decision.tier == .fastLocal)
        #expect(routed[1][0].selectedModel == "gemma4-e4b")

        let agg = BusinessOrchestratorCore.aggregate([
            SubtaskExecutionOutput(subtaskId: "t1", text: "Verified ICD-10", tier: .symbolicVerified, cacheHit: false, ttftSeconds: 0.9),
            SubtaskExecutionOutput(subtaskId: "t2", text: "def fmt(): pass", tier: .fastLocal, cacheHit: true, ttftSeconds: 0.2),
        ])
        #expect(agg.cacheHits == 1 && agg.tiersUsed.count == 2 && abs(agg.maxTTFTSeconds - 0.9) < 1e-6)
        #expect(agg.combinedText.contains("Verified ICD-10") && agg.combinedText.contains("def fmt()"))
    }

    @Test("BusinessOrchestratorCore rejects unknown dependencies and cyclic task DAGs")
    func dagValidationAndCycleDetection() throws {
        #expect(try BusinessOrchestratorCore.buildDAG([]).totalTasks == 0)
        #expect(throws: BasaltError.self) {
            _ = try BusinessOrchestratorCore.buildDAG([DecomposedSubtask(id: "a", prompt: "x", dependsOn: ["missing"])])
        }
        #expect(throws: BasaltError.self) {
            _ = try BusinessOrchestratorCore.buildDAG([
                DecomposedSubtask(id: "a", prompt: "x", dependsOn: ["b"]),
                DecomposedSubtask(id: "b", prompt: "y", dependsOn: ["a"]),
            ])
        }
        #expect(BusinessOrchestratorCore.decompose(query: "Check CVE vulnerability").first?.domainHint == .security)
        #expect(BusinessOrchestratorCore.decompose(query: "Check statute and law").first?.domainHint == .legal)
        #expect(BusinessOrchestratorCore.decompose(query: "Check ERP invoice").first?.domainHint == .finance)
    }
}
