import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import NIOCore
import Testing
@testable import BasaltCore
@testable import BasaltServer

@Suite("HITL MCP Governance (HMD 2026) and LAAS-CNRS Neuro-Symbolic Knowledge Loop (#619)")
struct HITLAndSymbolicKnowledgeTests {
    @Test("HITLStateStore enforces policy validation, mutating tool detection, queue cap, approval, rejection, and expiry")
    func hitlStateStoreLifecycle() throws {
        let policy = HITLToolPolicy(defaultTimeoutSeconds: 60, maxPendingRequests: 2)
        try policy.validate()
        #expect(policy.requiresApproval(toolName: "weights_delta"))
        #expect(policy.requiresApproval(toolName: "switch_model"))
        #expect(!policy.requiresApproval(toolName: "status"))
        #expect(throws: BasaltError.self) { try HITLToolPolicy(defaultTimeoutSeconds: 0).validate() }
        #expect(throws: BasaltError.self) { try HITLToolPolicy(maxPendingRequests: 0).validate() }

        var store = HITLStateStore(policy: policy)
        let r1 = try store.submit(id: "req-1", toolName: "weights_delta", argumentsSummary: "layer 5", now: 100)
        #expect(r1.status == .pending && r1.expiresAt == 160)
        #expect(throws: BasaltError.self) {
            _ = try store.submit(id: "req-1", toolName: "weights_delta", argumentsSummary: "dup", now: 101)
        }
        _ = try store.submit(id: "req-2", toolName: "switch_model", argumentsSummary: "gemma4", now: 102)
        #expect(throws: BasaltError.self) {
            _ = try store.submit(id: "req-3", toolName: "steer_commit", argumentsSummary: "overflow", now: 103)
        }
        #expect(store.pendingRequests().map(\.id) == ["req-1", "req-2"])

        let approved = try store.decide(id: "req-1", approved: true, reviewer: "alice", reason: "verified", now: 110)
        #expect(approved.approved && approved.reviewer == "alice")
        #expect(throws: BasaltError.self) {
            _ = try store.decide(id: "req-1", approved: false, reviewer: "bob", reason: "late", now: 111)
        }

        let timedOut = store.expirePending(now: 200)
        #expect(timedOut.count == 1 && !timedOut[0].approved && timedOut[0].reason == "approval_timeout")
        #expect(store.auditLog.count == 2)
        #expect(store.auditLog[0].status == .approved)
        #expect(store.auditLog[1].status == .timedOut)
    }

    @Test("HITLApprovalActor blocks mutating call until approved, rejected, or timed out via REST /v1/mcp/approvals")
    func hitlActorAndRestEndpoints() async throws {
        let actor = HITLApprovalActor(policy: HITLToolPolicy(defaultTimeoutSeconds: 30, maxPendingRequests: 4))
        #expect(await actor.requiresApproval(toolName: "adapter_apply"))
        _ = try await actor.enqueue(id: "op-1", toolName: "adapter_apply", argumentsSummary: "lora-v1", now: 10)
        _ = try await actor.enqueue(id: "op-2", toolName: "overlay_undo", argumentsSummary: "ov-1", now: 11)
        _ = try await actor.enqueue(id: "op-3", toolName: "steer_commit", argumentsSummary: "s-1", now: 12)

        let waiter1 = Task { await actor.awaitDecision(id: "op-1") }
        let waiter2 = Task { await actor.awaitDecision(id: "op-2") }

        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock"))
            .withHITLApprovals(actor, clock: { 20.0 })
        let app = Application(router: server.makeRouter())

        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/mcp/approvals", method: .get) { res in
                #expect(res.status == .ok)
                let body = String(buffer: res.body)
                #expect(body.contains("op-1") && body.contains("op-2"))
            }
            let approveBody = Data(#"{"reviewer":"secops","reason":"safe"}"#.utf8)
            try await client.execute(
                uri: "/v1/mcp/approvals/op-1/approve",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: approveBody)
            ) { res in
                #expect(res.status == .ok)
                #expect(String(buffer: res.body).contains("\"approved\":true"))
            }
            let emptyJson = ByteBuffer(bytes: Data("{}".utf8))
            try await client.execute(
                uri: "/v1/mcp/approvals/op-2/reject",
                method: .post,
                headers: [.contentType: "application/json"],
                body: emptyJson
            ) { res in
                #expect(res.status == .ok)
                #expect(String(buffer: res.body).contains("\"approved\":false"))
            }
            try await client.execute(
                uri: "/v1/mcp/approvals/unknown/approve",
                method: .post,
                headers: [.contentType: "application/json"],
                body: emptyJson
            ) { res in
                #expect(res.status == .notFound)
            }
        }

        let d1 = await waiter1.value
        let d2 = await waiter2.value
        #expect(d1.approved && d1.reviewer == "secops")
        #expect(!d2.approved && d2.reviewer == "operator")
        let cached = await actor.awaitDecision(id: "op-1")
        #expect(cached == d1)

        let expired = await actor.expire(now: 100)
        #expect(expired.count == 1 && expired[0].requestId == "op-3")
        #expect(await actor.auditLog().count == 3)
    }

    @Test("SymbolicKnowledgeGraph executes 4-step LAAS-CNRS loop, rejects logical conflicts, and caches verified facts")
    func laasCnrsFourStepSymbolicLoop() throws {
        var graph = SymbolicKnowledgeGraph()
        let gap = try #require(graph.detectGap(subject: "Basalt", requiredPredicates: ["has_status", "is_safe"]))
        #expect(gap.missingPredicates == ["has_status", "is_safe"])

        let ragPrompt = graph.buildInternalRAGPrompt(for: gap)
        #expect(ragPrompt.contains("Basalt") && ragPrompt.contains("has_status") && ragPrompt.contains("is_safe"))

        let llmOutput = """
        - Basalt | has_status | verified | 0.95
        (Basalt, is_safe, production, 0.92)
        Basalt | has_owner | uncertain | 0.30
        malformed line without separators
        """
        let parsed = SymbolicKnowledgeGraph.parseTriplets(from: llmOutput)
        #expect(parsed.count == 3)

        let firstCommit = graph.verifyAndCommit(parsed, minConfidence: 0.7)
        #expect(firstCommit.committed.count == 2)
        #expect(firstCommit.skippedLowConfidence == 1)
        #expect(firstCommit.conflicts.isEmpty)

        #expect(graph.detectGap(subject: "Basalt", requiredPredicates: ["has_status", "is_safe"]) == nil)
        #expect(graph.query(subject: "Basalt", predicate: "has_status").first?.object == "verified")

        let conflicting = [
            SemanticTriplet(subject: "Basalt", predicate: "has_status", object: "verified", confidence: 0.95),
            SemanticTriplet(subject: "Basalt", predicate: "has_status", object: "deprecated", confidence: 0.95),
            SemanticTriplet(subject: "Basalt", predicate: "is_hazardous", object: "production", confidence: 0.95),
        ]
        let secondCommit = graph.verifyAndCommit(conflicting, minConfidence: 0.7)
        #expect(secondCommit.committed.isEmpty)
        #expect(secondCommit.skippedDuplicates == 1)
        #expect(secondCommit.conflicts.count == 2)
        #expect(secondCommit.conflicts[0].reason == "functional_predicate_violation")
        #expect(secondCommit.conflicts[1].reason == "disjoint_predicate_violation")
    }
}
