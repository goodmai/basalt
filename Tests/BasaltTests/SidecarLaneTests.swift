import Testing
import Foundation
import Synchronization
@testable import BasaltCore

struct GateYieldingEmbeddingEngine: EmbeddingEngine {
    let models = [EmbeddingModelInfo(id: "org/sidecar-embed", task: .embedding)]
    let calls = EmbeddingCalls()

    func embed(_ inputs: [EmbeddingInput], dimensions: Int?, yieldingTo gate: GenerationGate?) async throws(BasaltError) -> Embeddings {
        if let gate {
            calls.events.withLock { $0.append("waiting") }
            while await !gate.idle(within: .seconds(60)) {
                guard !Task.isCancelled else { throw .inferenceError("cancelled") }
            }
        }
        calls.events.withLock { $0.append("embedded") }
        return Embeddings(vectors: inputs.map { _ in [1] }, promptTokens: inputs.count)
    }

    func rerank(query: String, documents: [String], yieldingTo gate: GenerationGate?) async throws(BasaltError) -> [Double] {
        calls.events.withLock { $0.append(gate == nil ? "rerank ungated" : "rerank gated") }
        return documents.map { _ in 0 }
    }
}

@Suite("Sidecars by rule: the compute plan's placement decides who runs beside decode and who waits for the generation gate (#399 phase 3)", .timeLimit(.minutes(1)))
struct SidecarLaneTests {
    static let chat = Workload(name: "gemma-4-12b", role: "chat", unit: .gpu, kind: .decode, residentBytes: 0)
    static let battery = PowerState(onAC: false, charging: false, batteryPercent: 40, adapterWatts: nil)

    static func plan(_ unit: ComputeUnit, _ power: PowerState) -> PlanVerdict {
        let sidecars = ["embed", "decide"].map { Workload(name: $0, role: $0, unit: unit, kind: .encoder, residentBytes: 0) }
        return ComputePlan.admit([chat] + sidecars, budgetBytes: 0, power: power)
    }

    @Test("Only an ANE encoder on adequate power runs beside decode; GPU encoders, a weak power source, decoders and absent roles do not")
    func besideDecodeByRole() {
        #expect(Self.plan(.ane, .desktop).runsBesideDecode("embed") && Self.plan(.ane, .desktop).runsBesideDecode("decide"))
        #expect(!Self.plan(.gpu, .desktop).runsBesideDecode("embed") && !Self.plan(.gpu, .desktop).runsBesideDecode("decide"))
        #expect(!Self.plan(.ane, Self.battery).runsBesideDecode("embed") && !Self.plan(.ane, Self.battery).runsBesideDecode("decide"))
        #expect(!Self.plan(.ane, .desktop).runsBesideDecode("chat") && !Self.plan(.ane, .desktop).runsBesideDecode("rerank"))
    }

    @Test("An engine beside decode is wrapped once and drops the gate only for embeddings; rerank keeps it, and a gated or absent engine is served as is")
    func embeddingWrapper() async throws {
        let engine = GateYieldingEmbeddingEngine()
        #expect(BesideDecodeEmbeddingEngine.serving(nil, besideDecode: true) == nil)
        #expect(BesideDecodeEmbeddingEngine.serving(engine, besideDecode: false) is GateYieldingEmbeddingEngine)
        let wrapped = try #require(BesideDecodeEmbeddingEngine.serving(engine, besideDecode: true) as? BesideDecodeEmbeddingEngine)
        #expect((BesideDecodeEmbeddingEngine.serving(wrapped, besideDecode: true) as? BesideDecodeEmbeddingEngine)?.inner is GateYieldingEmbeddingEngine)
        #expect(wrapped.models == engine.models)
        _ = try await wrapped.rerank(query: "q", documents: ["d"], yieldingTo: GenerationGate())
        #expect(engine.calls.events.withLock { $0 } == ["rerank gated"])
    }

    @Test("While a generation holds the gate, an ANE sidecar answers at once and a GPU sidecar, or any sidecar on a weak power source, waits for the release",
          arguments: [(ComputeUnit.ane, false, false), (.gpu, false, true), (.ane, true, true), (.gpu, true, true)])
    func waitsByPlacement(unit: ComputeUnit, onBattery: Bool, blocked: Bool) async throws {
        let plan = Self.plan(unit, onBattery ? Self.battery : .desktop)
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 4)
        let embedder = GateYieldingEmbeddingEngine()
        let decider = StubDecisionEngine()
        let embeddings = try #require(BesideDecodeEmbeddingEngine.serving(embedder, besideDecode: plan.runsBesideDecode("embed")))
        let inner: any DecisionEngine = unit == .ane ? BesideDecodeDecisionEngine() : decider
        let decisions = try #require(GatedDecisionEngine.serving(inner, gate: gate, besideDecode: plan.runsBesideDecode("decide")))
        let generation = try await gate.admit()
        let embedding = Task { try await embeddings.embed([.text("chunk")], dimensions: nil, yieldingTo: gate) }
        let decision = Task { try await decisions.decide(GatedDecisionEngineTests.request) }
        if blocked {
            while await gate.exclusiveWaiting < 1 || embedder.calls.events.withLock({ $0 }) != ["waiting"] {
                try Task.checkCancellation()
                await Task.yield()
            }
            #expect(decider.calls.requests.withLock { $0.isEmpty })
            await gate.release(generation)
            _ = try await embedding.value
            _ = try await decision.value
        } else {
            _ = try await embedding.value
            _ = try await decision.value
            #expect(await gate.inFlight == 1)
            await gate.release(generation)
        }
        #expect(embedder.calls.events.withLock { $0 } == (blocked ? ["waiting", "embedded"] : ["embedded"]))
        #expect(await gate.checkInvariants(idle: true).isEmpty)
    }
}
