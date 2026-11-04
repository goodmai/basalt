import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Fault injection at the engine's suspension points (#300)", .serialized)
struct FaultInjectionTests {
    static func request() -> GenerationRequest {
        greedy(Conversation.firstTurn, key: "ns")
    }

    static func orchestrator(_ engine: MLXInferenceEngine) -> ModelOrchestratorActor {
        ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
    }

    static func unknownModel() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("fault-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"model_type": "basalt-test-unknown"}"#.utf8).write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    static func expectRecovered(_ engine: MLXInferenceEngine, _ orchestrator: ModelOrchestratorActor) async throws {
        await engine.inject(FaultPlan())
        let fresh = try await Self.orchestrator(try await TinyModel.engine()).generate(request: request())
        let next = try await orchestrator.generate(request: request())
        #expect(next.generatedText == fresh.generatedText)
        #expect(await engine.checkInvariants(idle: true) == [])
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("A fault at a suspension point of a request on a warm prompt cache fails it with a 503, a cancellation as cancelled, and the next request answers like a fresh engine with no lease, pin or gate slot left behind",
          arguments: [FaultPlan.Point.tokenize, .snapshotBoundary, .restoreFromDisk, .anchorOffsets, .launch, .settle, .keep],
          FaultPlan.Fault.allCases)
    func requestFault(point: FaultPlan.Point, fault: FaultPlan.Fault) async throws {
        let engine = try await TinyModel.engine()
        let orchestrator = Self.orchestrator(engine)
        _ = try await orchestrator.generate(request: Self.request())
        await engine.inject(FaultPlan(point: point, fault: fault))
        let failure = await #expect(throws: BasaltError.self) { try await orchestrator.generate(request: Self.request()) }
        let endsStream = FaultPlan(point: point).endsStream
        let reason: String? = if case .inferenceHardwareFailure(let message)? = failure { message } else { nil }
        if fault == .cancellation, !endsStream {
            #expect(failure.map(ModelOrchestratorActor.logStatus) == "cancelled", "\(String(describing: failure))")
        } else {
            #expect(reason != nil, "\(String(describing: failure))")
            #expect((reason == "Stream finished without metadata") == endsStream)
        }
        try await Self.expectRecovered(engine, orchestrator)
    }

    @Test("A fault while unloading the old model or loading the new one fails the load as corrupted weights, a cancellation as cancelled, leaves no model, cache or gate hold behind, and the engine takes the next model",
          arguments: [FaultPlan.Point.unload, .load], FaultPlan.Fault.allCases)
    func loadFault(point: FaultPlan.Point, fault: FaultPlan.Fault) async throws {
        let engine = try await TinyModel.engine()
        let orchestrator = Self.orchestrator(engine)
        _ = try await orchestrator.generate(request: Self.request())
        let directory = try Self.unknownModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        await engine.inject(FaultPlan(point: point, fault: fault))
        let failure = await #expect(throws: BasaltError.self) { try await orchestrator.loadModel(path: directory.path) }
        let corrupted = if case .weightsCorrupted? = failure { true } else { false }
        let cancelled = failure.map(ModelOrchestratorActor.logStatus) == "cancelled"
        #expect(fault == .cancellation ? cancelled : corrupted, "\(String(describing: failure))")
        #expect(await !engine.isLoaded)
        #expect(await engine.readPromptCaches { $0.roots.isEmpty })
        await engine.attach(try TinyModel.container())
        try await Self.expectRecovered(engine, orchestrator)
    }

    @Test("Without a planned fault nothing fires: the same load fails on its unknown model type")
    func noFaultByDefault() async throws {
        let engine = try await TinyModel.engine()
        #expect(await engine.faults.point == nil)
        let directory = try Self.unknownModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        let failure = await #expect(throws: BasaltError.self) { try await engine.load(modelPath: directory.path) }
        let invalid = if case .modelConfigurationInvalid? = failure { true } else { false }
        #expect(invalid, "\(String(describing: failure))")
        #expect(await engine.checkInvariants(idle: true) == [])
    }
}
