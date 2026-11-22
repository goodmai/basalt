import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltInference
@testable import BasaltServer

@Suite("Round 8 audit CLI, quant & server regression tests (#704, #705, #706)", .serialized)
struct Round8AuditRegressionTests {
    @Test("ServeCommand.sessionLogWriter clamps extreme idleMinutes and maxTotalMB without overflow (#704)")
    func sessionLogWriterClampsExtremeFlags() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-r8-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = ServeCommand.sessionLogWriter(
            mode: .meta,
            directory: dir.path,
            retention: SessionLogRetention(days: 14, maxTotalMB: Int.max),
            redacts: true,
            idleMinutes: Int.max
        )
        #expect(writer != nil)
    }

    @Test("ModelsConvertCommand.copyToCacheSnapshot skips identical paths and rejects overlap (#704)")
    func convertCopyToCacheSnapshotOverlapGuard() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-r8-snap-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try ConvertSubcommand.copyToCacheSnapshot(from: dir, to: dir)
        #expect(FileManager.default.fileExists(atPath: dir.path))
        let child = dir.appendingPathComponent("sub")
        #expect(throws: (any Error).self) {
            try ConvertSubcommand.copyToCacheSnapshot(from: dir, to: child)
        }
    }

    @Test("SINQAbsorption.addsUnitOffset recognizes muse_glimmer centered RMSNorm (#705)")
    func sinqAbsorptionMuseGlimmerUnitOffset() {
        #expect(SINQAbsorption.addsUnitOffset(config: ["model_type": "muse_glimmer"]))
        #expect(SINQAbsorption.addsUnitOffset(config: ["text_config": ["model_type": "muse_glimmer_text"]]))
        #expect(!SINQAbsorption.addsUnitOffset(config: ["model_type": "qwen3"]))
    }

    @Test("DeadZoneBitAllocator terminates cleanly on NaN/Inf sensitivity metrics (#705)")
    func deadZoneBitAllocatorNonFiniteSafety() {
        let allocator = DeadZoneBitAllocator(targetBPW: 3.5)
        let items = [
            DeadZoneBitAllocator.ExpertSensitivity(
                id: "e0", weightCount: 1024, routerWeightedTrace: .nan, cascadeRouterKL: .infinity, routingMass: 0.5),
            DeadZoneBitAllocator.ExpertSensitivity(
                id: "e1", weightCount: 1024, routerWeightedTrace: 1.0, cascadeRouterKL: 0.1, routingMass: 0.5)
        ]
        let plan = allocator.allocate(items)
        #expect(plan.allocations.count == 2)
    }

    @Test("OpenAIController.requestedTools ignores primitive parameters without SIGABRT (#706)")
    func requestedToolsPrimitiveParametersSafe() throws {
        let body: [String: Any] = [
            "tools": [
                ["type": "function", "name": "f", "parameters": "not-an-object"]
            ]
        ]
        let result = try OpenAIController.requestedTools(in: body)
        #expect(result.tools?.count == 1)
        #expect(result.tools?.first?.parametersJSON == nil)
    }

    @Test("SystemOneController.request rejects primitive questions without SIGABRT (#706)")
    func systemOnePrimitiveQuestionsSafe() {
        let data = Data(#"{"state":"ok","questions":"bad"}"#.utf8)
        #expect(SystemOneController.request(from: data) == nil)
    }

    @Test("HITLApprovalActor.awaitDecision unblocks on task cancellation (#706)")
    func hitlApprovalAwaitCancellation() async throws {
        let actor = HITLApprovalActor(policy: HITLToolPolicy(defaultTimeoutSeconds: 300))
        _ = try await actor.enqueue(id: "req-cancel", toolName: "intervene", argumentsSummary: "{}", now: Date().timeIntervalSince1970)
        let waiter = Task { await actor.awaitDecision(id: "req-cancel") }
        waiter.cancel()
        let decision = await waiter.value
        #expect(!decision.approved)
    }
}
