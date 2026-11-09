import Foundation
import Testing
import MLX
import MLXLMCommon
import Synchronization
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Preemptible distillation on the server: MCP routing, single flight, model-directory guard, resume hint, divergence and teacher failures (#399 phase 2)", .serialized, .timeLimit(.minutes(1)))
struct PreemptibleTrainingServerTests {
    typealias Fixture = PreemptibleTrainingTests

    static func failure(_ body: @escaping @Sendable () async throws -> DistillRun) async -> String {
        await PreemptibleTrainingGuardTests.failure(body)
    }

    @Test("MCP distill holds no gate ticket for the run: with the only slot taken and no queue it still starts, where a research call is refused; checkpointEvery and resume parse")
    func mcpDistillIsPreemptible() async throws {
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 0)
        let orchestrator = ModelOrchestratorActor(engine: try await TinyModel.engine(.qwen3), telemetry: TelemetryTracker(), gate: gate)
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orchestrator) { capturer.capture($0) }.withResearchTools(true)
        let held = try await gate.admit()
        let arguments: [String: Any] = [
            "prompts": "/nonexistent-\(UUID().uuidString).jsonl", "teacherUrl": "https://teacher.invalid/v1", "teacherModel": "t",
            "heldOut": ["Hello."], "checkpointEvery": 7, "resume": true, "output": "/tmp/x",
        ]
        let call: [String: Any] = ["jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["name": "distill", "arguments": arguments]]
        await server.dispatch(String(decoding: try JSONSerialization.data(withJSONObject: call), as: UTF8.self))
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"overlays_list"}}"#)
        await gate.release(held)
        let lines = capturer.lines
        #expect(lines.count == 2)
        #expect(lines.first?.contains("could not read the prompts") == true, "\(lines.first ?? "")")
        #expect(lines.last?.contains("busy") == true || lines.last?.contains("503") == true, "\(lines.last ?? "")")
        let request = try DistillRequest(mcp: arguments)
        #expect(request.checkpointEvery == 7 && request.resume && request.output == "/tmp/x")
        #expect(DistillRequest(prompts: "", teacher: TeacherSpec(url: "u", model: "m"), gate: GateRequest(heldOut: [])).checkpointEvery == 50)
    }

    @Test("A second preemptible run is refused while one is active, and allowed once it ends; the research route no longer runs distill")
    func singleFlight() async throws {
        let orchestrator = ModelOrchestratorActor(engine: try await TinyModel.engine(.qwen3), telemetry: TelemetryTracker())
        let started = Mutex(false)
        let (release, releasing) = AsyncStream<Void>.makeStream()
        let first = Task {
            try await orchestrator.preemptibly { _, _ in
                started.withLock { $0 = true }
                for await _ in release { break }
                return 1
            }
        }
        while !started.withLock({ $0 }) { try Task.checkCancellation(); await Task.yield() }
        await #expect(throws: BasaltError.self) { try await orchestrator.preemptibly { _, _ in 2 } }
        releasing.yield()
        #expect(try await first.value == 1)
        #expect(try await orchestrator.preemptibly { _, _ in 3 } == 3)
        let request = DistillRequest(prompts: "", teacher: TeacherSpec(url: "u", model: "m"), gate: GateRequest(heldOut: []))
        await #expect(throws: BasaltError.self) { try await ResearchCall.distill(request).run(on: try await TinyModel.engine(.qwen3) as any ModelLab) }
    }

    @Test("The model-directory guard follows symlinks: an output under a link to the model, or the link itself, is inside")
    func symlinkGuard() throws {
        let root = try DistillTests.scratch().resolvingSymlinksInPath()
        let model = root.appending(path: "model")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data("w".utf8).write(to: model.appending(component: "model.safetensors"))
        let link = root.appending(path: "m")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: model)
        #expect(MLXInferenceEngine.isInside(link.appending(path: "adapter"), model))
        #expect(MLXInferenceEngine.isInside(link, model))
        #expect(MLXInferenceEngine.isInside(root.appending(path: "m/../m/adapter/deeper"), model))
        #expect(!MLXInferenceEngine.isInside(root.appending(path: "adapter"), model))
    }

    @Test("A failed run names its checkpoint and how to resume only when one exists, and only for runtime failures")
    func resumeHint() throws {
        let adapter = try DistillTests.scratch().appending(path: "adapter")
        let failure = BasaltError.inferenceError("distillation was cancelled; the model is unchanged")
        #expect(AdapterCheckpoint.withResumeHint(failure, in: adapter).localizedDescription == failure.localizedDescription)
        try FileManager.default.createDirectory(at: AdapterCheckpoint.directory(in: adapter), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: AdapterCheckpoint.directory(in: adapter).appending(component: "progress.json"))
        #expect(AdapterCheckpoint.withResumeHint(failure, in: adapter).localizedDescription.contains("resume with output \(adapter.path) and resume true"))
        let refusal = BasaltError.invalidRequestStructure(details: "distill: the checkpoint does not fit")
        #expect(AdapterCheckpoint.withResumeHint(refusal, in: adapter).localizedDescription == refusal.localizedDescription)
    }

    @Test("A non-finite loss stops the run and discards the checkpoint, so a resume cannot replay the failing batch")
    func nonFiniteLossDiscardsTheCheckpoint() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let output = try DistillTests.scratch().appending(path: "diverged")
        var request = Fixture.request(output: output)
        request.learningRate = 1e35
        let message = await Self.failure { [request] in try await Fixture.run(engine, request) }
        #expect(message.contains("non-finite") && message.contains("discarded"), "\(message)")
        #expect(Fixture.savedStep(output) == nil)
    }

    @Test("A run that fails in the teacher phase leaves no output directory behind, so a rerun with the same output can reserve it")
    func teacherFailureLeavesNoDirectory() async throws {
        let directory = try DistillTests.scratch()
        let host = StubURLProtocol.host { _, _ in (400, [:], Data("bad".utf8)) }
        let engine = try await TinyModel.engine(.qwen3)
        let output = directory.appending(path: "adapter")
        let request = DistillTests.request(host, prompts: try DistillTests.prompts(4, in: directory), output: output)
        let session = URLSession(configuration: StubURLProtocol.configuration)
        await #expect(throws: BasaltError.self) {
            try await engine.distill(request, session: session, home: directory, environment: [TeacherClient.urlVariable: "https://\(host)", TeacherClient.keyVariable: "k"])
        }
        #expect(!FileManager.default.fileExists(atPath: output.path))
        _ = try await engine.reservedDirectory(output.path)
    }
}
