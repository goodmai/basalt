import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Self-rewriting P3 harness: JSONL dataset loader, decontamination, repair damage fixture, MCP selfedit_start/selfedit_status, and CLI (#676, #770)", .serialized)
struct SelfRewritingP3HarnessTests {
    static let textOProj = "model.layers.0.self_attn.o_proj"

    static func writeJSONL(_ lines: [String], name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(path: "selfedit-p3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appending(path: name)
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url)
        return url
    }

    static func sampleDatasetLines() -> [String] {
        [
            #"{"id":"m1","prompt":"Compute the sum of the first ten prime numbers in ascending order.","expected_answer":"129"}"#,
            #"{"id":"m2","prompt":"What is the exact remainder when 2023 factorial is divided by 2024?","expected_answer":"0"}"#,
            #"{"id":"m3","prompt":"Solve for positive x when x squared minus nine equals sixteen.","expected_answer":"5"}"#,
            #"{"id":"m4","prompt":"Translate technical prose into Russian.","expected_answer":"Привет мир","source_text":"Hello world"}"#,
        ]
    }
}

extension SelfRewritingP3HarnessTests {
    @Test("SelfRewriteDatasetLoader loads DeepScaleR/MT JSONL, decontaminates against benchmarks, and enforces reserved split")
    func datasetLoaderAndDecontamination() throws {
        let practiceURL = try Self.writeJSONL(Self.sampleDatasetLines(), name: "practice.jsonl")
        let benchURL = try Self.writeJSONL([
            #"{"problem":"What is the exact remainder when 2023 factorial is divided by 2024?"}"#,
        ], name: "bench.jsonl")
        defer {
            try? FileManager.default.removeItem(at: practiceURL.deletingLastPathComponent())
            try? FileManager.default.removeItem(at: benchURL.deletingLastPathComponent())
        }

        let (split, removed) = try SelfRewriteDatasetLoader.buildSplit(
            practiceURL: practiceURL,
            decontaminateURL: benchURL,
            practiceCount: 2,
            ngramSize: 6,
            seed: 676
        )
        #expect(removed == 1)
        #expect(split.practice.count == 2 && split.reservedCount == 1)
        #expect(!split.practice.contains { $0.id == "m2" })

        #expect(throws: SelfRewriteSessionError.self) {
            _ = try SelfRewriteDatasetLoader.loadProblems(fromJSONL: Data("{\"id\":\"bad\"}\n".utf8))
        }
        #expect(throws: SelfRewriteSessionError.self) {
            _ = try SelfRewriteDatasetLoader.loadProblems(fromJSONL: Data("\n".utf8))
        }
    }

    @Test("SelfRewriteTaskMode parses repair specs and SelfRewriteSessionDriver.applyRepairDamage shifts Gemma 4 MoE logits and undoes bit-exact")
    func repairSpecParseAndDamageOnGemma4Moe() throws {
        #expect(try SelfRewriteTaskMode.parse("improve") == .improve)
        let parsed = try SelfRewriteTaskMode.parse("repair:\(Self.textOProj).weight:0,1,2:10.0")
        guard case .repair(let spec) = parsed else {
            Issue.record("expected repair mode, got \(parsed)")
            return
        }
        #expect(spec.rows == [0, 1, 2] && spec.scale == 10.0 && parsed.label.contains("repair:"))
        #expect(throws: SelfRewriteSessionError.self) {
            _ = try SelfRewriteTaskMode.parse("repair:only_two_parts:1")
        }
        #expect(throws: SelfRewriteSessionError.self) {
            _ = try SelfRewriteTaskMode.parse("repair:w:0,0:2.0")
        }

        let model = try TinyModel.make(.gemma4Moe)
        let origFP = ParameterFingerprint.capture(of: model)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)])[.newAxis]
        let baseOut = model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        eval(baseOut)
        let baseBits = baseOut.asType(.float32).flattened().asArray(Float.self)

        let damageAdapter = try SelfRewriteSessionDriver.applyRepairDamage(spec, to: model)
        let damagedOut = model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        eval(damagedOut)
        let damagedBits = damagedOut.asType(.float32).flattened().asArray(Float.self)
        #expect(damagedBits != baseBits)

        damageAdapter.unload(from: model)
        let restoredOut = model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        eval(restoredOut)
        #expect(restoredOut.asType(.float32).flattened().asArray(Float.self) == baseBits)
        #expect(ParameterFingerprint.capture(of: model).aggregateHash == origFP.aggregateHash)
    }
}

extension SelfRewritingP3HarnessTests {
    @Test("MCP selfedit_start and selfedit_status require --research-tools, enforce experimental & cloud opt-ins, and report persisted transcripts")
    func mcpSelfEditToolsGatingAndStatus() async throws {
        let practiceURL = try Self.writeJSONL(Self.sampleDatasetLines(), name: "mcp_practice.jsonl")
        let sessionDir = FileManager.default.temporaryDirectory.appending(path: "selfedit-mcp-dir-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: practiceURL.deletingLastPathComponent())
            try? FileManager.default.removeItem(at: sessionDir)
        }
        let persisted = sessionDir.appending(path: "prior-session.jsonl")
        let entry = EditSessionTranscriptEntry(turnIndex: 0, role: "editor", content: "describe")
        try Data((String(decoding: try JSONEncoder().encode(entry), as: UTF8.self) + "\n").utf8).write(to: persisted)

        let engine = ScriptedResearchEngine([])
        let lockedCapturer = OutputCapturer()
        let lockedServer = MCPServer(orchestrator: ModelOrchestratorActor(engine: engine)) { lockedCapturer.capture($0) }
        await lockedServer.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"selfedit_status","arguments":{}}}"#)
        #expect(lockedCapturer.lines[0].contains("-32601") && lockedCapturer.lines[0].contains("serve --research-tools"))

        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: engine)) { capturer.capture($0) }
            .withResearchTools(true)
            .withSelfEditDirectory(sessionDir)

        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"selfedit_start","arguments":{"experimentalSelfRewrite":false,"practicePath":"\#(practiceURL.path)"}}}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"selfedit_start","arguments":{"experimentalSelfRewrite":true,"practicePath":"\#(practiceURL.path)","editor":"cloud:gemini/gemini-2.5-pro","allowCloudEditor":false}}}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"selfedit_start","arguments":{"experimentalSelfRewrite":true,"practicePath":"\#(practiceURL.path)","practiceCount":2,"task":"repair:\#(Self.textOProj).weight:0,1:5.0"}}}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"selfedit_status","arguments":{}}}"#)

        let lines = capturer.lines
        #expect(lines.count == 5)
        #expect(lines[0].contains("selfedit_start") && lines[0].contains("selfedit_status"))
        #expect(lines[1].contains("-32602") && lines[1].contains("experimentalSelfRewrite=true"))
        #expect(lines[2].contains("-32602") && lines[2].contains("--allow-cloud-editor"))
        #expect(lines[3].contains("\"result\"") && lines[3].contains("practiceCount") && !lines[3].contains("\"error\""))
        #expect(lines[4].contains("\"result\"") && lines[4].contains("prior-session") && lines[4].contains("selfedit-"))
    }

    @Test("SelfEditCommand.prepareHarness wires --practice, --decontaminate, --task repair, and --transcript cleanly")
    func selfEditCommandPrepareHarness() throws {
        let practiceURL = try Self.writeJSONL(Self.sampleDatasetLines(), name: "cli_practice.jsonl")
        let benchURL = try Self.writeJSONL(["What is the exact remainder when 2023 factorial is divided by 2024?"], name: "cli_bench.txt")
        let transcriptURL = practiceURL.deletingLastPathComponent().appending(path: "out.jsonl")
        defer {
            try? FileManager.default.removeItem(at: practiceURL.deletingLastPathComponent())
            try? FileManager.default.removeItem(at: benchURL.deletingLastPathComponent())
        }

        let cmd = try SelfEditCommand.parse([
            "--experimental-self-rewrite",
            "--model", "mlx-community/gemma-4-26b-a4b-it-qat-4bit",
            "--practice", practiceURL.path,
            "--decontaminate", benchURL.path,
            "--practice-count", "2",
            "--task", "repair:\(Self.textOProj).weight:0,1:200",
            "--transcript", transcriptURL.path,
        ])
        let prepared = try cmd.prepareHarness(sessionId: "sess-cli-test")
        #expect(prepared.split.practice.count == 2 && prepared.split.reservedCount == 1 && prepared.removedDuplicates == 1)
        #expect(prepared.transcriptURL == transcriptURL)
        guard case .repair(let spec) = prepared.taskMode else {
            Issue.record("expected repair task mode")
            return
        }
        #expect(spec.scale == 200 && spec.rows == [0, 1])
    }
}
