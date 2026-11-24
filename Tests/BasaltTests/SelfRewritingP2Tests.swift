import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltCore
@testable import BasaltInference

private actor LocalStepActor {
    private let replies: [String]
    private var idx = 0

    init(_ replies: [String]) {
        self.replies = replies
    }

    func next() -> String {
        guard idx < replies.count else { return "{\"rationale\":\"end\",\"action\":{\"withdraw\":{\"rationale\":\"eof\"}}}" }
        defer { idx += 1 }
        return replies[idx]
    }
}

private actor ScriptedTestEditor: EditSessionEditor {
    let editorLabel = "scripted"
    private var turns: [EditEditorTurn]
    private var index = 0

    init(_ turns: [EditEditorTurn]) {
        self.turns = turns
    }

    func nextTurn(
        transcript: [EditSessionTranscriptEntry],
        remainingBudget: SelfRewriteBudget
    ) async throws -> EditEditorTurn {
        guard index < turns.count else {
            return EditEditorTurn(rationale: "done", action: .withdraw(rationale: "script exhausted"))
        }
        defer { index += 1 }
        return turns[index]
    }
}

@Suite("Self-rewriting P2: session reducer, privacy fence, cloud Gemini editor, scripted Gemma 4 driver, and selfedit CLI (#676, #770)", .serialized)
struct SelfRewritingP2Tests {
    static let textOProj = "model.layers.0.self_attn.o_proj"

    static func encode(_ text: String) -> [Int] {
        ByteTokenizer().encode(text: text, addSpecialTokens: false) + [ByteTokenizer.endOfUserTurn]
    }

    static func decode(_ tokens: [Int]) -> String {
        ByteTokenizer().decode(tokenIds: tokens)
    }

    static func makeSplit() -> PairedDatasetSplit {
        let practice = [
            PairedProblem(id: "pr-1", prompt: "Compute 2 + 2.", expectedAnswer: "4"),
            PairedProblem(id: "pr-2", prompt: "Compute 3 + 5.", expectedAnswer: "8"),
        ]
        let reserved = [
            PairedProblem(id: "res-1", prompt: "RESERVED SECRET PROMPT 991234567890", expectedAnswer: "10"),
        ]
        return PairedDatasetSplit(practice: practice, reserved: reserved)
    }

    static func makeSpec() -> PairedEvaluationSpec {
        PairedEvaluationSpec(
            encode: encode,
            decode: decode,
            grader: MathAnswerGrader(),
            config: PairedTestRunConfig(seeds: [7], maxTokens: 3, temperature: 0.0, stopTokenIds: [ByteTokenizer.endOfUserTurn], budgetCost: 1)
        )
    }

    static func tempDir(_ prefix: String) -> URL {
        FileManager.default.temporaryDirectory.appending(path: "\(prefix)-\(UUID().uuidString)")
    }
}

extension SelfRewritingP2Tests {
    @Test("SelfRewriteSessionCore drops stale epoch completions, enforces budget exhaustion, and merges pending edits on submit")
    func sessionCoreReducerEpochAndBudget() throws {
        let budget = try SelfRewriteBudget(maxForwardPasses: 4, maxToolCalls: 3, maxReservedTests: 1, maxEditApplies: 2, maxCloudUSD: 1.0)
        var core = SelfRewriteSessionCore(sessionId: "s-core", baseModelFingerprint: "fp-1", budget: budget)

        let e1 = try core.beginAction(EditEditorTurn(rationale: "t1", action: .describeModel))
        #expect(e1 == 1 && core.epoch == 1)
        let e2 = try core.beginAction(EditEditorTurn(rationale: "t2", action: .applyEdit(EditProgram(ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.5)]))))
        #expect(e2 == 2)

        #expect(try !core.completeEdit(program: EditProgram(ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.0)]), expectedEpoch: e1))
        #expect(core.appliedPrograms.isEmpty)
        #expect(try core.completeEdit(program: EditProgram(ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.5)]), expectedEpoch: e2))
        #expect(core.appliedPrograms.count == 1)

        let e3 = try core.beginAction(EditEditorTurn(rationale: "t3", action: .submit(rationale: "better"), costUSD: 0.25))
        let merged = try #require(try core.completeSubmit(rationale: "better", expectedEpoch: e3))
        #expect(merged.approval == .pending && merged.sessionId == "s-core" && merged.baseModelFingerprint == "fp-1")
        #expect(core.phase.isTerminal)

        #expect(throws: SelfRewriteSessionError.self) {
            _ = try core.beginAction(EditEditorTurn(rationale: "late", action: .describeModel))
        }

        var tightCore = SelfRewriteSessionCore(
            sessionId: "s-tight",
            baseModelFingerprint: "fp-2",
            budget: try SelfRewriteBudget(maxForwardPasses: 2, maxToolCalls: 1)
        )
        _ = try tightCore.beginAction(EditEditorTurn(rationale: "one", action: .describeModel))
        #expect(throws: SelfRewriteBudgetError.self) {
            _ = try tightCore.beginAction(EditEditorTurn(rationale: "two", action: .describeModel))
        }
        guard case .exhausted = tightCore.phase else {
            Issue.record("expected exhausted phase, got \(tightCore.phase)")
            return
        }
    }

    @Test("CloudPrivacyFence redacts secrets and refuses raw float arrays, filesystem paths, and reserved benchmark content")
    func cloudPrivacyFenceEnforcesBoundaries() throws {
        let safe = [EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: "{\"l2Norm\":1.0,\"token\":\"sk-or-v1-1234567890abcdef1234567890abcdef\"}")]
        let cleaned = try CloudPrivacyFence.sanitizeAndValidate(safe, privateStrings: ["RESERVED SECRET PROMPT 991234567890"])
        #expect(!cleaned[0].content.contains("1234567890abcdef"))

        let badPath = [EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: "loaded /Users/alice/model.safetensors")]
        #expect(throws: SelfRewriteSessionError.self) {
            _ = try CloudPrivacyFence.sanitizeAndValidate(badPath)
        }

        let bigArray = "[" + (0 ..< 70).map { String(format: "%.3f", Float($0) * 0.01) }.joined(separator: ",") + "]"
        #expect(throws: SelfRewriteSessionError.self) {
            _ = try CloudPrivacyFence.sanitizeAndValidate([EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: bigArray)])
        }

        let leakedReserved = [EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: "saw RESERVED SECRET PROMPT 991234567890 in output")]
        #expect(throws: SelfRewriteSessionError.self) {
            _ = try CloudPrivacyFence.sanitizeAndValidate(leakedReserved, privateStrings: ["RESERVED SECRET PROMPT 991234567890"])
        }
    }
}

extension SelfRewritingP2Tests {
    @Test("GeminiCloudEditor fails closed without opt-in, enforces USD caps, parses mock HTTP turns, and records cost in CloudGate")
    func geminiCloudEditorMockHTTPAndCaps() async throws {
        #expect(throws: SelfRewriteSessionError.self) {
            _ = try GeminiCloudEditor(config: GeminiCloudEditorConfig(model: "gemini-2.5-pro", apiKey: "k", allowCloudEditor: false))
        }
        #expect(throws: BasaltError.self) {
            _ = try GeminiCloudEditor(config: GeminiCloudEditorConfig(model: "gemini-2.5-pro", apiKey: "", allowCloudEditor: true))
        }

        let turnJSON = "{\"rationale\":\"inspect first\",\"action\":{\"describeModel\":{}}}"
        let chatPayload: [String: Any] = [
            "id": "cmpl-1",
            "model": "gemini-2.5-pro",
            "choices": [["message": ["role": "assistant", "content": "```json\n\(turnJSON)\n```"], "finish_reason": "stop"]],
            "usage": ["prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140, "cost": 0.15],
        ]
        let payloadData = try JSONSerialization.data(withJSONObject: chatPayload)
        let host = StubURLProtocol.host { _, _ in (200, ["Content-Type": "application/json"], payloadData) }
        let baseURL = try #require(URL(string: "https://\(host)/v1beta/openai"))
        let gate = CloudGate(limits: ["gemini-2.5-pro": 1.0], storage: nil)
        let editor = try GeminiCloudEditor(
            config: GeminiCloudEditorConfig(
                model: "gemini-2.5-pro",
                apiKey: "test-key",
                baseURL: baseURL,
                allowCloudEditor: true,
                maxSessionUSD: 0.20
            ),
            cloudGate: gate,
            privateStrings: ["RESERVED SECRET PROMPT 991234567890"],
            configuration: StubURLProtocol.configuration
        )
        let budget = try SelfRewriteBudget()
        let turn = try await editor.nextTurn(transcript: [], remainingBudget: budget)
        #expect(turn.action == .describeModel && abs(turn.costUSD - 0.15) < 1e-6)
        #expect(abs(await editor.spentUSD() - 0.15) < 1e-6)
        #expect(abs(await gate.spent("gemini-2.5-pro") - 0.15) < 1e-6)

        let captured = StubURLProtocol.requests(for: host)
        #expect(captured.count == 1)
        #expect(captured[0].0.value(forHTTPHeaderField: "Authorization") == "Bearer test-key")
        let bodyText = String(decoding: captured[0].1 ?? Data(), as: UTF8.self)
        #expect(!bodyText.contains("RESERVED SECRET PROMPT 991234567890"))

        await #expect(throws: SelfRewriteBudgetError.self) {
            _ = try await editor.nextTurn(transcript: [], remainingBudget: budget)
        }
    }

    @Test("Scripted editor drives SelfRewriteSessionDriver through full toolset and submit on tiny Gemma 4 MoE")
    func scriptedDriverSubmitOnGemma4Moe() async throws {
        let model = try TinyModel.make(.gemma4Moe)
        let origFingerprint = ParameterFingerprint.capture(of: model)
        let configObj = try #require(JSONSerialization.jsonObject(with: Data(TinyModel.Family.gemma4Moe.config.utf8)) as? [String: Any])
        let topology = Gemma4EditTopology(config: configObj)
        let outDir = Self.tempDir("selfedit-submit")
        let transcriptURL = outDir.deletingLastPathComponent().appending(path: "sess-g4-submit-\(UUID().uuidString).jsonl")
        defer {
            try? FileManager.default.removeItem(at: outDir)
            try? FileManager.default.removeItem(at: transcriptURL)
        }

        let submitEditor = ScriptedTestEditor([
            EditEditorTurn(rationale: "describe", action: .describeModel),
            EditEditorTurn(rationale: "inspect", action: .inspectTensor(name: Self.textOProj + ".weight")),
            EditEditorTurn(rationale: "rt", action: .measureRoundTrip(tensor: Self.textOProj + ".weight")),
            EditEditorTurn(rationale: "moe", action: .summarizeMoERouter(text: "Hello MoE", layers: [0])),
            EditEditorTurn(rationale: "svd", action: .registerSVD(key: "u0", tensor: Self.textOProj + ".weight", iterations: 10)),
            EditEditorTurn(rationale: "edit", action: .applyEdit(EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u0", alpha: 0.25)]))),
            EditEditorTurn(rationale: "practice", action: .pairedPractice(problemIds: ["pr-1"], seeds: [7])),
            EditEditorTurn(rationale: "reserved", action: .pairedReserved(seeds: [7])),
            EditEditorTurn(rationale: "submit", action: .submit(rationale: "verified improvement")),
        ])
        let submitResult = try await SelfRewriteSessionDriver.run(
            config: SelfRewriteSessionConfig(
                sessionId: "sess-g4-submit",
                budget: try SelfRewriteBudget(maxForwardPasses: 10, maxToolCalls: 20, maxReservedTests: 1),
                outputDirectory: outDir,
                topology: topology
            ).withTranscriptURL(transcriptURL),
            split: Self.makeSplit(),
            evalSpec: Self.makeSpec(),
            editor: submitEditor,
            original: model
        )
        #expect(submitResult.savedAdapterURL == outDir)
        #expect(submitResult.submittedProgram?.approval == .pending)
        #expect(EditAdapter.isEditAdapterDirectory(outDir))
        let reloaded = try EditAdapter.from(directory: outDir)
        #expect(reloaded.program.sessionId == "sess-g4-submit" && reloaded.program.ops.count == 1)
        let lines = try String(contentsOf: transcriptURL, encoding: .utf8).split(whereSeparator: \.isNewline)
        #expect(lines.count >= 9)
        let perms = try #require(try FileManager.default.attributesOfItem(atPath: transcriptURL.path)[.posixPermissions] as? Int)
        #expect(perms & 0o777 == 0o600)
        try origFingerprint.requireUnchanged(against: model)
    }
}

extension SelfRewritingP2Tests {
    @Test("Pending EditAdapter is refused for serving until explicitly approved via SelfEditCommand --accept-edit")
    func pendingEditAdapterRequiresHumanApprovalBeforeServing() async throws {
        let model = try TinyModel.make(.gemma4Moe)
        let dir = Self.tempDir("selfedit-approval")
        defer { try? FileManager.default.removeItem(at: dir) }

        let adapter = try EditAdapter(
            program: EditProgram(
                sessionId: "sess-pending",
                approval: .pending,
                ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.5)]
            )
        )
        try adapter.save(to: dir)

        #expect(throws: BasaltError.self) {
            _ = try MLXInferenceEngine.loadServedAdapter(at: dir, into: model) { reason in
                BasaltError.invalidRequestStructure(details: reason)
            }
        }

        var approveCmd = try SelfEditCommand.parse(["--accept-edit", dir.path])
        try await approveCmd.run()
        let updated = try EditAdapter.from(directory: dir)
        #expect(updated.program.approval == .approved)

        let served = try MLXInferenceEngine.loadServedAdapter(at: dir, into: model) { reason in
            BasaltError.invalidRequestStructure(details: reason)
        }
        #expect(served is EditAdapter)
    }

    @Test("LocalSelfRewriteEditor formats prompts, parses JSON turns, and drives a session on tiny Gemma 4 MoE")
    func localSelfRewriteEditorDrivesSession() async throws {
        let model = try TinyModel.make(.gemma4Moe)
        let origFingerprint = ParameterFingerprint.capture(of: model)
        let stepCounter = LocalStepActor([
            "{\"rationale\":\"inspect\",\"action\":{\"describeModel\":{}}}",
            "{\"rationale\":\"stop\",\"action\":{\"withdraw\":{\"rationale\":\"local check done\"}}}",
        ])
        let localEditor = LocalSelfRewriteEditor(editorLabel: "local:gemma-4-26b") { prompt in
            #expect(prompt.contains("weight-surgery editor"))
            return await stepCounter.next()
        }
        let result = try await SelfRewriteSessionDriver.run(
            config: SelfRewriteSessionConfig(sessionId: "sess-local", budget: try SelfRewriteBudget()),
            split: Self.makeSplit(),
            evalSpec: Self.makeSpec(),
            editor: localEditor,
            original: model
        )
        #expect(result.phase == .withdrawn(rationale: "local check done"))
        #expect(result.turnCount == 2)
        try origFingerprint.requireUnchanged(against: model)
    }

    @Test("Scripted editor handles undo+withdraw without saving and cleanly terminates on budget exhaustion")
    func scriptedDriverWithdrawAndExhaustionOnGemma4Moe() async throws {
        let model = try TinyModel.make(.gemma4Moe)
        let origFingerprint = ParameterFingerprint.capture(of: model)
        let withdrawDir = Self.tempDir("selfedit-withdraw")
        defer { try? FileManager.default.removeItem(at: withdrawDir) }
        let withdrawEditor = ScriptedTestEditor([
            EditEditorTurn(rationale: "clip", action: .applyEdit(EditProgram(ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.5)]))),
            EditEditorTurn(rationale: "undo", action: .undoLastEdit),
            EditEditorTurn(rationale: "withdraw", action: .withdraw(rationale: "no gain")),
        ])
        let withdrawResult = try await SelfRewriteSessionDriver.run(
            config: SelfRewriteSessionConfig(sessionId: "sess-g4-withdraw", budget: try SelfRewriteBudget(), outputDirectory: withdrawDir),
            split: Self.makeSplit(),
            evalSpec: Self.makeSpec(),
            editor: withdrawEditor,
            original: model
        )
        #expect(withdrawResult.phase == .withdrawn(rationale: "no gain"))
        #expect(!FileManager.default.fileExists(atPath: withdrawDir.path))
        try origFingerprint.requireUnchanged(against: model)

        let exhaustEditor = ScriptedTestEditor([
            EditEditorTurn(rationale: "p1", action: .pairedPractice(problemIds: ["pr-1"], seeds: [7])),
            EditEditorTurn(rationale: "p2", action: .pairedPractice(problemIds: ["pr-1"], seeds: [7])),
        ])
        let exhaustResult = try await SelfRewriteSessionDriver.run(
            config: SelfRewriteSessionConfig(
                sessionId: "sess-g4-exhaust",
                budget: try SelfRewriteBudget(maxForwardPasses: 1, maxToolCalls: 5)
            ),
            split: Self.makeSplit(),
            evalSpec: Self.makeSpec(),
            editor: exhaustEditor,
            original: model
        )
        guard case .exhausted = exhaustResult.phase else {
            Issue.record("expected exhausted phase, got \(exhaustResult.phase)")
            return
        }
        try origFingerprint.requireUnchanged(against: model)
    }

    @Test("SelfEditCommand validates --experimental-self-rewrite, --editor specifiers, --allow-cloud-editor opt-in, and --max-alpha")
    func selfEditCommandValidation() throws {
        var cmd = try SelfEditCommand.parse(["--model", "gemma-4-26b"])
        #expect(throws: (any Error).self) { _ = try cmd.validateConfiguration() }

        cmd = try SelfEditCommand.parse(["--experimental-self-rewrite", "--model", "gemma-4-26b", "--editor", "cloud:gemini/gemini-2.5-pro"])
        let cloudErr = #expect(throws: SelfRewriteSessionError.self) { _ = try cmd.validateConfiguration() }
        #expect(cloudErr == .cloudEditorDisabled)

        cmd = try SelfEditCommand.parse([
            "--experimental-self-rewrite", "--model", "gemma-4-26b",
            "--editor", "cloud:gemini/gemini-2.5-pro", "--allow-cloud-editor", "--max-cloud-usd", "3.5", "--max-alpha", "5.0",
        ])
        let valid = try cmd.validateConfiguration()
        #expect(valid.editorSpec == .cloudGemini(model: "gemini-2.5-pro"))
        #expect(valid.budget.maxCloudUSD == 3.5)

        #expect(throws: SelfRewriteSessionError.self) {
            _ = try SelfEditEditorSpec.parse("cloud:openai/gpt-4")
        }
    }
}
