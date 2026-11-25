import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

private actor RunnerScriptEditor: EditSessionEditor {
    let editorLabel = "runner-script"
    private let turns: [EditEditorTurn]
    private var idx = 0

    init(_ turns: [EditEditorTurn]) {
        self.turns = turns
    }

    func nextTurn(
        transcript: [EditSessionTranscriptEntry],
        remainingBudget: SelfRewriteBudget
    ) async throws -> EditEditorTurn {
        guard idx < turns.count else {
            return EditEditorTurn(rationale: "done", action: .withdraw(rationale: "eof"))
        }
        defer { idx += 1 }
        return turns[idx]
    }
}

@Suite("Self-rewriting P3 live runner: repair lifecycle, MLXInferenceEngine.runSelfRewrite, native local turns, privacy fence, and CLI dry-run (#676, #770)", .serialized)
struct SelfRewritingP3RunnerTests {
    static let textOProj = "model.layers.0.self_attn.o_proj"

    static func encode(_ text: String) -> [Int] {
        ByteTokenizer().encode(text: text, addSpecialTokens: false) + [ByteTokenizer.endOfUserTurn]
    }

    static func decode(_ ids: [Int]) -> String {
        ByteTokenizer().decode(tokenIds: ids)
    }

    static func makeSplit() -> PairedDatasetSplit {
        PairedDatasetSplit(
            practice: [PairedProblem(id: "pr-1", prompt: "Compute 2 + 2.", expectedAnswer: "4")],
            reserved: [PairedProblem(id: "res-1", prompt: "SECRET RESERVED BENCHMARK PROMPT 42424242", expectedAnswer: "9")]
        )
    }

    static func makeSpec() -> PairedEvaluationSpec {
        PairedEvaluationSpec(
            encode: encode,
            decode: decode,
            grader: MathAnswerGrader(),
            config: PairedTestRunConfig(seeds: [7], maxTokens: 3, temperature: 0.0, stopTokenIds: [ByteTokenizer.endOfUserTurn], budgetCost: 1)
        )
    }
}

extension SelfRewritingP3RunnerTests {
    @Test("SelfRewriteSessionDriver.runLive applies repair damage, runs session, unloads damage, verifies fingerprint, and persists 0600 JSON report")
    func runLiveRepairLifecycleAndReport() async throws {
        let model = try TinyModel.make(.gemma4Moe)
        let origFP = ParameterFingerprint.capture(of: model)
        let tempRoot = FileManager.default.temporaryDirectory.appending(path: "selfedit-live-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempRoot) }
        let adapterDir = tempRoot.appending(path: "adapter")
        let transcriptURL = tempRoot.appending(path: "session.jsonl")
        let reportURL = tempRoot.appending(path: "report.json")

        let repairSpec = try SelfRewriteRepairSpec(tensor: Self.textOProj + ".weight", rows: [0, 1], scale: 10.0)
        let sessionConfig = SelfRewriteSessionConfig(
            sessionId: "sess-live-repair",
            budget: try SelfRewriteBudget(maxForwardPasses: 6, maxToolCalls: 12),
            outputDirectory: adapterDir
        ).withTranscriptURL(transcriptURL)
        let liveConfig = SelfRewriteLiveConfig(
            session: sessionConfig,
            taskMode: .repair(repairSpec),
            evalConfig: PairedTestRunConfig(seeds: [7], maxTokens: 3),
            reportURL: reportURL
        )
        let editor = RunnerScriptEditor([
            EditEditorTurn(rationale: "clip", action: .applyEdit(EditProgram(ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.5)]))),
            EditEditorTurn(rationale: "practice", action: .pairedPractice(problemIds: ["pr-1"], seeds: [7])),
            EditEditorTurn(rationale: "submit", action: .submit(rationale: "repaired")),
        ])

        let report = try await SelfRewriteSessionDriver.runLive(
            config: liveConfig,
            split: Self.makeSplit(),
            evalSpec: Self.makeSpec(),
            editor: editor,
            original: model
        )
        #expect(report.sessionId == "sess-live-repair")
        #expect(report.integrity.originalFingerprintVerified)
        #expect(report.integrity.originalFingerprintHash == origFP.aggregateHash)
        #expect(report.artifacts.submittedOpsCount == 1)
        #expect(report.artifacts.savedAdapterPath == adapterDir.path)
        #expect(ParameterFingerprint.capture(of: model).aggregateHash == origFP.aggregateHash)

        let perms = try #require(try FileManager.default.attributesOfItem(atPath: reportURL.path)[.posixPermissions] as? Int)
        #expect(perms & 0o777 == 0o600)
        let reloaded = try JSONDecoder().decode(SelfRewriteLiveReport.self, from: try Data(contentsOf: reportURL))
        #expect(reloaded == report)
    }

    @Test("MLXInferenceEngine.runSelfRewrite executes inside ModelContainer.perform on TinyModel.gemma4 and loads Gemma4EditTopology from config.json")
    func engineRunSelfRewriteAndTopologyLoader() async throws {
        let engine = try await TinyModel.engine(.gemma4)
        let cfgDir = FileManager.default.temporaryDirectory.appending(path: "g4-cfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: cfgDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cfgDir) }
        try Data(TinyModel.Family.gemma4.config.utf8).write(to: cfgDir.appending(component: "config.json"))

        let topology = try #require(MLXInferenceEngine.loadGemma4Topology(fromModelDirectory: cfgDir))
        let session = SelfRewriteSessionConfig(
            sessionId: "sess-engine-g4",
            budget: try SelfRewriteBudget(maxForwardPasses: 4, maxToolCalls: 8),
            topology: topology
        )
        let liveConfig = SelfRewriteLiveConfig(
            session: session,
            taskMode: .improve,
            evalConfig: PairedTestRunConfig(seeds: [7], maxTokens: 3)
        )
        let editor = RunnerScriptEditor([
            EditEditorTurn(rationale: "describe", action: .describeModel),
            EditEditorTurn(rationale: "withdraw", action: .withdraw(rationale: "engine check ok")),
        ])
        let report = try await engine.runSelfRewrite(
            config: liveConfig,
            split: Self.makeSplit(),
            grader: MathAnswerGrader(),
            editor: editor
        )
        #expect(report.phase == .withdrawn(rationale: "engine check ok"))
        #expect(report.integrity.originalFingerprintVerified && report.turnCount == 2)
    }

    @Test("Native local turn generation cleans up repair damage on malformed JSON, reservedFenceStrings protects reserved prompts, and CLI dry-run succeeds")
    func nativeLocalTurnCleanupAndCLIDryRun() async throws {
        let split = Self.makeSplit()
        let fenceStrings = split.reservedFenceStrings()
        #expect(throws: SelfRewriteSessionError.self) {
            _ = try CloudPrivacyFence.sanitizeAndValidate(
                [EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: "saw SECRET RESERVED BENCHMARK PROMPT 42424242")],
                privateStrings: fenceStrings
            )
        }

        let model = try TinyModel.make(.gemma4)
        let origFP = ParameterFingerprint.capture(of: model)
        let repairSpec = try SelfRewriteRepairSpec(tensor: Self.textOProj + ".weight", rows: [0], scale: 5.0)
        let liveConfig = SelfRewriteLiveConfig(
            session: SelfRewriteSessionConfig(sessionId: "sess-native-err", budget: try SelfRewriteBudget()).withMaxEditorTokens(16),
            taskMode: .repair(repairSpec),
            evalConfig: PairedTestRunConfig(seeds: [7], maxTokens: 3)
        )
        await #expect(throws: SelfRewriteSessionError.self) {
            _ = try await SelfRewriteSessionDriver.runLive(
                config: liveConfig,
                split: split,
                evalSpec: Self.makeSpec(),
                editor: nil,
                original: model
            )
        }
        #expect(ParameterFingerprint.capture(of: model).aggregateHash == origFP.aggregateHash)

        let tmp = FileManager.default.temporaryDirectory.appending(path: "selfedit-dryrun-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Data("{\"id\":\"1\",\"prompt\":\"2+2?\",\"expected_answer\":\"4\"}\n{\"id\":\"2\",\"prompt\":\"3+3?\",\"expected_answer\":\"6\"}\n".utf8).write(to: tmp)
        var cmd = try SelfEditCommand.parse([
            "--experimental-self-rewrite",
            "--model", "org/tiny-gemma4",
            "--practice", tmp.path,
            "--seeds", "7,11",
            "--dry-run",
        ])
        #expect(try cmd.parseSeeds() == [7, 11])
        try await cmd.run()
    }
}

extension SelfRewritingP3RunnerTests {
    @Test("SelfEditCommand request builder wires local and cloud Gemini/OpenRouter editors, math/translation graders, topology, and rejects invalid paths/seeds")
    func cliRequestBuilderAndErrorPaths() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "selfedit-req-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(TinyModel.Family.gemma4.config.utf8).write(to: dir.appending(component: "config.json"))
        let practice = dir.appending(component: "practice.jsonl")
        try Data("{\"id\":\"1\",\"prompt\":\"2+2?\",\"expected_answer\":\"4\"}\n{\"id\":\"2\",\"prompt\":\"3+3?\",\"expected_answer\":\"6\"}\n".utf8).write(to: practice)

        let localCmd = try SelfEditCommand.parse([
            "--experimental-self-rewrite", "--model", dir.path, "--practice", practice.path,
            "--grader", "math", "--report", dir.appending(component: "r.json").path, "--max-eval-tokens", "64",
        ])
        let localPrep = try localCmd.prepareHarness(sessionId: "sess-local")
        #expect(try localCmd.resolveModelPath() == dir.path)
        let localReq = try localCmd.buildLiveRequest(sessionId: "sess-local", prepared: localPrep, modelPath: dir.path, environment: [:])
        #expect(localReq.editor == nil)
        #expect(localReq.config.session.topology != nil && localReq.config.evalConfig.maxTokens == 64)
        #expect(localReq.config.reportURL?.lastPathComponent == "r.json")

        let cloudCmd = try SelfEditCommand.parse([
            "--experimental-self-rewrite", "--model", dir.path, "--practice", practice.path,
            "--grader", "translation", "--editor", "cloud:gemini/gemini-2.5-pro", "--allow-cloud-editor", "--max-cloud-usd", "1.5",
        ])
        let cloudPrep = try cloudCmd.prepareHarness(sessionId: "sess-cloud")
        let geminiReq = try cloudCmd.buildLiveRequest(sessionId: "sess-cloud", prepared: cloudPrep, modelPath: dir.path, environment: ["GEMINI_API_KEY": "test-key"])
        #expect(geminiReq.editor != nil)
        let orReq = try cloudCmd.buildLiveRequest(sessionId: "sess-cloud", prepared: cloudPrep, modelPath: dir.path, environment: ["OPENROUTER_API_KEY": "or-key"])
        #expect(orReq.editor != nil)

        var badSeeds = localCmd
        badSeeds.seeds = "abc"
        #expect(throws: Error.self) { _ = try badSeeds.parseSeeds() }
        var missingModel = localCmd
        missingModel.model = "nonexistent-org/definitely-not-cached-model-xyz"
        #expect(throws: Error.self) { _ = try missingModel.resolveModelPath() }
        #expect(MLXInferenceEngine.loadGemma4Topology(fromModelDirectory: dir.appending(component: "missing")) == nil)

        let unloaded = MLXInferenceEngine()
        await #expect(throws: BasaltError.self) {
            _ = try await unloaded.runSelfRewrite(config: localReq.config, split: Self.makeSplit(), grader: MathAnswerGrader())
        }
    }
}
