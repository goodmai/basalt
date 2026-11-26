import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
@testable import BasaltCLI
@testable import BasaltCloud
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

    @Test("EditEditorTurn.parseJSON strips thinking tags and decoy braces; prompt formatter lists action schema and practice IDs only")
    func parseJSONStripsThinkingAndPromptListsSchema() throws {
        let withThink = """
            <think>Let me consider {"draft": 1} before choosing an action.</think>
            ```json
            {"rationale":"inspect first","action":{"describeModel":{}},"costUSD":0}
            ```
            """
        let parsed = try EditEditorTurn.parseJSON(from: withThink)
        #expect(parsed.action == .describeModel && parsed.rationale == "inspect first")

        let withGemmaThought = "<|thought|>thinking {unclosed<|thought|> {\"rationale\":\"done\",\"action\":{\"withdraw\":{\"rationale\":\"ok\"}}}"
        let parsed2 = try EditEditorTurn.parseJSON(from: withGemmaThought)
        #expect(parsed2.action == .withdraw(rationale: "ok"))

        let withUnclosedThink = "<think>still thinking {\"rationale\":\"no\",\"action\":{\"describeModel\":{}}}"
        #expect(throws: SelfRewriteSessionError.self) { _ = try EditEditorTurn.parseJSON(from: withUnclosedThink) }

        let prompt = SelfRewritePromptFormatter.formatSinglePrompt(
            entries: [EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: String(repeating: "x", count: 6000))],
            budget: try SelfRewriteBudget(),
            practiceIds: ["m01", "m02"]
        )
        #expect(prompt.contains("describeModel") && prompt.contains("pairedPractice") && prompt.contains("applyEdit"))
        #expect(prompt.contains("m01, m02") && prompt.contains("truncated 6000 chars"))
        #expect(SelfRewritePromptFormatter.compactObservation("short") == "short")
    }

    @Test("SelfRewriteSessionDriver recovers from malformed JSON and rejected tool actions via tool observations, and stops after 3 consecutive failures")
    func sessionDriverErrorRecovery() async throws {
        actor FlakyEditor: EditSessionEditor {
            let editorLabel = "flaky"
            var step = 0
            func nextTurn(transcript: [EditSessionTranscriptEntry], remainingBudget: SelfRewriteBudget) async throws -> EditEditorTurn {
                defer { step += 1 }
                switch step {
                case 0: throw SelfRewriteSessionError.invalidEditorTurn("bad json")
                case 1: return EditEditorTurn(rationale: "bad tensor", action: .inspectTensor(name: "missing.weight"))
                case 2: return EditEditorTurn(rationale: "ok", action: .describeModel)
                case 3: return EditEditorTurn(rationale: "missing handle", action: .applyEdit(EditProgram(ops: [.rank1Damp(module: SelfRewritingP3RunnerTests.textOProj, vectorKey: "unregistered", alpha: 0.2)])))
                case 4: return EditEditorTurn(rationale: "submit early", action: .submit(rationale: "too soon"))
                default: return EditEditorTurn(rationale: "recover", action: .withdraw(rationale: "recovered after errors"))
                }
            }
        }
        let model = try TinyModel.make(.gemma4)
        let config = SelfRewriteSessionConfig(sessionId: "sess-recover", budget: try SelfRewriteBudget(maxForwardPasses: 6, maxToolCalls: 12))
        let result = try await SelfRewriteSessionDriver.run(
            config: config,
            split: Self.makeSplit(),
            evalSpec: Self.makeSpec(),
            editor: FlakyEditor(),
            original: model
        )
        #expect(result.phase == .withdrawn(rationale: "recovered after errors") && result.turnCount == 5)
    }

    @Test("WeightCopy.makeSync clones wrapped MLXLLM.Qwen35Model and Qwen35MoEModel preserving model_type and sharing weights (#676)")
    func weightCopyClonesWrappedQwen35Model() throws {
        let wrappedJSON = "{\"model_type\":\"qwen3_5\",\"text_config\":\(TinyModel.Family.qwen35.config)}"
        let cfg = try JSONDecoder().decode(MLXLLM::Qwen35Configuration.self, from: Data(wrappedJSON.utf8))
        let original = MLXLLM::Qwen35Model(cfg)
        let copy = try WeightCopy.makeSync(from: original)
        #expect(copy.accounting.totalExtraBytes == 0)
        try copy.release(verifying: original)

        let moeJSON = "{\"model_type\":\"qwen3_5_moe\",\"text_config\":\(TinyModel.Family.qwen35Moe.config)}"
        let moeCfg = try JSONDecoder().decode(MLXLLM::Qwen35Configuration.self, from: Data(moeJSON.utf8))
        let moeOriginal = MLXLLM::Qwen35MoEModel(moeCfg)
        let moeCopy = try WeightCopy.makeSync(from: moeOriginal)
        #expect(moeCopy.accounting.totalExtraBytes == 0)
        try moeCopy.release(verifying: moeOriginal)
    }
}

extension SelfRewritingP3RunnerTests {
    @Test("EditEditorTurn.parseJSON handles odd quotes in prose before the JSON payload (A-C2)")
    func parseJSONHandlesOddProseQuote() throws {
        let oddQuote = "Don't use \"odd quotes before {\"rationale\":\"ok\",\"action\":{\"describeModel\":{}}}"
        #expect(try EditEditorTurn.parseJSON(from: oddQuote).action == .describeModel)
    }

    @Test("EditValidationError.invalidSlice formats without Int overflow on Int.max start (A-C3)")
    func invalidSliceDescriptionAvoidsOverflow() {
        let overflowDesc = EditValidationError.invalidSlice(tensor: "w", start: Int.max, count: 1, limit: 16).description
        #expect(overflowDesc.contains("start=\(Int.max)") && overflowDesc.contains("count=1"))
    }

    @Test("MathAnswerGrader preserves UTF-16 alignment when text contains Turkish dotted I (#797)")
    func mathAnswerGraderHandlesTurkishDottedI() {
        #expect(MathAnswerGrader().extractAnswer(from: "İİİ Answer: 42") == "42")
    }

    @Test("SelfRewriteSessionDriver classifies WeightCopyError recoverability exhaustively (#797)")
    func isRecoverableActionErrorClassifiesWeightCopyErrors() {
        #expect(!SelfRewriteSessionDriver.isRecoverableActionError(WeightCopyError.originalFingerprintMismatch([])))
        #expect(!SelfRewriteSessionDriver.isRecoverableActionError(WeightCopyError.copyReleased))
        #expect(!SelfRewriteSessionDriver.isRecoverableActionError(WeightCopyError.unsupportedModel("x")))
        #expect(!SelfRewriteSessionDriver.isRecoverableActionError(WeightCopyError.budgetExceeded(required: 10, budget: 5)))
        #expect(SelfRewriteSessionDriver.isRecoverableActionError(WeightCopyError.invalidOperation("bad")))
        #expect(SelfRewriteSessionDriver.isRecoverableActionError(WeightCopyError.sharedHandleMutationBlocked(tensor: "w")))
    }

    @Test("CloudPrivacyFence blocks nested numeric arrays and case-variant paths while allowing a 150-layer EditModelSummary (A-C7)")
    func privacyFenceNestedArraysAndLargeDescribeModel() throws {
        let nestedBypass = "[[0], " + (1 ... 80).map(String.init).joined(separator: ", ") + "]"
        let twoRows = "[" + "[" + (1 ... 40).map(String.init).joined(separator: ", ") + "], [" + (41 ... 80).map(String.init).joined(separator: ", ") + "]]"
        for bad in [nestedBypass, twoRows, "path /users/alice/m", "file WEIGHTS.SAFETENSORS"] {
            let entry = [EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: bad)]
            #expect(throws: SelfRewriteSessionError.self) {
                _ = try CloudPrivacyFence.sanitizeAndValidate(entry)
            }
        }

        let layers = (0 ..< 150).map { idx in
            EditEditableLayerSummary(path: "language_model.model.layers.\(idx).self_attn.o_proj", shape: [2048, 512], format: "quantized4", blastRadiusWarning: nil)
        }
        let summary = EditModelSummary(modelType: "gemma4", topology: nil, editableLayers: layers, totalParameterCount: 26_000_000_000)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let summaryJSON = String(decoding: try encoder.encode(summary), as: UTF8.self)
        let summaryEntry = [EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: summaryJSON)]
        let sanitized = try CloudPrivacyFence.sanitizeAndValidate(summaryEntry)
        #expect(sanitized.count == 1 && sanitized[0].content.contains("layers.149"))
    }
}

extension SelfRewritingP3RunnerTests {
    @Test("Three consecutive invalid JSON turns abort and emit valid JSON error observations escaping quotes (#797)")
    func threeInvalidJSONTurnsAbortWithValidJSONObservation() async throws {
        actor ThreeFailJSONEditor: EditSessionEditor {
            let editorLabel = "bad-json"
            var calls = 0
            var seenTranscript: [EditSessionTranscriptEntry] = []
            func nextTurn(transcript: [EditSessionTranscriptEntry], remainingBudget: SelfRewriteBudget) async throws -> EditEditorTurn {
                seenTranscript = transcript
                calls += 1
                guard calls <= 5 else { return EditEditorTurn(rationale: "stop", action: .withdraw(rationale: "cap regressed")) }
                throw SelfRewriteSessionError.invalidEditorTurn("has \"quoted\" text")
            }
        }
        let model = try TinyModel.make(.gemma4)
        let config = SelfRewriteSessionConfig(sessionId: "sess-cap-json", budget: try SelfRewriteBudget(maxForwardPasses: 6, maxToolCalls: 12))
        let editor = ThreeFailJSONEditor()
        await #expect(throws: SelfRewriteSessionError.self) {
            _ = try await SelfRewriteSessionDriver.run(
                config: config, split: Self.makeSplit(), evalSpec: Self.makeSpec(), editor: editor, original: model
            )
        }
        #expect(await editor.calls == 3)
        let transcript = await editor.seenTranscript
        let errors = transcript.filter { $0.role == "tool" && $0.content.contains("quoted") }
        #expect(errors.count == 2)
        for obs in errors {
            let parsed = try JSONSerialization.jsonObject(with: Data(obs.content.utf8)) as? [String: String]
            #expect(parsed?["error"]?.contains("has \"quoted\" text") == true)
        }
    }

    @Test("Three consecutive rejected tool actions abort and emit valid JSON error observations escaping quotes (#797)")
    func threeRejectedActionsAbortWithValidJSONObservation() async throws {
        actor ThreeFailActionEditor: EditSessionEditor {
            let editorLabel = "bad-action"
            var calls = 0
            var seenTranscript: [EditSessionTranscriptEntry] = []
            func nextTurn(transcript: [EditSessionTranscriptEntry], remainingBudget: SelfRewriteBudget) async throws -> EditEditorTurn {
                seenTranscript = transcript
                calls += 1
                guard calls <= 5 else { return EditEditorTurn(rationale: "stop", action: .withdraw(rationale: "cap regressed")) }
                return EditEditorTurn(rationale: "bad", action: .inspectTensor(name: "missing\"quote.weight"))
            }
        }
        let model = try TinyModel.make(.gemma4)
        let config = SelfRewriteSessionConfig(sessionId: "sess-cap-action", budget: try SelfRewriteBudget(maxForwardPasses: 6, maxToolCalls: 12))
        let editor = ThreeFailActionEditor()
        await #expect(throws: BasaltError.self) {
            _ = try await SelfRewriteSessionDriver.run(
                config: config, split: Self.makeSplit(), evalSpec: Self.makeSpec(), editor: editor, original: model
            )
        }
        #expect(await editor.calls == 3)
        let transcript = await editor.seenTranscript
        let errors = transcript.filter { $0.role == "tool" && $0.content.contains("action rejected") }
        #expect(errors.count == 2)
        for obs in errors {
            let parsed = try JSONSerialization.jsonObject(with: Data(obs.content.utf8)) as? [String: String]
            #expect(parsed?["error"]?.contains("quote.weight") == true && parsed?["error"]?.contains("\"") == true)
        }
    }

    @Test("Submitting a merged program exceeding 64 ops does not falsely mark session .submitted and returns a recoverable JSON error observation (#797)")
    func submitOver64MergedOpsStaysRunning() async throws {
        let op = EditOp.clip(tensor: Self.textOProj + ".weight", factor: 1.0)
        let fortyOps = EditProgram(ops: Array(repeating: op, count: 40))
        actor OverLimitSubmitEditor: EditSessionEditor {
            let editorLabel = "overlimit"
            let batch: EditProgram
            var step = 0
            var seenTranscript: [EditSessionTranscriptEntry] = []
            init(batch: EditProgram) { self.batch = batch }
            func nextTurn(transcript: [EditSessionTranscriptEntry], remainingBudget: SelfRewriteBudget) async throws -> EditEditorTurn {
                seenTranscript = transcript
                defer { step += 1 }
                switch step {
                case 0, 1: return EditEditorTurn(rationale: "apply 40", action: .applyEdit(batch))
                case 2: return EditEditorTurn(rationale: "submit 80", action: .submit(rationale: "too many"))
                default: return EditEditorTurn(rationale: "give up", action: .withdraw(rationale: "validated before submit"))
                }
            }
        }
        let model = try TinyModel.make(.gemma4)
        let config = SelfRewriteSessionConfig(sessionId: "sess-overlimit", budget: try SelfRewriteBudget(maxForwardPasses: 6, maxToolCalls: 12))
        let editor = OverLimitSubmitEditor(batch: fortyOps)
        let result = try await SelfRewriteSessionDriver.run(
            config: config,
            split: Self.makeSplit(),
            evalSpec: Self.makeSpec(),
            editor: editor,
            original: model
        )
        #expect(result.phase == .withdrawn(rationale: "validated before submit"))
        #expect(result.submittedProgram == nil && result.savedAdapterURL == nil)
        let transcript = await editor.seenTranscript
        let toolErrors = transcript.filter { $0.role == "tool" && $0.content.contains("action rejected") }
        #expect(toolErrors.count == 1)
        let parsed = try JSONSerialization.jsonObject(with: Data(toolErrors[0].content.utf8)) as? [String: String]
        #expect(parsed?["error"] != nil)
    }
}

extension SelfRewritingP3RunnerTests {
    @Test("EditEditorTurn.parseJSON normalizes multi-key actions, plain-string zero-arg actions, and Gemma 4 channel thought blocks (#770)")
    func parseJSONNormalizesLiveModelActionShapes() throws {
        let multiKey = """
            {"rationale":"extra keys","action":{"inspectTensor":{"name":"w.weight"},"describeModel":{}}}
            """
        #expect(try EditEditorTurn.parseJSON(from: multiKey).action == .inspectTensor(name: "w.weight"))

        let plainString = "{\"rationale\":\"revert\",\"action\":\"undoLastEdit\"}"
        #expect(try EditEditorTurn.parseJSON(from: plainString).action == .undoLastEdit)

        let gemmaChannel = "<|channel>thought\nchecking {\"bad\":1}<channel|>{\"rationale\":\"ok\",\"action\":{\"describeModel\":{}}}"
        #expect(try EditEditorTurn.parseJSON(from: gemmaChannel).action == .describeModel)
    }

    @Test("SelfRewritePromptFormatter windows long transcripts and GeminiCloudEditor includes practice problemIds (#770)")
    func promptWindowingAndCloudPracticeIds() throws {
        let entries = (0 ..< 24).map { idx in
            EditSessionTranscriptEntry(turnIndex: idx, role: idx.isMultiple(of: 2) ? "editor" : "tool", content: "entry-\(idx)")
        }
        let prompt = SelfRewritePromptFormatter.formatSinglePrompt(
            entries: entries,
            budget: try SelfRewriteBudget(),
            practiceIds: ["m01", "m02"]
        )
        #expect(prompt.contains("entry-0") && prompt.contains("entry-1"))
        #expect(prompt.contains("8 earlier transcript entries omitted"))
        #expect(!prompt.contains("entry-5"))
        #expect(prompt.contains("entry-10") && prompt.contains("entry-23"))

        let messages = GeminiCloudEditor.buildMessages(
            entries: [EditSessionTranscriptEntry(turnIndex: 0, role: "tool", content: String(repeating: "a", count: 9000))],
            budget: try SelfRewriteBudget(),
            practiceIds: ["m01", "m02"]
        )
        #expect(messages.first?.role == "system")
        #expect(messages.first?.content.contains("Available practice problemIds: m01, m02") == true)
        #expect(messages.last?.content.contains("truncated 9000 chars") == true)
    }

    @Test("SelfRewriteSessionDriver deduplicates repeated read actions (including non-consecutive A-B-A cycles) until a weight mutation, and exposes samples on practice only (#770, #676)")
    func repeatDedupAndPracticeSampleDiagnostics() async throws {
        actor RecordingEditor: EditSessionEditor {
            let editorLabel = "recording"
            let turns: [EditEditorTurn]
            var step = 0
            var seenTranscript: [EditSessionTranscriptEntry] = []
            init(_ turns: [EditEditorTurn]) { self.turns = turns }
            func nextTurn(transcript: [EditSessionTranscriptEntry], remainingBudget: SelfRewriteBudget) async throws -> EditEditorTurn {
                seenTranscript = transcript
                defer { step += 1 }
                return turns[min(step, turns.count - 1)]
            }
        }
        let tensor = Self.textOProj + ".weight"
        let editor = RecordingEditor([
            EditEditorTurn(rationale: "inspect 1", action: .inspectTensor(name: tensor)),
            EditEditorTurn(rationale: "describe", action: .describeModel),
            EditEditorTurn(rationale: "inspect A-B-A repeat", action: .inspectTensor(name: tensor)),
            EditEditorTurn(rationale: "clip", action: .applyEdit(EditProgram(ops: [.clip(tensor: tensor, factor: 1.5)]))),
            EditEditorTurn(rationale: "inspect post-edit", action: .inspectTensor(name: tensor)),
            EditEditorTurn(rationale: "practice", action: .pairedPractice(problemIds: ["pr-1"], seeds: [7])),
            EditEditorTurn(rationale: "reserved", action: .pairedReserved(seeds: [7])),
            EditEditorTurn(rationale: "done", action: .withdraw(rationale: "verified")),
        ])
        let model = try TinyModel.make(.gemma4)
        let config = SelfRewriteSessionConfig(sessionId: "sess-dedup", budget: try SelfRewriteBudget(maxForwardPasses: 6, maxToolCalls: 16))
        let result = try await SelfRewriteSessionDriver.run(
            config: config, split: Self.makeSplit(), evalSpec: Self.makeSpec(), editor: editor, original: model
        )
        #expect(result.phase == .withdrawn(rationale: "verified") && result.turnCount == 8)

        let toolObs = await editor.seenTranscript.filter { $0.role == "tool" }.map(\.content)
        #expect(toolObs.count == 7)
        #expect(toolObs[2].contains("repeated identical action"))
        #expect(toolObs[4].contains("\"name\":\"\(tensor)\""))
        #expect(toolObs[5].contains("\"samples\"") && toolObs[5].contains("\"problemId\":\"pr-1\""))
        #expect(toolObs[6].contains("\"base_accuracy\"") && !toolObs[6].contains("\"samples\""))
    }
}
