import BasaltCore
import Foundation
import MLX
import MLXLMCommon

public struct SelfRewriteLiveConfig: Sendable {
    public let session: SelfRewriteSessionConfig
    public let taskMode: SelfRewriteTaskMode
    public let evalConfig: PairedTestRunConfig
    public let reportURL: URL?

    public init(
        session: SelfRewriteSessionConfig,
        taskMode: SelfRewriteTaskMode = .improve,
        evalConfig: PairedTestRunConfig = PairedTestRunConfig(),
        reportURL: URL? = nil
    ) {
        self.session = session
        self.taskMode = taskMode
        self.evalConfig = evalConfig
        self.reportURL = reportURL
    }
}

public struct SelfRewriteIntegritySummary: Sendable, Codable, Equatable {
    public let originalFingerprintHash: String
    public let originalFingerprintVerified: Bool
    public let wallSeconds: Double

    public init(
        originalFingerprintHash: String,
        originalFingerprintVerified: Bool,
        wallSeconds: Double
    ) {
        self.originalFingerprintHash = originalFingerprintHash
        self.originalFingerprintVerified = originalFingerprintVerified
        self.wallSeconds = wallSeconds
    }
}

public struct SelfRewriteArtifactsSummary: Sendable, Codable, Equatable {
    public let savedAdapterPath: String?
    public let transcriptPath: String?
    public let submittedOpsCount: Int
    public let pairedTest: PairedTestSummary?

    public init(
        savedAdapterPath: String?,
        transcriptPath: String?,
        submittedOpsCount: Int,
        pairedTest: PairedTestSummary?
    ) {
        self.savedAdapterPath = savedAdapterPath
        self.transcriptPath = transcriptPath
        self.submittedOpsCount = submittedOpsCount
        self.pairedTest = pairedTest
    }
}

public struct SelfRewriteLiveReport: Sendable, Codable, Equatable {
    public let sessionId: String
    public let taskMode: SelfRewriteTaskMode
    public let phase: EditSessionPhase
    public let turnCount: Int
    public let integrity: SelfRewriteIntegritySummary
    public let artifacts: SelfRewriteArtifactsSummary

    public init(
        sessionId: String,
        taskMode: SelfRewriteTaskMode,
        phase: EditSessionPhase,
        turnCount: Int,
        integrity: SelfRewriteIntegritySummary,
        artifacts: SelfRewriteArtifactsSummary
    ) {
        self.sessionId = sessionId
        self.taskMode = taskMode
        self.phase = phase
        self.turnCount = turnCount
        self.integrity = integrity
        self.artifacts = artifacts
    }
}

extension SelfRewriteSessionDriver {
    static func generateLocalTurn(
        state: SessionRunState,
        turnIndex: Int
    ) throws -> EditEditorTurn {
        let prompt = SelfRewritePromptFormatter.formatSinglePrompt(
            entries: state.core.transcript,
            budget: state.core.budget,
            practiceIds: state.split.practice.map(\.id)
        )
        let promptIds = state.evalSpec.encode(prompt)
        let seed = UInt64(676 &+ turnIndex)
        let raw = try PairedTestRunner.generate(
            promptIds: promptIds,
            model: state.original,
            seed: seed,
            spec: state.evalSpec,
            maxTokensOverride: state.config.maxEditorTokens
        )
        return try EditEditorTurn.parseJSON(from: raw)
    }

    public static func runLive(
        config: SelfRewriteLiveConfig,
        split: PairedDatasetSplit,
        evalSpec: PairedEvaluationSpec,
        editor: (any EditSessionEditor)? = nil,
        original: any LanguageModel
    ) async throws -> SelfRewriteLiveReport {
        let start = ContinuousClock.now
        let pristineFP = ParameterFingerprint.capture(of: original)
        var damageAdapter: EditAdapter?
        if case .repair(let spec) = config.taskMode {
            damageAdapter = try applyRepairDamage(spec, to: original)
        }
        let sessionResult: Result<SelfRewriteSessionResult, any Error>
        do {
            sessionResult = .success(try await run(
                config: config.session,
                split: split,
                evalSpec: evalSpec,
                editor: editor,
                original: original
            ))
        } catch {
            sessionResult = .failure(error)
        }
        if let damageAdapter {
            damageAdapter.unload(from: original)
        }
        let afterFP = ParameterFingerprint.capture(of: original)
        let verified = afterFP.aggregateHash == pristineFP.aggregateHash
        let result = try sessionResult.get()
        let wall = ContinuousClock.now - start
        let report = SelfRewriteLiveReport(
            sessionId: result.sessionId,
            taskMode: config.taskMode,
            phase: result.phase,
            turnCount: result.turnCount,
            integrity: SelfRewriteIntegritySummary(
                originalFingerprintHash: pristineFP.aggregateHash,
                originalFingerprintVerified: verified,
                wallSeconds: wall.inSeconds
            ),
            artifacts: SelfRewriteArtifactsSummary(
                savedAdapterPath: result.savedAdapterURL?.path,
                transcriptPath: config.session.transcriptURL?.path,
                submittedOpsCount: result.submittedProgram?.ops.count ?? 0,
                pairedTest: result.submittedProgram?.pairedTest
            )
        )
        if let reportURL = config.reportURL {
            try persistReport(report, to: reportURL)
        }
        return report
    }

    static func persistReport(_ report: SelfRewriteLiveReport, to url: URL) throws {
        try withTaskCancellationShield {
            let files = FileManager.default
            try files.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(report)
            try data.write(to: url, options: .atomic)
            try files.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
    }
}

extension MLXInferenceEngine {
    public static func loadGemma4Topology(fromModelDirectory directory: URL) -> Gemma4EditTopology? {
        let configURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return Gemma4EditTopology(config: root)
    }

    static func encodeChatPrompt(_ text: String, tokenizer: any Tokenizer) -> [Int] {
        let messages: [[String: any Sendable]] = [["role": "user", "content": text]]
        if let ids = try? tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: ["enable_thinking": false]
        ), !ids.isEmpty {
            return ids
        }
        if let ids = try? tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil), !ids.isEmpty {
            return ids
        }
        return tokenizer.encode(text: text)
    }

    public func runSelfRewrite(
        config: SelfRewriteLiveConfig,
        split: PairedDatasetSplit,
        grader: any PairedAnswerGrader,
        editor: (any EditSessionEditor)? = nil
    ) async throws(BasaltError) -> SelfRewriteLiveReport {
        guard let container else { throw .modelNotLoaded }
        let report: SelfRewriteLiveReport
        do {
            report = try await container.perform { context in
                let stops = Self.stopTokenIds(context)
                let evalConfig = PairedTestRunConfig(
                    seeds: config.evalConfig.seeds,
                    maxTokens: config.evalConfig.maxTokens,
                    temperature: config.evalConfig.temperature,
                    stopTokenIds: stops.union(config.evalConfig.stopTokenIds),
                    budgetCost: config.evalConfig.budgetCost
                )
                let tokenizer = context.tokenizer
                let evalSpec = PairedEvaluationSpec(
                    encode: { text in Self.encodeChatPrompt(text, tokenizer: tokenizer) },
                    decode: { ids in tokenizer.decode(tokenIds: ids, skipSpecialTokens: true) },
                    grader: grader,
                    config: evalConfig
                )
                return try await SelfRewriteSessionDriver.runLive(
                    config: config,
                    split: split,
                    evalSpec: evalSpec,
                    editor: editor,
                    original: context.model
                )
            }
        } catch {
            throw Self.labFailure(error, "self-rewrite")
        }
        guard self.container === container else {
            throw .inferenceError("the model changed during self-rewrite")
        }
        return report
    }
}
