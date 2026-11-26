import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct SelfRewriteSessionConfig: Sendable {
    public let sessionId: String
    public let targetScope: EditTargetScope
    public let maxAlpha: Float?
    public let budget: SelfRewriteBudget
    public let outputDirectory: URL?
    public let topology: Gemma4EditTopology?
    public var transcriptURL: URL?
    public var maxEditorTokens: Int = 512

    public init(
        sessionId: String = "selfedit-\(UUID().uuidString.prefix(8))",
        targetScope: EditTargetScope = .textBackboneOnly,
        maxAlpha: Float? = nil,
        budget: SelfRewriteBudget,
        outputDirectory: URL? = nil,
        topology: Gemma4EditTopology? = nil
    ) {
        self.sessionId = sessionId
        self.targetScope = targetScope
        self.maxAlpha = maxAlpha
        self.budget = budget
        self.outputDirectory = outputDirectory
        self.topology = topology
        self.transcriptURL = nil
        self.maxEditorTokens = 512
    }

    public func withTranscriptURL(_ url: URL) -> SelfRewriteSessionConfig {
        var copy = self
        copy.transcriptURL = url
        return copy
    }

    public func withMaxEditorTokens(_ tokens: Int) -> SelfRewriteSessionConfig {
        var copy = self
        copy.maxEditorTokens = max(16, tokens)
        return copy
    }
}

public struct SelfRewriteSessionResult: Sendable, Equatable {
    public let sessionId: String
    public let phase: EditSessionPhase
    public let savedAdapterURL: URL?
    public let submittedProgram: EditProgram?
    public let budget: SelfRewriteBudget
    public let turnCount: Int

    public init(
        sessionId: String,
        phase: EditSessionPhase,
        savedAdapterURL: URL?,
        submittedProgram: EditProgram?,
        budget: SelfRewriteBudget,
        turnCount: Int
    ) {
        self.sessionId = sessionId
        self.phase = phase
        self.savedAdapterURL = savedAdapterURL
        self.submittedProgram = submittedProgram
        self.budget = budget
        self.turnCount = turnCount
    }
}

public enum SelfRewriteSessionDriver {
    public static func run(
        config: SelfRewriteSessionConfig,
        split: PairedDatasetSplit,
        evalSpec: PairedEvaluationSpec,
        editor: (any EditSessionEditor)? = nil,
        original: any LanguageModel
    ) async throws -> SelfRewriteSessionResult {
        var copy = try WeightCopy.makeSync(from: original)
        let core = SelfRewriteSessionCore(
            sessionId: config.sessionId,
            baseModelFingerprint: copy.originalFingerprint.aggregateHash,
            budget: config.budget
        )
        var state = SessionRunState(
            config: config,
            split: split,
            evalSpec: evalSpec,
            core: core,
            registry: SessionTensorRegistry(),
            original: original
        )
        let loopResult: Result<LoopSummary, any Error>
        do {
            loopResult = .success(try await runLoop(state: &state, copy: &copy, editor: editor))
        } catch {
            loopResult = .failure(error)
        }
        try copy.release(verifying: original)
        if let transcriptURL = config.transcriptURL {
            try persistTranscript(state.core.transcript, to: transcriptURL)
        }
        let summary = try loopResult.get()
        return SelfRewriteSessionResult(
            sessionId: config.sessionId,
            phase: state.core.phase,
            savedAdapterURL: summary.savedURL,
            submittedProgram: summary.submittedProgram,
            budget: state.core.budget,
            turnCount: summary.turnCount
        )
    }

    public static func applyRepairDamage(
        _ spec: SelfRewriteRepairSpec,
        to model: any LanguageModel
    ) throws -> EditAdapter {
        let dense = try resolveMatrix(spec.tensor, in: model)
        eval(dense)
        let rowCount = dense.dim(0)
        guard spec.rows.allSatisfy({ $0 >= 0 && $0 < rowCount }) else {
            throw WeightCopyError.invalidOperation("repair row index out of bounds for '\(spec.tensor)' with \(rowCount) rows")
        }
        let idxArray = MLXArray(spec.rows.map(Int32.init))
        let scaledRows = dense.take(idxArray, axis: 0) * MLXArray(spec.scale)
        eval(scaledRows)
        let key = "damage_rows"
        let adapter = try EditAdapter(program: spec.damageProgram(valuesKey: key), tensors: [key: scaledRows])
        try adapter.load(into: model)
        return adapter
    }

    private struct LoopSummary {
        var submittedProgram: EditProgram?
        var savedURL: URL?
        var turnCount: Int = 0
    }

    private static let maxConsecutiveErrors = 3

    private static func nextTurn(
        state: SessionRunState,
        turnCount: Int,
        editor: (any EditSessionEditor)?
    ) async throws -> EditEditorTurn {
        if let editor {
            return try await editor.nextTurn(transcript: state.core.transcript, remainingBudget: state.core.budget)
        }
        return try generateLocalTurn(state: state, turnIndex: turnCount)
    }

    static func isRecoverableActionError(_ error: any Error) -> Bool {
        if let weightError = error as? WeightCopyError {
            switch weightError {
            case .invalidOperation, .sharedHandleMutationBlocked: return true
            case .originalFingerprintMismatch, .copyReleased, .unsupportedModel, .budgetExceeded: return false
            }
        }
        if error is EditValidationError || error is TensorRegistryError { return true }
        if let sessionError = error as? SelfRewriteSessionError, sessionError == .noCandidateEdit { return true }
        if case .invalidRequestStructure = error as? BasaltError { return true }
        return false
    }

    private static func errorObservation(_ message: String) -> String {
        (try? encodeJSON(["error": message])) ?? "{\"error\":\"action rejected\"}"
    }

    private static func isIdempotentAction(_ action: EditSessionAction) -> Bool {
        switch action {
        case .describeModel, .inspectTensor, .measureRoundTrip, .summarizeMoERouter, .pairedPractice:
            return true
        case .registerSVD, .applyEdit, .undoLastEdit, .pairedReserved, .submit, .withdraw:
            return false
        }
    }

    private static func isWeightMutation(_ action: EditSessionAction) -> Bool {
        switch action {
        case .applyEdit, .undoLastEdit: true
        default: false
        }
    }

    private static func recordActionFailure(
        _ error: any Error,
        consecutiveErrors: inout Int,
        state: inout SessionRunState
    ) throws {
        consecutiveErrors += 1
        guard !state.core.phase.isTerminal, isRecoverableActionError(error), consecutiveErrors < maxConsecutiveErrors else {
            throw error
        }
        let extra = error is TensorRegistryError ? " (registered keys: \(state.registry.keys))" : ""
        state.core.appendObservation(errorObservation("action rejected: \(error)\(extra)"))
    }

    private static func recordRepeatedAction(
        consecutiveErrors: inout Int,
        state: inout SessionRunState
    ) -> Bool {
        consecutiveErrors += 1
        if consecutiveErrors >= maxConsecutiveErrors {
            state.core.markExhausted(reason: "repeated identical read action with unchanged weights \(maxConsecutiveErrors) times consecutively")
            return true
        }
        state.core.appendObservation(errorObservation("repeated identical action with unchanged weights; inspect a different tensor, apply or undo an edit, or submit/withdraw"))
        return false
    }

    private static func runLoop(
        state: inout SessionRunState,
        copy: inout WeightCopy,
        editor: (any EditSessionEditor)?
    ) async throws -> LoopSummary {
        var summary = LoopSummary()
        var consecutiveErrors = 0
        var completedIdempotentActions: [EditSessionAction] = []
        while !state.core.phase.isTerminal {
            try Task.checkCancellation()
            if state.core.budget.isExhausted {
                state.core.markExhausted(reason: "session forward-pass or tool-call budget exhausted")
                break
            }
            let turn: EditEditorTurn
            do {
                turn = try await nextTurn(state: state, turnCount: summary.turnCount, editor: editor)
            } catch let error as SelfRewriteSessionError {
                consecutiveErrors += 1
                guard case .invalidEditorTurn(let detail) = error, consecutiveErrors < maxConsecutiveErrors else { throw error }
                state.core.appendObservation(errorObservation("invalid JSON turn: \(detail). Reply ONLY with one valid EditEditorTurn JSON object."))
                continue
            }
            summary.turnCount += 1
            let epoch: UInt64
            do {
                epoch = try state.core.beginAction(turn)
            } catch is SelfRewriteBudgetError {
                break
            }
            if isIdempotentAction(turn.action), completedIdempotentActions.contains(turn.action) {
                if recordRepeatedAction(consecutiveErrors: &consecutiveErrors, state: &state) { break }
                continue
            }
            do {
                let outcome = try executeAction(turn.action, epoch: epoch, state: &state, copy: &copy)
                consecutiveErrors = 0
                if isIdempotentAction(turn.action) {
                    completedIdempotentActions.append(turn.action)
                } else if isWeightMutation(turn.action) {
                    completedIdempotentActions.removeAll(keepingCapacity: true)
                }
                if let program = outcome.submitted {
                    summary.submittedProgram = program
                    summary.savedURL = outcome.savedURL
                }
            } catch let budgetError as SelfRewriteBudgetError {
                state.core.markExhausted(reason: budgetError.description)
                break
            } catch {
                try recordActionFailure(error, consecutiveErrors: &consecutiveErrors, state: &state)
            }
        }
        return summary
    }
}

extension SelfRewriteSessionDriver {
    struct ActionOutcome {
        var submitted: EditProgram?
        var savedURL: URL?
    }

    struct SessionRunState {
        let config: SelfRewriteSessionConfig
        let split: PairedDatasetSplit
        let evalSpec: PairedEvaluationSpec
        var core: SelfRewriteSessionCore
        var registry: SessionTensorRegistry
        let original: any LanguageModel
    }

    static func executeAction(
        _ action: EditSessionAction,
        epoch: UInt64,
        state: inout SessionRunState,
        copy: inout WeightCopy
    ) throws -> ActionOutcome {
        switch action {
        case .describeModel, .inspectTensor, .measureRoundTrip, .summarizeMoERouter, .registerSVD:
            try executeReadAction(action, state: &state, copy: copy)
            return ActionOutcome()
        case .applyEdit, .undoLastEdit, .pairedPractice, .pairedReserved, .submit, .withdraw:
            return try executeWriteOrEvalAction(action, epoch: epoch, state: &state, copy: &copy)
        }
    }

    private static func executeReadAction(
        _ action: EditSessionAction,
        state: inout SessionRunState,
        copy: borrowing WeightCopy
    ) throws {
        switch action {
        case .describeModel:
            let summary = Gemma4SessionInspector.describeForEditing(model: try copy.model, scope: state.config.targetScope)
            state.core.appendObservation(try encodeJSON(summary))
        case .inspectTensor(let name):
            let info = try Gemma4SessionInspector.inspectTensor(
                name, in: try copy.model, topology: state.config.topology, scope: state.config.targetScope
            )
            state.core.appendObservation(try encodeJSON(info))
        case .measureRoundTrip(let tensor):
            let rt = try copy.measureRoundTrip(tensor: tensor)
            state.core.appendObservation(try encodeJSON(rt))
        case .summarizeMoERouter(let text, let layers):
            let tokens = state.evalSpec.encode(text).map(Int32.init)
            let stats = Gemma4SessionInspector.summarizeMoERouter(
                model: try copy.model, tokens: tokens, layers: layers.map(Set.init)
            )
            state.core.appendObservation(try encodeJSON(stats))
        case .registerSVD(let key, let tensor, let iterations):
            let matrix = try resolveMatrix(tensor, in: try copy.model)
            let summary = try state.registry.registerTopLeftSingularVector(
                key: key, matrix: matrix, iterations: iterations ?? 16
            )
            state.core.appendObservation(try encodeJSON(summary))
        default:
            break
        }
    }

    private static func executeWriteOrEvalAction(
        _ action: EditSessionAction,
        epoch: UInt64,
        state: inout SessionRunState,
        copy: inout WeightCopy
    ) throws -> ActionOutcome {
        switch action {
        case .applyEdit(let rawProgram):
            let program = rawProgram.withPolicy(targetScope: state.config.targetScope, maxAlpha: state.config.maxAlpha)
            let operands = try state.registry.resolveOperands(for: program)
            let metrics = try copy.apply(program, tensors: operands, verifying: state.original)
            _ = try state.core.completeEdit(program: program, expectedEpoch: epoch)
            state.core.appendObservation(try encodeJSON(metrics))
            return ActionOutcome()
        case .undoLastEdit:
            try copy.undoLast(verifying: state.original)
            _ = state.core.completeUndo(expectedEpoch: epoch)
            state.core.appendObservation("{\"undone\":true}")
            return ActionOutcome()
        case .pairedPractice(let problemIds, let seeds):
            let batch = try state.split.practiceBatch(ids: problemIds)
            let spec = overriddenSpec(state.evalSpec, seeds: seeds)
            let outcome = try PairedTestRunner.evaluatePractice(
                problems: batch, original: state.original, copy: copy, spec: spec, budget: &state.core.budget
            )
            _ = state.core.completePractice(summary: outcome.summary, expectedEpoch: epoch)
            state.core.appendObservation(try encodePracticeObservation(outcome))
            return ActionOutcome()
        case .pairedReserved(let seeds):
            let spec = overriddenSpec(state.evalSpec, seeds: seeds)
            let summary = try PairedTestRunner.evaluateReserved(
                split: state.split, original: state.original, copy: copy, spec: spec, budget: &state.core.budget
            )
            _ = state.core.completeReserved(summary: summary, expectedEpoch: epoch)
            state.core.appendObservation(try encodeJSON(summary))
            return ActionOutcome()
        case .submit(let rationale):
            guard epoch == state.core.epoch, case .running = state.core.phase else {
                return ActionOutcome()
            }
            guard let candidate = state.core.mergedCandidateProgram() else {
                throw SelfRewriteSessionError.noCandidateEdit
            }
            let operands = try state.registry.resolveOperands(for: candidate)
            let adapter = try EditAdapter(program: candidate, tensors: operands)
            guard let merged = try state.core.completeSubmit(rationale: rationale, expectedEpoch: epoch) else {
                return ActionOutcome()
            }
            if let dir = state.config.outputDirectory {
                try adapter.save(to: dir)
            }
            return ActionOutcome(submitted: merged, savedURL: state.config.outputDirectory)
        case .withdraw(let rationale):
            _ = state.core.completeWithdraw(rationale: rationale, expectedEpoch: epoch)
            return ActionOutcome()
        default:
            return ActionOutcome()
        }
    }
}

extension SelfRewriteSessionDriver {
    static func resolveMatrix(_ tensorName: String, in model: Module) throws -> MLXArray {
        guard tensorName.hasSuffix(".weight") else {
            throw WeightCopyError.invalidOperation("registerSVD requires a .weight tensor name, got '\(tensorName)'")
        }
        let path = String(tensorName.dropLast(".weight".count))
        guard let module = model.namedModules().first(where: { $0.0 == path })?.1 else {
            throw WeightCopyError.invalidOperation("module '\(path)' not found")
        }
        if let lora = module as? LoRALayer, let base = lora.reverted() as? Linear {
            return TensorPacker.dense(base).asType(.float32)
        }
        guard let linear = module as? Linear else {
            throw WeightCopyError.invalidOperation("module '\(path)' is not a Linear layer")
        }
        return TensorPacker.dense(linear).asType(.float32)
    }

    static func overriddenSpec(_ spec: PairedEvaluationSpec, seeds: [UInt64]?) -> PairedEvaluationSpec {
        guard let seeds, !seeds.isEmpty else { return spec }
        let updated = PairedTestRunConfig(
            seeds: seeds,
            maxTokens: spec.config.maxTokens,
            temperature: spec.config.temperature,
            stopTokenIds: spec.config.stopTokenIds,
            budgetCost: spec.config.budgetCost
        )
        return PairedEvaluationSpec(encode: spec.encode, decode: spec.decode, grader: spec.grader, config: updated)
    }

    struct PracticeSampleObservation: Encodable {
        let problemId: String
        let basePassed: Bool
        let baseExtracted: String?
        let baseDetail: String?
        let editedPassed: Bool
        let editedExtracted: String?
        let editedDetail: String?
    }

    struct PracticeObservationPayload: Encodable {
        let problemCount: Int
        let baseAccuracy: Double
        let editedAccuracy: Double
        let delta: Double
        let standardError: Double
        let wrongToRight: Int
        let rightToWrong: Int
        let samples: [PracticeSampleObservation]

        enum CodingKeys: String, CodingKey {
            case problemCount = "problem_count"
            case baseAccuracy = "base_accuracy"
            case editedAccuracy = "edited_accuracy"
            case delta
            case standardError = "standard_error"
            case wrongToRight = "wrong_to_right"
            case rightToWrong = "right_to_wrong"
            case samples
        }
    }

    private static func truncatedField(_ value: String?, limit: Int = 160) -> String? {
        guard let value else { return nil }
        return value.count > limit ? String(value.prefix(limit)) + "..." : value
    }

    static func encodePracticeObservation(_ outcome: PairedPracticeOutcome) throws -> String {
        let s = outcome.summary
        let details = outcome.samples.map {
            PracticeSampleObservation(
                problemId: $0.problemId,
                basePassed: $0.baseGrade.passed,
                baseExtracted: truncatedField($0.baseGrade.extractedAnswer),
                baseDetail: truncatedField($0.baseGrade.detail),
                editedPassed: $0.editedGrade.passed,
                editedExtracted: truncatedField($0.editedGrade.extractedAnswer),
                editedDetail: truncatedField($0.editedGrade.detail)
            )
        }
        return try encodeJSON(PracticeObservationPayload(
            problemCount: s.problemCount,
            baseAccuracy: s.baseAccuracy,
            editedAccuracy: s.editedAccuracy,
            delta: s.delta,
            standardError: s.standardError,
            wrongToRight: s.wrongToRight,
            rightToWrong: s.rightToWrong,
            samples: details
        ))
    }

    static func encodeJSON<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(value)
        return String(decoding: data, as: UTF8.self)
    }

    static func persistTranscript(_ entries: [EditSessionTranscriptEntry], to url: URL) throws {
        try withTaskCancellationShield {
            let files = FileManager.default
            let dir = url.deletingLastPathComponent()
            try files.createDirectory(at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            var lines: [String] = []
            lines.reserveCapacity(entries.count)
            for entry in entries {
                let data = try encoder.encode(entry)
                lines.append(String(decoding: data, as: UTF8.self))
            }
            let payload = lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n"
            try Data(payload.utf8).write(to: url, options: .atomic)
            try files.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
    }
}
