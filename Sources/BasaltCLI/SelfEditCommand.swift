import ArgumentParser
import BasaltBench
import BasaltCloud
import BasaltCore
import BasaltInference
import Foundation

public enum SelfEditGraderKind: String, ExpressibleByArgument, Sendable, CaseIterable {
    case math
    case translation
}

public enum SelfEditEditorSpec: Sendable, Equatable {
    case local
    case cloudGemini(model: String)

    public static func parse(_ raw: String) throws(SelfRewriteSessionError) -> SelfEditEditorSpec {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == "local" {
            return .local
        }
        if trimmed.hasPrefix("cloud:gemini/") {
            let model = String(trimmed.dropFirst("cloud:gemini/".count))
            guard !model.isEmpty else { throw .invalidEditorSpecifier(raw) }
            return .cloudGemini(model: model)
        }
        throw .invalidEditorSpecifier(raw)
    }
}

public struct SelfEditCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "selfedit",
        abstract: "Experimental copy-on-write self-rewriting session (#676, #770)"
    )

    @Flag(name: .long, help: "Required opt-in gate for experimental self-rewriting sessions.")
    public var experimentalSelfRewrite: Bool = false

    @Option(name: .long, help: "Local subject model path or Hugging Face repo id.")
    public var model: String?

    @Option(name: .long, help: "Explicitly approve a submitted EditAdapter directory for serving.")
    public var acceptEdit: String?

    @Option(name: .long, help: "Editor mode: 'local' or 'cloud:gemini/<model>'.")
    public var editor: String = "local"

    @Flag(name: .long, help: "Explicit opt-in required to send summary stats to a cloud Gemini editor.")
    public var allowCloudEditor: Bool = false

    @Option(name: .long, help: "Maximum cloud spend in USD for this session.")
    public var maxCloudUsd: Double = 2.0

    @Option(name: .long, help: "Maximum forward-pass budget for this session.")
    public var maxForwardPasses: Int = 20

    @Option(name: .long, help: "Maximum tool-call budget for this session.")
    public var maxToolCalls: Int = 200

    @Option(name: .long, help: "Answer grader: 'math' or 'translation'.")
    public var grader: SelfEditGraderKind = .math

    @Option(name: .long, help: "Target scope: 'text_backbone_only' or 'include_multimodal'.")
    public var scope: String = EditTargetScope.textBackboneOnly.rawValue

    @Option(name: .long, help: "Optional research alpha cap (<= 10.0).")
    public var maxAlpha: Float?

    @Option(name: .long, help: "Output directory for the human-gated EditAdapter (saved only on submit).")
    public var output: String?

    @Option(name: .long, help: "Path to JSONL practice dataset (DeepScaleR / MATH / WMT24++ / FLORES+).")
    public var practice: String?

    @Option(name: .long, help: "Optional benchmark JSONL/text path for n-gram decontamination.")
    public var decontaminate: String?

    @Option(name: .long, help: "Optional practice split size (remaining items stay private reserved).")
    public var practiceCount: Int?

    @Option(name: .long, help: "Task mode: 'improve' or 'repair:<tensor>:<row0,row1,...>:<scale>'.")
    public var task: String = "improve"

    @Option(name: .long, help: "Optional JSONL session transcript path (defaults to ~/.basalt/selfedit/<session>.jsonl).")
    public var transcript: String?

    @Option(name: .long, help: "Optional JSON report output path for session metrics and fingerprint integrity.")
    public var report: String?

    @Option(name: .long, help: "Max generation tokens per paired-test answer.")
    public var maxEvalTokens: Int = 512

    @Option(name: .long, help: "Max generation tokens per editor turn (defaults to 1024 local, 4096 cloud).")
    public var maxEditorTokens: Int?

    @Option(name: .long, help: "Comma-separated deterministic seeds for paired evaluation (default: 42).")
    public var seeds: String = "42"

    @Flag(name: .long, help: "Validate flags, dataset split, and repair task without loading model weights.")
    public var dryRun: Bool = false

    public init() {}

    public struct PreparedHarness: Sendable {
        public let editorSpec: SelfEditEditorSpec
        public let scope: EditTargetScope
        public let budget: SelfRewriteBudget
        public let taskMode: SelfRewriteTaskMode
        public let split: PairedDatasetSplit
        public let removedDuplicates: Int
        public let transcriptURL: URL
    }

    public func validateConfiguration() throws -> (editorSpec: SelfEditEditorSpec, scope: EditTargetScope, budget: SelfRewriteBudget) {
        guard experimentalSelfRewrite else {
            throw ValidationError("pass --experimental-self-rewrite to run a self-editing session")
        }
        guard let model, !model.isEmpty else {
            throw ValidationError("--model is required to run a self-editing session")
        }
        let spec = try SelfEditEditorSpec.parse(editor)
        if case .cloudGemini = spec, !allowCloudEditor {
            throw SelfRewriteSessionError.cloudEditorDisabled
        }
        guard let parsedScope = EditTargetScope(rawValue: scope) else {
            throw ValidationError("invalid --scope '\(scope)' (expected text_backbone_only or include_multimodal)")
        }
        if let maxAlpha, (!maxAlpha.isFinite || maxAlpha <= 0 || maxAlpha > EditProgramValidator.maxAllowedResearchAlpha) {
            throw ValidationError("--max-alpha must be finite in (0, \(EditProgramValidator.maxAllowedResearchAlpha)]")
        }
        let cloudLimit: Double? = if case .cloudGemini = spec { maxCloudUsd } else { nil }
        let budget = try SelfRewriteBudget(
            maxForwardPasses: maxForwardPasses,
            maxToolCalls: maxToolCalls,
            maxReservedTests: 1,
            maxEditApplies: 32,
            maxCloudUSD: cloudLimit
        )
        return (spec, parsedScope, budget)
    }

    public func prepareHarness(sessionId: String = "selfedit-\(UUID().uuidString.prefix(8))") throws -> PreparedHarness {
        let validated = try validateConfiguration()
        let taskMode = try SelfRewriteTaskMode.parse(task)
        guard let practice, !practice.isEmpty else {
            throw ValidationError("--practice <dataset.jsonl> is required to load practice/reserved splits")
        }
        let splitInfo = try SelfRewriteDatasetLoader.buildSplit(
            practiceURL: URL(fileURLWithPath: practice),
            decontaminateURL: decontaminate.map { URL(fileURLWithPath: $0) },
            practiceCount: practiceCount
        )
        let transcriptURL = transcript.map { URL(fileURLWithPath: $0) }
            ?? SelfEditSessionRecord.defaultTranscriptURL(sessionId: sessionId)
        return PreparedHarness(
            editorSpec: validated.editorSpec,
            scope: validated.scope,
            budget: validated.budget,
            taskMode: taskMode,
            split: splitInfo.split,
            removedDuplicates: splitInfo.removedDuplicates,
            transcriptURL: transcriptURL
        )
    }

    public mutating func run() async throws {
        if let acceptEdit {
            let dir = URL(fileURLWithPath: acceptEdit)
            let program = try EditAdapter.setApproval(.approved, at: dir)
            print("Approved edit adapter at \(dir.path) (session=\(program.sessionId), ops=\(program.ops.count))")
            return
        }
        let sessionId = "selfedit-\(UUID().uuidString.prefix(8))"
        let prepared = try prepareHarness(sessionId: sessionId)
        if dryRun {
            print("SelfEdit dry-run: model=\(model ?? "") editor=\(prepared.editorSpec) task=\(prepared.taskMode.label) practice=\(prepared.split.practice.count) reserved=\(prepared.split.reservedCount) removed=\(prepared.removedDuplicates) transcript=\(prepared.transcriptURL.path)")
            return
        }
        let liveReport = try await executeLiveSession(sessionId: sessionId, prepared: prepared)
        print("SelfEdit finished: session=\(liveReport.sessionId) phase=\(liveReport.phase) turns=\(liveReport.turnCount) verified=\(liveReport.integrity.originalFingerprintVerified) ops=\(liveReport.artifacts.submittedOpsCount) adapter=\(liveReport.artifacts.savedAdapterPath ?? "-")")
    }
}
