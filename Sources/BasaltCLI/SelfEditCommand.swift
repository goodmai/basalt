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

    public init() {}

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

    public mutating func run() async throws {
        if let acceptEdit {
            let dir = URL(fileURLWithPath: acceptEdit)
            let program = try EditAdapter.setApproval(.approved, at: dir)
            print("Approved edit adapter at \(dir.path) (session=\(program.sessionId), ops=\(program.ops.count))")
            return
        }
        let validated = try validateConfiguration()
        print("SelfEdit validated: model=\(model ?? "") editor=\(validated.editorSpec) scope=\(validated.scope.rawValue) budget=\(validated.budget.maxForwardPasses) passes / \(validated.budget.maxToolCalls) tools")
        throw ValidationError("interactive dataset loading for `basalt selfedit` lands with P3 harness wiring (#676, #770)")
    }
}
