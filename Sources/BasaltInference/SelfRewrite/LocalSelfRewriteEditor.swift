import BasaltCore
import Foundation

public struct LocalSelfRewriteEditor: EditSessionEditor {
    public let editorLabel: String
    private let generate: @Sendable (String) async throws -> String

    public init(
        editorLabel: String = "local",
        generate: @escaping @Sendable (String) async throws -> String
    ) {
        self.editorLabel = editorLabel
        self.generate = generate
    }

    public func nextTurn(
        transcript: [EditSessionTranscriptEntry],
        remainingBudget: SelfRewriteBudget
    ) async throws -> EditEditorTurn {
        let prompt = SelfRewritePromptFormatter.formatSinglePrompt(entries: transcript, budget: remainingBudget)
        let reply = try await generate(prompt)
        return try EditEditorTurn.parseJSON(from: reply)
    }
}
