import Foundation

public enum SelfRewriteSessionError: Error, Sendable, Equatable, CustomStringConvertible {
    case sessionAlreadyEnded(EditSessionPhase)
    case staleEpoch(expected: UInt64, actual: UInt64)
    case noCandidateEdit
    case invalidEditorTurn(String)
    case cloudEditorDisabled
    case privacyFenceViolation(String)
    case invalidEditorSpecifier(String)

    public var description: String {
        switch self {
        case .sessionAlreadyEnded(let phase):
            return "self-rewrite session has already ended (\(phase))"
        case .staleEpoch(let expected, let actual):
            return "dropped stale session result (expected epoch \(expected), actual \(actual))"
        case .noCandidateEdit:
            return "cannot submit self-rewrite session with no applied edits"
        case .invalidEditorTurn(let detail):
            return "invalid editor turn: \(detail)"
        case .cloudEditorDisabled:
            return "cloud editor is disabled; pass --allow-cloud-editor explicitly to enable external Gemini surgery"
        case .privacyFenceViolation(let detail):
            return "blocked outbound cloud editor payload: \(detail)"
        case .invalidEditorSpecifier(let value):
            return "invalid editor specifier '\(value)' (expected 'local' or 'cloud:gemini/<model>')"
        }
    }
}

public enum EditSessionPhase: Sendable, Codable, Equatable {
    case running
    case submitted(rationale: String, pairedTest: PairedTestSummary?)
    case withdrawn(rationale: String)
    case exhausted(reason: String)

    public var isTerminal: Bool {
        switch self {
        case .running: false
        case .submitted, .withdrawn, .exhausted: true
        }
    }
}

public enum EditSessionAction: Sendable, Codable, Equatable {
    case describeModel
    case inspectTensor(name: String)
    case measureRoundTrip(tensor: String)
    case summarizeMoERouter(text: String, layers: [Int]?)
    case registerSVD(key: String, tensor: String, iterations: Int?)
    case applyEdit(EditProgram)
    case undoLastEdit
    case pairedPractice(problemIds: [String], seeds: [UInt64]?)
    case pairedReserved(seeds: [UInt64]?)
    case submit(rationale: String)
    case withdraw(rationale: String)
}

public struct EditEditorTurn: Sendable, Codable, Equatable {
    public let rationale: String
    public let action: EditSessionAction
    public let costUSD: Double

    enum CodingKeys: String, CodingKey {
        case rationale
        case action
        case costUSD
    }

    public init(rationale: String, action: EditSessionAction, costUSD: Double = 0) {
        self.rationale = rationale
        self.action = action
        self.costUSD = costUSD
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.rationale = try container.decodeIfPresent(String.self, forKey: .rationale) ?? ""
        self.action = try container.decode(EditSessionAction.self, forKey: .action)
        self.costUSD = try container.decodeIfPresent(Double.self, forKey: .costUSD) ?? 0
    }

    public static func parseJSON(from content: String) throws(SelfRewriteSessionError) -> EditEditorTurn {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        let jsonString: String
        if let start = trimmed.firstIndex(of: "{"), let end = trimmed.lastIndex(of: "}"), start <= end {
            jsonString = String(trimmed[start ... end])
        } else {
            jsonString = trimmed
        }
        guard let data = jsonString.data(using: .utf8) else {
            throw .invalidEditorTurn("response is not valid UTF-8 JSON")
        }
        do {
            return try JSONDecoder().decode(EditEditorTurn.self, from: data)
        } catch {
            throw .invalidEditorTurn("could not decode EditEditorTurn JSON: \(error)")
        }
    }
}

public enum SelfRewritePromptFormatter {
    public static func systemPrompt(budget: SelfRewriteBudget) -> String {
        """
        You are a weight-surgery editor operating strictly through symbolic handles and aggregate stats on a copy-on-write WeightCopy.
        Respond with a single JSON object matching EditEditorTurn: {"rationale":"...","action":{...},"costUSD":0}.
        Remaining budget: forwardPasses=\(budget.remainingForwardPasses), toolCalls=\(budget.remainingToolCalls), reservedTests=\(budget.remainingReservedTests).
        """
    }

    public static func formatSinglePrompt(
        entries: [EditSessionTranscriptEntry],
        budget: SelfRewriteBudget
    ) -> String {
        var parts = [systemPrompt(budget: budget)]
        if entries.isEmpty {
            parts.append("user: Begin session. Inspect model topology or tensor stats first.")
        } else {
            for entry in entries {
                let role = entry.role == "editor" ? "assistant" : "user"
                parts.append("\(role): \(entry.content)")
            }
        }
        parts.append("assistant:")
        return parts.joined(separator: "\n")
    }
}

public struct EditSessionTranscriptEntry: Sendable, Codable, Equatable {
    public let turnIndex: Int
    public let role: String
    public let content: String

    public init(turnIndex: Int, role: String, content: String) {
        self.turnIndex = turnIndex
        self.role = role
        self.content = content
    }
}

public protocol EditSessionEditor: Sendable {
    var editorLabel: String { get }
    func nextTurn(
        transcript: [EditSessionTranscriptEntry],
        remainingBudget: SelfRewriteBudget
    ) async throws -> EditEditorTurn
}

public struct SelfRewriteSessionCore: Sendable, Equatable {
    public let sessionId: String
    public let baseModelFingerprint: String
    public private(set) var phase: EditSessionPhase = .running
    public private(set) var epoch: UInt64 = 0
    public var budget: SelfRewriteBudget
    public private(set) var appliedPrograms: [EditProgram] = []
    public private(set) var lastPracticeSummary: PairedTestSummary?
    public private(set) var lastReservedSummary: PairedTestSummary?
    public private(set) var transcript: [EditSessionTranscriptEntry] = []

    public init(
        sessionId: String,
        baseModelFingerprint: String,
        budget: SelfRewriteBudget
    ) {
        self.sessionId = sessionId
        self.baseModelFingerprint = baseModelFingerprint
        self.budget = budget
    }

    public mutating func appendObservation(_ text: String) {
        transcript.append(EditSessionTranscriptEntry(turnIndex: transcript.count, role: "tool", content: text))
    }

    public mutating func beginAction(_ turn: EditEditorTurn) throws -> UInt64 {
        guard case .running = phase else {
            throw SelfRewriteSessionError.sessionAlreadyEnded(phase)
        }
        do {
            try budget.consumeToolCall()
            if turn.costUSD > 0 {
                try budget.consumeCloudCost(usd: turn.costUSD)
            }
        } catch {
            phase = .exhausted(reason: error.description)
            epoch &+= 1
            throw error
        }
        epoch &+= 1
        transcript.append(EditSessionTranscriptEntry(turnIndex: transcript.count, role: "editor", content: turn.rationale))
        return epoch
    }

    public mutating func completeEdit(program: EditProgram, expectedEpoch: UInt64) throws -> Bool {
        guard expectedEpoch == epoch, case .running = phase else { return false }
        try budget.consumeEditApply()
        appliedPrograms.append(program)
        return true
    }

    public mutating func completeUndo(expectedEpoch: UInt64) -> Bool {
        guard expectedEpoch == epoch, case .running = phase else { return false }
        _ = appliedPrograms.popLast()
        return true
    }

    public mutating func completePractice(summary: PairedTestSummary, expectedEpoch: UInt64) -> Bool {
        guard expectedEpoch == epoch, case .running = phase else { return false }
        lastPracticeSummary = summary
        return true
    }

    public mutating func completeReserved(summary: PairedTestSummary, expectedEpoch: UInt64) -> Bool {
        guard expectedEpoch == epoch, case .running = phase else { return false }
        lastReservedSummary = summary
        return true
    }

    public mutating func completeSubmit(rationale: String, expectedEpoch: UInt64) throws -> EditProgram? {
        guard expectedEpoch == epoch, case .running = phase else { return nil }
        guard let merged = mergedCandidateProgram() else {
            throw SelfRewriteSessionError.noCandidateEdit
        }
        phase = .submitted(rationale: rationale, pairedTest: merged.pairedTest)
        epoch &+= 1
        return merged
    }

    public mutating func completeWithdraw(rationale: String, expectedEpoch: UInt64) -> Bool {
        guard expectedEpoch == epoch, case .running = phase else { return false }
        phase = .withdrawn(rationale: rationale)
        epoch &+= 1
        return true
    }

    public mutating func markExhausted(reason: String) {
        guard case .running = phase else { return }
        phase = .exhausted(reason: reason)
        epoch &+= 1
    }

    public func mergedCandidateProgram() -> EditProgram? {
        let allOps = appliedPrograms.flatMap(\.ops)
        guard !allOps.isEmpty else { return nil }
        let scope = appliedPrograms.compactMap(\.targetScope).last
        let maxAlpha = appliedPrograms.compactMap(\.maxAlpha).max()
        return EditProgram(
            sessionId: sessionId,
            baseModelFingerprint: baseModelFingerprint,
            approval: .pending,
            pairedTest: lastReservedSummary ?? lastPracticeSummary,
            ops: allOps
        ).withPolicy(targetScope: scope, maxAlpha: maxAlpha)
    }
}

public enum CloudPrivacyFence {
    public static let maxInlineNumericTokens = 64
    private static let forbiddenPathSubstrings = [
        "/Users/", "/private/", "/Volumes/", "/home/", ".safetensors",
    ]

    public static func sanitizeAndValidate(
        _ entries: [EditSessionTranscriptEntry],
        privateStrings: Set<String> = []
    ) throws(SelfRewriteSessionError) -> [EditSessionTranscriptEntry] {
        var sanitized: [EditSessionTranscriptEntry] = []
        sanitized.reserveCapacity(entries.count)
        for entry in entries {
            let redacted = SecretRedactor.redact(entry.content)
            try verifyNoLeakage(redacted, privateStrings: privateStrings)
            sanitized.append(EditSessionTranscriptEntry(
                turnIndex: entry.turnIndex,
                role: entry.role,
                content: redacted
            ))
        }
        return sanitized
    }

    private static func verifyNoLeakage(
        _ text: String,
        privateStrings: Set<String>
    ) throws(SelfRewriteSessionError) {
        for forbidden in forbiddenPathSubstrings where text.contains(forbidden) {
            throw .privacyFenceViolation("outbound text contains forbidden path or checkpoint token '\(forbidden)'")
        }
        for secret in privateStrings where secret.count >= 16 && text.contains(secret) {
            throw .privacyFenceViolation("outbound text contains reserved benchmark content")
        }
        if exceedsInlineFloatLimit(text) {
            throw .privacyFenceViolation("outbound text contains inline numeric array exceeding \(maxInlineNumericTokens) elements")
        }
    }

    private static func exceedsInlineFloatLimit(_ text: String) -> Bool {
        var inBracket = false
        var numericRun = 0
        var token = ""
        for ch in text {
            if ch == "[" {
                inBracket = true
                numericRun = 0
                token.removeAll(keepingCapacity: true)
            } else if ch == "]" {
                if inBracket, isNumericToken(token) { numericRun += 1 }
                if numericRun > maxInlineNumericTokens { return true }
                inBracket = false
                numericRun = 0
                token.removeAll(keepingCapacity: true)
            } else if inBracket {
                if ch == "," || ch.isWhitespace {
                    if isNumericToken(token) { numericRun += 1 }
                    if numericRun > maxInlineNumericTokens { return true }
                    token.removeAll(keepingCapacity: true)
                } else {
                    token.append(ch)
                }
            }
        }
        return false
    }

    private static func isNumericToken(_ token: String) -> Bool {
        guard !token.isEmpty else { return false }
        return Double(token) != nil
    }
}
