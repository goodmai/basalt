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

    public var summaryLabel: String {
        switch self {
        case .describeModel: "describeModel"
        case .inspectTensor(let name): "inspectTensor(\(name))"
        case .measureRoundTrip(let tensor): "measureRoundTrip(\(tensor))"
        case .summarizeMoERouter(_, let layers): "summarizeMoERouter(layers:\(layers?.count ?? 0))"
        case .registerSVD(let key, let tensor, _): "registerSVD(\(key)<-\(tensor))"
        case .applyEdit(let program): "applyEdit(\(program.ops.count) ops)"
        case .undoLastEdit: "undoLastEdit"
        case .pairedPractice(let ids, _): "pairedPractice(\(ids.count))"
        case .pairedReserved: "pairedReserved"
        case .submit: "submit"
        case .withdraw: "withdraw"
        }
    }
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

    public static let knownActionKeys: [String] = [
        "submit", "withdraw", "applyEdit", "undoLastEdit", "pairedReserved",
        "pairedPractice", "registerSVD", "summarizeMoERouter", "measureRoundTrip",
        "inspectTensor", "describeModel",
    ]

    public static func stripThinkingBlocks(_ text: String) -> String {
        var result = text
        let delimiters = [
            ("<think>", "</think>"),
            ("<|thought|>", "<|thought|>"),
            ("<thought>", "</thought>"),
            ("<|channel>thought", "<channel|>"),
        ]
        for (open, close) in delimiters {
            while let start = result.range(of: open) {
                if let end = result.range(of: close, range: start.upperBound ..< result.endIndex) {
                    result.removeSubrange(start.lowerBound ..< end.upperBound)
                } else {
                    result.removeSubrange(start.lowerBound ..< result.endIndex)
                }
            }
        }
        return result
    }

    static func normalizeActionPayload(_ data: Data) -> Data {
        guard var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return data }
        if let name = obj["action"] as? String, knownActionKeys.contains(name) {
            obj["action"] = [name: [String: Any]()]
            return (try? JSONSerialization.data(withJSONObject: obj)) ?? data
        }
        if let actionDict = obj["action"] as? [String: Any], actionDict.count > 1,
           let key = knownActionKeys.first(where: { actionDict[$0] != nil }),
           let value = actionDict[key] {
            obj["action"] = [key: value]
            return (try? JSONSerialization.data(withJSONObject: obj)) ?? data
        }
        return data
    }

    public static func extractBalancedJSONObjects(from text: String) -> [String] {
        var objects: [String] = []
        var depth = 0
        var inString = false
        var escaped = false
        var startIndex: String.Index?
        for idx in text.indices {
            let ch = text[idx]
            if escaped {
                escaped = false
                continue
            }
            if inString {
                if ch == "\\" { escaped = true } else if ch == "\"" { inString = false }
                continue
            }
            if depth > 0, ch == "\"" {
                inString = true
            } else if ch == "{" {
                if depth == 0 { startIndex = idx }
                depth += 1
            } else if ch == "}" && depth > 0 {
                depth -= 1
                if depth == 0, let start = startIndex {
                    objects.append(String(text[start ... idx]))
                    startIndex = nil
                }
            }
        }
        return objects
    }

    public static func parseJSON(from content: String) throws(SelfRewriteSessionError) -> EditEditorTurn {
        let cleaned = stripThinkingBlocks(content).trimmingCharacters(in: .whitespacesAndNewlines)
        let candidates = extractBalancedJSONObjects(from: cleaned)
        let decoder = JSONDecoder()
        var lastError: (any Error)?
        for candidate in candidates.reversed() {
            if let data = candidate.data(using: .utf8) {
                do {
                    return try decoder.decode(EditEditorTurn.self, from: normalizeActionPayload(data))
                } catch {
                    lastError = error
                }
            }
        }
        guard let fallbackData = cleaned.data(using: .utf8) else {
            throw .invalidEditorTurn("response is not valid UTF-8 JSON")
        }
        do {
            return try decoder.decode(EditEditorTurn.self, from: normalizeActionPayload(fallbackData))
        } catch {
            throw .invalidEditorTurn("could not decode EditEditorTurn JSON: \(lastError ?? error)")
        }
    }
}

public enum SelfRewritePromptFormatter {
    public static func systemPrompt(budget: SelfRewriteBudget) -> String {
        """
        You are a weight-surgery editor operating strictly through symbolic handles and aggregate stats on a copy-on-write WeightCopy.
        Respond with ONLY one JSON object matching EditEditorTurn: {"rationale":"brief reason","action":<Action>,"costUSD":0}.
        Valid <Action> shapes:
        - {"describeModel":{}}
        - {"inspectTensor":{"name":"<tensor>"}}
        - {"measureRoundTrip":{"tensor":"<tensor>"}}
        - {"summarizeMoERouter":{"text":"<probe>","layers":[0]}}
        - {"registerSVD":{"key":"<k>","tensor":"<tensor>","iterations":6}}
        - {"applyEdit":{"_0":{"format_version":1,"session_id":"s","base_model_fingerprint":"","approval":"pending","ops":[{"op":"clip","tensor":"<t>","factor":1.5}]}}}
        - {"undoLastEdit":{}}
        - {"pairedPractice":{"problemIds":["id1"],"seeds":[42]}}
        - {"pairedReserved":{"seeds":[42]}}
        - {"submit":{"rationale":"why"}}
        - {"withdraw":{"rationale":"why"}}
        Supported EditOp kinds: clip(tensor,factor), rank1_damp(module,vector_key,alpha) [uses 1 registered SVD key on module path without '.weight'], rank1(module,output_key,input_key,scale) [needs 2 registered keys], shift(tensor,start,count,value_key), rows(tensor,row_indices,values_key).
        Rules: call describeModel at most once; never repeat an already completed action; use exact '.weight' tensor names for inspectTensor/measureRoundTrip/registerSVD/clip and module path (no '.weight') with a registered SVD key for rank1_damp; run pairedPractice before submitting or withdrawing.
        Remaining budget: forwardPasses=\(budget.remainingForwardPasses), toolCalls=\(budget.remainingToolCalls), reservedTests=\(budget.remainingReservedTests).
        """
    }

    public static let maxObservationChars = 2400
    public static let maxPromptEntries = 16

    public static func compactObservation(_ text: String, limit: Int = maxObservationChars) -> String {
        guard text.count > limit, limit > 500 else { return text }
        let head = text.prefix(limit - 450)
        let tail = text.suffix(350)
        return "\(head) ... [truncated \(text.count) chars] ... \(tail)"
    }

    private static func formatEntry(_ entry: EditSessionTranscriptEntry) -> String {
        let role = entry.role == "editor" ? "assistant" : "user"
        let body = entry.role == "tool" ? compactObservation(entry.content) : entry.content
        return "\(role): \(body)"
    }

    public static func formatSinglePrompt(
        entries: [EditSessionTranscriptEntry],
        budget: SelfRewriteBudget,
        practiceIds: [String] = []
    ) -> String {
        var parts = [systemPrompt(budget: budget)]
        if !practiceIds.isEmpty {
            parts.append("Available practice problemIds: \(practiceIds.joined(separator: ", "))")
        }
        if entries.isEmpty {
            parts.append("user: Begin session. Inspect model topology or tensor stats first, then test edits on practice problems before submitting or withdrawing.")
        } else if entries.count <= maxPromptEntries {
            parts.append(contentsOf: entries.map(formatEntry))
        } else {
            let headCount = 2
            let tailCount = maxPromptEntries - headCount
            let dropped = entries.count - maxPromptEntries
            parts.append(contentsOf: entries.prefix(headCount).map(formatEntry))
            parts.append("user: [... \(dropped) earlier transcript entries omitted ...]")
            parts.append(contentsOf: entries.suffix(tailCount).map(formatEntry))
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
        let summary = turn.rationale.isEmpty ? turn.action.summaryLabel : "[\(turn.action.summaryLabel)] \(turn.rationale)"
        transcript.append(EditSessionTranscriptEntry(turnIndex: transcript.count, role: "editor", content: summary))
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
        for forbidden in forbiddenPathSubstrings where text.range(of: forbidden, options: .caseInsensitive) != nil {
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
        var depth = 0
        var numericRun = 0
        var token = ""
        for ch in text {
            if ch == "[" {
                if depth == 0 {
                    numericRun = 0
                    token.removeAll(keepingCapacity: true)
                } else if flushNumericToken(&token, run: &numericRun) {
                    return true
                }
                depth += 1
            } else if ch == "]" {
                if depth > 0, flushNumericToken(&token, run: &numericRun) { return true }
                depth = max(0, depth - 1)
                if depth == 0 { numericRun = 0 }
            } else if depth > 0 {
                if ch == "," || ch.isWhitespace {
                    if flushNumericToken(&token, run: &numericRun) { return true }
                } else {
                    token.append(ch)
                }
            }
        }
        return false
    }

    private static func flushNumericToken(_ token: inout String, run: inout Int) -> Bool {
        defer { token.removeAll(keepingCapacity: true) }
        guard !token.isEmpty else { return false }
        if Double(token) != nil {
            run += 1
            return run > maxInlineNumericTokens
        }
        run = 0
        return false
    }
}
