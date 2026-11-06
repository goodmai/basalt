import Foundation

public struct SessionTemporalCheck: Sendable {
    public enum Property: String, CaseIterable, Sendable {
        case identicalTriple = "P1"
        case retryAfterError = "P2"
        case retryAfterRejection = "P3"
        case silentRounds = "P4"
    }

    public struct Violation: Hashable, Sendable {
        public let property: Property
        public let messageIndex: Int
        public let tool: String
    }

    enum Atom: Int {
        case call, repeatsCall, toolError, rejection, retriesRejected, silentRound, textTurn

        var bit: UInt64 { 1 << UInt64(rawValue) }
        var formula: PastFormula { .atom(rawValue) }
    }

    static let silentRoundLimit = 8

    static func formula(_ property: Property) -> PastFormula {
        let sinceLastCall = { (event: Atom) in PastFormula.yesterday(.since(.not(Atom.call.formula), event.formula)) }
        switch property {
        case .identicalTriple:
            return .and(Atom.repeatsCall.formula, sinceLastCall(.repeatsCall))
        case .retryAfterError:
            return .and(Atom.repeatsCall.formula, sinceLastCall(.toolError))
        case .retryAfterRejection:
            return .and(Atom.retriesRejected.formula,
                        .yesterday(.since(.not(Atom.retriesRejected.formula), Atom.rejection.formula)))
        case .silentRounds:
            return (1..<silentRoundLimit).reduce(Atom.silentRound.formula) { inner, _ in
                .and(Atom.silentRound.formula, .yesterday(.since(.not(Atom.textTurn.formula), inner)))
            }
        }
    }

    private var monitors = Property.allCases.map { ($0, PastMonitor(Self.formula($0))) }
    private var lastCall: String?
    private var rejectedTool: String?

    public init() {}

    public mutating func feed(_ message: ChatMessageInput, at index: Int) -> [Violation] {
        let text = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch message.role {
        case .system:
            return []
        case .user:
            return step(Atom.textTurn.bit, at: index, tool: "")
        case .tool:
            return step(Self.isError(text) ? Atom.toolError.bit : 0, at: index, tool: message.toolName ?? "")
        case .assistant:
            return assistant(text: text, calls: message.toolCalls ?? [], at: index)
        }
    }

    private mutating func assistant(text: String, calls: [ToolCallOutput], at index: Int) -> [Violation] {
        var atoms: UInt64 = text.isEmpty ? (calls.isEmpty ? 0 : Atom.silentRound.bit) : Atom.textTurn.bit
        if let rejected = Self.rejectedTool(in: text) {
            rejectedTool = rejected
            atoms |= Atom.rejection.bit
        }
        var violations = step(atoms, at: index, tool: "")
        for call in calls {
            let key = call.name + "\u{0}" + Self.canonical(call.arguments)
            var callAtoms = Atom.call.bit
            if key == lastCall { callAtoms |= Atom.repeatsCall.bit }
            if call.name == rejectedTool { callAtoms |= Atom.retriesRejected.bit }
            lastCall = key
            violations += step(callAtoms, at: index, tool: call.name)
        }
        return violations
    }

    private mutating func step(_ atoms: UInt64, at index: Int, tool: String) -> [Violation] {
        var violations: [Violation] = []
        for slot in monitors.indices where monitors[slot].1.step(atoms) {
            violations.append(Violation(property: monitors[slot].0, messageIndex: index, tool: tool))
        }
        return violations
    }

    static func isError(_ text: String) -> Bool {
        text.hasPrefix("Error") || text.contains("<tool_use_error>")
    }

    static func rejectedTool(in text: String) -> String? {
        guard let start = text.range(of: "[tool call to ", options: .backwards),
              let end = text.range(of: " rejected: ", range: start.upperBound..<text.endIndex)
        else { return nil }
        return String(text[start.upperBound..<end.lowerBound])
    }

    static func canonical(_ arguments: String) -> String {
        guard let object = try? JSONSerialization.jsonObject(with: Data(arguments.utf8), options: .fragmentsAllowed),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .fragmentsAllowed])
        else { return arguments }
        return String(decoding: data, as: UTF8.self)
    }

    public static func check(_ messages: [ChatMessageInput]) -> Set<Violation> {
        var monitor = SessionTemporalCheck()
        return Set(messages.enumerated().flatMap { monitor.feed($1, at: $0) })
    }

    public static func check(_ lines: [SessionLogReader.Line]) -> Set<Violation> {
        let requests = lines.filter { $0.type == "request" && $0.content != nil }
        let parents = Set(requests.compactMap(\.content?.parent))
        return requests.compactMap(\.id).filter { !parents.contains($0) }.reduce(into: []) { found, id in
            guard var messages = SessionLogReader.conversation(of: id, in: lines) else { return }
            if let output = lines.last(where: { $0.type == "response" && $0.requestId == id })?.output {
                messages.append(ChatMessageInput(role: .assistant, text: output.text ?? "", toolCalls: output.toolCalls))
            }
            found.formUnion(check(messages))
        }
    }
}
