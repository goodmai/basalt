import Foundation

public struct ToolCallGate: Sendable {
    private let declared: Set<String>
    private let forced: String?
    private let parallel: Bool
    private let callsDisabled: Bool
    private var admitted = 0

    public init(request: GenerationRequest) {
        declared = Set(request.tools?.map(\.name) ?? [])
        forced = request.toolChoice?.forcedTool
        parallel = request.toolChoice?.parallel ?? true
        callsDisabled = request.toolChoice?.callsDisabled ?? false
    }

    public var isSaturated: Bool { !parallel && admitted > 0 }

    public mutating func admit(_ call: ToolCallOutput) -> StreamChunk? {
        if callsDisabled { return Self.rejection(call.name, reason: "tool_choice is none") }
        guard declared.contains(call.name) else { return Self.rejection(call.name, reason: "undeclared") }
        if let forced, forced != call.name { return Self.rejection(call.name, reason: "tool_choice is \(forced)") }
        guard parallel || admitted == 0 else { return nil }
        admitted += 1
        return .toolCall(call)
    }

    public mutating func admit(_ chunk: StreamChunk) -> StreamChunk? {
        guard case .toolCall(let call) = chunk else { return chunk }
        return admit(call)
    }

    public mutating func admit(_ chunks: [StreamChunk]) -> [StreamChunk] {
        var admittedChunks: [StreamChunk] = []
        for chunk in chunks {
            guard !isSaturated else { break }
            if let admittedChunk = admit(chunk) { admittedChunks.append(admittedChunk) }
        }
        return admittedChunks
    }

    static func rejection(_ name: String, reason: String) -> StreamChunk {
        .text("\n[tool call to \(name) rejected: \(reason)]")
    }
}

public struct NemotronToolCallScanner: Sendable {
    static let opening = "<TOOLCALL>"
    static let closing = "</TOOLCALL>"
    private var pending = ""

    public init() {}

    public mutating func feed(_ text: String) -> [StreamChunk] {
        pending += text
        var chunks: [StreamChunk] = []
        while let start = pending.range(of: Self.opening) {
            appendText(String(pending[..<start.lowerBound]), to: &chunks)
            guard let end = pending.range(of: Self.closing, range: start.upperBound..<pending.endIndex) else {
                pending = String(pending[start.lowerBound...])
                return chunks
            }
            let body = String(pending[start.upperBound..<end.lowerBound])
            chunks += Self.calls(in: body)?.map(StreamChunk.toolCall) ?? [.text(Self.opening + body + Self.closing)]
            pending = String(pending[end.upperBound...])
        }
        let held = (1..<Self.opening.count).last { pending.hasSuffix(Self.opening.prefix($0)) } ?? 0
        appendText(String(pending.dropLast(held)), to: &chunks)
        pending = String(pending.suffix(held))
        return chunks
    }

    public mutating func finish() -> [StreamChunk] {
        defer { pending = "" }
        guard pending.hasPrefix(Self.opening) else { return pending.isEmpty ? [] : [.text(pending)] }
        return Self.calls(in: String(pending.dropFirst(Self.opening.count)))?.map(StreamChunk.toolCall) ?? [.text(pending)]
    }

    private func appendText(_ text: String, to chunks: inout [StreamChunk]) {
        if !text.isEmpty { chunks.append(.text(text)) }
    }

    static func calls(in body: String) -> [ToolCallOutput]? {
        guard let json = try? JSONSerialization.jsonObject(with: Data(body.utf8)),
              let objects = (json as? [[String: Any]]) ?? (json as? [String: Any]).map({ [$0] }),
              !objects.isEmpty else { return nil }
        var calls: [ToolCallOutput] = []
        for object in objects {
            guard let name = object["name"] as? String, let arguments = Self.arguments(object["arguments"]) else { return nil }
            calls.append(ToolCallOutput(id: "call_\(UUID().uuidString.prefix(24))", name: name, arguments: arguments))
        }
        return calls
    }

    private static func arguments(_ value: Any?) -> String? {
        switch value {
        case nil:
            return "{}"
        case let text as String:
            return (try? JSONSerialization.jsonObject(with: Data(text.utf8))) is [String: Any] ? text : encoded(["input": text])
        case let object as [String: Any]:
            return encoded(object)
        default:
            return nil
        }
    }

    private static func encoded(_ object: [String: Any]) -> String? {
        (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]))
            .flatMap { String(data: $0, encoding: .utf8) }
    }
}
