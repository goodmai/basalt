public struct ToolCallGate: Sendable {
    private let declared: Set<String>
    private let forced: String?
    private let parallel: Bool
    private var admitted = 0

    public init(request: GenerationRequest) {
        declared = Set(request.tools?.map(\.name) ?? [])
        forced = request.toolChoice?.forcedTool
        parallel = request.toolChoice?.parallel ?? true
    }

    public mutating func admit(_ call: ToolCallOutput) -> StreamChunk? {
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

    static func rejection(_ name: String, reason: String) -> StreamChunk {
        .text("\n[tool call to \(name) rejected: \(reason)]")
    }
}
