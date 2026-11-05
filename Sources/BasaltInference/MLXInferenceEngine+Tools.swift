import Foundation
import CoreImage
import MLX
import MLXLMCommon
import Tokenizers
import Jinja
import BasaltCore

extension MLXInferenceEngine {
    public static func buildTemplateContext(
        request: GenerationRequest,
        defaultReasoningEffort: String? = nil,
        profile: (any ModelProfile)? = nil
    ) -> [String: any Sendable]? {
        let policy = reasoningPolicy(request: request, defaultReasoningEffort: defaultReasoningEffort)
        let context = (profile ?? GenericProfile()).templateContext(for: policy)
        return context.isEmpty ? nil : context
    }

    static func reasoningPolicy(request: GenerationRequest, defaultReasoningEffort: String?) -> ReasoningPolicy {
        request.enableThinking == false
            ? .disabled
            : ReasoningPolicy(rawValue: request.reasoningEffort ?? defaultReasoningEffort) ?? .modelDefault
    }
    static func chat(_ messages: [ChatMessageInput], lastText: String) -> [Chat.Message] {
        messages.enumerated().map { index, message in
            let text = index == messages.count - 1 ? lastText : message.text
            let images = message.images.compactMap { CIImage(data: $0.data) }.map(UserInput.Image.ciImage)
            switch message.role {
            case .system:    return .system(text, images: images)
            case .user:      return .user(text, images: images)
            case .assistant: return .assistant(text, images: images, toolCalls: message.toolCalls?.map(toolCall))
            case .tool:      return .tool(text, id: message.toolCallId, name: message.toolName)
            }
        }
    }

    static func chatInput(
        _ messages: [ChatMessageInput], normalisation: String, tools: [ToolSpec]?, context: [String: any Sendable]?
    ) -> (@Sendable (String) -> UserInput, String?, String) {
        let last = messages[messages.count - 1]
        return (
            { UserInput(chat: Self.chat(messages, lastText: $0), tools: tools, additionalContext: context) },
            HybridPrefixSnapshot.templateKey([tools ?? [], context ?? [:], last.role.rawValue, last.toolCalls != nil, normalisation]),
            last.text
        )
    }

    static func templateFallbacks(_ turns: [ChatMessageInput]) -> [(String, [ChatMessageInput])] {
        [("leadingSystem", ChatMessageInput.leadingSystem(turns)), ("alternating", ChatMessageInput.alternating(turns))]
    }

    static func briefly(_ error: any Error, limit: Int = 300) -> String {
        let text = String(describing: error)
        return text.count > limit ? String(text.prefix(limit)) + "…" : text
    }

    static func isTemplateRefusal(_ error: any Error) -> Bool {
        if error is TemplateException { return true }
        if case JinjaError.runtime = error { return true }
        return false
    }

    static func toolCall(_ call: ToolCallOutput) -> ToolCall {
        let arguments = (try? JSONDecoder().decode([String: JSONValue].self, from: Data(call.arguments.utf8)))
            ?? ["input": .string(call.arguments)]
        return ToolCall(function: .init(name: call.name, arguments: arguments), id: call.id.isEmpty ? nil : call.id)
    }

    static func toolSpecs(_ tools: [ToolDefinition]?) -> [ToolSpec]? {
        guard let tools, !tools.isEmpty else { return nil }
        return tools.map { tool in
            var function: [String: any Sendable] = ["name": tool.name]
            if let description = tool.description { function["description"] = description }
            if let json = tool.parametersJSON,
               let schema = try? JSONSerialization.jsonObject(with: Data(json.utf8)) {
                function["parameters"] = sendable(schema)
            }
            return ["type": "function", "function": function]
        }
    }

    private static func sendable(_ value: Any) -> any Sendable {
        switch value {
        case let dict as [String: Any]: return dict.mapValues { sendable($0) }
        case let array as [Any]:        return array.map { sendable($0) }
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue }
            return number.doubleValue.rounded() == number.doubleValue ? number.intValue : number.doubleValue
        case let string as String:      return string
        default:                        return String(describing: value)
        }
    }

    static func resolveRejectedToolCall(
        reason: String, toolName: String?, callID: String?,
        rawText: String, truncated: Bool, declaredTools: Set<String>
    ) -> StreamChunk {
        if reason == "malformed_syntax",
           let call = ToolCallRepairer.repairToolCall(
               name: toolName, callID: callID, rawText: rawText, truncated: truncated, declaredTools: declaredTools) {
            return .toolCall(call)
        }
        let tool = toolName.map { " to \($0)" } ?? ""
        return .text("\n[tool call\(tool) rejected: \(reason)]")
    }

    static func toolCallOutput(_ call: ToolCall) -> ToolCallOutput {
        let arguments = (try? JSONEncoder().encode(call.function.arguments)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return ToolCallOutput(id: "call_\(UUID().uuidString.prefix(24))", name: call.function.name, arguments: arguments)
    }
}
