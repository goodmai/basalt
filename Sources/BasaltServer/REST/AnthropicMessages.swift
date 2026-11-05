import Foundation
import BasaltCore

extension AnthropicController {
    struct ContentBlock: Codable, Sendable {
        let type: String
        var text: String? = nil
        var id: String? = nil
        var name: String? = nil
        var input: AnyCodable? = nil
        var toolUseId: String? = nil
        var content: MessageContent? = nil
        var isError: Bool? = nil
        var thinking: String? = nil
        var signature: String? = nil
        var source: ImageSource? = nil

        init(type: String = "text", text: String) {
            self.type = type
            self.text = text
        }

        init(thinking: String) {
            type = "thinking"
            self.thinking = thinking
            signature = ""
        }

        init(toolUse call: ToolCallOutput, input: AnyCodable?) {
            type = "tool_use"
            id = AnthropicController.toolUseID(call.id)
            name = call.name
            self.input = input ?? AnyCodable([String: Any]())
        }

        var promptText: String? {
            type == "image" ? ToolTranscript.imageOmitted : text
        }

        func image() throws(BasaltError) -> ImageInput {
            switch source?.type {
            case "base64": return try ImageInput(base64: source?.data ?? "", mime: source?.mediaType ?? "")
            case "url": throw .invalidRequestStructure(details: ImageInput.remoteRefusal)
            default: throw .invalidRequestStructure(details: "an image block needs a source of type base64")
            }
        }

        var inputJSON: String {
            input.flatMap { try? Self.sortedEncoder.encode($0) }.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        }

        private static let sortedEncoder: JSONEncoder = {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            return encoder
        }()

        enum CodingKeys: String, CodingKey {
            case type, text, id, name, input, content, thinking, signature, source
            case toolUseId = "tool_use_id"
            case isError = "is_error"
        }

        func toolResult(toolNames: [String: String]) -> ChatMessageInput {
            let output = content?.textValue ?? ""
            return ChatMessageInput(
                role: .tool, text: isError == true ? "Error: \(output)" : output,
                toolCallId: toolUseId, toolName: toolUseId.flatMap { toolNames[$0] })
        }
    }

    struct ImageSource: Codable, Sendable {
        let type: String
        var mediaType: String? = nil
        var data: String? = nil

        enum CodingKeys: String, CodingKey {
            case type, data
            case mediaType = "media_type"
        }
    }

    struct Message: Codable, Sendable {
        let role: String
        let content: MessageContent
    }

    enum MessageContent: Codable, Sendable {
        case text(String)
        case blocks([ContentBlock])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .text(str)
            } else {
                let blocks = try container.decode([ContentBlock].self)
                self = .blocks(blocks)
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .text(let str): try container.encode(str)
            case .blocks(let b): try container.encode(b)
            }
        }

        var contentBlocks: [ContentBlock] {
            switch self {
            case .text(let s):   [ContentBlock(text: s)]
            case .blocks(let b): b
            }
        }

        var textValue: String {
            switch self {
            case .text(let s): return s
            case .blocks(let b): return b.compactMap(\.promptText).joined(separator: "\n")
            }
        }
    }

    enum SystemPrompt: Codable, Sendable {
        case text(String)
        case blocks([ContentBlock])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let str = try? c.decode(String.self) {
                self = .text(str)
            } else {
                self = .blocks((try? c.decode([ContentBlock].self)) ?? [])
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .text(let s):   try c.encode(s)
            case .blocks(let b): try c.encode(b)
            }
        }

        var textValue: String {
            switch self {
            case .text(let s):   return s
            case .blocks(let b): return b.compactMap(\.promptText).joined(separator: "\n")
            }
        }
    }
}

extension AnthropicController {
    struct MessagesRequest: Codable, Sendable {
        let model: String?
        let messages: [Message]
        let system: SystemPrompt?
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let minP: Double?
        let dynamicTemperature: Bool?
        let stream: Bool?
        let thinking: ThinkingConfig?
        let reasoningEffort: String?
        let topK: Int?
        let speculativeLookup: Bool?
        let grammarJumpForward: Bool?

        enum CodingKeys: String, CodingKey {
            case model, messages, system, temperature, stream, thinking
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case minP = "min_p"
            case dynamicTemperature = "dynamic_temperature"
            case reasoningEffort = "reasoning_effort"
            case topK = "top_k"
            case speculativeLookup = "speculative_lookup"
            case grammarJumpForward = "grammar_jump_forward"
        }

        var effectiveMaxTokens: Int { maxTokens ?? 8192 }

        var thinkingBudget: Int? { thinking?.type == "enabled" ? thinking?.budgetTokens : nil }

        var thinkingBudgetProblem: String? {
            guard let budget = thinkingBudget, budget < 1 || budget >= effectiveMaxTokens else { return nil }
            return "thinking.budget_tokens must be >= 1 and less than max_tokens (\(effectiveMaxTokens))"
        }

        var effectiveReasoningEffort: String? {
            if let reasoningEffort { return reasoningEffort }
            guard let thinking else { return nil }
            if thinking.type == "disabled" { return "off" }
            return thinking.budgetTokens.map { $0 > 8000 ? "high" : ($0 > 2000 ? "medium" : "low") } ?? "medium"
        }

        func sampling(_ antiRepetition: ChatPipeline.AntiRepetition) -> ChatPipeline.Sampling {
            ChatPipeline.Sampling(
                temperature: temperature, topP: topP, minP: minP, dynamicTemperature: dynamicTemperature, topK: topK,
                speculativeLookup: speculativeLookup, grammarJumpForward: grammarJumpForward, antiRepetition: antiRepetition)
        }
    }

    struct ThinkingConfig: Codable, Sendable {
        let type: String
        let budgetTokens: Int?

        enum CodingKeys: String, CodingKey {
            case type
            case budgetTokens = "budget_tokens"
        }
    }

    struct MessagesResponse: Codable, Sendable {
        let id: String
        let type: String
        let role: String
        let content: [ContentBlock]
        let model: String
        let stopReason: String
        let stopSequence: String?
        let usage: Usage

        struct Usage: Codable, Sendable {
            let inputTokens: Int
            let cacheReadInputTokens: Int
            let outputTokens: Int

            init(_ result: GenerationResponse) {
                inputTokens = result.promptTokens - result.cachedPromptTokens
                cacheReadInputTokens = result.cachedPromptTokens
                outputTokens = result.completionTokens
            }

            enum CodingKeys: String, CodingKey {
                case inputTokens = "input_tokens"
                case cacheReadInputTokens = "cache_read_input_tokens"
                case outputTokens = "output_tokens"
            }
        }

        enum CodingKeys: String, CodingKey {
            case id, type, role, content, model, usage
            case stopReason = "stop_reason"
            case stopSequence = "stop_sequence"
        }
    }
}

extension AnthropicController.MessagesResponse {
    init(result: GenerationResponse, model: String) {
        let calls = result.toolCalls ?? []
        self = Self(
            id: AnthropicController.messageID(), type: "message", role: "assistant",
            content: AnthropicController.responseContent(reasoning: result.reasoningText, text: result.generatedText, toolCalls: calls),
            model: model, stopReason: AnthropicController.stopReason(result.finishReason, calledTools: !calls.isEmpty),
            stopSequence: nil, usage: Usage(result))
    }
}

extension AnthropicController {
    static func messageID() -> String {
        "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
    }

    static func objectInput(_ arguments: String) -> AnyCodable? {
        let decoded = try? JSONDecoder().decode(AnyCodable.self, from: Data(arguments.utf8))
        return decoded?.rawValue is [String: Any] ? decoded : nil
    }

    static func nonObjectNote(_ call: ToolCallOutput) -> String {
        "\n[tool call to \(call.name): arguments are not an object, sent as {}: \(call.arguments)]"
    }

    static func toolUseID(_ id: String) -> String {
        id.hasPrefix("toolu_") ? id : "toolu_\(id.hasPrefix("call_") ? id.dropFirst(5) : Substring(id))"
    }

    static func stopReason(_ finish: GenerationResponse.FinishReason, calledTools: Bool) -> String {
        calledTools ? "tool_use" : (finish.isTruncated ? "max_tokens" : "end_turn")
    }

    static func responseContent(reasoning: String? = nil, text answer: String, toolCalls: [ToolCallOutput]) -> [ContentBlock] {
        let inputs = toolCalls.map { objectInput($0.arguments) }
        let text = answer + zip(toolCalls, inputs).filter { $0.1 == nil }.map { nonObjectNote($0.0) }.joined()
        let keepText = toolCalls.isEmpty || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return (reasoning.map { [ContentBlock(thinking: $0)] } ?? []) + (keepText ? [ContentBlock(text: text)] : [])
            + zip(toolCalls, inputs).map { ContentBlock(toolUse: $0.0, input: $0.1) }
    }

    static func chatMessages(system: SystemPrompt?, messages: [Message]) throws(BasaltError) -> [ChatMessageInput] {
        let toolNames = toolNames(in: messages)
        var chat: [ChatMessageInput] = []
        if let text = system?.textValue, !text.isEmpty {
            chat.append(ChatMessageInput(role: .system, text: text))
        }
        for message in messages { chat += try turns(for: message, toolNames: toolNames) }
        return chat
    }

    static func toolNames(in messages: [Message]) -> [String: String] {
        var toolNames: [String: String] = [:]
        for case .blocks(let blocks) in messages.map(\.content) {
            for block in blocks where block.type == "tool_use" {
                if let id = block.id, let name = block.name { toolNames[id] = name }
            }
        }
        return toolNames
    }

    static func turns(for message: Message, toolNames: [String: String]) throws(BasaltError) -> [ChatMessageInput] {
        let role: ChatMessageInput.Role = message.role == "assistant" ? .assistant : .user
        var turns: [ChatMessageInput] = []
        var text: [String] = []
        var calls: [ToolCallOutput] = []
        var images: [ImageInput] = []
        func flush() {
            guard !text.isEmpty || !calls.isEmpty || !images.isEmpty else { return }
            turns.append(ChatMessageInput(
                role: role, text: text.joined(separator: "\n"), toolCalls: calls.isEmpty ? nil : calls, images: images))
            text = []
            calls = []
            images = []
        }
        for block in message.content.contentBlocks {
            switch block.type {
            case "image":
                images.append(try block.image())
            case "tool_use":
                calls.append(ToolCallOutput(id: block.id ?? "", name: block.name ?? "tool", arguments: block.inputJSON))
            case "tool_result":
                flush()
                turns.append(block.toolResult(toolNames: toolNames))
            default:
                if let t = block.promptText { text.append(t) }
            }
        }
        flush()
        return turns.isEmpty ? [ChatMessageInput(role: role, text: "")] : turns
    }
}
