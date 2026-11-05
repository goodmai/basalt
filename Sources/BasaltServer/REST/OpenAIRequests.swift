import Foundation
import BasaltCore

extension OpenAIController {
    enum MessageContent: Codable, Sendable {
        case text(String)
        case blocks([ContentBlock])

        struct ContentBlock: Codable, Sendable {
            let type: String?
            let text: String?
            var imageUrl: ImageURL? = nil

            var isImage: Bool { type?.contains("image") == true }

            enum CodingKeys: String, CodingKey {
                case type, text
                case imageUrl = "image_url"
            }
        }

        struct ImageURL: Codable, Sendable {
            let url: String?

            enum CodingKeys: String, CodingKey {
                case url
            }

            init(from decoder: Decoder) throws {
                if let url = try? decoder.singleValueContainer().decode(String.self) {
                    self.url = url
                } else {
                    url = try? decoder.container(keyedBy: CodingKeys.self).decodeIfPresent(String.self, forKey: .url)
                }
            }
        }

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

        var textValue: String { text(imagesAs: ToolTranscript.imageOmitted) }

        func text(imagesAs placeholder: String?) -> String {
            switch self {
            case .text(let s):   s
            case .blocks(let b): b.compactMap { $0.isImage ? placeholder : $0.text }.joined(separator: "\n")
            }
        }

        func images() throws(BasaltError) -> [ImageInput] {
            guard case .blocks(let blocks) = self else { return [] }
            var images: [ImageInput] = []
            for block in blocks where block.isImage {
                guard let url = block.imageUrl?.url else {
                    throw .invalidRequestStructure(details: "an image content part needs an image_url")
                }
                images.append(try ImageInput(url: url))
            }
            return images
        }
    }

    struct ChatMessage: Codable, Sendable {
        let type: String?
        let role: String?
        let content: MessageContent?
        let name: String?
        let arguments: String?
        let input: String?
        let output: MessageContent?
        let toolCalls: [ChatToolCall]?
        var callId: String? = nil
        var toolCallId: String? = nil
        var reasoning: String? = nil

        init(role: String, content: MessageContent?, toolCalls: [ChatToolCall]?, reasoning: String?) {
            self.type = nil
            self.role = role
            self.content = content
            self.name = nil
            self.arguments = nil
            self.input = nil
            self.output = nil
            self.toolCalls = toolCalls
            self.reasoning = reasoning
        }

        enum CodingKeys: String, CodingKey {
            case type, role, content, name, arguments, input, output
            case toolCalls = "tool_calls"
            case callId = "call_id"
            case toolCallId = "tool_call_id"
            case reasoning
        }

        enum ReasoningContentKey: String, CodingKey {
            case reasoningContent = "reasoning_content"
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encodeIfPresent(type, forKey: .type)
            try c.encodeIfPresent(role, forKey: .role)
            try c.encode(content, forKey: .content)
            try c.encodeIfPresent(reasoning, forKey: .reasoning)
            var compatible = encoder.container(keyedBy: ReasoningContentKey.self)
            try compatible.encodeIfPresent(reasoning, forKey: .reasoningContent)
            try c.encodeIfPresent(name, forKey: .name)
            try c.encodeIfPresent(arguments, forKey: .arguments)
            try c.encodeIfPresent(input, forKey: .input)
            try c.encodeIfPresent(output, forKey: .output)
            try c.encodeIfPresent(toolCalls, forKey: .toolCalls)
            try c.encodeIfPresent(callId, forKey: .callId)
            try c.encodeIfPresent(toolCallId, forKey: .toolCallId)
        }
    }

    struct ChatToolCall: Codable, Sendable {
        var id: String?
        var type: String? = "function"
        let function: Function

        struct Function: Codable, Sendable {
            let name: String
            let arguments: String?
        }

        init(_ call: ToolCallOutput) {
            id = call.id
            function = Function(name: call.name, arguments: call.arguments)
        }
    }
}

extension OpenAIController {
    struct ChatCompletionRequest: Codable, Sendable {
        let model: String?
        let messages: [ChatMessage]
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let minP: Double?
        let dynamicTemperature: Bool?
        let stream: Bool?
        let reasoningEffort: String?
        let topK: Int?
        let presencePenalty: Double?
        let repetitionPenalty: Double?
        let thinkingBudget: Int?
        let speculativeLookup: Bool?
        let grammarJumpForward: Bool?
        let streamOptions: StreamOptions?

        struct StreamOptions: Codable, Sendable {
            let includeUsage: Bool?

            enum CodingKeys: String, CodingKey {
                case includeUsage = "include_usage"
            }
        }

        enum CodingKeys: String, CodingKey {
            case model, messages, temperature, stream
            case streamOptions = "stream_options"
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case minP = "min_p"
            case dynamicTemperature = "dynamic_temperature"
            case reasoningEffort = "reasoning_effort"
            case topK = "top_k"
            case presencePenalty = "presence_penalty"
            case repetitionPenalty = "repetition_penalty"
            case thinkingBudget = "thinking_budget"
            case speculativeLookup = "speculative_lookup"
            case grammarJumpForward = "grammar_jump_forward"
        }

        func sampling(_ antiRepetition: ChatPipeline.AntiRepetition) -> ChatPipeline.Sampling {
            ChatPipeline.Sampling(
                temperature: temperature, topP: topP, minP: minP, dynamicTemperature: dynamicTemperature, topK: topK,
                presencePenalty: presencePenalty, repetitionPenalty: repetitionPenalty,
                speculativeLookup: speculativeLookup, grammarJumpForward: grammarJumpForward, antiRepetition: antiRepetition)
        }
    }

    struct ResponsesRequest: Codable, Sendable {
        let model: String?
        let instructions: String?
        let input: ResponseInput?
        let messages: [ChatMessage]?
        let maxOutputTokens: Int?
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let minP: Double?
        let dynamicTemperature: Bool?
        let stream: Bool?
        let reasoning: ReasoningConfig?
        let reasoningEffort: String?
        let tools: [ResponseTool]?
        let additionalTools: [ResponseTool]?
        let topK: Int?
        let presencePenalty: Double?
        let repetitionPenalty: Double?
        let thinkingBudget: Int?
        let speculativeLookup: Bool?
        let grammarJumpForward: Bool?

        struct ReasoningConfig: Codable, Sendable {
            let effort: String?
        }

        var effectiveReasoningEffort: String? { reasoning?.effort ?? reasoningEffort }

        enum CodingKeys: String, CodingKey {
            case model, instructions, input, messages, temperature, stream, tools, reasoning
            case maxOutputTokens = "max_output_tokens"
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case minP = "min_p"
            case dynamicTemperature = "dynamic_temperature"
            case reasoningEffort = "reasoning_effort"
            case additionalTools = "additional_tools"
            case topK = "top_k"
            case presencePenalty = "presence_penalty"
            case repetitionPenalty = "repetition_penalty"
            case thinkingBudget = "thinking_budget"
            case speculativeLookup = "speculative_lookup"
            case grammarJumpForward = "grammar_jump_forward"
        }

        func conversation() throws(BasaltError) -> [ChatMessageInput] {
            switch input {
            case .text(let t):        t.isEmpty ? [] : [ChatMessageInput(role: .user, text: t)]
            case .messages(let msgs): try OpenAIController.chatMessages(msgs)
            case nil:                 try OpenAIController.chatMessages(messages ?? [])
            }
        }

        var instructionMessages: [ChatMessageInput] {
            instructions.flatMap { $0.isEmpty ? nil : [ChatMessageInput(role: .system, text: $0)] } ?? []
        }

        func sampling(_ antiRepetition: ChatPipeline.AntiRepetition) -> ChatPipeline.Sampling {
            ChatPipeline.Sampling(
                temperature: temperature, topP: topP, minP: minP, dynamicTemperature: dynamicTemperature, topK: topK,
                presencePenalty: presencePenalty, repetitionPenalty: repetitionPenalty,
                speculativeLookup: speculativeLookup, grammarJumpForward: grammarJumpForward, antiRepetition: antiRepetition)
        }
    }

    enum ResponseInput: Codable, Sendable {
        case text(String)
        case messages([ChatMessage])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .text(str)
                return
            }
            let msgs = try container.decode([ChatMessage].self)
            self = .messages(msgs)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .text(let s): try container.encode(s)
            case .messages(let m): try container.encode(m)
            }
        }
    }

    struct ResponseTool: Codable, Sendable {
        let type: String
        let name: String?
        let description: String?
    }
}

extension OpenAIController {
    struct RequestedTools {
        var tools: [ToolDefinition]?
        var choice: ToolChoice?
    }

    struct UndeclaredForcedTool: Error {
        let name: String
        var message: String { "tool_choice names \(name), which is not in tools" }
    }

    static func requestedTools(fromRequestBody data: Data) throws(UndeclaredForcedTool) -> RequestedTools {
        guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return RequestedTools() }
        let choice = body["tool_choice"] as? [String: Any]
        let forced = ((choice?["function"] as? [String: Any])?["name"] as? String) ?? (choice?["name"] as? String)
        let parallel = body["parallel_tool_calls"] as? Bool != false && choice?["disable_parallel_tool_use"] as? Bool != true
        let raw = ((body["tools"] as? [[String: Any]]) ?? []) + ((body["additional_tools"] as? [[String: Any]]) ?? [])
        let tools: [ToolDefinition] = raw.compactMap { entry in
            guard entry["type"] as? String == "function" || entry["input_schema"] != nil else { return nil }
            let spec = (entry["function"] as? [String: Any]) ?? entry
            guard let name = spec["name"] as? String, !name.isEmpty else { return nil }
            let parameters = (spec["parameters"] ?? spec["input_schema"])
                .flatMap { try? JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys, .withoutEscapingSlashes]) }
                .flatMap { String(data: $0, encoding: .utf8) }
            return ToolDefinition(name: name, description: spec["description"] as? String, parametersJSON: parameters)
        }
        let offered = tools.isEmpty ? nil : tools
        if ((body["tool_choice"] as? String) ?? (choice?["type"] as? String)) == "none" {
            return RequestedTools(tools: offered, choice: ToolChoice(callsDisabled: true))
        }
        if let forced, !tools.contains(where: { $0.name == forced }) { throw UndeclaredForcedTool(name: forced) }
        return RequestedTools(tools: offered,
                              choice: forced == nil && parallel ? nil : ToolChoice(forcedTool: forced, parallel: parallel))
    }

    struct InvalidResponseFormat: Error {
        let message: String
    }

    static func responseFormat(fromRequestBody data: Data) throws(InvalidResponseFormat) -> ResponseFormat? {
        guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let format = (body["response_format"] as? [String: Any])
                ?? ((body["text"] as? [String: Any])?["format"] as? [String: Any]) else { return nil }
        switch format["type"] as? String {
        case "text":
            return nil
        case "json_object":
            return .jsonObject
        case "json_schema":
            let declaration = (format["json_schema"] as? [String: Any]) ?? format
            let ordered = OrderedJSON.parse(data, keeping: ["response_format", "text"])
            let orderedFormat = ordered?["response_format"] ?? ordered?["text"]?["format"]
            guard declaration["schema"] != nil, let text = (orderedFormat?["json_schema"] ?? orderedFormat)?["schema"]?.text else {
                throw InvalidResponseFormat(message: "response_format json_schema needs a schema")
            }
            let requested = ResponseFormat.jsonSchema(text, strict: declaration["strict"] as? Bool == true)
            let prepared: PreparedJSONSchema
            do {
                prepared = try requested.prepared()
            } catch {
                throw InvalidResponseFormat(message: "response_format: \(error.message)")
            }
            if !prepared.relaxations.isEmpty {
                BasaltLogger(module: "OpenAIController").info(
                    "response_format without strict: \(prepared.relaxations.joined(separator: "; "))")
            }
            return requested
        case let type:
            throw InvalidResponseFormat(message:
                "response_format type \(type.map { "'\($0)' is not supported" } ?? "is missing"); expected text, json_object or json_schema")
        }
    }

    static func chatMessages(_ messages: [ChatMessage]) throws(BasaltError) -> [ChatMessageInput] {
        let toolNames = toolNames(in: messages)
        var chat: [ChatMessageInput] = []
        for msg in messages {
            switch msg.type {
            case "function_call", "custom_tool_call":
                append(ToolCallOutput(id: msg.callId ?? "", name: msg.name ?? "tool", arguments: msg.arguments ?? msg.input ?? "{}"),
                       to: &chat)
            case "function_call_output", "custom_tool_call_output":
                chat.append(ChatMessageInput(role: .tool, text: msg.output?.textValue ?? "",
                                             toolCallId: msg.callId, toolName: msg.callId.flatMap { toolNames[$0] }))
            case "reasoning":
                continue
            default:
                if let input = try msg.chatInput(toolNames: toolNames) { chat.append(input) }
            }
        }
        return chat
    }

    static func toolNames(in messages: [ChatMessage]) -> [String: String] {
        var toolNames: [String: String] = [:]
        for msg in messages {
            if let id = msg.callId, let name = msg.name { toolNames[id] = name }
            for call in msg.toolCalls ?? [] { if let id = call.id { toolNames[id] = call.function.name } }
        }
        return toolNames
    }

    private static func append(_ call: ToolCallOutput, to chat: inout [ChatMessageInput]) {
        if let last = chat.last, last.role == .assistant {
            chat[chat.count - 1] = ChatMessageInput(role: .assistant, text: last.text, toolCalls: (last.toolCalls ?? []) + [call])
        } else {
            chat.append(ChatMessageInput(role: .assistant, text: "", toolCalls: [call]))
        }
    }
}

extension OpenAIController.ChatMessage {
    var chatRole: ChatMessageInput.Role {
        switch role {
        case "system", "developer": .system
        case "assistant":           .assistant
        case "tool":                .tool
        default:                    .user
        }
    }

    func chatInput(toolNames: [String: String]) throws(BasaltError) -> ChatMessageInput? {
        let role = chatRole
        if role == .tool {
            return ChatMessageInput(
                role: .tool, text: content?.textValue ?? "", toolCallId: toolCallId, toolName: toolCallId.flatMap { toolNames[$0] })
        }
        let text = content?.text(imagesAs: nil) ?? ""
        let images = try content?.images() ?? []
        let calls = (toolCalls ?? []).map {
            ToolCallOutput(id: $0.id ?? "", name: $0.function.name, arguments: $0.function.arguments ?? "{}")
        }
        guard !text.isEmpty || !calls.isEmpty || !images.isEmpty else { return nil }
        return ChatMessageInput(role: role, text: text, toolCalls: calls.isEmpty ? nil : calls, images: images)
    }
}
