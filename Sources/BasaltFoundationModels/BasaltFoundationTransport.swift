import BasaltAppCore
import BasaltCore
import Foundation

public struct BasaltFoundationRequest: Sendable, Equatable {
    public var model: String?
    public var messages: [ChatMessageInput]
    public var tools: [ToolDefinition]
    public var toolChoice: ToolChoice?
    public var responseFormat: ResponseFormat?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var seed: UInt64?
    public var maxTokens: Int?
    public var enableThinking: Bool?

    public init(messages: [ChatMessageInput], model: String? = nil) {
        self.model = model
        self.messages = messages
        self.tools = []
    }

    public func toGenerationRequest() -> GenerationRequest {
        let promptText = messages.last?.text ?? ChatMessageInput.plainText(messages)
        var request = GenerationRequest(
            prompt: promptText.isEmpty ? " " : promptText,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            tools: tools.isEmpty ? nil : tools,
            messages: messages.isEmpty ? nil : messages
        )
        request.toolChoice = toolChoice
        request.responseFormat = responseFormat
        request.topK = topK
        request.enableThinking = enableThinking
        return request
    }

    public func httpURLRequest(for endpoint: EngineEndpoint) throws(BasaltError) -> URLRequest {
        let body = try openAIJSONBody()
        var request = URLRequest(url: endpoint.base.appending(path: "v1/chat/completions"))
        request.httpMethod = "POST"
        request.httpBody = body
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token = endpoint.token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    public func openAIJSONBody() throws(BasaltError) -> Data {
        var payload: [String: Any] = [
            "messages": messages.map(Self.serializeMessage),
            "stream": true,
            "stream_options": ["include_usage": true],
        ]
        if let model, !model.isEmpty { payload["model"] = model }
        if let temperature { payload["temperature"] = temperature }
        if let topP { payload["top_p"] = topP }
        if let topK { payload["top_k"] = topK }
        if let seed { payload["seed"] = seed }
        if let maxTokens { payload["max_tokens"] = maxTokens }
        if let enableThinking {
            payload["enable_thinking"] = enableThinking
            if !enableThinking { payload["reasoning_effort"] = "none" }
        }
        if !tools.isEmpty {
            payload["tools"] = try tools.map(Self.serializeTool)
        }
        if let choiceValue = Self.serializeToolChoice(toolChoice) {
            payload["tool_choice"] = choiceValue
        }
        if let responseFormat {
            payload["response_format"] = try Self.serializeResponseFormat(responseFormat)
        }
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        else {
            throw .invalidRequestStructure(details: "failed to encode /v1/chat/completions request body")
        }
        return data
    }
}

extension BasaltFoundationRequest {
    private static func serializeMessage(_ message: ChatMessageInput) -> [String: Any] {
        var dict: [String: Any] = ["role": message.role.rawValue]
        if message.images.isEmpty {
            dict["content"] = message.text
        } else {
            var blocks: [[String: Any]] = []
            if !message.text.isEmpty {
                blocks.append(["type": "text", "text": message.text])
            }
            for image in message.images {
                let uri = "data:\(image.mime);base64,\(image.data.base64EncodedString())"
                blocks.append(["type": "image_url", "image_url": ["url": uri]])
            }
            dict["content"] = blocks
        }
        if let calls = message.toolCalls, !calls.isEmpty {
            dict["tool_calls"] = calls.map { call in
                [
                    "id": call.id,
                    "type": "function",
                    "function": ["name": call.name, "arguments": call.arguments],
                ] as [String: Any]
            }
        }
        if let toolCallId = message.toolCallId { dict["tool_call_id"] = toolCallId }
        if let toolName = message.toolName { dict["name"] = toolName }
        return dict
    }

    private static func serializeTool(_ tool: ToolDefinition) throws(BasaltError) -> [String: Any] {
        var function: [String: Any] = ["name": tool.name]
        if let description = tool.description { function["description"] = description }
        if let paramsJSON = tool.parametersJSON, !paramsJSON.isEmpty {
            guard let object = try? JSONSerialization.jsonObject(with: Data(paramsJSON.utf8)) else {
                throw .invalidRequestStructure(details: "tool '\(tool.name)' has invalid JSON parameters")
            }
            function["parameters"] = object
        } else {
            function["parameters"] = ["type": "object", "properties": [String: Any]()]
        }
        return ["type": "function", "function": function]
    }

    private static func serializeToolChoice(_ choice: ToolChoice?) -> Any? {
        guard let choice else { return nil }
        if choice.callsDisabled { return "none" }
        if let forced = choice.forcedTool {
            return ["type": "function", "function": ["name": forced]]
        }
        return nil
    }

    private static func serializeResponseFormat(_ format: ResponseFormat) throws(BasaltError) -> [String: Any] {
        switch format {
        case .jsonObject:
            return ["type": "json_object"]
        case .jsonSchema(let rawSchema, let strict):
            guard let schemaObject = try? JSONSerialization.jsonObject(with: Data(rawSchema.utf8)) else {
                throw .invalidRequestStructure(details: "responseFormat jsonSchema is not valid JSON")
            }
            return [
                "type": "json_schema",
                "json_schema": [
                    "name": "basalt_structured_output",
                    "strict": strict,
                    "schema": schemaObject,
                ] as [String: Any],
            ]
        case .regex(let pattern):
            return ["type": "regex", "regex": pattern]
        case .grammar(let source):
            return ["type": "grammar", "grammar": source]
        case .structuralTag(let tagJSON):
            guard let object = (try? JSONSerialization.jsonObject(with: Data(tagJSON.utf8))) as? [String: Any] else {
                throw .invalidRequestStructure(details: "responseFormat structuralTag is not a JSON object")
            }
            return object
        }
    }
}

public struct BasaltFoundationResponse: Sendable, Equatable {
    public var text: String
    public var reasoning: String
    public var toolCalls: [ToolCallOutput]
    public var promptTokens: Int
    public var completionTokens: Int

    public init(
        text: String = "",
        reasoning: String = "",
        toolCalls: [ToolCallOutput] = [],
        promptTokens: Int = 0,
        completionTokens: Int = 0
    ) {
        self.text = text
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

public struct BasaltFoundationTransport: Sendable {
    public var stream: @Sendable (BasaltFoundationRequest) -> AsyncThrowingStream<ChatDelta, any Error>
    public var health: @Sendable () async -> Bool

    public init(
        stream: @escaping @Sendable (BasaltFoundationRequest) -> AsyncThrowingStream<ChatDelta, any Error>,
        health: @escaping @Sendable () async -> Bool = { true }
    ) {
        self.stream = stream
        self.health = health
    }

    public static func rest(
        endpoint: EngineEndpoint,
        model: String? = nil,
        session: URLSession = .engine
    ) -> BasaltFoundationTransport {
        BasaltFoundationTransport(
            stream: { incoming in
                var effective = incoming
                if effective.model == nil { effective.model = model }
                return restStream(endpoint: endpoint, request: effective, session: session)
            },
            health: {
                var probe = URLRequest(url: endpoint.base.appending(path: "api/v1/health"))
                if let token = endpoint.token, !token.isEmpty {
                    probe.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                }
                guard let (_, response) = try? await session.data(for: probe) else { return false }
                return (response as? HTTPURLResponse)?.statusCode == 200
            }
        )
    }

    public static func inProcess(engine: any InferenceEngine) -> BasaltFoundationTransport {
        BasaltFoundationTransport(
            stream: { request in inProcessStream(engine: engine, request: request) },
            health: { await engine.isLoaded }
        )
    }

    public func collect(_ request: BasaltFoundationRequest) async throws -> BasaltFoundationResponse {
        var response = BasaltFoundationResponse()
        for try await delta in stream(request) {
            apply(delta, into: &response)
        }
        if response.toolCalls.isEmpty, !request.tools.isEmpty, request.toolChoice?.callsDisabled != true {
            let declared = Set(request.tools.map(\.name))
            let extracted = BasaltFoundationSchemaAdapter.parseEmbeddedToolCalls(
                from: response.text,
                declaredTools: declared
            )
            if !extracted.isEmpty {
                response.toolCalls = extracted
                response.text = ""
            }
        }
        return response
    }
}

extension BasaltFoundationTransport {
    private func apply(_ delta: ChatDelta, into response: inout BasaltFoundationResponse) {
        switch delta {
        case .text(let chunk):
            response.text += chunk
        case .reasoning(let chunk):
            response.reasoning += chunk
        case .toolCall(let incoming):
            mergeToolCall(incoming, into: &response.toolCalls)
        case .usage(let prompt, let completion):
            response.promptTokens = prompt
            response.completionTokens = completion
        case .finish:
            break
        }
    }

    private func mergeToolCall(_ incoming: ToolCallOutput, into calls: inout [ToolCallOutput]) {
        if !incoming.id.isEmpty, let index = calls.firstIndex(where: { $0.id == incoming.id }) {
            let existing = calls[index]
            let mergedName = existing.name.isEmpty ? incoming.name : existing.name
            calls[index] = ToolCallOutput(
                id: existing.id,
                name: mergedName,
                arguments: existing.arguments + incoming.arguments
            )
            return
        }
        if incoming.id.isEmpty, !calls.isEmpty, incoming.name.isEmpty {
            let lastIndex = calls.count - 1
            let existing = calls[lastIndex]
            calls[lastIndex] = ToolCallOutput(
                id: existing.id,
                name: existing.name,
                arguments: existing.arguments + incoming.arguments
            )
            return
        }
        let assignedID = incoming.id.isEmpty ? "call_\(calls.count)" : incoming.id
        calls.append(ToolCallOutput(id: assignedID, name: incoming.name, arguments: incoming.arguments))
    }

    private static func restStream(
        endpoint: EngineEndpoint,
        request: BasaltFoundationRequest,
        session: URLSession
    ) -> AsyncThrowingStream<ChatDelta, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let urlRequest = try request.httpURLRequest(for: endpoint)
                    let (bytes, response) = try await session.bytes(for: urlRequest)
                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        var bodyData = Data()
                        for try await byte in bytes.prefix(4096) { bodyData.append(byte) }
                        let message = String(decoding: bodyData.prefix(300), as: UTF8.self)
                        throw EngineError.refused(status: http.statusCode, message: message)
                    }
                    try await relaySSE(bytes: bytes, to: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: EngineError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func relaySSE(
        bytes: URLSession.AsyncBytes,
        to continuation: AsyncThrowingStream<ChatDelta, any Error>.Continuation
    ) async throws {
        var parser = SSEParser()
        var pending: [UInt8] = []
        var toolCallIDs: [Int: String] = [:]
        for try await byte in bytes {
            try Task.checkCancellation()
            pending.append(byte)
            guard byte == UInt8(ascii: "\n") else { continue }
            for event in parser.feed(pending) {
                switch try ChatDeltaDecoder.decode(event, toolCallIDs: &toolCallIDs) {
                case .done: return
                case .deltas(let deltas):
                    for delta in deltas { continuation.yield(delta) }
                }
            }
            pending.removeAll(keepingCapacity: true)
        }
        if !pending.isEmpty {
            pending.append(UInt8(ascii: "\n"))
        }
        pending.append(UInt8(ascii: "\n"))
        for event in parser.feed(pending) {
            switch try ChatDeltaDecoder.decode(event, toolCallIDs: &toolCallIDs) {
            case .done: return
            case .deltas(let deltas):
                for delta in deltas { continuation.yield(delta) }
            }
        }
        throw Task.isCancelled ? EngineError.cancelled : EngineError.transport("chat stream ended before [DONE]")
    }

    private static func inProcessStream(
        engine: any InferenceEngine,
        request: BasaltFoundationRequest
    ) -> AsyncThrowingStream<ChatDelta, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let stream = try await engine.generateStream(request: request.toGenerationRequest())
                    for await chunk in stream {
                        try Task.checkCancellation()
                        switch chunk {
                        case .text(let text):
                            continuation.yield(.text(text))
                        case .reasoning(let reasoning):
                            continuation.yield(.reasoning(reasoning))
                        case .toolCall(let call):
                            continuation.yield(.toolCall(call))
                        case .metadata(let meta):
                            continuation.yield(.usage(prompt: meta.promptTokens, completion: meta.completionTokens))
                            continuation.yield(.finish(meta.finishReason.rawValue))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
