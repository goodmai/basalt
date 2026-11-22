import BasaltAppCore
import BasaltCore
import CoreImage
import Foundation

#if canImport(FoundationModels, _version: 2)
import FoundationModels
#endif

public struct BasaltFoundationModelProvider: Sendable {
    public let modelID: String
    public let transport: BasaltFoundationTransport

    public init(modelID: String = "basalt-local", transport: BasaltFoundationTransport) {
        self.modelID = modelID
        self.transport = transport
    }

    public static func remote(
        endpoint: EngineEndpoint,
        modelID: String = "default",
        session: URLSession = .engine
    ) -> BasaltFoundationModelProvider {
        BasaltFoundationModelProvider(
            modelID: modelID,
            transport: .rest(endpoint: endpoint, model: modelID, session: session)
        )
    }

    public static func inProcess(
        engine: any InferenceEngine,
        modelID: String = "basalt-local"
    ) -> BasaltFoundationModelProvider {
        BasaltFoundationModelProvider(
            modelID: modelID,
            transport: .inProcess(engine: engine)
        )
    }

    public func checkAvailability() async -> Bool {
        await transport.health()
    }

    public func respond(_ request: BasaltFoundationRequest) async throws -> BasaltFoundationResponse {
        var effective = request
        if effective.model == nil { effective.model = modelID }
        return try await transport.collect(effective)
    }

    public func generateStructuredJSON(
        prompt: String,
        instructions: String? = nil,
        jsonSchema: String,
        strict: Bool = true,
        maxTokens: Int = 4096
    ) async throws -> String {
        let validated = try BasaltFoundationSchemaAdapter.validatedJSONSchema(jsonSchema, strict: strict)
        var messages: [ChatMessageInput] = []
        if let instructions, !instructions.isEmpty {
            messages.append(ChatMessageInput(role: .system, text: instructions))
        }
        messages.append(ChatMessageInput(role: .user, text: prompt))
        var request = BasaltFoundationRequest(messages: messages, model: modelID)
        request.responseFormat = .jsonSchema(validated, strict: strict)
        request.temperature = 0
        request.maxTokens = maxTokens
        request.enableThinking = false
        let response = try await transport.collect(request)
        return BasaltFoundationSchemaAdapter.stripMarkdownFence(response.text)
    }

    public func executeToolLoop(
        messages: [ChatMessageInput],
        tools: [ToolDefinition],
        maxRounds: Int = 4,
        handler: @Sendable (ToolCallOutput) async throws -> String
    ) async throws -> BasaltFoundationResponse {
        var history = messages
        let rounds = max(1, maxRounds)
        for _ in 0 ..< rounds {
            try Task.checkCancellation()
            var turnRequest = BasaltFoundationRequest(messages: history, model: modelID)
            turnRequest.tools = tools
            let reply = try await transport.collect(turnRequest)
            guard !reply.toolCalls.isEmpty else { return reply }
            history.append(
                ChatMessageInput(role: .assistant, text: reply.text, toolCalls: reply.toolCalls)
            )
            for call in reply.toolCalls {
                let output = try await handler(call)
                history.append(
                    ChatMessageInput(
                        role: .tool,
                        text: output,
                        toolCallId: call.id,
                        toolName: call.name
                    )
                )
            }
        }
        var finalRequest = BasaltFoundationRequest(messages: history, model: modelID)
        finalRequest.toolChoice = ToolChoice(callsDisabled: true)
        return try await transport.collect(finalRequest)
    }
}

#if canImport(FoundationModels, _version: 2)
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
public struct BasaltLanguageModel: FoundationModels.LanguageModel, Sendable {
    public let modelID: String
    public let capabilities: LanguageModelCapabilities
    public let transport: BasaltFoundationTransport

    public var executorConfiguration: Executor.Configuration {
        Executor.Configuration(modelID: modelID)
    }

    public init(
        modelID: String,
        capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration, .toolCalling, .reasoning],
        transport: BasaltFoundationTransport
    ) {
        self.modelID = modelID
        self.capabilities = LanguageModelCapabilities(capabilities)
        self.transport = transport
    }

    public init(
        endpoint: EngineEndpoint,
        modelID: String = "default",
        capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration, .toolCalling, .reasoning],
        session: URLSession = .engine
    ) {
        self.modelID = modelID
        self.capabilities = LanguageModelCapabilities(capabilities)
        self.transport = .rest(endpoint: endpoint, model: modelID, session: session)
    }

    public init(
        engine: any InferenceEngine,
        modelID: String = "basalt-local",
        capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration, .toolCalling, .reasoning]
    ) {
        self.modelID = modelID
        self.capabilities = LanguageModelCapabilities(capabilities)
        self.transport = .inProcess(engine: engine)
    }

    public struct Executor: FoundationModels.LanguageModelExecutor, Sendable {
        public struct Configuration: Hashable, Sendable {
            public let modelID: String

            public init(modelID: String) {
                self.modelID = modelID
            }
        }

        public let modelID: String

        public init(configuration: Configuration) throws {
            self.modelID = configuration.modelID
        }

        public func prewarm(model: BasaltLanguageModel, transcript: Transcript) {
            Task {
                _ = await model.transport.health()
            }
        }

        public func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: BasaltLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let messages = try BasaltTranscriptBridge.messages(
                from: request.transcript,
                supportsVision: model.capabilities.contains(.vision)
            )
            let foundationRequest = try BasaltTranscriptBridge.buildRequest(
                from: request,
                model: model,
                messages: messages
            )
            try await BasaltTranscriptBridge.streamResponse(
                foundationRequest: foundationRequest,
                requestID: request.id,
                model: model,
                into: channel
            )
        }
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum BasaltTranscriptBridge {
    static func messages(
        from transcript: Transcript,
        supportsVision: Bool
    ) throws -> [ChatMessageInput] {
        var converted: [ChatMessageInput] = []
        for entry in transcript {
            if let message = try convertEntry(entry, supportsVision: supportsVision) {
                converted.append(message)
            }
        }
        return converted.isEmpty ? [ChatMessageInput(role: .user, text: "")] : converted
    }

    private static func convertEntry(
        _ entry: Transcript.Entry,
        supportsVision: Bool
    ) throws -> ChatMessageInput? {
        switch entry {
        case .instructions(let instructions):
            guard let text = extractText(from: instructions.segments) else { return nil }
            return ChatMessageInput(role: .system, text: text)
        case .prompt(let prompt):
            let text = extractText(from: prompt.segments) ?? ""
            let images = try extractImages(from: prompt.segments, entry: entry, supportsVision: supportsVision)
            guard !text.isEmpty || !images.isEmpty else { return nil }
            return ChatMessageInput(role: .user, text: text, images: images)
        case .response(let response):
            guard let text = extractText(from: response.segments) else { return nil }
            return ChatMessageInput(role: .assistant, text: text)
        case .reasoning:
            return nil
        case .toolCalls(let toolCalls):
            let calls = toolCalls.map {
                ToolCallOutput(id: $0.id, name: $0.toolName, arguments: $0.arguments.jsonString)
            }
            return calls.isEmpty ? nil : ChatMessageInput(role: .assistant, text: "", toolCalls: calls)
        case .toolOutput(let output):
            let text = extractToolOutput(from: output.segments)
            return ChatMessageInput(role: .tool, text: text, toolCallId: output.id, toolName: output.toolName)
        default:
            return nil
        }
    }

    private static func extractText(from segments: [Transcript.Segment]) -> String? {
        let parts = segments.compactMap { segment -> String? in
            if case .text(let textSegment) = segment { return textSegment.content }
            if case .structure(let structured) = segment { return structured.content.jsonString }
            return nil
        }
        let joined = parts.joined(separator: "\n")
        return joined.isEmpty ? nil : joined
    }

    private static func extractToolOutput(from segments: [Transcript.Segment]) -> String {
        extractText(from: segments) ?? ""
    }

    private static func extractImages(
        from segments: [Transcript.Segment],
        entry: Transcript.Entry,
        supportsVision: Bool
    ) throws -> [ImageInput] {
        var images: [ImageInput] = []
        let context = CIContext(options: [.cacheIntermediates: false])
        for segment in segments {
            guard case .attachment(let attachment) = segment else { continue }
            switch attachment.content {
            case .image(let imageAttachment):
                guard supportsVision else {
                    throw LanguageModelError.unsupportedCapability(
                        .init(capability: .vision, debugDescription: "Declare .vision capability to send image attachments")
                    )
                }
                let oriented = imageAttachment.ciImage.oriented(imageAttachment.orientation)
                let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
                guard let pngData = context.pngRepresentation(of: oriented, format: .RGBA8, colorSpace: colorSpace) else {
                    throw LanguageModelError.unsupportedTranscriptContent(
                        .init(unsupportedContent: [entry], debugDescription: "Failed to encode image attachment as PNG")
                    )
                }
                images.append(try ImageInput(data: pngData, mime: "image/png"))
            default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(unsupportedContent: [entry], debugDescription: "Unsupported transcript attachment content")
                )
            }
        }
        return images
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension BasaltTranscriptBridge {
    static func buildRequest(
        from request: LanguageModelExecutorGenerationRequest,
        model: BasaltLanguageModel,
        messages: [ChatMessageInput]
    ) throws -> BasaltFoundationRequest {
        var out = BasaltFoundationRequest(messages: messages, model: model.modelID)
        out.maxTokens = request.generationOptions.maximumResponseTokens ?? 4096
        if let temp = request.generationOptions.temperature {
            out.temperature = max(0, temp)
        }
        applySampling(request.generationOptions.samplingMode, into: &out)
        try applyTools(
            mode: request.generationOptions.toolCallingMode ?? .allowed,
            definitions: request.enabledToolDefinitions,
            capabilities: model.capabilities,
            into: &out
        )
        if let schema = request.schema {
            guard model.capabilities.contains(.guidedGeneration) else {
                throw LanguageModelError.unsupportedCapability(
                    .init(capability: .guidedGeneration, debugDescription: "Declare .guidedGeneration to use schemas")
                )
            }
            let encoded = try BasaltFoundationSchemaAdapter.encodeGenerationSchema(schema)
            if out.responseFormat == nil {
                out.responseFormat = .jsonSchema(encoded, strict: false)
            }
            out.enableThinking = false
        } else if !model.capabilities.contains(.reasoning) {
            out.enableThinking = false
        }
        return out
    }

    private static func applySampling(
        _ mode: GenerationOptions.SamplingMode?,
        into request: inout BasaltFoundationRequest
    ) {
        guard let kind = mode?.kind else { return }
        switch kind {
        case .greedy:
            request.temperature = 0
        case .randomTopK(let k, let seed):
            request.topK = k
            request.seed = seed
        case .randomProbabilityThreshold(let threshold, let seed):
            request.topP = threshold
            request.seed = seed
        @unknown default:
            break
        }
    }

    private static func applyTools(
        mode: GenerationOptions.ToolCallingMode,
        definitions: [Transcript.ToolDefinition],
        capabilities: LanguageModelCapabilities,
        into request: inout BasaltFoundationRequest
    ) throws {
        if mode.kind == .disallowed {
            request.toolChoice = ToolChoice(callsDisabled: true)
            return
        }
        guard !definitions.isEmpty else {
            if mode.kind == .required {
                throw LanguageModelError.unsupportedCapability(
                    .init(capability: .toolCalling, debugDescription: "Required tool mode needs at least one tool")
                )
            }
            return
        }
        guard capabilities.contains(.toolCalling) else {
            throw LanguageModelError.unsupportedCapability(
                .init(capability: .toolCalling, debugDescription: "Declare .toolCalling to invoke tools")
            )
        }
        let converted = try definitions.map(BasaltFoundationSchemaAdapter.convertToolDefinition)
        request.tools = converted
        if mode.kind == .required {
            if converted.count == 1 {
                request.toolChoice = ToolChoice(forcedTool: converted[0].name)
            } else {
                let tag = try BasaltFoundationSchemaAdapter.encodeToolCallingStructuralTag(tools: converted)
                request.responseFormat = .structuralTag(tag)
            }
        }
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension BasaltTranscriptBridge {
    static func streamResponse(
        foundationRequest: BasaltFoundationRequest,
        requestID: UUID,
        model: BasaltLanguageModel,
        into channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let responseEntryID = UUID().uuidString
        let reasoningEntryID = UUID().uuidString
        let toolEntryID = UUID().uuidString
        await channel.send(
            .response(
                entryID: responseEntryID,
                action: .updateMetadata(["modelID": model.modelID, "requestID": requestID.uuidString])
            )
        )
        let bufferForTools = !foundationRequest.tools.isEmpty && foundationRequest.toolChoice?.callsDisabled != true
        guard !bufferForTools else {
            let collected = try await model.transport.collect(foundationRequest)
            await emitCollected(
                collected,
                ids: (response: responseEntryID, reasoning: reasoningEntryID, tools: toolEntryID),
                declaresReasoning: model.capabilities.contains(.reasoning),
                into: channel
            )
            return
        }
        for try await delta in model.transport.stream(foundationRequest) {
            await emitDelta(
                delta,
                ids: (response: responseEntryID, reasoning: reasoningEntryID, tools: toolEntryID),
                declaresReasoning: model.capabilities.contains(.reasoning),
                into: channel
            )
        }
    }

    private static func emitCollected(
        _ response: BasaltFoundationResponse,
        ids: (response: String, reasoning: String, tools: String),
        declaresReasoning: Bool,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        if declaresReasoning, !response.reasoning.isEmpty {
            await channel.send(.reasoning(entryID: ids.reasoning, action: .appendText(response.reasoning, tokenCount: 1)))
        }
        if !response.toolCalls.isEmpty {
            for call in response.toolCalls {
                await channel.send(
                    .toolCalls(
                        entryID: ids.tools,
                        action: .toolCall(id: call.id, name: call.name, action: .appendArguments(call.arguments, tokenCount: 1))
                    )
                )
            }
        } else if !response.text.isEmpty {
            await channel.send(.response(entryID: ids.response, action: .appendText(response.text, tokenCount: 1)))
        }
        if response.promptTokens > 0 || response.completionTokens > 0 {
            let inputUsage = LanguageModelExecutorGenerationChannel.Usage.Input(
                totalTokenCount: response.promptTokens,
                cachedTokenCount: 0
            )
            let outputUsage = LanguageModelExecutorGenerationChannel.Usage.Output(
                totalTokenCount: response.completionTokens,
                reasoningTokenCount: 0
            )
            await channel.send(
                .response(
                    entryID: ids.response,
                    action: .updateUsage(input: inputUsage, output: outputUsage)
                )
            )
        }
    }

    private static func emitDelta(
        _ delta: ChatDelta,
        ids: (response: String, reasoning: String, tools: String),
        declaresReasoning: Bool,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        switch delta {
        case .text(let chunk) where !chunk.isEmpty:
            await channel.send(.response(entryID: ids.response, action: .appendText(chunk, tokenCount: 1)))
        case .reasoning(let chunk) where declaresReasoning && !chunk.isEmpty:
            await channel.send(.reasoning(entryID: ids.reasoning, action: .appendText(chunk, tokenCount: 1)))
        case .toolCall(let call):
            let callID = call.id.isEmpty ? UUID().uuidString : call.id
            await channel.send(
                .toolCalls(
                    entryID: ids.tools,
                    action: .toolCall(id: callID, name: call.name, action: .appendArguments(call.arguments, tokenCount: 1))
                )
            )
        case .usage(let prompt, let completion):
            let inputUsage = LanguageModelExecutorGenerationChannel.Usage.Input(
                totalTokenCount: prompt,
                cachedTokenCount: 0
            )
            let outputUsage = LanguageModelExecutorGenerationChannel.Usage.Output(
                totalTokenCount: completion,
                reasoningTokenCount: 0
            )
            await channel.send(
                .response(
                    entryID: ids.response,
                    action: .updateUsage(input: inputUsage, output: outputUsage)
                )
            )
        default:
            break
        }
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension BasaltFoundationModelProvider {
    public func languageModel(
        capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration, .toolCalling, .reasoning]
    ) -> BasaltLanguageModel {
        BasaltLanguageModel(modelID: modelID, capabilities: capabilities, transport: transport)
    }

    public func makeSession(
        tools: [any FoundationModels.Tool] = [],
        instructions: String? = nil,
        capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration, .toolCalling, .reasoning]
    ) -> LanguageModelSession {
        let model = languageModel(capabilities: capabilities)
        if let instructions {
            return LanguageModelSession(model: model, tools: tools, instructions: Instructions(instructions))
        }
        return LanguageModelSession(model: model, tools: tools, instructions: nil as Instructions?)
    }

    public func generate<Content: Generable>(
        _ type: Content.Type = Content.self,
        prompt: String,
        instructions: String? = nil,
        temperature: Double? = 0,
        maxTokens: Int = 4096
    ) async throws -> Content {
        let schemaJSON = try BasaltFoundationSchemaAdapter.encodeGenerationSchema(Content.generationSchema)
        var messages: [ChatMessageInput] = []
        if let instructions, !instructions.isEmpty {
            messages.append(ChatMessageInput(role: .system, text: instructions))
        }
        messages.append(ChatMessageInput(role: .user, text: prompt))
        var request = BasaltFoundationRequest(messages: messages, model: modelID)
        request.responseFormat = .jsonSchema(schemaJSON, strict: false)
        request.temperature = temperature
        request.maxTokens = maxTokens
        request.enableThinking = false
        let response = try await transport.collect(request)
        return try BasaltFoundationSchemaAdapter.decodeGenerable(Content.self, from: response.text)
    }
}
#endif
