import Foundation
import MLX
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    public func promptTokenCount(for request: GenerationRequest) async -> Int? {
        guard let container else { return nil }
        let text = request.messages.map { ChatMessageInput.plainText($0) } ?? request.prompt
        return await container.perform { $0.tokenizer.encode(text: text, addSpecialTokens: false).count }
    }

    public func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        let stream = try await generateStream(request: request)
        var textChunks: [String] = []
        var reasoningChunks: [String] = []
        var toolCalls: [ToolCallOutput] = []
        var finalResponse: GenerationResponse?

        for await chunk in stream {
            switch chunk {
            case .text(let t):
                textChunks.append(t)
            case .reasoning(let r):
                reasoningChunks.append(r)
            case .metadata(let m):
                finalResponse = m
            case .toolCall(let call):
                toolCalls.append(call)
            }
        }

        guard let finalResponse else {
            throw .inferenceHardwareFailure(reason: "Stream finished without metadata")
        }
        guard finalResponse.finishReason != .error else { throw .nonFiniteLogits }

        let fullReasoning = reasoningChunks.joined()
        return GenerationResponse(
            generatedText: textChunks.joined(),
            reasoningText: fullReasoning.isEmpty ? nil : fullReasoning,
            promptTokens: finalResponse.promptTokens,
            cachedPromptTokens: finalResponse.cachedPromptTokens,
            completionTokens: finalResponse.completionTokens,
            tokensPerSecond: finalResponse.tokensPerSecond,
            generationTime: finalResponse.generationTime,
            timeToFirstToken: finalResponse.timeToFirstToken,
            memory: finalResponse.memory,
            finishReason: finalResponse.finishReason,
            toolCalls: toolCalls.isEmpty ? nil : toolCalls,
            reasoningTokens: finalResponse.reasoningTokens,
            draftTokens: finalResponse.draftTokens,
            acceptedDraftTokens: finalResponse.acceptedDraftTokens,
            forcedTokens: finalResponse.forcedTokens,
            modelCalls: finalResponse.modelCalls,
            toolCallRepairs: finalResponse.toolCallRepairs,
            toolCallRejections: finalResponse.toolCallRejections
        ).reporting(
            degenerate: finalResponse.degenerate, draftSource: finalResponse.draftSource,
            modelModified: finalResponse.modelModified, confidence: finalResponse.confidence
        ).withCacheTelemetry(finalResponse.cacheTelemetry).withTokenTelemetry(finalResponse.tokenTelemetry)
    }

    struct GenerationLaunch {
        let stream: AsyncStream<Generation>
        let task: Task<Void, Never>?
        let cache: PromptCache?
        var anchors: [PromptCache] = []
        let reusedTokens: TokenCount
        var exactTokens = TokenCount(Int.max)
        var prefillSeconds: TimeInterval = 0
        var mtpPrefix: MTPPrefix?
        var onyxReasoning: OnyxReasoningBox?
    }

    static func launch(input: LMInput, prompt: CachedPrompt, options: LaunchOptions, context: ModelContext) throws -> GenerationLaunch {
        let start = GenerationLauncher(
            input: input, params: options.params.reportingPrefill.paced(by: options.pacer), components: options.components, tools: options.tools, context: context, started: .now)
        if options.mtp, let drafter = context.mtpDrafter, !carriesMoreThanTokens(input) {
            return try start.mtp(
                MTPRequest(drafter: drafter, blockSize: options.mtpBlockSize, prefix: options.mtpPrefix), prompt: prompt,
                keepsCache: options.keepsCache)
        }
        guard options.keepsCache else {
            return try start.fresh(FreshCache.plan(options: options), batch: options.batch, lookup: options.lookup)
        }
        return try start.cached(prompt, batch: options.batch, lookup: options.lookup, rows: options.kEqVRow)
    }

    static func carriesMoreThanTokens(_ input: LMInput) -> Bool {
        input.image != nil || input.video != nil || input.audio != nil
            || input.text.mask.map { !all($0).item(Bool.self) } ?? false
    }

    static func appending(_ ids: [Int], to input: LMInput) -> LMInput {
        let tokens = input.text.tokens
        let extra = MLXArray(ids.map(Int32.init)).asType(tokens.dtype)
            .reshaped(Array(tokens.shape.dropLast()) + [ids.count])
        return LMInput(tokens: concatenated([tokens, extra], axis: -1))
    }

    static func handOff<T: TokenIteratorProtocol>(
        promptTokenCount: Int, context: ModelContext, iterator: consuming T, tools: [ToolSpec]?,
        toolCallPolicy: ToolCallPolicy, since started: ContinuousClock.Instant = .now
    ) -> (AsyncStream<Generation>, Task<Void, Never>, TimeInterval) {
        let result = handOffWithOnyx(
            promptTokenCount: promptTokenCount, context: context, iterator: iterator, tools: tools,
            toolCallPolicy: toolCallPolicy, since: started)
        return (result.0, result.1, result.2)
    }

    static func handOffWithOnyx<T: TokenIteratorProtocol>(
        promptTokenCount: Int, context: ModelContext, iterator: consuming T, tools: [ToolSpec]?,
        toolCallPolicy: ToolCallPolicy, since started: ContinuousClock.Instant = .now
    ) -> (AsyncStream<Generation>, Task<Void, Never>, TimeInterval, OnyxReasoningBox?) {
        Stream.defaultStream.synchronize()
        let prefill = (ContinuousClock.now - started).inSeconds
        var config = context.configuration
        if config.toolCallFormat == nil { config.toolCallFormat = context.model.toolCallFormat }
        if config.reasoningConfig == nil { config.reasoningConfig = context.model.reasoningConfig }
        if config.toolCallFormat == .atem {
            config.extraEOSTokens.formUnion(["<|eot|>", "<|end_of_text|>"])
            if let parser = OnyxReasoningStateMachine(tokenizer: context.tokenizer) {
                let box = OnyxReasoningBox()
                let tapped = OnyxReasoningTapIterator(base: iterator, parser: parser, box: box)
                let (stream, task) = generateTask(
                    promptTokenCount: promptTokenCount, modelConfiguration: config, tokenizer: context.tokenizer,
                    iterator: tapped, tools: tools, toolCallPolicy: toolCallPolicy)
                return (stream, task, prefill, box)
            }
        }
        if config.toolCallFormat == .inkling, let parser = InklingReasoningStateMachine(tokenizer: context.tokenizer) {
            let box = OnyxReasoningBox()
            let (stream, task) = generateTask(
                promptTokenCount: promptTokenCount, modelConfiguration: config, tokenizer: context.tokenizer,
                iterator: OnyxReasoningTapIterator(base: iterator, parser: parser, box: box), tools: tools, toolCallPolicy: toolCallPolicy)
            return (stream, task, prefill, box)
        }
        let (stream, task) = generateTask(
            promptTokenCount: promptTokenCount, modelConfiguration: config, tokenizer: context.tokenizer,
            iterator: iterator, tools: tools, toolCallPolicy: toolCallPolicy)
        return (stream, task, prefill, nil)
    }

    func batchedDecoder(_ container: ModelContainer, epoch: ModelEpoch) async -> BatchedDecoder? {
        guard configuration.batchSlots > 1, epoch == modelEpoch, appliedMemoryDirective < .tight else { return nil }
        if let batchDecoder { return batchDecoder }
        let slots = configuration.batchSlots
        await container.update { if $0.batch == nil { $0.batch = BatchCore(model: $0.model, slots: slots) } }
        let made = await container.perform { $0.batch?.decoder }
        guard epoch == modelEpoch else { return nil }
        batchDecoder = made
        return made
    }

    var batchedRows: Int { batchDecoder?.admittedRows ?? 0 }
}
