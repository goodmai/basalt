import BasaltCore
import Foundation
import MLX
import MLXLMCommon

struct GenerationRelay: Sendable {
    struct Setup: Sendable {
        var request: GenerationRequest
        var promptTokens = 0
        var reusedTokens = 0
        var alarm = NonFiniteLogitAlarm()
        var controlTokens: [String] = []
        var startsThinking = false
        var scansToolCallText = false
        var echoedPrefix: String?
        var forcedTool: String?
        var prefixAnswer = ""
        var guardsRepetition = true
        var draftSource: DraftSource?
        var confidenceProbe: ConfidenceProbe?
        var onyxReasoning: OnyxReasoningBox?
    }

    let setup: Setup
    private let declaredTools: Set<String>
    private var splitter: ThinkingSplitter
    private var gate: ToolCallGate
    private var scanner: NemotronToolCallScanner?
    private var repetition: RepetitionGuard?
    private var health: DegenerateOutputDetector
    private var forcedArguments: String
    private(set) var reasoningText = ""
    private var lastInfo: GenerateCompletionInfo?
    private var tokenCount = 0
    private var toolCallRepairs = 0
    private var toolCallRejections = 0
    private var stoppedOnRepetition = false

    init(_ setup: Setup) {
        self.setup = setup
        declaredTools = Set(setup.request.tools?.map(\.name) ?? [])
        splitter = ThinkingSplitter(startsThinking: setup.startsThinking)
        gate = ToolCallGate(request: setup.request)
        scanner = setup.scansToolCallText ? NemotronToolCallScanner() : nil
        repetition = setup.guardsRepetition ? RepetitionGuard() : nil
        health = DegenerateOutputDetector(controlTokens: setup.controlTokens)
        forcedArguments = setup.prefixAnswer
    }

    mutating func step(_ generation: Generation) -> (chunks: [StreamChunk], stops: Bool) {
        let onyxPrefix = drainOnyxReasoning()
        switch generation {
        case .chunk(let text):
            tokenCount += 1
            let chunks = onyxPrefix + route(splitter.feed(text))
            if gate.isSaturated || setup.alarm.isTripped { return (chunks, true) }
            stoppedOnRepetition = repetition?.shouldStop(after: text) == true
            return (chunks, stoppedOnRepetition)
        case .info(let info):
            lastInfo = info
            return (onyxPrefix, false)
        case .toolCall(let call):
            let res = called(.toolCall(MLXInferenceEngine.toolCallOutput(call)))
            return (onyxPrefix + res.chunks, res.stops)
        case .rejectedToolCall(let rejection):
            let res = called(resolved(rejection))
            return (onyxPrefix + res.chunks, res.stops)
        @unknown default:
            break
        }
        return (onyxPrefix, false)
    }

    mutating func drained(_ info: GenerateCompletionInfo?) {
        lastInfo = info ?? lastInfo
    }

    mutating func finish() -> [StreamChunk] {
        var onyxTail: [StreamChunk] = []
        if let unemitted = setup.onyxReasoning?.drainUnemittedInitial() {
            for text in unemitted { onyxTail.append(.reasoning(text)) }
        }
        onyxTail += drainOnyxReasoning()
        let chunks = onyxTail + route(splitter.feed("", final: true)) + gate.admit(scanner?.finish() ?? [])
        guard let forcedTool = setup.forcedTool else { return chunks }
        let arguments = forcedArguments.trimmingCharacters(in: .whitespacesAndNewlines)
        if (try? JSONSerialization.jsonObject(with: Data(arguments.utf8))) is [String: Any] {
            return chunks + gate.admit([.toolCall(ToolCallOutput(
                id: "call_\(UUID().uuidString.prefix(24))", name: forcedTool, arguments: arguments))])
        }
        let continued = String(forcedArguments.dropFirst(setup.prefixAnswer.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        return continued.isEmpty ? chunks : chunks + [.text(continued)]
    }

    func response(
        generationTime: TimeInterval, timeToFirstToken: TimeInterval, memory: GenerationResponse.MemoryUsage,
        reasoningTokens: Int?, forcedTokens: Int?, modelModified: Bool
    ) -> GenerationResponse {
        let telemetry = lastInfo?.speculativeDecodingTelemetry
        var confidence = setup.confidenceProbe?.summary
        if var measured = confidence, measured.totalHighConfidenceDrafts == 0,
           let telemetry, telemetry.draftTokenCount > 0 {
            measured.totalHighConfidenceDrafts = telemetry.draftTokenCount
            measured.highConfidenceDraftRejections = Swift.max(0, telemetry.draftTokenCount - telemetry.acceptedDraftTokenCount)
            confidence = measured
        }
        return GenerationResponse(
            generatedText: "",
            reasoningText: nil,
            promptTokens: lastInfo.map { setup.reusedTokens + $0.promptTokenCount } ?? setup.promptTokens,
            cachedPromptTokens: setup.reusedTokens,
            completionTokens: lastInfo?.generationTokenCount ?? tokenCount,
            tokensPerSecond: lastInfo?.tokensPerSecond ?? 0,
            generationTime: generationTime,
            timeToFirstToken: timeToFirstToken,
            memory: memory,
            finishReason: finishReason,
            reasoningTokens: reasoningTokens,
            draftTokens: telemetry?.draftTokenCount,
            acceptedDraftTokens: telemetry?.acceptedDraftTokenCount,
            forcedTokens: forcedTokens,
            modelCalls: telemetry?.targetModelCallCount,
            toolCallRepairs: toolCallRepairs,
            toolCallRejections: toolCallRejections
        ).reporting(
            degenerate: health.verdict(answerExpected: gate.admitted == 0 && finishReason != .length),
            draftSource: setup.draftSource, modelModified: modelModified,
            confidence: confidence
        )
    }

    private var finishReason: GenerationResponse.FinishReason {
        if setup.alarm.isTripped { return .error }
        if stoppedOnRepetition { return .repetition }
        let limit = setup.request.maxTokens ?? 1024
        return limit > 0 && (lastInfo?.generationTokenCount ?? tokenCount) >= limit ? .length : .stop
    }

    private mutating func called(_ chunk: StreamChunk) -> (chunks: [StreamChunk], stops: Bool) {
        var chunks = route(splitter.feed("", final: true))
        if let admitted = gate.admit(chunk) { chunks.append(admitted) }
        return (chunks, gate.isSaturated)
    }

    private mutating func resolved(_ rejection: RejectedToolCall) -> StreamChunk {
        let chunk = MLXInferenceEngine.resolveRejectedToolCall(
            reason: rejection.reason.rawValue, toolName: rejection.toolName,
            callID: rejection.callID, rawText: rejection.rawTextPreview,
            truncated: rejection.isPreviewTruncated, declaredTools: declaredTools)
        if case .toolCall = chunk {
            toolCallRepairs += 1
        } else {
            toolCallRejections += 1
            BasaltLogger(module: "MLXInferenceEngine").warn("Tool call rejected: \(rejection.reason.rawValue)")
        }
        return chunk
    }

    private mutating func route(_ segments: [ThinkingSplitter.Segment]) -> [StreamChunk] {
        var chunks: [StreamChunk] = []
        for segment in segments {
            health.feed(segment)
            switch segment {
            case .reasoning(let text):
                reasoningText += text
                chunks.append(.reasoning(text))
            case .answer(let text):
                chunks += answered(text)
            }
        }
        return chunks
    }

    private mutating func drainOnyxReasoning() -> [StreamChunk] {
        guard let drained = setup.onyxReasoning?.drainRelayStep() else { return [] }
        for text in drained.accountedInitial {
            health.feed(.reasoning(text))
            reasoningText += text
        }
        var chunks: [StreamChunk] = []
        for text in drained.additionalChunks {
            health.feed(.reasoning(text))
            reasoningText += text
            chunks.append(.reasoning(text))
        }
        return chunks
    }

    private mutating func answered(_ text: String) -> [StreamChunk] {
        guard setup.forcedTool == nil else {
            forcedArguments += text
            return []
        }
        guard let scanned = scanner?.feed(text) else { return [.text(text)] }
        return gate.admit(scanned)
    }
    var isThinkingAtEnd: Bool { splitter.isThinking }
    var completionTokenCount: Int { lastInfo?.generationTokenCount ?? tokenCount }
    var finalFinishReason: GenerationResponse.FinishReason { finishReason }

    func attachInitial(into continuation: AsyncStream<StreamChunk>.Continuation, now: ContinuousClock.Instant) -> ContinuousClock.Instant? {
        if let echoed = setup.echoedPrefix { continuation.yield(.text(echoed)) }
        guard let box = setup.onyxReasoning else { return nil }
        return box.attachInitialEmitter { continuation.yield(.reasoning($0)) } ? now : nil
    }
}

extension MLXInferenceEngine {
    struct RelayInputs: Sendable {
        let stream: AsyncStream<Generation>
        let task: Task<Void, Never>?
        let prefillSeconds: TimeInterval
        let promptTokens: [Int]
        let namespace: String?
        let leaseID: Int
        let epoch: ModelEpoch
        let constraint: OutputConstraint?
        let container: ModelContainer
        let faults: FaultPlan
    }

    nonisolated func relay(_ core: GenerationRelay, _ inputs: RelayInputs, into continuation: AsyncStream<StreamChunk>.Continuation) async {
        var relay = core
        let clock = ContinuousClock()
        let startTime = clock.now
        var firstTokenAt = relay.attachInitial(into: continuation, now: startTime)
        var previousTokenAt = firstTokenAt
        var itlSamples: [Double] = []
        var stoppedAfterCall = false
        var generations = inputs.stream.makeAsyncIterator()
        while let generation = await generations.next() {
            if Task.isCancelled { break }
            Self.recordChunkTimestamp(generation, now: clock.now, firstTokenAt: &firstTokenAt, previousTokenAt: &previousTokenAt, itlSamples: &itlSamples)
            let step = relay.step(generation)
            for chunk in step.chunks { continuation.yield(chunk) }
            guard step.stops else { continue }
            if let task = inputs.task {
                task.cancel()
                let info = await Self.drainCompletionInfo(&generations)
                relay.drained(info)
            } else {
                stoppedAfterCall = true
            }
            break
        }
        for chunk in relay.finish() { continuation.yield(chunk) }
        let completed = inputs.faults.point != .settle && !(Task.isCancelled || stoppedAfterCall)
        await Self.settle(inputs.task, cancelled: !completed)
        if completed, let session = inputs.constraint?.session { await self.recycle(session, epoch: inputs.epoch) }
        await self.keep(tokens: inputs.promptTokens, namespace: relay.setup.alarm.isTripped ? nil : inputs.namespace, releasing: inputs.leaseID, epoch: inputs.epoch)
        if !Task.isCancelled, !inputs.faults.endsStream {
            let generationTime = (clock.now - startTime).inSeconds + inputs.prefillSeconds
            let timeToFirstToken = (firstTokenAt.map { ($0 - startTime).inSeconds } ?? 0) + inputs.prefillSeconds
            let response = await metadata(
                relay, inputs, generationTime: generationTime,
                timeToFirstToken: timeToFirstToken, itlSamplesSeconds: itlSamples)
            continuation.yield(.metadata(response))
        }
        continuation.finish()
    }

    private nonisolated func metadata(
        _ relay: GenerationRelay, _ inputs: RelayInputs, generationTime: TimeInterval,
        timeToFirstToken: TimeInterval, itlSamplesSeconds: [Double]
    ) async -> GenerationResponse {
        let mem = Memory.snapshot()
        let reasoning = relay.reasoningText
        let thinkingIDs: [Int] = reasoning.isEmpty ? [] : await inputs.container.perform {
            $0.tokenizer.encode(text: reasoning, addSpecialTokens: false)
        }
        let reasoningTokens = thinkingIDs.isEmpty ? nil : thinkingIDs.count
        let forcedTokens = inputs.constraint?.fastForward == true ? inputs.constraint?.session.totalForcedTokens : nil
        let modified = await self.modelModified(since: inputs.leaseID)
        let cacheTelemetry = await self.drainPromptCacheTelemetry()
        let tokenTelemetry = await Self.buildTokenTelemetry(for: relay.setup.request, container: inputs.container)
        let reasoningTurn = Self.buildReasoningTurn(relay: relay, thinkingIDs: thinkingIDs, tokenTelemetry: tokenTelemetry)
        return relay.response(
            generationTime: generationTime, timeToFirstToken: timeToFirstToken,
            memory: GenerationResponse.MemoryUsage(peakBytes: mem.peakMemory, activeBytes: mem.activeMemory, cacheBytes: mem.cacheMemory),
            reasoningTokens: reasoningTokens, forcedTokens: forcedTokens, modelModified: modified
        ).withCacheTelemetry(cacheTelemetry)
            .withTokenTelemetry(tokenTelemetry)
            .withServingTiming(prefillSeconds: inputs.prefillSeconds, itlSamplesSeconds: itlSamplesSeconds)
            .withReasoningTurn(reasoningTurn)
    }
}

extension MLXInferenceEngine {
    fileprivate static func buildReasoningTurn(
        relay: GenerationRelay,
        thinkingIDs: [Int],
        tokenTelemetry: [OutputTokenTelemetry]?
    ) -> ReasoningTurnRecord? {
        guard !thinkingIDs.isEmpty || relay.isThinkingAtEnd else { return nil }
        let answerTokens = max(0, relay.completionTokenCount - thinkingIDs.count)
        let truncated = relay.isThinkingAtEnd && relay.finalFinishReason == .length
        let (thinkEntropy, answerEntropy) = splitSpanEntropies(tokenTelemetry: tokenTelemetry, thinkingCount: thinkingIDs.count)
        return ReasoningTurnRecord(
            thinkingTokens: thinkingIDs.count,
            answerTokens: answerTokens,
            hitTokenBudgetBeforeClose: truncated,
            thinkingTokenIDs: thinkingIDs,
            meanThinkingEntropy: thinkEntropy,
            meanAnswerEntropy: answerEntropy
        )
    }

    private static func splitSpanEntropies(
        tokenTelemetry: [OutputTokenTelemetry]?,
        thinkingCount: Int
    ) -> (thinking: Double?, answer: Double?) {
        guard let entries = tokenTelemetry, !entries.isEmpty else { return (nil, nil) }
        let split = min(max(0, thinkingCount), entries.count)
        let thinkSlice = entries.prefix(split).map { Double($0.entropy) }.filter(\.isFinite)
        let answerSlice = entries.dropFirst(split).map { Double($0.entropy) }.filter(\.isFinite)
        let meanThink = thinkSlice.isEmpty ? nil : thinkSlice.reduce(0, +) / Double(thinkSlice.count)
        let meanAnswer = answerSlice.isEmpty ? nil : answerSlice.reduce(0, +) / Double(answerSlice.count)
        return (meanThink, meanAnswer)
    }

    fileprivate static func recordChunkTimestamp(
        _ generation: Generation,
        now: ContinuousClock.Instant,
        firstTokenAt: inout ContinuousClock.Instant?,
        previousTokenAt: inout ContinuousClock.Instant?,
        itlSamples: inout [Double]
    ) {
        guard case .chunk = generation else { return }
        if firstTokenAt == nil {
            firstTokenAt = now
        } else if let previous = previousTokenAt {
            itlSamples.append((now - previous).inSeconds)
        }
        previousTokenAt = now
    }

    fileprivate static func buildTokenTelemetry(
        for request: GenerationRequest,
        container: ModelContainer
    ) async -> [OutputTokenTelemetry]? {
        guard request.logprobs == true, let probe = request.logitProbe else { return nil }
        let steps = probe.steps.filter { $0.token >= 0 }
        let topLimit = max(0, min(20, request.topLogprobs ?? 5))
        return await container.perform { context in
            steps.enumerated().map { index, step in
                let text = context.tokenizer.decode(tokenIds: [step.token], skipSpecialTokens: false)
                let lp = step.sampledLogprob ?? step.logprobs.first ?? -20.0
                var item = OutputTokenTelemetry(
                    index: index,
                    token: text,
                    tokenId: step.token,
                    logprob: lp,
                    entropy: step.entropy ?? 0.0,
                    margin: step.margin ?? 1.0
                )
                let n = min(topLimit, min(step.ids.count, step.logprobs.count))
                if n > 0 {
                    item.topLogprobs = (0 ..< n).map { k in
                        let tid = step.ids[k]
                        let tStr = context.tokenizer.decode(tokenIds: [tid], skipSpecialTokens: false)
                        return TokenCandidateProbability(token: tStr, tokenId: tid, logprob: step.logprobs[k])
                    }
                }
                return item
            }
        }
    }
}
