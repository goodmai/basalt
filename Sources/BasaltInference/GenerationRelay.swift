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
        tokenCount += 1
        switch generation {
        case .chunk(let text):
            let chunks = route(splitter.feed(text))
            if gate.isSaturated || setup.alarm.isTripped { return (chunks, true) }
            stoppedOnRepetition = repetition?.shouldStop(after: text) == true
            return (chunks, stoppedOnRepetition)
        case .info(let info):
            lastInfo = info
        case .toolCall(let call):
            return called(.toolCall(MLXInferenceEngine.toolCallOutput(call)))
        case .rejectedToolCall(let rejection):
            return called(resolved(rejection))
        @unknown default:
            break
        }
        return ([], false)
    }

    mutating func drained(_ info: GenerateCompletionInfo?) {
        lastInfo = info ?? lastInfo
    }

    mutating func finish() -> [StreamChunk] {
        let chunks = route(splitter.feed("", final: true)) + gate.admit(scanner?.finish() ?? [])
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
            confidence: setup.confidenceProbe?.summary
        )
    }

    private var finishReason: GenerationResponse.FinishReason {
        if setup.alarm.isTripped { return .error }
        if stoppedOnRepetition { return .repetition }
        return (lastInfo?.generationTokenCount ?? 0) >= (setup.request.maxTokens ?? 1024) ? .length : .stop
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

    private mutating func answered(_ text: String) -> [StreamChunk] {
        guard setup.forcedTool == nil else {
            forcedArguments += text
            return []
        }
        guard let scanned = scanner?.feed(text) else { return [.text(text)] }
        return gate.admit(scanned)
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
        var firstTokenAt: ContinuousClock.Instant?
        var stoppedAfterCall = false
        var generations = inputs.stream.makeAsyncIterator()
        if let echoed = relay.setup.echoedPrefix { continuation.yield(.text(echoed)) }
        while let generation = await generations.next() {
            if Task.isCancelled { break }
            if firstTokenAt == nil, case .chunk = generation { firstTokenAt = clock.now }
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
            let response = await metadata(relay, inputs, generationTime: generationTime, timeToFirstToken: timeToFirstToken)
            continuation.yield(.metadata(response))
        }
        continuation.finish()
    }

    private nonisolated func metadata(
        _ relay: GenerationRelay, _ inputs: RelayInputs, generationTime: TimeInterval, timeToFirstToken: TimeInterval
    ) async -> GenerationResponse {
        let mem = Memory.snapshot()
        let reasoning = relay.reasoningText
        let reasoningTokens = reasoning.isEmpty ? nil : await inputs.container.perform {
            $0.tokenizer.encode(text: reasoning, addSpecialTokens: false).count
        }
        let forcedTokens = inputs.constraint?.fastForward == true ? inputs.constraint?.session.totalForcedTokens : nil
        let modified = await self.modelModified(since: inputs.leaseID)
        return relay.response(
            generationTime: generationTime, timeToFirstToken: timeToFirstToken,
            memory: GenerationResponse.MemoryUsage(peakBytes: mem.peakMemory, activeBytes: mem.activeMemory, cacheBytes: mem.cacheMemory),
            reasoningTokens: reasoningTokens, forcedTokens: forcedTokens, modelModified: modified)
    }
}
