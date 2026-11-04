import Foundation
import CryptoKit

extension SessionLogRecord {
    struct Stamp: Sendable, Equatable {
        let ts: String
        let session: String
    }
}

extension SessionLogRecord.Event {
    init(_ type: String, at stamp: SessionLogRecord.Stamp) {
        self.init(v: SessionLogRecord.schemaVersion, type: type, ts: stamp.ts, session: stamp.session)
    }
}

struct SessionLogFormat: Sendable {
    typealias Parent = (id: String, count: Int)

    let mode: SessionLogMode
    let redacts: Bool

    static func milliseconds(_ duration: Duration) -> Int {
        Int(duration.components.seconds * 1_000 + duration.components.attoseconds / 1_000_000_000_000_000)
    }

    static func line(_ record: some Encodable, sorted: Bool = true) -> Data? {
        let encoder = JSONEncoder()
        if sorted { encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes] }
        guard var line = try? encoder.encode(record) else { return nil }
        line.append(0x0A)
        return line
    }

    func scrub(_ text: String) -> String {
        redacts ? SecretRedactor.redact(text) : text
    }

    func request(
        _ request: GenerationRequest, id: String, origin: RequestOrigin, modelRequested: String?,
        parent: Parent?, at stamp: SessionLogRecord.Stamp
    ) -> SessionLogRecord.Request {
        SessionLogRecord.Request(
            v: SessionLogRecord.schemaVersion, type: "request", ts: stamp.ts, session: stamp.session, id: id,
            source: origin.source, endpoint: origin.endpoint, stream: origin.stream, service: request.isServiceRequest == true,
            modelRequested: modelRequested,
            params: .init(maxTokens: request.maxTokens, temperature: request.temperature, topP: request.topP,
                          topK: request.topK, minP: request.minP, reasoningEffort: request.reasoningEffort,
                          thinkingBudget: request.thinkingBudget),
            cacheKey: request.cacheKey, messagesCount: request.messages?.count ?? 1, toolsCount: request.tools?.count ?? 0,
            prompt: .init(request.messages.map(ChatMessageInput.plainText) ?? request.prompt),
            content: mode == .full ? content(of: request, parent: parent) : nil)
    }

    func content(of request: GenerationRequest, parent: Parent?) -> SessionLogRecord.RequestContent {
        guard let messages = request.messages else {
            return .init(messages: nil, prompt: scrub(request.prompt), tools: request.tools)
        }
        let stored = parent.map { Array(messages.dropFirst($0.count)) } ?? messages
        var content = SessionLogRecord.RequestContent(
            messages: redacts ? SecretRedactor.redact(stored) : stored, prompt: nil, tools: request.tools)
        content.parent = parent?.id
        content.prefixCount = parent?.count
        return content
    }

    func response(
        id: String, status: String, result: SessionLogResult, queueWait: Duration, error: String?,
        at stamp: SessionLogRecord.Stamp
    ) -> SessionLogRecord.Response {
        let meta = result.metadata
        return SessionLogRecord.Response(
            v: SessionLogRecord.schemaVersion, type: "response", ts: stamp.ts, session: stamp.session, requestId: id,
            status: status, finishReason: meta?.finishReason.rawValue,
            tokens: meta.map { .init(prompt: $0.promptTokens, completion: $0.completionTokens, cached: $0.cachedPromptTokens) },
            timing: meta.map {
                .init(queueWaitMs: Self.milliseconds(queueWait), ttftMs: Int(($0.timeToFirstToken * 1_000).rounded()),
                      generationMs: Int(($0.generationTime * 1_000).rounded()), tokensPerSecond: $0.tokensPerSecond)
            },
            memory: meta.map { .init(activeBytes: $0.memory.activeBytes, peakBytes: $0.memory.peakBytes) },
            output: output(of: result),
            toolCallsCount: result.toolCalls.count, toolCallRepairs: meta?.toolCallRepairs,
            toolCallRejections: meta?.toolCallRejections, error: error)
    }

    func output(of result: SessionLogResult) -> SessionLogRecord.Output {
        let full = mode == .full
        return .init(
            chars: result.text.count, sha256: SessionLogRecord.Digest(result.text).sha256,
            text: full ? scrub(result.text) : nil,
            reasoning: full && !result.reasoning.isEmpty ? scrub(result.reasoning) : nil,
            toolCalls: full && !result.toolCalls.isEmpty ? (redacts ? SecretRedactor.redact(result.toolCalls) : result.toolCalls) : nil)
    }
}

struct SessionConversations: Sendable {
    typealias Entry = (id: String, digests: [SHA256.Digest])

    static let window = 8
    static let capacity = 4_096

    private var recent: [String: [Entry]] = [:]

    static func digests(_ messages: [ChatMessageInput]) -> [SHA256.Digest] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        return messages.map { SHA256.hash(data: (try? encoder.encode($0)) ?? Data($0.text.utf8)) }
    }

    func session(continuing digests: [SHA256.Digest]) -> String? {
        recent.first { _, entries in entries.contains { digests.starts(with: $0.digests) } }?.key
    }

    func knows(_ session: String) -> Bool {
        recent[session] != nil
    }

    mutating func seed(_ session: String, with entries: [Entry]) {
        recent[session] = entries
    }

    mutating func remember(_ messages: [ChatMessageInput]?, id: String, session: String) -> SessionLogFormat.Parent? {
        guard let messages else { return nil }
        let digests = Self.digests(messages)
        var entries = recent[session] ?? []
        let parent = entries.last { digests.starts(with: $0.digests) }
        entries.append((id, digests))
        if recent[session] == nil, recent.count >= Self.capacity { recent.removeAll() }
        recent[session] = Array(entries.suffix(Self.window))
        return parent.map { ($0.id, $0.digests.count) }
    }
}

struct SessionActivity: Sendable {
    enum Start: Equatable {
        case resumed(gapS: Int)
        case firstSeen
        case seen
    }

    private(set) var lastActivity: [String: Date] = [:]
    private var suspendedAt: [String: Date] = [:]
    private var primed: Set<String> = []

    mutating func start(_ session: String, at now: Date) -> Start {
        lastActivity[session] = now
        if let suspended = suspendedAt.removeValue(forKey: session) {
            return .resumed(gapS: Int(now.timeIntervalSince(suspended)))
        }
        return primed.insert(session).inserted ? .firstSeen : .seen
    }

    mutating func touch(_ session: String, at now: Date) {
        lastActivity[session] = now
    }

    static func idleSeconds(since last: Date, now: Date, timeout: TimeInterval) -> Int? {
        let idle = now.timeIntervalSince(last)
        return idle >= timeout ? Int(idle) : nil
    }

    mutating func suspend(_ session: String, at now: Date) {
        suspendedAt[session] = now
        lastActivity[session] = nil
    }
}

struct SessionLogRetention: Sendable {
    struct File: Sendable, Equatable {
        let url: URL
        let modified: Date
        let size: Int64
    }

    let days: Int
    let maxTotalBytes: Int64

    func deletions(among files: [File], now: Date) -> [URL] {
        let cutoff = now.addingTimeInterval(-Double(max(0, days)) * 86_400)
        let kept = files.filter { $0.modified >= cutoff }
        var total = kept.reduce(0) { $0 + $1.size }
        var oversize: [URL] = []
        for file in kept.sorted(by: { $0.modified < $1.modified }) where total > maxTotalBytes {
            oversize.append(file.url)
            total -= file.size
        }
        return files.filter { $0.modified < cutoff }.map(\.url) + oversize
    }
}
