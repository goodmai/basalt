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
            toolCallRejections: meta?.toolCallRejections, confidence: meta?.confidence, error: error)
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

    func parent(of digests: [SHA256.Digest], in session: String) -> SessionLogFormat.Parent? {
        recent[session]?.last { digests.starts(with: $0.digests) }.map { ($0.id, $0.digests.count) }
    }

    mutating func remember(_ digests: [SHA256.Digest], id: String, session: String) {
        let entries = (recent[session] ?? []) + [(id, digests)]
        if recent[session] == nil, recent.count >= Self.capacity { recent.removeAll() }
        recent[session] = Array(entries.suffix(Self.window))
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

    func start(of session: String, at now: Date) -> Start {
        if let suspended = suspendedAt[session] { return .resumed(gapS: Int(now.timeIntervalSince(suspended))) }
        return primed.contains(session) ? .seen : .firstSeen
    }

    mutating func begin(_ session: String, at now: Date) {
        lastActivity[session] = now
        if suspendedAt.removeValue(forKey: session) == nil { primed.insert(session) }
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

struct SessionLogState: Sendable {
    enum Event: Sendable {
        case started(id: String, session: String, at: Date)
        case resumed(session: String, from: URL, seed: [SessionConversations.Entry]?)
        case remembered(id: String, session: String, digests: [SHA256.Digest]?)
        case answered(id: String, session: String, at: Date)
        case appended(session: String, line: Data?, limit: Int)
        case dropsReported(session: String, line: Data?)
        case drained
        case opened(session: String, fd: Int32)
        case suspended(session: String, at: Date)
        case closed
        case failed
    }

    var pending: [(session: String, line: Data)] = []
    var dropped = 0
    var disabled = false
    var handles: [String: Int32] = [:]
    var conversations = SessionConversations()
    var activity = SessionActivity()
    var requestSessions: [String: String] = [:]
    var resumedPaths: [String: URL] = [:]

    func checkInvariants(after previous: SessionLogState) -> [String] {
        disabled && !pending.isEmpty ? ["a disabled log buffers \(pending.count) lines"] : []
    }

    static func reduce(_ state: SessionLogState, _ event: Event) -> SessionLogState {
        var next = state
        switch event {
        case .started(let id, let session, let now):
            next.requestSessions[id] = session
            next.activity.begin(session, at: now)
        case .resumed(let session, let url, let seed):
            next.resumedPaths[session] = url
            if let seed { next.conversations.seed(session, with: seed) }
        case .remembered(let id, let session, let digests):
            if let digests { next.conversations.remember(digests, id: id, session: session) }
        case .answered(let id, let session, let now):
            next.requestSessions[id] = nil
            next.activity.touch(session, at: now)
        case .appended(let session, let line, let limit):
            next.buffer(line, session: session, limit: limit)
        case .dropsReported(let session, let line):
            next.dropped = 0
            next.buffer(line, session: session, limit: .max)
        case .drained:
            next.pending.removeAll(keepingCapacity: true)
        case .opened(let session, let fd):
            next.handles[session] = fd
        case .suspended(let session, let now):
            next.handles[session] = nil
            next.activity.suspend(session, at: now)
        case .closed:
            next.handles.removeAll()
        case .failed:
            next.disabled = true
            next.pending.removeAll()
        }
        return next
    }

    private mutating func buffer(_ line: Data?, session: String, limit: Int) {
        guard !disabled else { return }
        guard pending.count < limit else {
            dropped += 1
            return
        }
        if let line { pending.append((session, line)) }
    }
}
