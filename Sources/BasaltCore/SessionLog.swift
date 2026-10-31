import Foundation
import CryptoKit

public enum SessionLogMode: String, Sendable, CaseIterable {
    case off
    case meta
    case full
}

public struct RequestOrigin: Sendable, Equatable {
    public let source: String
    public let endpoint: String
    public let stream: Bool

    public init(source: String, endpoint: String, stream: Bool) {
        self.source = source
        self.endpoint = endpoint
        self.stream = stream
    }

    public static let unspecified = RequestOrigin(source: "api", endpoint: "", stream: false)
}

public enum SessionLogRecord {
    public static let schemaVersion = 1

    public struct Digest: Codable, Sendable, Equatable {
        public let chars: Int
        public let sha256: String

        public init(_ text: String) {
            chars = text.count
            sha256 = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
        }
    }

    public struct Params: Codable, Sendable, Equatable {
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let topK: Int?
        let minP: Double?
        let reasoningEffort: String?
        let thinkingBudget: Int?

        enum CodingKeys: String, CodingKey {
            case maxTokens = "max_tokens", temperature, topP = "top_p", topK = "top_k", minP = "min_p"
            case reasoningEffort = "reasoning_effort", thinkingBudget = "thinking_budget"
        }
    }

    public struct RequestContent: Codable, Sendable, Equatable {
        let messages: [ChatMessageInput]?
        let prompt: String?
        let tools: [ToolDefinition]?
        var parent: String? = nil
        var prefixCount: Int? = nil

        enum CodingKeys: String, CodingKey { case messages, prompt, tools, parent, prefixCount = "prefix_count" }
    }

    public struct Request: Codable, Sendable {
        let v: Int
        let type: String
        let ts: String
        let session: String
        let id: String
        let source: String
        let endpoint: String
        let stream: Bool
        let service: Bool
        let modelRequested: String?
        let params: Params
        let cacheKey: String?
        let messagesCount: Int
        let toolsCount: Int
        let prompt: Digest
        let content: RequestContent?

        enum CodingKeys: String, CodingKey {
            case v, type, ts, session, id, source, endpoint, stream, service
            case modelRequested = "model_requested", params, cacheKey = "cache_key"
            case messagesCount = "messages_count", toolsCount = "tools_count", prompt, content
        }
    }

    public struct Tokens: Codable, Sendable, Equatable {
        let prompt: Int
        let completion: Int
        let cached: Int
    }

    public struct Timing: Codable, Sendable, Equatable {
        let queueWaitMs: Int
        let ttftMs: Int
        let generationMs: Int
        let tokensPerSecond: Double

        enum CodingKeys: String, CodingKey {
            case queueWaitMs = "queue_wait_ms", ttftMs = "ttft_ms", generationMs = "generation_ms"
            case tokensPerSecond = "tokens_per_second"
        }
    }

    public struct Memory: Codable, Sendable, Equatable {
        let activeBytes: Int
        let peakBytes: Int

        enum CodingKeys: String, CodingKey { case activeBytes = "active_bytes", peakBytes = "peak_bytes" }
    }

    public struct Output: Codable, Sendable, Equatable {
        let chars: Int
        let sha256: String
        let text: String?
        let reasoning: String?
        let toolCalls: [ToolCallOutput]?

        enum CodingKeys: String, CodingKey { case chars, sha256, text, reasoning, toolCalls = "tool_calls" }
    }

    public struct Response: Codable, Sendable {
        let v: Int
        let type: String
        let ts: String
        let session: String
        let requestId: String
        let status: String
        let finishReason: String?
        let tokens: Tokens?
        let timing: Timing?
        let memory: Memory?
        let output: Output
        let toolCallsCount: Int
        let error: String?

        enum CodingKeys: String, CodingKey {
            case v, type, ts, session, requestId = "request_id", status, finishReason = "finish_reason"
            case tokens, timing, memory, output, toolCallsCount = "tool_calls_count", error
        }
    }

    public struct Rejection: Codable, Sendable {
        let v: Int
        let type: String
        let ts: String
        let session: String
        let source: String
        let endpoint: String
        let code: Int
        let message: String
        let bodyBytes: Int

        enum CodingKeys: String, CodingKey { case v, type, ts, session, source, endpoint, code, message, bodyBytes = "body_bytes" }
    }

    public struct Event: Codable, Sendable {
        let v: Int
        let type: String
        let ts: String
        let session: String
        var version: String? = nil
        var pid: Int32? = nil
        var os: String? = nil
        var mode: String? = nil
        var model: String? = nil
        var from: String? = nil
        var to: String? = nil
        var durationMs: Int? = nil
        var count: Int? = nil
        var idleS: Int? = nil
        var gapS: Int? = nil
        var resumedFrom: String? = nil

        enum CodingKeys: String, CodingKey {
            case v, type, ts, session, version, pid, os, mode, model, from, to, durationMs = "duration_ms", count
            case idleS = "idle_s", gapS = "gap_s", resumedFrom = "resumed_from"
        }
    }
}

public struct SessionLogResult: Sendable {
    public var text = ""
    public var reasoning = ""
    public var toolCalls: [ToolCallOutput] = []
    public var metadata: GenerationResponse?

    public init() {}

    public mutating func absorb(_ chunk: StreamChunk) {
        switch chunk {
        case .text(let t): text += t
        case .reasoning(let r): reasoning += r
        case .toolCall(let call): toolCalls.append(call)
        case .metadata(let m): metadata = m
        }
    }

    public init(_ response: GenerationResponse) {
        text = response.generatedText
        reasoning = response.reasoningText ?? ""
        toolCalls = response.toolCalls ?? []
        metadata = response
    }
}

public actor SessionLogWriter {
    nonisolated public let mode: SessionLogMode
    nonisolated public let directory: URL
    nonisolated public let defaultSession: String
    let bufferLimit: Int

    private var pending: [(session: String, line: Data)] = []
    private var dropped = 0
    private var handles: [String: Int32] = [:]
    private var disabled = false
    private var flushTask: Task<Void, Never>?
    private let clock: @Sendable () -> Date

    nonisolated public let redacts: Bool
    static let prefixWindow = 8
    private var recentConversations: [String: [(id: String, digests: [SHA256.Digest])]] = [:]

    static func digests(_ messages: [ChatMessageInput]) -> [SHA256.Digest] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        return messages.map { SHA256.hash(data: (try? encoder.encode($0)) ?? Data($0.text.utf8)) }
    }
    private var requestSessions: [String: String] = [:]
    private var lastActivity: [String: Date] = [:]
    private var suspendedAt: [String: Date] = [:]
    private var resumedPaths: [String: URL] = [:]
    private var primed: Set<String> = []
    nonisolated public let idleTimeout: TimeInterval

    public init(
        mode: SessionLogMode, directory: URL, bufferLimit: Int = 1_024, redacts: Bool = true,
        defaultSession: String = SessionLogWriter.processSession(), idleTimeout: TimeInterval = 1_800,
        clock: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.idleTimeout = idleTimeout
        self.mode = mode
        self.redacts = redacts
        self.directory = directory
        self.bufferLimit = bufferLimit
        self.defaultSession = defaultSession
        self.clock = clock
    }

    public static func processSession() -> String {
        let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .gmt, formatOptions: [.withFullDate, .withTime])
        return "server-\(stamp)-\(ProcessInfo.processInfo.processIdentifier)"
    }

    public static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".basalt/sessions")
    }

    public var isEnabled: Bool { mode != .off && !disabled }

    public func startFlushing(every interval: Duration = .seconds(1)) {
        guard flushTask == nil, mode != .off else { return }
        flushTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                await self?.flush()
            }
        }
    }

    public func close() {
        flushTask?.cancel()
        flushTask = nil
        flush()
        for fd in handles.values { Darwin.close(fd) }
        handles.removeAll()
    }

    func timestamp() -> String {
        ISO8601DateFormatter.string(from: clock(), timeZone: .gmt, formatOptions: [.withInternetDateTime, .withFractionalSeconds])
    }

    func session(for request: GenerationRequest) -> String {
        if let explicit = Self.sanitized(request.sessionId) { return explicit }
        guard let messages = request.messages, !messages.isEmpty else { return defaultSession }
        let digests = Self.digests(messages)
        let continued = recentConversations.first { _, conversations in
            conversations.contains { $0.digests.count <= digests.count && digests.starts(with: $0.digests) }
        }
        return continued?.key ?? "s-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12).lowercased()
    }

    func prime(_ session: String) {
        let now = clock()
        defer { lastActivity[session] = now }
        if let suspended = suspendedAt.removeValue(forKey: session) {
            appendEvent("session_resume", session: session) { $0.gapS = Int(now.timeIntervalSince(suspended)) }
            return
        }
        guard primed.insert(session).inserted, handles[session] == nil,
              let existing = SessionLogReader.file(for: session, in: directory)
        else { return }
        resumedPaths[session] = existing
        let modified = (try? existing.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? now
        appendEvent("session_resume", session: session) {
            $0.gapS = max(0, Int(now.timeIntervalSince(modified)))
            $0.resumedFrom = existing.deletingLastPathComponent().lastPathComponent + "/" + existing.lastPathComponent
        }
        if mode == .full, recentConversations[session] == nil {
            recentConversations[session] = SessionLogReader.recentConversations(
                in: SessionLogReader.lines(of: existing), limit: Self.prefixWindow).map { ($0.id, Self.digests($0.messages)) }
        }
    }

    func suspendIdleSessions() {
        let now = clock()
        for (session, last) in lastActivity where handles[session] != nil && now.timeIntervalSince(last) >= idleTimeout {
            appendEvent("session_suspend", session: session) { $0.idleS = Int(now.timeIntervalSince(last)) }
            writePending()
            if let fd = handles.removeValue(forKey: session) { Darwin.close(fd) }
            suspendedAt[session] = now
            lastActivity[session] = nil
        }
    }

    private func appendEvent(_ type: String, session: String, _ fill: (inout SessionLogRecord.Event) -> Void) {
        var event = SessionLogRecord.Event(v: SessionLogRecord.schemaVersion, type: type, ts: timestamp(), session: session)
        fill(&event)
        append(event, session: session)
    }

    static func sanitized(_ id: String?) -> String? {
        guard let id, !id.isEmpty else { return nil }
        let safe = String(id.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) || "-_.".unicodeScalars.contains($0) ? Character($0) : "_" })
        return safe.hasPrefix(".") ? "_" + safe.dropFirst() : safe
    }

    public func logRequest(_ request: GenerationRequest, id: String, origin: RequestOrigin, modelRequested: String?) {
        guard isEnabled else { return }
        let session = session(for: request)
        requestSessions[id] = session
        prime(session)
        let parent = remember(request, id: id, session: session)
        let prompt = request.messages.map(ChatMessageInput.plainText) ?? request.prompt
        let record = SessionLogRecord.Request(
            v: SessionLogRecord.schemaVersion, type: "request", ts: timestamp(), session: session, id: id,
            source: origin.source, endpoint: origin.endpoint, stream: origin.stream, service: request.isServiceRequest == true,
            modelRequested: modelRequested,
            params: .init(maxTokens: request.maxTokens, temperature: request.temperature, topP: request.topP,
                          topK: request.topK, minP: request.minP, reasoningEffort: request.reasoningEffort,
                          thinkingBudget: request.thinkingBudget),
            cacheKey: request.cacheKey, messagesCount: request.messages?.count ?? 1, toolsCount: request.tools?.count ?? 0,
            prompt: .init(prompt),
            content: mode == .full ? content(of: request, parent: parent) : nil)
        append(record, session: record.session)
    }

    public func logResponse(
        _ request: GenerationRequest, id: String, status: String, result: SessionLogResult, queueWait: Duration, error: String? = nil
    ) {
        guard isEnabled else { return }
        let session = requestSessions.removeValue(forKey: id) ?? session(for: request)
        lastActivity[session] = clock()
        let meta = result.metadata
        let record = SessionLogRecord.Response(
            v: SessionLogRecord.schemaVersion, type: "response", ts: timestamp(), session: session, requestId: id,
            status: status, finishReason: meta?.finishReason.rawValue,
            tokens: meta.map { .init(prompt: $0.promptTokens, completion: $0.completionTokens, cached: $0.cachedPromptTokens) },
            timing: meta.map {
                .init(queueWaitMs: Self.milliseconds(queueWait), ttftMs: Int(($0.timeToFirstToken * 1_000).rounded()),
                      generationMs: Int(($0.generationTime * 1_000).rounded()), tokensPerSecond: $0.tokensPerSecond)
            },
            memory: meta.map { .init(activeBytes: $0.memory.activeBytes, peakBytes: $0.memory.peakBytes) },
            output: .init(
                chars: result.text.count, sha256: SessionLogRecord.Digest(result.text).sha256,
                text: mode == .full ? scrub(result.text) : nil,
                reasoning: mode == .full && !result.reasoning.isEmpty ? scrub(result.reasoning) : nil,
                toolCalls: mode == .full && !result.toolCalls.isEmpty ? (redacts ? SecretRedactor.redact(result.toolCalls) : result.toolCalls) : nil),
            toolCallsCount: result.toolCalls.count, error: error)
        append(record, session: record.session)
    }

    func scrub(_ text: String) -> String {
        redacts ? SecretRedactor.redact(text) : text
    }

    func remember(_ request: GenerationRequest, id: String, session: String) -> (id: String, count: Int)? {
        guard let messages = request.messages else { return nil }
        let digests = Self.digests(messages)
        var recent = recentConversations[session] ?? []
        let parent = recent.reversed().first { earlier in
            earlier.digests.count <= digests.count && digests.starts(with: earlier.digests)
        }
        recent.append((id, digests))
        if recentConversations[session] == nil, recentConversations.count >= 4_096 { recentConversations.removeAll() }
        recentConversations[session] = Array(recent.suffix(Self.prefixWindow))
        return parent.map { ($0.id, $0.digests.count) }
    }

    func content(of request: GenerationRequest, parent: (id: String, count: Int)?) -> SessionLogRecord.RequestContent {
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

    public func logRejection(origin: RequestOrigin, sessionId: String?, code: Int, message: String, bodyBytes: Int) {
        guard isEnabled else { return }
        let record = SessionLogRecord.Rejection(
            v: SessionLogRecord.schemaVersion, type: "error", ts: timestamp(), session: Self.sanitized(sessionId) ?? defaultSession,
            source: origin.source, endpoint: origin.endpoint, code: code, message: message, bodyBytes: bodyBytes)
        append(record, session: record.session)
    }

    public func logModelSwitch(from: String, to: String, duration: Duration) {
        guard isEnabled else { return }
        var event = SessionLogRecord.Event(v: SessionLogRecord.schemaVersion, type: "model_switch", ts: timestamp(), session: defaultSession)
        event.from = from
        event.to = to
        event.durationMs = Self.milliseconds(duration)
        append(event, session: defaultSession)
    }

    static func milliseconds(_ duration: Duration) -> Int {
        Int(duration.components.seconds * 1_000 + duration.components.attoseconds / 1_000_000_000_000_000)
    }

    private func append<R: Encodable>(_ record: R, session: String) {
        guard pending.count < bufferLimit else {
            dropped += 1
            return
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard var line = try? encoder.encode(record) else { return }
        line.append(0x0A)
        pending.append((session, line))
    }

    public func flush() {
        guard isEnabled else { return }
        suspendIdleSessions()
        guard !pending.isEmpty || dropped > 0 else { return }
        if dropped > 0 {
            var event = SessionLogRecord.Event(v: SessionLogRecord.schemaVersion, type: "dropped", ts: timestamp(), session: defaultSession)
            event.count = dropped
            dropped = 0
            if var line = try? JSONEncoder().encode(event) {
                line.append(0x0A)
                pending.append((defaultSession, line))
            }
        }
        writePending()
    }

    private func writePending() {
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        for entry in batch {
            guard let fd = handle(for: entry.session) else { return }
            let written = entry.line.withUnsafeBytes { Darwin.write(fd, $0.baseAddress, $0.count) }
            if written != entry.line.count {
                fail("write to the session log failed (errno \(errno))")
                return
            }
        }
    }

    private func handle(for session: String) -> Int32? {
        if let fd = handles[session] { return fd }
        let day = ISO8601DateFormatter.string(from: clock(), timeZone: .gmt, formatOptions: [.withFullDate])
        let folder = resumedPaths[session]?.deletingLastPathComponent() ?? directory.appendingPathComponent(day)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        } catch {
            fail("cannot create \(folder.path): \(error.localizedDescription)")
            return nil
        }
        let path = folder.appendingPathComponent("\(session).jsonl").path
        let isNew = !FileManager.default.fileExists(atPath: path)
        let fd = Darwin.open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0o600)
        guard fd >= 0 else {
            fail("cannot open \(path) (errno \(errno))")
            return nil
        }
        fchmod(fd, 0o600)
        handles[session] = fd
        if isNew {
            var start = SessionLogRecord.Event(v: SessionLogRecord.schemaVersion, type: "session_start", ts: timestamp(), session: session)
            start.version = HealthResponse.version
            start.pid = ProcessInfo.processInfo.processIdentifier
            start.os = ProcessInfo.processInfo.operatingSystemVersionString
            start.mode = mode.rawValue
            if var line = try? JSONEncoder().encode(start) {
                line.append(0x0A)
                _ = line.withUnsafeBytes { Darwin.write(fd, $0.baseAddress, $0.count) }
            }
        }
        return fd
    }

    private func fail(_ reason: String) {
        guard !disabled else { return }
        disabled = true
        pending.removeAll()
        BasaltLogger(module: "SessionLog").warn("Session log disabled until restart: \(reason)")
    }

    public static func applyRetention(directory: URL, days: Int, maxTotalBytes: Int64, now: Date = Date()) {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]) else { return }
        var files: [(url: URL, modified: Date, size: Int64)] = []
        for case let url as URL in enumerator where url.pathExtension == "jsonl" {
            let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            files.append((url, values?.contentModificationDate ?? now, Int64(values?.fileSize ?? 0)))
        }
        let cutoff = now.addingTimeInterval(-Double(max(0, days)) * 86_400)
        var kept: [(url: URL, modified: Date, size: Int64)] = []
        for file in files {
            if file.modified < cutoff { try? fm.removeItem(at: file.url) } else { kept.append(file) }
        }
        var total = kept.reduce(0) { $0 + $1.size }
        for file in kept.sorted(by: { $0.modified < $1.modified }) where total > maxTotalBytes {
            try? fm.removeItem(at: file.url)
            total -= file.size
        }
        if let days = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            for day in days where (try? fm.contentsOfDirectory(atPath: day.path))?.isEmpty == true {
                try? fm.removeItem(at: day)
            }
        }
    }
}
