import Foundation
import CryptoKit
import System

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
        let toolCallRepairs: Int?
        let toolCallRejections: Int?
        let error: String?

        enum CodingKeys: String, CodingKey {
            case v, type, ts, session, requestId = "request_id", status, finishReason = "finish_reason"
            case tokens, timing, memory, output, toolCallsCount = "tool_calls_count", error
            case toolCallRepairs = "tool_call_repairs", toolCallRejections = "tool_call_rejections"
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
    nonisolated public let redacts: Bool
    nonisolated public let idleTimeout: TimeInterval
    static let prefixWindow = SessionConversations.window
    let bufferLimit: Int
    private let clock: @Sendable () -> Date

    private var pending: [(session: String, line: Data)] = []
    private var dropped = 0
    private var handles: [String: Int32] = [:]
    private var disabled = false
    private var flushTask: Task<Void, Never>?
    private var conversations = SessionConversations()
    private var activity = SessionActivity()
    private var requestSessions: [String: String] = [:]
    private var resumedPaths: [String: URL] = [:]

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

    public var isEnabled: Bool { mode != .off && !disabled }

    nonisolated var format: SessionLogFormat { .init(mode: mode, redacts: redacts) }
}

extension SessionLogWriter {
    public static func processSession() -> String {
        let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .gmt, formatOptions: [.withFullDate, .withTime])
        return "server-\(stamp)-\(ProcessInfo.processInfo.processIdentifier)"
    }

    public static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".basalt/sessions")
    }

    static func sanitized(_ id: String?) -> String? {
        guard let id, !id.isEmpty else { return nil }
        let safe = String(id.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) || "-_.".unicodeScalars.contains($0) ? Character($0) : "_" })
        return safe.hasPrefix(".") ? "_" + safe.dropFirst() : safe
    }

    public static func applyRetention(directory: URL, days: Int, maxTotalBytes: Int64, now: Date = Date()) {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]) else { return }
        var files: [SessionLogRetention.File] = []
        for case let url as URL in enumerator where url.pathExtension == "jsonl" {
            let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            files.append(.init(url: url, modified: values?.contentModificationDate ?? now, size: Int64(values?.fileSize ?? 0)))
        }
        for url in SessionLogRetention(days: days, maxTotalBytes: maxTotalBytes).deletions(among: files, now: now) {
            try? fm.removeItem(at: url)
        }
        if let days = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            for day in days where (try? fm.contentsOfDirectory(atPath: day.path))?.isEmpty == true {
                try? fm.removeItem(at: day)
            }
        }
    }
}

extension SessionLogWriter {
    public func logRequest(_ request: GenerationRequest, id: String, origin: RequestOrigin, modelRequested: String?) {
        guard isEnabled else { return }
        let session = session(for: request)
        requestSessions[id] = session
        prime(session)
        let parent = conversations.remember(request.messages, id: id, session: session)
        let record = format.request(request, id: id, origin: origin, modelRequested: modelRequested, parent: parent, at: stamp(session))
        append(record, session: session)
    }

    public func logResponse(
        _ request: GenerationRequest, id: String, status: String, result: SessionLogResult, queueWait: Duration, error: String? = nil
    ) {
        guard isEnabled else { return }
        let session = requestSessions.removeValue(forKey: id) ?? session(for: request)
        activity.touch(session, at: clock())
        let record = format.response(id: id, status: status, result: result, queueWait: queueWait, error: error, at: stamp(session))
        append(record, session: session)
    }

    public func logRejection(origin: RequestOrigin, sessionId: String?, code: Int, message: String, bodyBytes: Int) {
        guard isEnabled else { return }
        let stamped = stamp(Self.sanitized(sessionId) ?? defaultSession)
        let record = SessionLogRecord.Rejection(
            v: SessionLogRecord.schemaVersion, type: "error", ts: stamped.ts, session: stamped.session,
            source: origin.source, endpoint: origin.endpoint, code: code, message: message, bodyBytes: bodyBytes)
        append(record, session: stamped.session)
    }

    public func logModelSwitch(from: String, to: String, duration: Duration) {
        guard isEnabled else { return }
        appendEvent("model_switch", session: defaultSession) {
            $0.from = from
            $0.to = to
            $0.durationMs = SessionLogFormat.milliseconds(duration)
        }
    }

    func session(for request: GenerationRequest) -> String {
        if let explicit = Self.sanitized(request.sessionId) { return explicit }
        guard let messages = request.messages, !messages.isEmpty else { return defaultSession }
        return conversations.session(continuing: SessionConversations.digests(messages))
            ?? "s-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12).lowercased()
    }

    func prime(_ session: String) {
        let now = clock()
        switch activity.start(session, at: now) {
        case .resumed(let gapS): appendEvent("session_resume", session: session) { $0.gapS = gapS }
        case .firstSeen where handles[session] == nil: resumeFromDisk(session, now: now)
        case .firstSeen, .seen: break
        }
    }

    private func resumeFromDisk(_ session: String, now: Date) {
        guard let existing = SessionLogReader.file(for: session, in: directory) else { return }
        resumedPaths[session] = existing
        let modified = (try? existing.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? now
        appendEvent("session_resume", session: session) {
            $0.gapS = max(0, Int(now.timeIntervalSince(modified)))
            $0.resumedFrom = existing.deletingLastPathComponent().lastPathComponent + "/" + existing.lastPathComponent
        }
        if mode == .full, !conversations.knows(session) {
            conversations.seed(session, with: SessionLogReader.recentConversations(
                in: SessionLogReader.lines(of: existing), limit: Self.prefixWindow).map { ($0.id, SessionConversations.digests($0.messages)) })
        }
    }

    func suspendIdleSessions() {
        let now = clock()
        for (session, last) in activity.lastActivity where handles[session] != nil {
            guard let idle = SessionActivity.idleSeconds(since: last, now: now, timeout: idleTimeout) else { continue }
            appendEvent("session_suspend", session: session) { $0.idleS = idle }
            writePending()
            if let fd = handles.removeValue(forKey: session) { Darwin.close(fd) }
            activity.suspend(session, at: now)
        }
    }

    private func stamp(_ session: String) -> SessionLogRecord.Stamp {
        .init(ts: ISO8601DateFormatter.string(from: clock(), timeZone: .gmt, formatOptions: [.withInternetDateTime, .withFractionalSeconds]),
              session: session)
    }

    private func appendEvent(_ type: String, session: String, _ fill: (inout SessionLogRecord.Event) -> Void) {
        var event = SessionLogRecord.Event(type, at: stamp(session))
        fill(&event)
        append(event, session: session)
    }
}

extension SessionLogWriter {
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

    public func flush() {
        guard isEnabled else { return }
        suspendIdleSessions()
        guard !pending.isEmpty || dropped > 0 else { return }
        if dropped > 0 {
            var event = SessionLogRecord.Event("dropped", at: stamp(defaultSession))
            event.count = dropped
            dropped = 0
            if let line = SessionLogFormat.line(event, sorted: false) { pending.append((defaultSession, line)) }
        }
        writePending()
    }

    private func append(_ record: some Encodable, session: String) {
        guard pending.count < bufferLimit else {
            dropped += 1
            return
        }
        guard let line = SessionLogFormat.line(record) else { return }
        pending.append((session, line))
    }

    private func writePending() {
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        for entry in batch {
            guard let fd = handle(for: entry.session) else { return }
            do {
                try FileDescriptor(rawValue: fd).writeAll(entry.line)
            } catch {
                fail("write to the session log failed (errno \((error as? Errno)?.rawValue ?? errno))")
                return
            }
        }
    }

    private func handle(for session: String) -> Int32? {
        if let fd = handles[session] { return fd }
        guard let folder = folder(for: session) else { return nil }
        let path = folder.appendingPathComponent("\(session).jsonl").path
        let isNew = !FileManager.default.fileExists(atPath: path)
        let fd = Darwin.open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0o600)
        guard fd >= 0 else {
            fail("cannot open \(path) (errno \(errno))")
            return nil
        }
        fchmod(fd, 0o600)
        handles[session] = fd
        if isNew { writeStart(of: session, to: fd) }
        return fd
    }

    private func folder(for session: String) -> URL? {
        let day = ISO8601DateFormatter.string(from: clock(), timeZone: .gmt, formatOptions: [.withFullDate])
        let folder = resumedPaths[session]?.deletingLastPathComponent() ?? directory.appendingPathComponent(day)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
            return folder
        } catch {
            fail("cannot create \(folder.path): \(error.localizedDescription)")
            return nil
        }
    }

    private func writeStart(of session: String, to fd: Int32) {
        var start = SessionLogRecord.Event("session_start", at: stamp(session))
        start.version = HealthResponse.version
        start.pid = ProcessInfo.processInfo.processIdentifier
        start.os = ProcessInfo.processInfo.operatingSystemVersionString
        start.mode = mode.rawValue
        if let line = SessionLogFormat.line(start, sorted: false) { _ = try? FileDescriptor(rawValue: fd).writeAll(line) }
    }

    private func fail(_ reason: String) {
        guard !disabled else { return }
        disabled = true
        pending.removeAll()
        BasaltLogger(module: "SessionLog").warn("Session log disabled until restart: \(reason)")
    }
}
