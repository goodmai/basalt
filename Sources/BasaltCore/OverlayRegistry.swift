import Foundation
import System

public struct OverlayRegistry: Sendable {
    public struct Overlay: Sendable {
        public var info: OverlayInfo
        public var intervention: Intervention?
    }

    struct JournalEntry: Codable, Equatable {
        var event: String
        var id: Int
        var description: String
        var at: String
        var expiresAt: String?
    }

    public static let defaultLifetime: TimeInterval = 30 * 60
    public static let defaultJournal = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".basalt/experiments")

    public var journal: URL?
    public private(set) var overlays: [Overlay] = []
    private var lastID = 0

    public init(journal: URL? = OverlayRegistry.defaultJournal) {
        self.journal = journal
    }

    public mutating func add(
        _ intervention: Intervention?, description: String, at now: Date, lifetime: TimeInterval? = defaultLifetime
    ) -> OverlayInfo {
        lastID += 1
        let info = OverlayInfo(id: OverlayID(lastID), description: description, createdAt: now, expiresAt: lifetime.map { now.addingTimeInterval($0) })
        overlays.append(Overlay(info: info, intervention: intervention))
        return info
    }

    public mutating func remove(_ id: OverlayID) -> Overlay? {
        overlays.firstIndex { $0.info.id == id }.map { overlays.remove(at: $0) }
    }

    public mutating func expire(at now: Date) -> [Overlay] {
        let expired = overlays.filter { Self.isExpired($0, at: now) }
        overlays.removeAll { Self.isExpired($0, at: now) }
        return expired
    }

    public mutating func removeAll() {
        overlays.removeAll()
    }

    static func isExpired(_ overlay: Overlay, at now: Date) -> Bool {
        overlay.info.expiresAt.map { $0 <= now } ?? false
    }

    static func journalLine(_ event: String, _ info: OverlayInfo, at now: Date) -> Data? {
        SessionLogFormat.line(JournalEntry(
            event: event, id: info.id.rawValue, description: info.description,
            at: now.formatted(.iso8601), expiresAt: info.expiresAt?.formatted(.iso8601)))
    }

    public func record(_ event: String, _ info: OverlayInfo, at now: Date, fingerprint: String) throws {
        guard let journal, let line = Self.journalLine(event, info, at: now) else { return }
        try FileManager.default.createDirectory(at: journal, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: journal.path)
        let file = FilePath(journal.appendingPathComponent("\(fingerprint).jsonl").path)
        let fd = try FileDescriptor.open(file, .writeOnly, options: [.create, .append, .closeOnExec], permissions: .ownerReadWrite)
        _ = try fd.closeAfter { try fd.writeAll(line) }
    }
}
