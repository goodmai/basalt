import BasaltCore
import Foundation

public struct EngineRecord: Codable, Sendable, Equatable {
    public var pid: Int32
    public var port: Int
    public var token: String
    public var version: String
    public var arguments: [String]?

    public init(pid: Int32, port: Int, token: String, version: String = HealthResponse.version, arguments: [String]? = nil) {
        (self.pid, self.port, self.token, self.version, self.arguments) = (pid, port, token, version, arguments)
    }
}

public struct EngineRecords: Sendable {
    public var load: @Sendable () -> EngineRecord?
    public var save: @Sendable (EngineRecord) -> Void
    public var remove: @Sendable () -> Void

    public init(load: @escaping @Sendable () -> EngineRecord?, save: @escaping @Sendable (EngineRecord) -> Void, remove: @escaping @Sendable () -> Void) {
        (self.load, self.save, self.remove) = (load, save, remove)
    }

    public static func file(_ url: URL) -> EngineRecords {
        EngineRecords(
            load: { (try? Data(contentsOf: url)).flatMap { try? JSONDecoder().decode(EngineRecord.self, from: $0) } },
            save: { record in
                try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                guard let data = try? JSONEncoder().encode(record) else { return }
                FileManager.default.createFile(atPath: url.path, contents: data, attributes: [.posixPermissions: 0o600])
            },
            remove: { try? FileManager.default.removeItem(at: url) })
    }

    public static var standard: EngineRecords {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return .file(support.appending(path: "Basalt/engine.json"))
    }
}

public struct EngineEnvironment: Sendable {
    public var now: @Sendable () -> Duration
    public var sleep: @Sendable (Duration) async throws -> Void
    public var probe: @Sendable (EngineEndpoint) async -> Bool
    public var records: EngineRecords
    public var isAlive: @Sendable (Int32) -> Bool

    public init(
        now: @escaping @Sendable () -> Duration, sleep: @escaping @Sendable (Duration) async throws -> Void,
        probe: @escaping @Sendable (EngineEndpoint) async -> Bool, records: EngineRecords,
        isAlive: @escaping @Sendable (Int32) -> Bool = { kill($0, 0) == 0 }
    ) {
        (self.now, self.sleep, self.probe, self.records, self.isAlive) = (now, sleep, probe, records, isAlive)
    }

    public static var live: EngineEnvironment {
        let origin = ContinuousClock.now
        return EngineEnvironment(
            now: { ContinuousClock.now - origin }, sleep: { try await Task.sleep(for: $0) },
            probe: { await EngineClient.http($0).health() }, records: .standard)
    }
}
