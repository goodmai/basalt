import Foundation

struct ComposeRecord: Codable, Sendable, Equatable {
    var name: String
    var pid: Int32
    var port: Int
    var file: String
    var started: Int?

    static let directory = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/compose")

    static func url(_ name: String) -> URL { directory.appending(path: "\(name).json") }

    static func write(name: String, port: Int, file: String, pid: Int32 = ProcessInfo.processInfo.processIdentifier) throws -> String {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let record = ComposeRecord(name: name, pid: pid, port: port, file: file, started: startTime(pid))
        try JSONEncoder().encode(record).write(to: url(name), options: .atomic)
        return name
    }

    static func remove(_ name: String) {
        guard read(name)?.pid == ProcessInfo.processInfo.processIdentifier else { return }
        try? FileManager.default.removeItem(at: url(name))
    }

    static func read(_ name: String) -> ComposeRecord? {
        (try? Data(contentsOf: url(name))).flatMap { try? JSONDecoder().decode(ComposeRecord.self, from: $0) }
    }

    static func running(_ name: String) -> ComposeRecord? {
        guard let record = read(name) else { return nil }
        guard alive(record) else {
            try? FileManager.default.removeItem(at: url(name))
            return nil
        }
        return record
    }

    static func alive(_ record: ComposeRecord) -> Bool {
        guard kill(record.pid, 0) == 0 else { return false }
        return record.started.map { $0 == startTime(record.pid) } ?? true
    }

    static func startTime(_ pid: Int32) -> Int? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.size)
        guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else { return nil }
        return Int(info.pbi_start_tvsec)
    }

    static func all() -> [ComposeRecord] {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return files.filter { $0.hasSuffix(".json") }.compactMap { running(String($0.dropLast(5))) }.sorted { $0.name < $1.name }
    }
}
