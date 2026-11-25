import Darwin
import Foundation
import Synchronization

public enum BasaltLogLevel: Int, Comparable, Sendable {
    case trace = 0
    case debug
    case info
    case warn
    case error
    
    public static func < (lhs: BasaltLogLevel, rhs: BasaltLogLevel) -> Bool {
        return lhs.rawValue < rhs.rawValue
    }
}

public struct BasaltLogger: Sendable {
    private static let level = Mutex(BasaltLogLevel.info)

    public static var globalLevel: BasaltLogLevel {
        get { level.withLock { $0 } }
        set { level.withLock { $0 = newValue } }
    }
    
    public let module: String
    
    public init(module: String) {
        self.module = module
    }
    
    private static let dateFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return df
    }()
    
    /// `~/.basalt/logs/app.log` — not `./logs`.
    ///
    /// An installed binary is run from wherever the user happens to be: that
    /// either litters their directory or, somewhere read-only like `/`, fails
    /// and prints a Cocoa error on every start. The launcher already keeps its
    /// server log in `~/.basalt`, so this is the same place.
    private static let logFileURL: URL? = {
        let fm = FileManager.default
        let logsDir = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".basalt")
            .appendingPathComponent("logs")
        do {
            try fm.createDirectory(at: logsDir, withIntermediateDirectories: true)
            return logsDir.appendingPathComponent("app.log")
        } catch {
            // File logging is a convenience; stderr already has everything.
            FileHandle.standardError.write(
                Data("[log] cannot use \(logsDir.path): \(error.localizedDescription)\n".utf8))
            return nil
        }
    }()
    
    public func log(_ level: BasaltLogLevel, _ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        guard level >= BasaltLogger.globalLevel else { return }
        
        let dateStr = BasaltLogger.dateFormatter.string(from: Date())
        let levelStr: String
        switch level {
        case .trace: levelStr = "🔘 TRACE"
        case .debug: levelStr = "🟢 DEBUG"
        case .info:  levelStr = "🔵 INFO "
        case .warn:  levelStr = "🟠 WARN "
        case .error: levelStr = "🔴 ERROR"
        }
        
        let filename = (file as NSString).lastPathComponent
        let formatted = "\(dateStr) \(levelStr) [\(module)] \(filename):\(line) - \(message)"
        
        fputs(formatted + "\n", stderr)
        if let url = BasaltLogger.logFileURL { Self.append(formatted + "\n", to: url.path) }
    }

    static func append(_ text: String, to path: String) {
        let descriptor = Darwin.open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else { return }
        defer { _ = Darwin.close(descriptor) }
        _ = text.withCString { Darwin.write(descriptor, $0, strlen($0)) }
    }
    
    public func trace(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(.trace, message, file: file, function: function, line: line)
    }
    
    public func debug(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(.debug, message, file: file, function: function, line: line)
    }
    
    public func info(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(.info, message, file: file, function: function, line: line)
    }
    
    public func warn(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(.warn, message, file: file, function: function, line: line)
    }
    
    public func error(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(.error, message, file: file, function: function, line: line)
    }
}
