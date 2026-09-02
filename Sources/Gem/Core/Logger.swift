import Foundation

public enum GemLogLevel: Int, Comparable, Sendable {
    case trace = 0
    case debug
    case info
    case warn
    case error
    
    public static func < (lhs: GemLogLevel, rhs: GemLogLevel) -> Bool {
        return lhs.rawValue < rhs.rawValue
    }
}

public struct GemLogger: Sendable {
    public static nonisolated(unsafe) var globalLevel: GemLogLevel = .info
    
    public let module: String
    
    public init(module: String) {
        self.module = module
    }
    
    private static let dateFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return df
    }()
    
    /// `~/.gemm/logs/app.log` — not `./logs`.
    ///
    /// An installed binary is run from wherever the user happens to be: that
    /// either litters their directory or, somewhere read-only like `/`, fails
    /// and prints a Cocoa error on every start. The launcher already keeps its
    /// server log in `~/.gemm`, so this is the same place.
    private static let logFileURL: URL? = {
        let fm = FileManager.default
        let logsDir = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".gemm")
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
    
    public func log(_ level: GemLogLevel, _ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        guard level >= GemLogger.globalLevel else { return }
        
        let dateStr = GemLogger.dateFormatter.string(from: Date())
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
        
        // Always write to stderr — stdout is reserved for MCP JSON-RPC in stdio mode
        fputs(formatted + "\n", stderr)
        
        if let url = GemLogger.logFileURL {
            let logData = (formatted + "\n").data(using: .utf8)
            if FileManager.default.fileExists(atPath: url.path) {
                if let fileHandle = try? FileHandle(forWritingTo: url) {
                    fileHandle.seekToEndOfFile()
                    if let data = logData {
                        fileHandle.write(data)
                    }
                    fileHandle.closeFile()
                }
            } else {
                try? logData?.write(to: url)
            }
        }
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
