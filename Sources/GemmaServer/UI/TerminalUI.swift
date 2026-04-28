import Foundation
@preconcurrency import Rainbow

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Epic 16.1: Rich Terminal UI Foundation
/// Provides colored and styled terminal output using Rainbow library
public enum TerminalUI: Sendable {
    
    // MARK: - Configuration
    
    /// Global flag to enable/disable colors
    /// Auto-detected based on TTY and environment variables
    /// Note: Using nonisolated(unsafe) for test compatibility
    public nonisolated(unsafe) static var colorsEnabled: Bool = {
        let enabled = detectColorSupport()
        // Synchronize with Rainbow
        Rainbow.enabled = enabled
        return enabled
    }() {
        didSet {
            Rainbow.enabled = colorsEnabled
        }
    }
    
    private static func detectColorSupport() -> Bool {
        // Check NO_COLOR environment variable (https://no-color.org/)
        if ProcessInfo.processInfo.environment["NO_COLOR"] != nil {
            return false
        }
        
        // Check FORCE_COLOR environment variable
        if ProcessInfo.processInfo.environment["FORCE_COLOR"] != nil {
            return true
        }
        
        // Auto-detect TTY
        #if canImport(Darwin)
        return isatty(fileno(stdout)) == 1
        #else
        return false
        #endif
    }
    
    // MARK: - TTY Detection
    
    /// Check if stdout is connected to a terminal
    public static var isStdoutTTY: Bool {
        #if canImport(Darwin)
        return isatty(fileno(stdout)) == 1
        #else
        return false
        #endif
    }
    
    /// Check if stderr is connected to a terminal
    public static var isStderrTTY: Bool {
        #if canImport(Darwin)
        return isatty(fileno(stderr)) == 1
        #else
        return false
        #endif
    }
    
    // MARK: - Color Helpers
    
    /// Format text as success (green, bold)
    public static func success(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.green.bold
    }
    
    /// Format text as error (red, bold)
    public static func error(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.red.bold
    }
    
    /// Format text as warning (yellow)
    public static func warning(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.yellow
    }
    
    /// Format text as info (blue)
    public static func info(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.blue
    }
    
    /// Format text as dimmed/subtle (gray, dim)
    public static func dim(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.dim
    }
    
    /// Format text as code (cyan, for inline code)
    public static func code(_ text: String) -> String {
        guard colorsEnabled else { return "`\(text)`" }
        return text.cyan
    }
    
    // MARK: - Style Helpers
    
    /// Make text bold
    public static func bold(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.bold
    }
    
    /// Underline text
    public static func underline(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.underline
    }
    
    /// Italic text (not widely supported in terminals)
    public static func italic(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.italic
    }
    
    // MARK: - Composite Helpers
    
    /// Format text as heading (bold, blue)
    public static func heading(_ text: String) -> String {
        guard colorsEnabled else { return text }
        return text.blue.bold
    }
    
    /// Format text as code block (cyan background, black text)
    public static func codeBlock(_ text: String) -> String {
        guard colorsEnabled else { return "```\n\(text)\n```" }
        return text.onCyan.black
    }
}

// MARK: - Output Mode (Epic 16.9)

/// Output mode for CLI commands
public enum OutputMode: String, Sendable, CaseIterable {
    case json    // Machine-readable JSON output
    case plain   // Plain text without colors
    case pretty  // Human-readable with colors and formatting
    
    /// Auto-detect output mode based on TTY
    public static var auto: OutputMode {
        TerminalUI.isStdoutTTY ? .pretty : .plain
    }
}

/// Formatter for consistent output across different modes
public struct OutputFormatter: Sendable {
    public let mode: OutputMode
    
    public init(mode: OutputMode) {
        self.mode = mode
    }
    
    /// Format success output (string)
    public func formatSuccess(_ value: String) -> String {
        switch mode {
        case .json:
            let dict = ["result": value]
            return formatJSON(dict)
        case .plain:
            return value
        case .pretty:
            return TerminalUI.success(value)
        }
    }
    
    /// Format success output (Codable)
    public func formatSuccess<T>(_ value: T) -> String where T: Encodable {
        switch mode {
        case .json:
            return formatJSON(value)
        case .plain, .pretty:
            if let customString = value as? CustomStringConvertible {
                return mode == .pretty ? TerminalUI.success(customString.description) : customString.description
            }
            return formatJSON(value)
        }
    }
    
    /// Format error output
    public func formatError(_ error: Error) -> String {
        let message = error.localizedDescription
        
        switch mode {
        case .json:
            let errorDict = ["error": message]
            return formatJSON(errorDict)
        case .plain:
            return "error: \(message)"
        case .pretty:
            return "\(TerminalUI.error("Error:"))\n\(message)"
        }
    }
    
    // MARK: - Private Helpers
    
    private func formatJSON<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        guard let data = try? encoder.encode(value),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        
        return json
    }
}

// MARK: - Convenience Extensions

extension String {
    /// Quick access to TerminalUI helpers
    public var asSuccess: String { TerminalUI.success(self) }
    public var asError: String { TerminalUI.error(self) }
    public var asWarning: String { TerminalUI.warning(self) }
    public var asInfo: String { TerminalUI.info(self) }
    public var asDim: String { TerminalUI.dim(self) }
    public var asCode: String { TerminalUI.code(self) }
}
