import Foundation

/// Epic 16.6: Clipboard Integration
/// Copy/Paste functionality with cross-platform support

// MARK: - Clipboard Errors

public enum ClipboardError: Error, LocalizedError {
    case toolNotAvailable
    case operationFailed(String)
    case notSupported
    
    public var errorDescription: String? {
        switch self {
        case .toolNotAvailable:
            return "Clipboard tool not available. Install pbcopy (macOS) or xclip/xsel (Linux)."
        case .operationFailed(let message):
            return "Clipboard operation failed: \(message)"
        case .notSupported:
            return "Clipboard not supported on this platform"
        }
    }
}

// MARK: - Clipboard Tool

/// Available clipboard tools
public enum ClipboardTool: String, Sendable {
    case pbcopy    // macOS
    case pbpaste   // macOS
    case xclip     // Linux
    case xsel      // Linux
    case none      // Not available
    
    var copyCommand: [String]? {
        switch self {
        case .pbcopy:
            return ["/usr/bin/pbcopy"]
        case .xclip:
            return ["/usr/bin/xclip", "-selection", "clipboard"]
        case .xsel:
            return ["/usr/bin/xsel", "--clipboard", "--input"]
        default:
            return nil
        }
    }
    
    var pasteCommand: [String]? {
        switch self {
        case .pbpaste:
            return ["/usr/bin/pbpaste"]
        case .xclip:
            return ["/usr/bin/xclip", "-selection", "clipboard", "-o"]
        case .xsel:
            return ["/usr/bin/xsel", "--clipboard", "--output"]
        default:
            return nil
        }
    }
}

// MARK: - Clipboard Manager

/// Cross-platform clipboard manager
public actor ClipboardManager {
    
    private var detectedTool: ClipboardTool?
    
    public init() {}
    
    // MARK: - Main API
    
    /// Copy text to clipboard
    public func copy(_ text: String) async throws {
        // Don't try to copy in CI environments
        if Self.isRunningInCI() {
            throw ClipboardError.notSupported
        }
        
        let tool = await detectClipboardTool()
        
        guard tool != .none else {
            throw ClipboardError.toolNotAvailable
        }
        
        guard let command = tool.copyCommand else {
            throw ClipboardError.toolNotAvailable
        }
        
        let _ = try await runClipboardCommand(command, input: text)
    }
    
    /// Paste text from clipboard
    public func paste() async throws -> String {
        // Don't try to paste in CI environments
        if Self.isRunningInCI() {
            throw ClipboardError.notSupported
        }
        
        let tool = await detectPasteTool()
        
        guard tool != .none else {
            throw ClipboardError.toolNotAvailable
        }
        
        guard let command = tool.pasteCommand else {
            throw ClipboardError.toolNotAvailable
        }
        
        return try await runClipboardCommand(command)
    }
    
    /// Check if clipboard is available
    public func isAvailable() async -> Bool {
        if Self.isRunningInCI() {
            return false
        }
        
        let tool = await detectClipboardTool()
        return tool != .none
    }
    
    // MARK: - Tool Detection
    
    /// Detect available clipboard tool for copying
    public func detectClipboardTool() async -> ClipboardTool {
        if let cached = detectedTool {
            return cached
        }
        
        let tool: ClipboardTool
        
        #if os(macOS)
        // macOS always has pbcopy
        tool = FileManager.default.fileExists(atPath: "/usr/bin/pbcopy") ? .pbcopy : .none
        #elseif os(Linux)
        // Check xclip first, then xsel
        if FileManager.default.fileExists(atPath: "/usr/bin/xclip") {
            tool = .xclip
        } else if FileManager.default.fileExists(atPath: "/usr/bin/xsel") {
            tool = .xsel
        } else {
            tool = .none
        }
        #else
        tool = .none
        #endif
        
        detectedTool = tool
        return tool
    }
    
    /// Detect available clipboard tool for pasting
    private func detectPasteTool() async -> ClipboardTool {
        let copyTool = await detectClipboardTool()
        
        switch copyTool {
        case .pbcopy:
            return .pbpaste
        case .xclip:
            return .xclip
        case .xsel:
            return .xsel
        default:
            return .none
        }
    }
    
    // MARK: - Process Execution
    
    /// Run clipboard command with optional input
    private func runClipboardCommand(_ command: [String], input: String? = nil) async throws -> String {
        guard !command.isEmpty else {
            throw ClipboardError.operationFailed("Empty command")
        }
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command[0])
        
        if command.count > 1 {
            process.arguments = Array(command.dropFirst())
        }
        
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        
        if input != nil {
            process.standardInput = inputPipe
        }
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        do {
            try process.run()
            
            // Write input if provided
            if let input = input {
                let inputData = Data(input.utf8)
                try inputPipe.fileHandleForWriting.write(contentsOf: inputData)
                try inputPipe.fileHandleForWriting.close()
            }
            
            process.waitUntilExit()
            
            // Check exit status
            guard process.terminationStatus == 0 else {
                let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
                let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
                throw ClipboardError.operationFailed(errorMessage)
            }
            
            // Read output
            let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: outputData, encoding: .utf8) ?? ""
            
        } catch let error as ClipboardError {
            throw error
        } catch {
            throw ClipboardError.operationFailed(error.localizedDescription)
        }
    }
    
    // MARK: - Static Helpers
    
    /// Get tool path for a specific tool
    public static func toolPath(for tool: ClipboardTool) -> String? {
        switch tool {
        case .pbcopy:
            return "/usr/bin/pbcopy"
        case .pbpaste:
            return "/usr/bin/pbpaste"
        case .xclip:
            return "/usr/bin/xclip"
        case .xsel:
            return "/usr/bin/xsel"
        case .none:
            return nil
        }
    }
    
    /// Check if running in CI environment
    public static func isRunningInCI() -> Bool {
        let ciKeys = ["CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI"]
        let env = ProcessInfo.processInfo.environment
        
        return ciKeys.contains { env[$0] != nil }
    }
    
    /// Get installation instructions for missing tools
    public static func installationInstructions() -> String {
        #if os(macOS)
        return "Clipboard is built-in on macOS (pbcopy/pbpaste)"
        #elseif os(Linux)
        return """
        Install clipboard tools:
        
        Ubuntu/Debian:
          sudo apt-get install xclip
          # or
          sudo apt-get install xsel
        
        Fedora:
          sudo dnf install xclip
          # or
          sudo dnf install xsel
        
        Arch:
          sudo pacman -S xclip
          # or
          sudo pacman -S xsel
        """
        #else
        return "Clipboard not supported on this platform"
        #endif
    }
}

// MARK: - String Extension

extension String {
    /// Copy this string to clipboard
    public func copyToClipboard() async throws {
        let manager = ClipboardManager()
        try await manager.copy(self)
    }
}

// MARK: - User-Friendly Helpers

extension ClipboardManager {
    /// Copy with user feedback
    public func copyWithFeedback(_ text: String) async -> Bool {
        do {
            try await copy(text)
            print("✅ Copied to clipboard")
            return true
        } catch ClipboardError.toolNotAvailable {
            print("⚠️  Clipboard not available")
            print(Self.installationInstructions())
            return false
        } catch ClipboardError.notSupported {
            // Silently fail in CI
            return false
        } catch {
            print("❌ Failed to copy: \(error.localizedDescription)")
            return false
        }
    }
    
    /// Paste with user feedback
    public func pasteWithFeedback() async -> String? {
        do {
            let text = try await paste()
            if !text.isEmpty {
                print("📋 Pasted \(text.count) characters from clipboard")
            }
            return text
        } catch ClipboardError.toolNotAvailable {
            print("⚠️  Clipboard not available")
            print(Self.installationInstructions())
            return nil
        } catch ClipboardError.notSupported {
            // Silently fail in CI
            return nil
        } catch {
            print("❌ Failed to paste: \(error.localizedDescription)")
            return nil
        }
    }
}
