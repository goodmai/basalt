import Foundation
#if canImport(Glibc)
import Glibc
#endif

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

public enum ClipboardTool: String, Sendable {
    case pbcopy
    case pbpaste
    case xclip
    case xsel
    case none

    var copyCommand: [String]? {
        switch self {
        case .pbcopy:
            return ["/usr/bin/pbcopy"]
        case .xclip:
            return ["/usr/bin/xclip", "-selection", "clipboard"]
        case .xsel:
            return ["/usr/bin/xsel", "--clipboard", "--input"]
        case .pbpaste, .none:
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
        case .pbcopy, .none:
            return nil
        }
    }
}

private struct ClipboardPipePump {
    static let chunkSize = 65_536

    var stdinHandle: FileHandle?
    var stdoutHandle: FileHandle?
    var stderrHandle: FileHandle?
    var offset = 0
    var stdoutData = Data()
    var stderrData = Data()

    var isActive: Bool {
        stdinHandle != nil || stdoutHandle != nil || stderrHandle != nil
    }

    mutating func closeAll() {
        try? stdinHandle?.close()
        try? stdoutHandle?.close()
        try? stderrHandle?.close()
        stdinHandle = nil
        stdoutHandle = nil
        stderrHandle = nil
    }

    static func configure(_ handle: FileHandle, isWrite: Bool) {
        let fd = handle.fileDescriptor
        let flags = fcntl(fd, F_GETFL)
        if flags >= 0 {
            _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        }
        #if canImport(Darwin)
        if isWrite {
            _ = fcntl(fd, F_SETNOSIGPIPE, 1)
        }
        #endif
    }

    static func drainStdin(_ handle: FileHandle?, bytes: borrowing Span<UInt8>, offset: inout Int) -> FileHandle? {
        guard let handle else { return nil }
        let fd = handle.fileDescriptor
        while offset < bytes.count {
            let written = bytes.extracting(offset...).withUnsafeBytes { raw -> Int in
                guard let base = raw.baseAddress, raw.count > 0 else { return 0 }
                return write(fd, base, raw.count)
            }
            if written > 0 {
                offset += written
                continue
            }
            if written < 0 && errno == EINTR { continue }
            if written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) { return handle }
            try? handle.close()
            return nil
        }
        try? handle.close()
        return nil
    }

    static func drainOutput(_ handle: FileHandle?, into data: inout Data, scratch: UnsafeMutableBufferPointer<UInt8>) -> FileHandle? {
        guard let handle, let base = scratch.baseAddress else { return nil }
        let fd = handle.fileDescriptor
        while true {
            let count = read(fd, base, scratch.count)
            if count > 0 {
                data.append(base, count: count)
                continue
            }
            if count == 0 {
                try? handle.close()
                return nil
            }
            if errno == EINTR { continue }
            if errno == EAGAIN || errno == EWOULDBLOCK { return handle }
            try? handle.close()
            return nil
        }
    }

    mutating func step(bytes: borrowing Span<UInt8>, scratch: UnsafeMutableBufferPointer<UInt8>) throws {
        var fds: [3 of pollfd] = [
            pollfd(fd: stdinHandle?.fileDescriptor ?? -1, events: Int16(POLLOUT), revents: 0),
            pollfd(fd: stdoutHandle?.fileDescriptor ?? -1, events: Int16(POLLIN), revents: 0),
            pollfd(fd: stderrHandle?.fileDescriptor ?? -1, events: Int16(POLLIN), revents: 0),
        ]
        var fdSpan = fds.mutableSpan
        let ready = fdSpan.withUnsafeMutableBufferPointer { ptr in
            poll(ptr.baseAddress, nfds_t(ptr.count), 50)
        }
        if ready == 0 { return }
        if ready < 0 {
            if errno == EINTR { return }
            throw ClipboardError.operationFailed(String(cString: strerror(errno)))
        }
        if fds[0].revents & Int16(POLLOUT | POLLHUP | POLLERR) != 0 {
            stdinHandle = Self.drainStdin(stdinHandle, bytes: bytes, offset: &offset)
        } else if fds[0].revents & Int16(POLLNVAL) != 0 {
            stdinHandle = nil
        }
        if fds[1].revents & Int16(POLLIN | POLLHUP | POLLERR) != 0 {
            stdoutHandle = Self.drainOutput(stdoutHandle, into: &stdoutData, scratch: scratch)
        } else if fds[1].revents & Int16(POLLNVAL) != 0 {
            stdoutHandle = nil
        }
        if fds[2].revents & Int16(POLLIN | POLLHUP | POLLERR) != 0 {
            stderrHandle = Self.drainOutput(stderrHandle, into: &stderrData, scratch: scratch)
        } else if fds[2].revents & Int16(POLLNVAL) != 0 {
            stderrHandle = nil
        }
    }

    static func run(
        stdin writer: FileHandle?,
        inputBytes: borrowing Span<UInt8>,
        stdout reader: FileHandle,
        stderr errorReader: FileHandle
    ) throws -> (stdout: Data, stderr: Data) {
        var pump = ClipboardPipePump(stdinHandle: writer, stdoutHandle: reader, stderrHandle: errorReader)
        defer { pump.closeAll() }
        if let currentWriter = pump.stdinHandle {
            if inputBytes.isEmpty {
                try? currentWriter.close()
                pump.stdinHandle = nil
            } else {
                configure(currentWriter, isWrite: true)
            }
        }
        configure(reader, isWrite: false)
        configure(errorReader, isWrite: false)
        let scratch = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: chunkSize)
        defer { scratch.deallocate() }
        while pump.isActive {
            try Task.checkCancellation()
            try pump.step(bytes: inputBytes, scratch: scratch)
        }
        return (pump.stdoutData, pump.stderrData)
    }
}

public actor ClipboardManager {

    private var detectedTool: ClipboardTool?

    public init() {}

    public func copy(_ text: String) async throws {
        if Self.isRunningInCI() {
            throw ClipboardError.notSupported
        }
        let tool = await detectClipboardTool()
        guard tool != .none, let command = tool.copyCommand else {
            throw ClipboardError.toolNotAvailable
        }
        _ = try await runClipboardCommand(command, input: text)
    }

    public func paste() async throws -> String {
        if Self.isRunningInCI() {
            throw ClipboardError.notSupported
        }
        let tool = await detectPasteTool()
        guard tool != .none, let command = tool.pasteCommand else {
            throw ClipboardError.toolNotAvailable
        }
        return try await runClipboardCommand(command)
    }

    public func isAvailable() async -> Bool {
        if Self.isRunningInCI() {
            return false
        }
        let tool = await detectClipboardTool()
        return tool != .none
    }

    public func detectClipboardTool() async -> ClipboardTool {
        if let cached = detectedTool {
            return cached
        }
        let tool: ClipboardTool
        #if os(macOS)
        tool = FileManager.default.fileExists(atPath: "/usr/bin/pbcopy") ? .pbcopy : .none
        #elseif os(Linux)
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

    private func detectPasteTool() async -> ClipboardTool {
        let copyTool = await detectClipboardTool()
        switch copyTool {
        case .pbcopy:
            return .pbpaste
        case .xclip:
            return .xclip
        case .xsel:
            return .xsel
        case .pbpaste, .none:
            return .none
        }
    }

    func runClipboardCommand(_ command: [String], input: String? = nil) async throws -> String {
        guard !command.isEmpty else {
            throw ClipboardError.operationFailed("Empty command")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command[0])
        if command.count > 1 {
            process.arguments = Array(command.dropFirst())
        }
        let inputPipe = input.map { _ in Pipe() }
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        if let inputPipe {
            process.standardInput = inputPipe
        }
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        var launched = false
        defer {
            if launched {
                if process.isRunning { process.terminate() }
                process.waitUntilExit()
            }
        }
        do {
            try process.run()
            launched = true
            let inputBytes = ContiguousArray((input ?? "").utf8)
            let (outputData, errorData) = try ClipboardPipePump.run(
                stdin: inputPipe?.fileHandleForWriting,
                inputBytes: inputBytes.span,
                stdout: outputPipe.fileHandleForReading,
                stderr: errorPipe.fileHandleForReading
            )
            process.waitUntilExit()
            launched = false
            guard process.terminationStatus == 0 else {
                throw ClipboardError.operationFailed(String(data: errorData, encoding: .utf8) ?? "Unknown error")
            }
            return String(data: outputData, encoding: .utf8) ?? ""
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ClipboardError {
            throw error
        } catch {
            throw ClipboardError.operationFailed(error.localizedDescription)
        }
    }

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

    public static func isRunningInCI() -> Bool {
        let ciKeys = ["CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI"]
        let env = ProcessInfo.processInfo.environment
        return ciKeys.contains { env[$0] != nil }
    }

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

extension String {
    public func copyToClipboard() async throws {
        let manager = ClipboardManager()
        try await manager.copy(self)
    }
}

extension ClipboardManager {
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
            return false
        } catch {
            print("❌ Failed to copy: \(error.localizedDescription)")
            return false
        }
    }

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
            return nil
        } catch {
            print("❌ Failed to paste: \(error.localizedDescription)")
            return nil
        }
    }
}
