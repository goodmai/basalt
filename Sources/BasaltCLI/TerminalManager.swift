import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import BasaltCore
import BasaltUI

public enum TerminalEvent: Sendable {
    case lineSubmitted(String)
    case interrupt
    case exit
    case EOF
}

public actor TerminalManager {
    private var originalTermios: termios
    private var isRawMode = false
    
    private var inputBuffer: String = ""
    private var promptText: String = "Basalt > "
    private var isBusy: Bool = false
    private var debugInfo: String = "Ready"
    
    private var workspacePath: String = FileManager.default.currentDirectoryPath
    private var gitBranch: String = "main"
    private var chipModel: String = "Apple Silicon"
    private var currentModel: String = "None"
    
    public init() {
        self.originalTermios = termios()
        tcgetattr(STDIN_FILENO, &originalTermios)
        
        Task {
            let resources = await SystemProfiler().detectResources()
            await self.updateInfo(chip: resources.chipModel)
        }
    }
    
    public func updateInfo(chip: String? = nil, model: String? = nil, debug: String? = nil) {
        if let c = chip { self.chipModel = c }
        if let m = model { self.currentModel = m }
        if let d = debug { self.debugInfo = d }
        
        Task {
            await refreshGitAsync()
        }
        
        if isRawMode { refreshLine() }
    }
    
    private func refreshGitAsync() async {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["branch", "--show-current"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        defer { try? pipe.fileHandleForReading.close() }
        do {
            try process.run()
            let data = try pipe.fileHandleForReading.readToEnd()
            process.waitUntilExit()
            if let b = data.flatMap({ String(data: $0, encoding: .utf8) })?
                .trimmingCharacters(in: .whitespacesAndNewlines), !b.isEmpty {
                if self.gitBranch != b {
                    self.gitBranch = b
                    if isRawMode { refreshLine() }
                }
            }
        } catch {}
    }

    public func enableRawMode() {
        guard !isRawMode else { return }
        var raw = originalTermios
        raw.c_lflag &= ~tcflag_t(ECHO | ICANON | ISIG)
        raw.c_iflag &= ~tcflag_t(IXON | ICRNL)
        #if canImport(Darwin)
        raw.c_cc.16 = 1; raw.c_cc.17 = 0
        #else
        raw.c_cc.6 = 1; raw.c_cc.5 = 0
        #endif
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
        isRawMode = true
        
        fputs("\n\n\n\n\n", stdout)
        refreshLine()
    }

    public func disableRawMode() {
        guard isRawMode else { return }
        fputs("\r", stdout)
        let linesToClear = calculateTotalFooterLines()
        for _ in 0..<linesToClear {
            fputs("\u{1B}[K\n", stdout)
        }
        fputs("\u{1B}[\(linesToClear)A", stdout)
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &originalTermios)
        isRawMode = false
    }

    private func calculateTotalFooterLines() -> Int {
        let width = getTerminalWidth()
        let promptLen = isBusy ? 15 : 8
        let inputLines = (promptLen + inputBuffer.count + width - 1) / width
        return 4 + max(1, inputLines)
    }

    public func setBusy(_ busy: Bool) {
        self.isBusy = busy
        self.promptText = busy ? "Basalt (busy) > " : "Basalt > "
        refreshLine()
    }
    
    public func printOutput(_ text: String) {
        let linesToMove = calculateTotalFooterLines()
        fputs("\u{1B}[\(linesToMove)A\r", stdout)
        fputs("\u{1B}[J", stdout)
        fputs(text, stdout)
        fflush(stdout)
        renderFooter()
    }
    
    private func renderFooter() {
        let width = getTerminalWidth()
        let divider = TerminalUI.dim(String(repeating: "─", count: width))
        
        let shortPath = workspacePath.replacingOccurrences(of: NSHomeDirectory(), with: "~")
        let colorPrompt = isBusy ? TerminalUI.warning(promptText) : TerminalUI.success(promptText)
        
        var ui = "\r"
        ui += "\(divider)\n"
        ui += "\u{1B}[2K \(TerminalUI.dim("info:")) \(debugInfo)\n"
        ui += "\u{1B}[2K\(colorPrompt)\(inputBuffer)\n"
        ui += "\u{1B}[2K \(TerminalUI.dim("workspace:")) \(shortPath)    \(TerminalUI.dim("branch:")) \(TerminalUI.success(gitBranch))\n"
        ui += "\u{1B}[2K \(TerminalUI.dim("chip:")) \(TerminalUI.info(chipModel))    \(TerminalUI.dim("model:")) \(TerminalUI.code(currentModel))"
        
        fputs(ui, stdout)
        fputs("\u{1B}[2A\r", stdout)
        
        let promptLen = isBusy ? 15 : 8
        let totalInputLen = promptLen + inputBuffer.count
        let cursorX = totalInputLen % width
        if cursorX > 0 {
            fputs("\u{1B}[\(cursorX)C", stdout)
        }
        
        fflush(stdout)
    }
    
    private func refreshLine() {
        let linesToMove = calculateTotalFooterLines()
        fputs("\u{1B}[\(linesToMove)A\r", stdout)
        renderFooter()
    }
    
    private func getTerminalWidth() -> Int {
        var w = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0, w.ws_col > 0 { return Int(w.ws_col) }
        return 80
    }
    
    nonisolated public func readEvents() -> AsyncStream<TerminalEvent> {
        AsyncStream { continuation in
            let reader = Task {
                await self.enableRawMode()
                defer {
                    Task { await self.disableRawMode() }
                }
                var utf8Buffer = Data()
                do {
                    for try await byte in FileHandle.standardInput.bytes {
                        guard !Task.isCancelled else { break }
                        if byte == 3 {
                            if await self.isBusy { continuation.yield(.interrupt) }
                            else { continuation.yield(.exit) }
                            utf8Buffer.removeAll(); continue
                        }
                        if byte == 27 {
                            continuation.yield(.interrupt)
                            utf8Buffer.removeAll(); continue
                        }
                        if byte == 10 || byte == 13 {
                            let currentBuffer = await self.inputBuffer
                            if !currentBuffer.isEmpty {
                                await self.finalizeInputLine()
                                continuation.yield(.lineSubmitted(currentBuffer))
                                await self.clearBuffer()
                            } else {
                                await self.refreshLine()
                            }
                            utf8Buffer.removeAll(); continue
                        }
                        if byte == 127 || byte == 8 {
                            await self.removeLastCharacter()
                            await self.refreshLine()
                            utf8Buffer.removeAll(); continue
                        }
                        utf8Buffer.append(byte)
                        if let str = String(data: utf8Buffer, encoding: .utf8) {
                            await self.appendCharacter(str)
                            await self.refreshLine()
                            utf8Buffer.removeAll()
                        }
                    }
                    if !Task.isCancelled { continuation.yield(.EOF) }
                } catch {}
                await self.disableRawMode()
                continuation.finish()
            }
            continuation.onTermination = { _ in reader.cancel() }
        }
    }
    
    private func finalizeInputLine() {
        fputs("\u{1B}[2A\r", stdout)
        let colorPrompt = isBusy ? TerminalUI.warning(promptText) : TerminalUI.success(promptText)
        fputs("\(colorPrompt)\(inputBuffer)\n\n\n\n\n", stdout) 
        fflush(stdout)
    }
    
    private func appendCharacter(_ char: String) { inputBuffer.append(char) }
    private func removeLastCharacter() { if !inputBuffer.isEmpty { inputBuffer.removeLast() } }
    private func clearBuffer() { inputBuffer = ""; refreshLine() }
    private var _isBusy: Bool { isBusy }
}
