import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public enum TerminalEvent: Sendable {
    case lineSubmitted(String)
    case lineQueued(String)
    case interrupt
    case EOF
}

/// Epic 16.10: Async Spinner System
/// Manages terminal interaction and provides a rich UI with an async spinner,
/// process phases (enum), and context statistics (files, MCP, skills).
public actor TerminalManager {
    private var originalTermios: termios
    private var isRawMode = false
    
    private var inputBuffer: String = ""
    private var promptText: String = "Gemma > "
    private var isBusy: Bool = false
    
    // Spinner state
    private var loadingState: LoadingState? = nil
    private var contextStats: ContextStats? = nil
    private var spinnerFrame: Int = 0
    private let spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    private var spinnerTimer: Task<Void, Never>? = nil
    
    public init() {
        self.originalTermios = termios()
        tcgetattr(STDIN_FILENO, &originalTermios)
    }
    
    public func enableRawMode() {
        guard !isRawMode else { return }
        var raw = originalTermios
        raw.c_lflag &= ~tcflag_t(ECHO | ICANON | ISIG)
        raw.c_iflag &= ~tcflag_t(IXON | ICRNL)
        
        #if canImport(Darwin)
        raw.c_cc.16 = 1 // VMIN
        raw.c_cc.17 = 0 // VTIME
        #else
        raw.c_cc.6 = 1 // VMIN
        raw.c_cc.5 = 0 // VTIME
        #endif
        
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
        isRawMode = true
    }
    
    public func disableRawMode() {
        guard isRawMode else { return }
        stopSpinner()
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &originalTermios)
        isRawMode = false
    }
    
    public func setBusy(_ busy: Bool) {
        self.isBusy = busy
        self.promptText = busy ? "Gemma (busy) > " : "Gemma > "
        refreshLine()
    }
    
    /// Epic 16.10: Start showing a spinner with state and stats above the input line
    public func startSpinner(state: LoadingState, stats: ContextStats? = nil) {
        self.loadingState = state
        self.contextStats = stats
        self.spinnerFrame = 0
        
        // Start a background task for animation
        spinnerTimer?.cancel()
        spinnerTimer = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 80_000_000) // 80ms
                await self?.advanceSpinner()
            }
        }
        refreshLine()
    }
    
    /// Epic 16.10: Update the current state or stats while spinner is running
    public func updateSpinner(state: LoadingState? = nil, stats: ContextStats? = nil) {
        if let state = state {
            self.loadingState = state
        }
        if let stats = stats {
            self.contextStats = stats
        }
        refreshLine()
    }
    
    /// Epic 16.10: Stop the spinner and clear its line
    public func stopSpinner() {
        spinnerTimer?.cancel()
        spinnerTimer = nil
        
        if loadingState != nil {
            // Move up, clear line, move back down
            fputs("\u{1B}[1A\r\u{1B}[2K\u{1B}[1B", stdout)
            self.loadingState = nil
            self.contextStats = nil
            refreshLine()
        }
    }
    
    private func advanceSpinner() {
        spinnerFrame = (spinnerFrame + 1) % spinnerFrames.count
        refreshLine()
    }
    
    public func printOutput(_ text: String) {
        // If spinner is active, clear it first
        if loadingState != nil {
            fputs("\u{1B}[1A\r\u{1B}[2K\u{1B}[1B", stdout)
        }
        
        fputs("\r\u{1B}[2K", stdout)
        fputs(text, stdout)
        fflush(stdout)
        refreshLine()
    }
    
    private func refreshLine() {
        // 1. If we have a loading state, render it above with stats
        if let state = loadingState {
            let frame = spinnerFrames[spinnerFrame]
            let coloredFrame = TerminalUI.info(frame)
            let coloredMessage = TerminalUI.bold(state.description)
            
            var statsString = ""
            if let stats = contextStats, stats.hasContext {
                var parts: [String] = []
                if stats.files > 0 { parts.append("Files: \(stats.files)") }
                if stats.systemPrompts > 0 { parts.append("Sys: \(stats.systemPrompts)") }
                if stats.mcpServers > 0 { parts.append("MCP: \(stats.mcpServers)") }
                if stats.skills > 0 { parts.append("Skills: \(stats.skills)") }
                
                let joined = parts.joined(separator: " | ")
                statsString = TerminalUI.dim(" [\(joined)]")
            }
            
            // Move up, clear line, print spinner + state + stats, move down
            fputs("\u{1B}[1A\r\u{1B}[2K\(coloredFrame) \(coloredMessage)\(statsString)\n", stdout)
        }
        
        // 2. Render the prompt and buffer
        fputs("\r\u{1B}[2K", stdout)
        let colorPrompt = isBusy ? TerminalUI.warning(promptText) : TerminalUI.success(promptText)
        fputs("\(colorPrompt)\(inputBuffer)", stdout)
        fflush(stdout)
    }
    
    public func readEvents() -> AsyncStream<TerminalEvent> {
        return AsyncStream { continuation in
            Task {
                await self.enableRawMode()
                await self.refreshLine()
                
                let handle = FileHandle.standardInput
                var escapeSequence = false
                
                while true {
                    guard let data = try? handle.read(upToCount: 1), !data.isEmpty else {
                        continuation.yield(.EOF)
                        break
                    }
                    
                    let byte = data[0]
                    if byte == 3 {
                        continuation.yield(.interrupt)
                        continue
                    }
                    
                    if byte == 10 || byte == 13 {
                        let currentBuffer = await self.inputBuffer
                        fputs("\n", stdout)
                        
                        // Clear spinner space if active
                        if await self.loadingState != nil {
                            fputs("\u{1B}[1A\r\u{1B}[2K\u{1B}[1B", stdout)
                        }
                        
                        if !currentBuffer.isEmpty {
                            continuation.yield(.lineSubmitted(currentBuffer))
                            await self.clearBuffer()
                        } else {
                            await self.refreshLine()
                        }
                        continue
                    }
                    
                    if byte == 9 {
                        let currentBuffer = await self.inputBuffer
                        if !currentBuffer.isEmpty {
                            fputs("\n", stdout)
                            continuation.yield(.lineQueued(currentBuffer))
                            await self.clearBuffer()
                        }
                        continue
                    }
                    
                    if byte == 127 || byte == 8 {
                        let currentBuffer = await self.inputBuffer
                        if !currentBuffer.isEmpty {
                            await self.removeLastCharacter()
                            await self.refreshLine()
                        }
                        continue
                    }
                    
                    if byte == 27 {
                        escapeSequence = true
                        continue
                    }
                    
                    if escapeSequence {
                        if (byte >= 64 && byte <= 126) {
                            escapeSequence = false
                        }
                        continue
                    }
                    
                    if let str = String(data: data, encoding: .utf8) {
                        await self.appendCharacter(str)
                        await self.refreshLine()
                    }
                }
                
                await self.disableRawMode()
                continuation.finish()
            }
        }
    }
    
    private func appendCharacter(_ char: String) {
        inputBuffer.append(char)
    }
    
    private func removeLastCharacter() {
        if !inputBuffer.isEmpty {
            inputBuffer.removeLast()
        }
    }
    
    private func clearBuffer() {
        inputBuffer = ""
        refreshLine()
    }
}
