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

public actor TerminalManager {
    private var originalTermios: termios
    private var isRawMode = false
    
    private var inputBuffer: String = ""
    private var promptText: String = "Gemma > "
    private var isBusy: Bool = false
    
    public init() {
        self.originalTermios = termios()
        tcgetattr(STDIN_FILENO, &originalTermios)
    }
    
    deinit {
        // Can't easily disable raw mode in deinit of actor, but we provide a cleanup method
    }
    
    public func enableRawMode() {
        guard !isRawMode else { return }
        var raw = originalTermios
        
        // Disable ECHO, ICANON, ISIG (to catch Ctrl+C manually)
        raw.c_lflag &= ~tcflag_t(ECHO | ICANON | ISIG)
        // Disable IXON (Ctrl+S/Ctrl+Q)
        raw.c_iflag &= ~tcflag_t(IXON | ICRNL)
        
        // We keep OPOST enabled so \n translates to \r\n on output
        // raw.c_oflag &= ~tcflag_t(OPOST)
        
        // VMIN = 1, VTIME = 0 (blocking read until 1 byte)
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
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &originalTermios)
        isRawMode = false
    }
    
    public func setBusy(_ busy: Bool) {
        self.isBusy = busy
        self.promptText = busy ? "Gemma (busy) > " : "Gemma > "
        refreshLine()
    }
    
    public func printOutput(_ text: String) {
        // Move cursor to beginning of line, clear line
        fputs("\r\u{1B}[2K", stdout)
        // Print output
        fputs(text, stdout)
        fflush(stdout)
        // Redraw prompt and buffer
        refreshLine()
    }
    
    private func refreshLine() {
        fputs("\r\u{1B}[2K", stdout)
        let colorPrompt = isBusy ? "\u{1B}[33m\(promptText)\u{1B}[0m" : "\u{1B}[32m\(promptText)\u{1B}[0m"
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
                    
                    // Handle Ctrl+C (End of Text)
                    if byte == 3 {
                        continuation.yield(.interrupt)
                        continue
                    }
                    
                    // Handle Enter (CR or LF)
                    if byte == 10 || byte == 13 {
                        let currentBuffer = await self.inputBuffer
                        if !currentBuffer.isEmpty {
                            // Print newline to move past the prompt
                            fputs("\n", stdout)
                            continuation.yield(.lineSubmitted(currentBuffer))
                            await self.clearBuffer()
                        } else {
                            fputs("\n", stdout)
                            await self.refreshLine()
                        }
                        continue
                    }
                    
                    // Handle Tab
                    if byte == 9 {
                        let currentBuffer = await self.inputBuffer
                        if !currentBuffer.isEmpty {
                            fputs("\n", stdout)
                            continuation.yield(.lineQueued(currentBuffer))
                            await self.clearBuffer()
                        }
                        continue
                    }
                    
                    // Handle Backspace (127 or 8)
                    if byte == 127 || byte == 8 {
                        let currentBuffer = await self.inputBuffer
                        if !currentBuffer.isEmpty {
                            await self.removeLastCharacter()
                            await self.refreshLine()
                        }
                        continue
                    }
                    
                    // Ignore simple ANSI escape sequences for now (arrows)
                    if byte == 27 {
                        escapeSequence = true
                        continue
                    }
                    
                    if escapeSequence {
                        // Very naive escape sequence handling: skip until a letter
                        if (byte >= 64 && byte <= 126) {
                            escapeSequence = false
                        }
                        continue
                    }
                    
                    // Standard printable character
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
