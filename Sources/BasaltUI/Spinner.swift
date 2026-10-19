import Foundation

/// Epic 16.8: Progress indication for long-running operations
/// Thread-safe spinner implementation using actor model
public actor Spinner {
    
    public enum Style: Sendable {
        case dots      // ⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
        case line      // | / - \
        case pulse     // ░ ▒ ▓ █ ▓ ▒ ░
        case bounce    // (●   ) ( ●  ) (  ● ) (   ●)
        
        var frames: [String] {
            switch self {
            case .dots:   return ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            case .line:   return ["|", "/", "-", "\\"]
            case .pulse:  return ["░", "▒", "▓", "█", "▓", "▒"]
            case .bounce: return ["(●   )", "( ●  )", "(  ● )", "(   ●)", "(  ● )", "( ●  )"]
            }
        }
        
        var interval: Duration {
            switch self {
            case .dots:   return .milliseconds(80)
            case .line:   return .milliseconds(100)
            case .pulse:  return .milliseconds(120)
            case .bounce: return .milliseconds(150)
            }
        }
    }
    
    private let style: Style
    private let message: String
    private var isRunning = false
    private var currentFrame = 0
    private var animationTask: Task<Void, Never>?
    
    public init(style: Style = .dots, message: String = "") {
        self.style = style
        self.message = message
    }
    
    /// Start the animation
    public func start() {
        guard !isRunning else { return }
        isRunning = true
        currentFrame = 0
        
        // Hide cursor
        fputs("\u{1B}[?25l", stderr)
        fflush(stderr)
        
        animationTask = Task { [weak self] in
            while let self = self, await self.isRunning {
                await self.render()
                try? await Task.sleep(for: self.style.interval)
            }
        }
    }
    
    /// Stop the animation and clean up
    public func stop(finalMessage: String? = nil) {
        isRunning = false
        animationTask?.cancel()
        animationTask = nil
        
        // Clear line and show cursor
        fputs("\r\u{1B}[K", stderr)
        if let msg = finalMessage {
            fputs("\(msg)\n", stderr)
        }
        fputs("\u{1B}[?25h", stderr)
        fflush(stderr)
    }
    
    private func render() {
        guard isRunning else { return }
        
        let frame = style.frames[currentFrame]
        let coloredFrame = TerminalUI.info(frame)
        let output = "\r\(coloredFrame) \(message)\u{1B}[K"
        
        fputs(output, stderr)
        fflush(stderr)
        
        currentFrame = (currentFrame + 1) % style.frames.count
    }
    
    deinit {
        // Ensure cleanup
        isRunning = false
        fputs("\u{1B}[?25h", stderr)
        fflush(stderr)
    }
}
