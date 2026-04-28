import Foundation

/// Epic 16.10: Async Spinner System
/// Provides an animated spinner for terminal output
public final class Spinner: @unchecked Sendable {
    
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
        
        var interval: TimeInterval {
            switch self {
            case .dots:   return 0.08
            case .line:   return 0.1
            case .pulse:  return 0.12
            case .bounce: return 0.15
            }
        }
    }
    
    private let style: Style
    private let message: String
    private var isRunning = false
    private var currentFrame = 0
    private var timer: Timer?
    private let queue = DispatchQueue(label: "com.gemmaserver.spinner")
    
    public init(style: Style = .dots, message: String = "") {
        self.style = style
        self.message = message
    }
    
    /// Start the animation
    public func start() {
        queue.async {
            guard !self.isRunning else { return }
            self.isRunning = true
            
            // Hide cursor
            fputs("\u{1B}[?25l", stderr)
            
            self.timer = Timer.scheduledTimer(withTimeInterval: self.style.interval, repeats: true) { [weak self] _ in
                self?.render()
            }
            
            RunLoop.current.add(self.timer!, forMode: .common)
            RunLoop.current.run()
        }
    }
    
    /// Stop the animation and clean up
    public func stop(finalMessage: String? = nil) {
        queue.async {
            guard self.isRunning else { return }
            self.isRunning = false
            self.timer?.invalidate()
            self.timer = nil
            
            // Clear line and show cursor
            fputs("\r\u{1B}[K", stderr)
            if let msg = finalMessage {
                fputs("\(msg)\n", stderr)
            }
            fputs("\u{1B}[?25h", stderr)
            fflush(stderr)
        }
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
}
