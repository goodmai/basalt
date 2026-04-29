import Foundation

/// Epic 16.8: Progress Bar System
/// Beautiful, flexible progress indicators for CLI
public struct ProgressBar: Sendable {
    
    // MARK: - Style
    
    public enum Style: Sendable {
        case bar(width: Int)       // [=====>    ] 50%
        case percentage            // 50%
        case spinner               // ⠋ Loading...
        case detailed              // [=====>    ] 50% (500/1000) 10MB/s ETA: 30s
        
        public static var `default`: Style {
            .bar(width: 40)
        }
    }
    
    // MARK: - Properties
    
    public let total: Int
    public private(set) var current: Int
    public let title: String
    public let style: Style
    public private(set) var currentMessage: String?
    
    private var startTime: Date
    private var lastUpdateTime: Date
    private var lastUpdateCurrent: Int
    
    // MARK: - Computed Properties
    
    /// Current percentage (0-100)
    public var percentage: Double {
        guard total > 0 else { return 0.0 }
        return min(100.0, (Double(current) / Double(total)) * 100.0)
    }
    
    /// Is the progress complete?
    public var isComplete: Bool {
        current >= total
    }
    
    /// Estimated speed (units per second)
    public var estimatedSpeed: Double {
        let elapsed = Date().timeIntervalSince(startTime)
        guard elapsed > 0 else { return 0 }
        return Double(current) / elapsed
    }
    
    /// Estimated time remaining (seconds)
    public var estimatedTimeRemaining: TimeInterval {
        guard current > 0 else { return 0 }
        let speed = estimatedSpeed
        guard speed > 0 else { return 0 }
        let remaining = total - current
        return Double(remaining) / speed
    }
    
    // MARK: - Initialization
    
    public init(
        total: Int,
        title: String = "",
        style: Style = .default
    ) {
        self.total = total
        self.current = 0
        self.title = title
        self.style = style
        self.startTime = Date()
        self.lastUpdateTime = Date()
        self.lastUpdateCurrent = 0
    }
    
    // MARK: - Update
    
    /// Update current progress value
    public mutating func update(current: Int, message: String? = nil) {
        self.current = min(current, total)
        self.currentMessage = message
        self.lastUpdateTime = Date()
        self.lastUpdateCurrent = current
    }
    
    /// Increment progress by amount
    public mutating func increment(by amount: Int = 1) {
        update(current: current + amount)
    }
    
    /// Mark as complete
    public mutating func markComplete() {
        update(current: total)
    }
    
    // MARK: - Rendering
    
    /// Render progress bar to string
    public func render() -> String {
        switch style {
        case .bar(let width):
            return renderBar(width: width)
        case .percentage:
            return renderPercentage()
        case .spinner:
            return renderSpinner()
        case .detailed:
            return renderDetailed()
        }
    }
    
    private func renderBar(width: Int) -> String {
        let filledWidth = Int(Double(width) * (percentage / 100.0))
        let emptyWidth = width - filledWidth
        
        let filled = String(repeating: "█", count: filledWidth)
        let empty = String(repeating: "░", count: emptyWidth)
        
        var output = ""
        
        if !title.isEmpty {
            output += TerminalUI.dim("\(title): ")
        }
        
        output += "[\(filled)\(empty)] "
        output += TerminalUI.bold(String(format: "%.1f%%", percentage))
        
        if let message = currentMessage {
            output += " " + TerminalUI.dim(message)
        }
        
        return output
    }
    
    private func renderPercentage() -> String {
        var output = ""
        
        if !title.isEmpty {
            output += "\(title): "
        }
        
        output += String(format: "%.1f%%", percentage)
        
        if let message = currentMessage {
            output += " - \(message)"
        }
        
        return output
    }
    
    private func renderSpinner() -> String {
        let spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        let frameIndex = (current / 10) % spinnerFrames.count
        let spinner = spinnerFrames[frameIndex]
        
        var output = "\(spinner) "
        
        if !title.isEmpty {
            output += "\(title)"
        }
        
        if let message = currentMessage {
            output += " - \(message)"
        } else {
            output += " - \(current)/\(total)"
        }
        
        return output
    }
    
    private func renderDetailed() -> String {
        let filledWidth = Int(40.0 * (percentage / 100.0))
        let emptyWidth = 40 - filledWidth
        
        let filled = String(repeating: "█", count: filledWidth)
        let empty = String(repeating: "░", count: emptyWidth)
        
        var output = ""
        
        if !title.isEmpty {
            output += TerminalUI.dim("\(title)\n")
        }
        
        output += "[\(filled)\(empty)] "
        output += TerminalUI.bold(String(format: "%.1f%%", percentage))
        output += " (\(current)/\(total))"
        
        if current > 0 && !isComplete {
            let speed = estimatedSpeed
            let eta = estimatedTimeRemaining
            
            if speed > 0 {
                output += " | \(Self.formatSpeed(speed))"
            }
            
            if eta > 0 && eta < 3600 {  // Only show if < 1 hour
                output += " | ETA: \(Self.formatDuration(Int(eta)))"
            }
        }
        
        if let message = currentMessage {
            output += "\n" + TerminalUI.dim(message)
        }
        
        return output
    }
    
    /// Print progress bar (with carriage return for inline update)
    public func display() {
        let output = render()
        print("\r\u{001B}[K\(output)", terminator: "")
        fflush(stdout)
    }
    
    /// Print final progress (with newline)
    public func finish() {
        let output = render()
        print("\r\u{001B}[K\(output)")
    }
    
    // MARK: - Formatting Helpers
    
    /// Format bytes to human-readable format
    public static func formatBytes(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var value = Double(bytes)
        var unitIndex = 0
        
        while value >= 1024 && unitIndex < units.count - 1 {
            value /= 1024
            unitIndex += 1
        }
        
        if unitIndex == 0 {
            return "\(bytes) B"
        } else {
            return String(format: "%.1f %@", value, units[unitIndex])
        }
    }
    
    /// Format duration in seconds to human-readable format
    public static func formatDuration(_ seconds: Int) -> String {
        if seconds < 60 {
            return "\(seconds)s"
        } else if seconds < 3600 {
            let minutes = seconds / 60
            let secs = seconds % 60
            return "\(minutes)m \(secs)s"
        } else {
            let hours = seconds / 3600
            let minutes = (seconds % 3600) / 60
            let secs = seconds % 60
            return "\(hours)h \(minutes)m \(secs)s"
        }
    }
    
    /// Format speed (units per second)
    private static func formatSpeed(_ speed: Double) -> String {
        if speed > 1_000_000 {
            return String(format: "%.1f M/s", speed / 1_000_000)
        } else if speed > 1000 {
            return String(format: "%.1f K/s", speed / 1000)
        } else {
            return String(format: "%.1f/s", speed)
        }
    }
}

// MARK: - MultiProgressBar

/// Manage multiple concurrent progress bars (thread-safe actor)
public actor MultiProgressBar {
    
    public struct TaskID: Hashable, Sendable {
        let id: UUID
        
        fileprivate init() {
            self.id = UUID()
        }
    }
    
    private var tasks: [TaskID: ProgressBar]
    
    public var taskCount: Int {
        tasks.count
    }
    
    public var overallPercentage: Double {
        guard !tasks.isEmpty else { return 0.0 }
        let totalPercentage = tasks.values.reduce(0.0) { $0 + $1.percentage }
        return totalPercentage / Double(tasks.count)
    }
    
    public init() {
        self.tasks = [:]
    }
    
    // MARK: - Task Management
    
    @discardableResult
    public func addTask(title: String, total: Int, style: ProgressBar.Style = .bar(width: 30)) -> TaskID {
        let id = TaskID()
        let progress = ProgressBar(total: total, title: title, style: style)
        tasks[id] = progress
        return id
    }
    
    public func updateTask(_ id: TaskID, current: Int, message: String? = nil) {
        tasks[id]?.update(current: current, message: message)
    }
    
    public func incrementTask(_ id: TaskID, by amount: Int = 1) {
        tasks[id]?.increment(by: amount)
    }
    
    public func completeTask(_ id: TaskID) {
        tasks[id]?.markComplete()
    }
    
    public func removeTask(_ id: TaskID) {
        tasks.removeValue(forKey: id)
    }
    
    public func getProgress(for id: TaskID) -> ProgressBar? {
        tasks[id]
    }
    
    // MARK: - Rendering
    
    /// Render all progress bars
    public func render() -> String {
        var output = ""
        
        for (_, progress) in tasks.sorted(by: { $0.value.title < $1.value.title }) {
            output += progress.render()
            output += "\n"
        }
        
        // Overall progress
        if tasks.count > 1 {
            output += TerminalUI.dim("━" * 60)
            output += "\n"
            output += "Overall: \(TerminalUI.bold(String(format: "%.1f%%", overallPercentage)))"
        }
        
        return output
    }
    
    /// Display all progress bars (inline update)
    public func display() {
        let output = render()
        let lineCount = tasks.count + (tasks.count > 1 ? 2 : 0)  // +2 for separator and overall
        
        // Move cursor up by lineCount
        for _ in 0..<lineCount {
            print("\u{001B}[A\u{001B}[K", terminator: "")
        }
        
        print("\r\(output)", terminator: "")
        fflush(stdout)
    }
    
    /// Print final state (with newline)
    public func finish() {
        let output = render()
        print(output)
    }
}

// MARK: - String Multiplication Extension

fileprivate extension String {
    static func * (lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
