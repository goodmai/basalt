import Foundation

public actor CostTracker: Sendable {
    
    public struct Budget: Sendable, Codable {
        public let dailyLimit: Double
        public let monthlyLimit: Double
        
        public init(dailyLimit: Double, monthlyLimit: Double) {
            self.dailyLimit = dailyLimit
            self.monthlyLimit = monthlyLimit
        }
    }
    
    public struct UsageStats: Sendable, Codable {
        public var dailyUsage: Double
        public var monthlyUsage: Double
        public var modelBreakdown: [String: Double]
        public var lastUpdatedDay: String     // YYYY-MM-DD
        public var lastUpdatedMonth: String   // YYYY-MM
        
        public init() {
            self.dailyUsage = 0.0
            self.monthlyUsage = 0.0
            self.modelBreakdown = [:]
            
            let (day, month) = UsageStats.currentDateStrings()
            self.lastUpdatedDay = day
            self.lastUpdatedMonth = month
        }
        
        static func currentDateStrings() -> (day: String, month: String) {
            let formatter = DateFormatter()
            formatter.timeZone = TimeZone(secondsFromGMT: 0)
            
            formatter.dateFormat = "yyyy-MM-dd"
            let day = formatter.string(from: Date())
            
            formatter.dateFormat = "yyyy-MM"
            let month = formatter.string(from: Date())
            
            return (day, month)
        }
    }
    
    private let storageURL: URL
    private let budget: Budget
    private var stats: UsageStats
    
    private let logger = Logger()
    
    public init(storageURL: URL, budget: Budget) throws {
        self.storageURL = storageURL
        self.budget = budget
        
        // Create directory if it doesn't exist
        let dir = storageURL.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        
        // Load existing stats if available
        var loadedStats: UsageStats
        if let data = try? Data(contentsOf: storageURL),
           let decoded = try? JSONDecoder().decode(UsageStats.self, from: data) {
            loadedStats = decoded
        } else {
            loadedStats = UsageStats()
        }
        
        let (currentDay, currentMonth) = UsageStats.currentDateStrings()
        if loadedStats.lastUpdatedDay != currentDay {
            loadedStats.dailyUsage = 0.0
            loadedStats.lastUpdatedDay = currentDay
        }
        
        if loadedStats.lastUpdatedMonth != currentMonth {
            loadedStats.monthlyUsage = 0.0
            loadedStats.lastUpdatedMonth = currentMonth
        }
        
        self.stats = loadedStats
    }
    
    public func recordUsage(model: String, cost: Double) throws {
        guard cost > 0 else { return }
        
        resetCountersIfNeeded()
        
        // Check budget
        if stats.dailyUsage + cost > budget.dailyLimit {
            throw GemError.invalidRequestStructure(
                details: "Daily budget exceeded. Limit: $\(budget.dailyLimit), Requested total: $\(stats.dailyUsage + cost)"
            )
        }
        
        if stats.monthlyUsage + cost > budget.monthlyLimit {
            throw GemError.invalidRequestStructure(
                details: "Monthly budget exceeded. Limit: $\(budget.monthlyLimit), Requested total: $\(stats.monthlyUsage + cost)"
            )
        }
        
        // Warning at 80% (just log for now, could be an event)
        if stats.dailyUsage < budget.dailyLimit * 0.8 && (stats.dailyUsage + cost) >= budget.dailyLimit * 0.8 {
            log("⚠️ Warning: 80% of daily budget consumed.")
        }
        
        stats.dailyUsage += cost
        stats.monthlyUsage += cost
        stats.modelBreakdown[model, default: 0.0] += cost
        
        try save()
    }
    
    public func getStats() -> UsageStats {
        resetCountersIfNeeded()
        return stats
    }
    
    // MARK: - Private
    
    private func resetCountersIfNeeded() {
        let (currentDay, currentMonth) = UsageStats.currentDateStrings()
        var changed = false
        
        if stats.lastUpdatedDay != currentDay {
            stats.dailyUsage = 0.0
            stats.lastUpdatedDay = currentDay
            changed = true
        }
        
        if stats.lastUpdatedMonth != currentMonth {
            stats.monthlyUsage = 0.0
            stats.lastUpdatedMonth = currentMonth
            changed = true
        }
        
        if changed {
            try? save()
        }
    }
    
    private func save() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(stats)
        try data.write(to: storageURL, options: .atomic)
    }
    
    private func log(_ message: String) {
        fputs("[CostTracker] \(message)\n", stderr)
    }
}

private struct Logger {
    func warning(_ msg: String) {
        fputs("[WARNING] \(msg)\n", stderr)
    }
}
