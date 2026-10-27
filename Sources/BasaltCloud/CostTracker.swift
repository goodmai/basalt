import Foundation
import BasaltCore

public actor CostTracker: Sendable {
    
    public struct Budget: Sendable, Codable {
        public let dailyLimit: Double
        public let monthlyLimit: Double
        
        public init(dailyLimit: Double, monthlyLimit: Double) {
            self.dailyLimit = dailyLimit
            self.monthlyLimit = monthlyLimit
        }
    }
    
    public enum BudgetScope: Sendable, Hashable {
        case session(String)
        case virtualKey(String)
    }

    public struct UsageStats: Sendable, Codable {
        public var dailyUsage: Double
        public var monthlyUsage: Double
        public var modelBreakdown: [String: Double]
        public var lastUpdatedDay: String     // YYYY-MM-DD
        public var lastUpdatedMonth: String   // YYYY-MM
        public var virtualKeyUsage: [String: Double]?
        
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
    private var scopeLimits: [BudgetScope: Double] = [:]
    private var sessionUsage: [String: Double] = [:]
    
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
    
    public func setMaxBudget(_ limit: Double, for scope: BudgetScope) {
        scopeLimits[scope] = limit
    }

    public func spent(in scope: BudgetScope) -> Double {
        switch scope {
        case .session(let id): sessionUsage[id] ?? 0
        case .virtualKey(let id): stats.virtualKeyUsage?[id] ?? 0
        }
    }

    public func isExhausted(scopes: [BudgetScope]) -> Bool {
        resetCountersIfNeeded()
        if stats.dailyUsage >= budget.dailyLimit || stats.monthlyUsage >= budget.monthlyLimit { return true }
        return scopes.contains { scope in
            scopeLimits[scope].map { spent(in: scope) >= $0 } ?? false
        }
    }

    public func remaining(scopes: [BudgetScope]) -> Double {
        resetCountersIfNeeded()
        let scoped = scopes.compactMap { scope in scopeLimits[scope].map { $0 - spent(in: scope) } }
        return ([budget.dailyLimit - stats.dailyUsage, budget.monthlyLimit - stats.monthlyUsage] + scoped).min()!
    }

    public func recordUsage(model: String, cost: Double, scopes: [BudgetScope] = []) throws {
        guard cost > 0 else { return }
        
        resetCountersIfNeeded()

        for scope in Set(scopes) {
            switch scope {
            case .session(let id): sessionUsage[id, default: 0] += cost
            case .virtualKey(let id): stats.virtualKeyUsage = (stats.virtualKeyUsage ?? [:]).merging([id: cost], uniquingKeysWith: +)
            }
        }
        
        if stats.dailyUsage < budget.dailyLimit * 0.8 && (stats.dailyUsage + cost) >= budget.dailyLimit * 0.8 {
            log("⚠️ Warning: 80% of daily budget consumed.")
        }
        
        stats.dailyUsage += cost
        stats.monthlyUsage += cost
        stats.modelBreakdown[model, default: 0.0] += cost
        try save()

        if stats.dailyUsage > budget.dailyLimit {
            throw BasaltError.invalidRequestStructure(
                details: "Daily budget exceeded. Limit: $\(budget.dailyLimit), Requested total: $\(stats.dailyUsage)"
            )
        }
        if stats.monthlyUsage > budget.monthlyLimit {
            throw BasaltError.invalidRequestStructure(
                details: "Monthly budget exceeded. Limit: $\(budget.monthlyLimit), Requested total: $\(stats.monthlyUsage)"
            )
        }
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
