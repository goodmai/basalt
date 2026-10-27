import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("CostTrackerTests")
struct CostTrackerTests {
    
    private func createTempFile() -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("usage_\(UUID().uuidString).json")
        return fileURL
    }
    
    @Test("Record usage within budget succeeds")
    func testRecordUsageWithinBudget() async throws {
        let fileURL = createTempFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }
        
        let tracker = try CostTracker(
            storageURL: fileURL,
            budget: .init(dailyLimit: 10.0, monthlyLimit: 100.0)
        )
        
        try await tracker.recordUsage(model: "gpt-4", cost: 5.0)
        let stats = await tracker.getStats()
        
        #expect(stats.dailyUsage == 5.0)
        #expect(stats.monthlyUsage == 5.0)
        #expect(stats.modelBreakdown["gpt-4"] == 5.0)
    }
    
    @Test("Exceeding daily budget throws error")
    func testExceedingDailyBudget() async throws {
        let fileURL = createTempFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }
        
        let tracker = try CostTracker(
            storageURL: fileURL,
            budget: .init(dailyLimit: 10.0, monthlyLimit: 100.0)
        )
        
        try await tracker.recordUsage(model: "gpt-4", cost: 8.0) // Within budget
        
        await #expect(throws: BasaltError.self) {
            try await tracker.recordUsage(model: "gpt-4", cost: 3.0) // Exceeds 10.0 daily
        }
    }
    
    @Test("Persistence across instances")
    func testPersistence() async throws {
        let fileURL = createTempFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }
        
        let budget = CostTracker.Budget(dailyLimit: 10.0, monthlyLimit: 100.0)
        
        // First instance
        let tracker1 = try CostTracker(storageURL: fileURL, budget: budget)
        try await tracker1.recordUsage(model: "claude-3.5", cost: 2.5)
        
        // Second instance loading from same file
        let tracker2 = try CostTracker(storageURL: fileURL, budget: budget)
        let stats = await tracker2.getStats()
        
        #expect(stats.dailyUsage == 2.5)
        #expect(stats.modelBreakdown["claude-3.5"] == 2.5)
    }

    @Test("Spend that overshoots the daily limit is still recorded, so the budget reads exhausted afterwards")
    func overshootIsRecorded() async throws {
        let fileURL = createTempFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let tracker = try CostTracker(storageURL: fileURL, budget: .init(dailyLimit: 10.0, monthlyLimit: 100.0))
        try await tracker.recordUsage(model: "gpt-4", cost: 8.0)
        await #expect(throws: BasaltError.self) {
            try await tracker.recordUsage(model: "gpt-4", cost: 3.0)
        }
        #expect(await tracker.getStats().dailyUsage == 11.0)
        #expect(await tracker.isExhausted(scopes: []))
    }

    @Test("Session and virtual-key budgets are separate; key spend survives a restart, session spend does not")
    func scopedBudgets() async throws {
        let fileURL = createTempFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let budget = CostTracker.Budget(dailyLimit: 100.0, monthlyLimit: 1_000.0)
        let tracker = try CostTracker(storageURL: fileURL, budget: budget)
        await tracker.setMaxBudget(1.0, for: .session("s1"))
        await tracker.setMaxBudget(2.0, for: .virtualKey("k1"))
        try await tracker.recordUsage(model: "m", cost: 1.0, scopes: [.session("s1"), .virtualKey("k1")])

        #expect(await tracker.isExhausted(scopes: [.session("s1")]))
        #expect(await !tracker.isExhausted(scopes: [.virtualKey("k1")]))
        #expect(await !tracker.isExhausted(scopes: [.session("s2"), .virtualKey("k1")]))
        #expect(await tracker.isExhausted(scopes: [.session("s2"), .session("s1")]))

        let restarted = try CostTracker(storageURL: fileURL, budget: budget)
        #expect(await restarted.spent(in: .virtualKey("k1")) == 1.0)
        #expect(await restarted.spent(in: .session("s1")) == 0)
    }
}
