import Testing
import Foundation
@testable import GemCore

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
        
        await #expect(throws: GemError.self) {
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
}
