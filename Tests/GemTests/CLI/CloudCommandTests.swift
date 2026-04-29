import XCTest
import ArgumentParser
@testable import GemCore

final class CloudCommandTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Clean up test config files if needed
        let configURL = CloudCommand.configURL()
        try? FileManager.default.removeItem(at: configURL)
        let costURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".gem/cost.json")
        try? FileManager.default.removeItem(at: costURL)
    }
    
    override func tearDown() {
        // Clean up test config files
        let configURL = CloudCommand.configURL()
        try? FileManager.default.removeItem(at: configURL)
        let costURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".gem/cost.json")
        try? FileManager.default.removeItem(at: costURL)
        super.tearDown()
    }
    
    func testCloudConfigure() async throws {
        var command = try CloudCommand.Configure.parse(["--api-key", "test-key", "--daily-budget", "10.0", "--monthly-budget", "100.0"])
        try await command.run()
        
        let config = CloudCommand.loadConfig()
        XCTAssertNotNil(config)
        XCTAssertEqual(config?.apiKey, "test-key")
        XCTAssertEqual(config?.dailyBudget, 10.0)
        XCTAssertEqual(config?.monthlyBudget, 100.0)
    }
}
