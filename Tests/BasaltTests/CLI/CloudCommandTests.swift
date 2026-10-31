import Foundation
import Testing
import ArgumentParser
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("Cloud CLI commands")
struct CloudCommandTests {
    private static func cleanup() {
        let configURL = CloudCommand.configURL()
        try? FileManager.default.removeItem(at: configURL)
        let costURL = CloudCommand.costURL()
        try? FileManager.default.removeItem(at: costURL)
    }

    @Test("Cloud configure command persists settings")
    func cloudConfigure() async throws {
        Self.cleanup()
        defer { Self.cleanup() }

        var command = try CloudCommand.Configure.parse(["--api-key", "test-key", "--daily-budget", "10.0", "--monthly-budget", "100.0"])
        try await command.run()

        let config = CloudCommand.loadConfig()
        #expect(config != nil)
        #expect(config?.apiKey == "test-key")
        #expect(config?.dailyBudget == 10.0)
        #expect(config?.monthlyBudget == 100.0)
    }
}
