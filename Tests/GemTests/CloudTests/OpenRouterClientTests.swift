import Testing
import Foundation
@testable import GemCore

@Suite("OpenRouterClientTests")
struct OpenRouterClientTests {
    
    @Test("Initialization fails without API key")
    func testInitializationWithoutKey() async throws {
        let config = OpenRouterClient.Config(
            apiKey: "",
            baseURL: URL(string: "https://openrouter.ai/api/v1")!,
            timeout: 60.0,
            maxRetries: 3
        )
        
        #expect(throws: GemError.self) {
            _ = try OpenRouterClient(config: config)
        }
    }
    
    @Test("Initialization succeeds with valid config")
    func testInitializationWithKey() async throws {
        let config = OpenRouterClient.Config(
            apiKey: "sk-or-test-key",
            baseURL: URL(string: "https://openrouter.ai/api/v1")!,
            timeout: 60.0,
            maxRetries: 3
        )
        
        let client = try OpenRouterClient(config: config)
        let metrics = await client.getMetrics()
        #expect(metrics.requests == 0)
        #expect(metrics.errors == 0)
    }
}
