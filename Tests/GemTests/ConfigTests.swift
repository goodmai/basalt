import Testing
import Foundation
@testable import GemCore

@Suite("ServerConfig logic")
struct ConfigTests {

    @Test("ServerConfig default values")
    func testDefaults() {
        let config = ServerConfig(modelPath: "path")
        #expect(config.restPort == 8080)
        #expect(config.host == "127.0.0.1")
        #expect(config.maxTokens == 65536)
        #expect(config.logLevel == .info)
    }

    @Test("ServerConfig custom values round-trip")
    func testCustom() {
        let config = ServerConfig(
            modelPath: "/models/qwen",
            modelId:   "mlx-community/Qwen3.5-4B-4bit",
            restPort:  9090,
            host:      "0.0.0.0",
            maxTokens: 8192,
            logLevel:  .debug
        )
        #expect(config.modelId == "mlx-community/Qwen3.5-4B-4bit")
        #expect(config.restPort == 9090)
        #expect(config.maxTokens == 8192)
        #expect(config.logLevel == .debug)
    }
}
