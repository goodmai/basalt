import Testing
import Foundation
@testable import GemmaServerCore

@Suite("ServerConfig logic")
struct ConfigTests {

    @Test("ServerConfig picks up defaults")
    func testDefaults() {
        let config = ServerConfig(modelPath: "path")
        #expect(!config.jwtSecret.isEmpty) // Default is UUID().uuidString
        #expect(config.dbPath == "auth.sqlite3")
        #expect(config.maxTokens == 65536)
    }

    @Test("ServerConfig custom values")
    func testCustom() {
        let config = ServerConfig(
            modelPath: "path",
            jwtSecret: "custom-secret",
            dbPath: "custom.db"
        )
        #expect(config.jwtSecret == "custom-secret")
        #expect(config.dbPath == "custom.db")
    }
}
