import Testing
import Foundation
@testable import GemCore

@Suite("ServerConfig logic")
struct ConfigTests {

    @Test("ServerConfig picks up defaults")
    func testDefaults() {
        let config = ServerConfig(modelPath: "path")
        #expect(config.jwtSecret == nil) // SECURITY: No default value
        #expect(config.dbPath == "auth.sqlite3")
        #expect(config.maxTokens == 65536)
    }

    @Test("ServerConfig custom values")
    func testCustom() {
        let config = ServerConfig(
            modelPath: "path",
            jwtSecret: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            dbPath: "custom.db"
        )
        #expect(config.jwtSecret == "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        #expect(config.dbPath == "custom.db")
    }
}
