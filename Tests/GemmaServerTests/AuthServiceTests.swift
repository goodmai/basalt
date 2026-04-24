import Testing
import Foundation
@testable import GemmaServerCore

@Suite("AuthService logic")
struct AuthServiceTests {

    @Test("login with default credentials succeeds")
    func testLoginSuccess() async throws {
        let dbPath = "test_auth.sqlite3"
        // Clean up before test
        try? FileManager.default.removeItem(atPath: dbPath)
        
        let auth = try AuthService(dbPath: dbPath)
        
        let token = try await auth.login(user: "admin", pass: "admin")
        #expect(!token.isEmpty)
        
        let user = try await auth.verify(token: token)
        #expect(user == "admin")
        
        // Clean up
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("login with wrong password fails")
    func testLoginFailure() async throws {
        let dbPath = "test_auth_fail.sqlite3"
        try? FileManager.default.removeItem(atPath: dbPath)
        
        let auth = try AuthService(dbPath: dbPath)
        
        await #expect(throws: GemmaServerError.self) {
            try await auth.login(user: "admin", pass: "wrong")
        }
        
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("logout invalidates token")
    func testLogout() async throws {
        let dbPath = "test_auth_logout.sqlite3"
        try? FileManager.default.removeItem(atPath: dbPath)
        
        let auth = try AuthService(dbPath: dbPath)
        let token = try await auth.login(user: "admin", pass: "admin")
        
        try await auth.logout(token: token)
        
        await #expect(throws: GemmaServerError.self) {
            try await auth.verify(token: token)
        }
        
        try? FileManager.default.removeItem(atPath: dbPath)
    }
}
