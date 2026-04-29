import Testing
import Foundation
@testable import GemCore

@Suite("AuthService logic")
struct AuthServiceTests {

    @Test("login with default credentials succeeds")
    func testLoginSuccess() async throws {
        let dbPath = "test_auth.sqlite3"
        // Clean up before test
        try? FileManager.default.removeItem(atPath: dbPath)
        
        let auth = try AuthService(dbPath: dbPath, jwtSecret: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        try await auth.createUser(username: "admin123", password: "admin123")
        
        let token = try await auth.login(user: "admin123", pass: "admin123")
        #expect(!token.isEmpty)
        
        let user = try await auth.verify(token: token)
        #expect(user == "admin123")
        
        // Clean up
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("login with wrong password fails")
    func testLoginFailure() async throws {
        let dbPath = "test_auth_fail.sqlite3"
        try? FileManager.default.removeItem(atPath: dbPath)
        
        let auth = try AuthService(dbPath: dbPath, jwtSecret: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        try await auth.createUser(username: "admin123", password: "admin123")
        
        await #expect(throws: GemError.self) {
            try await auth.login(user: "admin123", pass: "wrong")
        }
        
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("logout invalidates token")
    func testLogout() async throws {
        let dbPath = "test_auth_logout.sqlite3"
        try? FileManager.default.removeItem(atPath: dbPath)
        
        let auth = try AuthService(dbPath: dbPath, jwtSecret: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        try await auth.createUser(username: "admin123", password: "admin123")
        let token = try await auth.login(user: "admin123", pass: "admin123")
        
        try await auth.logout(token: token)
        
        await #expect(throws: GemError.self) {
            try await auth.verify(token: token)
        }
        
        try? FileManager.default.removeItem(atPath: dbPath)
    }
    
    // MARK: — TC-2.3.1.4: Concurrent session operations
    
    @Test("Concurrent logins and verifications work correctly")
    func testConcurrentSessions() async throws {
        let dbPath = "test_auth_concurrent.sqlite3"
        try? FileManager.default.removeItem(atPath: dbPath)
        
        let auth = try AuthService(dbPath: dbPath, jwtSecret: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        try await auth.createUser(username: "admin123", password: "admin123")
        
        // Create multiple sessions concurrently
        try await withThrowingTaskGroup(of: String.self) { group in
            for _ in 1...10 {
                group.addTask {
                    let token = try await auth.login(user: "admin123", pass: "admin123")
                    let user = try await auth.verify(token: token)
                    #expect(user == "admin123")
                    return token
                }
            }
            
            // Collect all tokens
            var tokens: [String] = []
            for try await token in group {
                tokens.append(token)
            }
            
            // All tokens should be unique
            let uniqueTokens = Set(tokens)
            #expect(uniqueTokens.count == tokens.count)
        }
        
        try? FileManager.default.removeItem(atPath: dbPath)
    }
    
    @Test("Concurrent logout operations are safe")
    func testConcurrentLogouts() async throws {
        let dbPath = "test_auth_concurrent_logout.sqlite3"
        try? FileManager.default.removeItem(atPath: dbPath)
        
        let auth = try AuthService(dbPath: dbPath, jwtSecret: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        try await auth.createUser(username: "admin123", password: "admin123")
        
        // Create multiple sessions
        var tokens: [String] = []
        for _ in 1...10 {
            let token = try await auth.login(user: "admin123", pass: "admin123")
            tokens.append(token)
        }
        
        // Logout all concurrently
        try await withThrowingTaskGroup(of: Void.self) { group in
            for token in tokens {
                group.addTask {
                    try await auth.logout(token: token)
                }
            }
            
            // Wait for all logouts to complete
            for try await _ in group {}
        }
        
        // Verify all tokens are revoked
        for token in tokens {
            await #expect(throws: GemError.self) {
                try await auth.verify(token: token)
            }
        }
        
        try? FileManager.default.removeItem(atPath: dbPath)
    }
}
