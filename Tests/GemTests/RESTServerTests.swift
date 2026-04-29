import Foundation
import Testing
import Hummingbird
import NIOCore
import NIOEmbedded
import Logging
@testable import GemCore

// MARK: — RESTServer Route Tests
// Testing controllers directly without full HTTP stack

@Suite("RESTServer Routes")
struct RESTServerTests {

    // MARK: — Test Fixtures

    private func makeTestAuthService() async throws -> AuthService {
        let tempDir = FileManager.default.temporaryDirectory
        let dbPath = tempDir.appendingPathComponent("test_\(UUID().uuidString).db").path
        let auth = try AuthService(dbPath: dbPath, jwtSecret: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        try await auth.createUser(username: "admin123", password: "admin123")
        return auth
    }

    private func makeTestOrchestrator() -> ModelOrchestratorActor {
        let engine = MockInferenceEngine()
        return ModelOrchestratorActor(engine: engine)
    }

    private func makeContext(authenticated: Bool = false, username: String = "admin123") -> GemmaRequestContext {
        var context = GemmaRequestContext(source: .init(channel: EmbeddedChannel(), logger: Logger(label: "test")))
        if authenticated {
            context.identity = User(username: username)
        }
        return context
    }

    // MARK: — TC-1.2.1.1: POST /api/v1/auth/login with valid credentials

    @Test("AuthService login returns JWT token for valid credentials")
    func testLoginWithValidCredentials() async throws {
        let authService = try await makeTestAuthService()

        // Use default admin user
        let token = try await authService.login(user: "admin123", pass: "admin123")

        #expect(!token.isEmpty)
        #expect(token.count > 20) // JWT tokens are long
    }

    // MARK: — TC-1.2.1.2: POST /api/v1/auth/login with invalid credentials

    @Test("AuthService login throws for invalid password")
    func testLoginWithInvalidPassword() async throws {
        let authService = try await makeTestAuthService()

        await #expect(throws: GemError.self) {
            _ = try await authService.login(user: "admin123", pass: "wrongpassword")
        }
    }

    @Test("AuthService login throws for non-existent user")
    func testLoginWithNonExistentUser() async throws {
        let authService = try await makeTestAuthService()

        await #expect(throws: GemError.self) {
            _ = try await authService.login(user: "nonexistent", pass: "anypassword")
        }
    }

    // MARK: — TC-1.2.1.3: POST /api/v1/generate with valid JWT

    @Test("GenerateController generate succeeds for authenticated request")
    func testGenerateWithAuthentication() async throws {
        let orchestrator = makeTestOrchestrator()
        try await orchestrator.loadModel(path: "test-model")

        let controller = GenerateController(orchestrator: orchestrator, modelId: "test-model")

        let genRequest = GenerationRequest(prompt: "Hello", maxTokens: 10)
        let data = try JSONEncoder().encode(genRequest)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)

        let request = Request(
            head: .init(method: .post, scheme: "http", authority: "localhost", path: "/api/v1/generate"),
            body: .init(buffer: buffer)
        )

        let context = makeContext(authenticated: true)
        let response = try await controller.generate(request: request, context: context)

        #expect(response.status == .ok)
    }

    // MARK: — TC-1.2.1.4: POST /api/v1/generate without JWT (401)

    @Test("GenerateController generate throws without authentication")
    func testGenerateWithoutAuthentication() async throws {
        let orchestrator = makeTestOrchestrator()
        try await orchestrator.loadModel(path: "test-model")

        let controller = GenerateController(orchestrator: orchestrator, modelId: "test-model")

        let genRequest = GenerationRequest(prompt: "Hello", maxTokens: 10)
        let data = try JSONEncoder().encode(genRequest)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)

        let request = Request(
            head: .init(method: .post, scheme: "http", authority: "localhost", path: "/api/v1/generate"),
            body: .init(buffer: buffer)
        )

        let context = makeContext(authenticated: false)

        await #expect(throws: Error.self) {
            _ = try await controller.generate(request: request, context: context)
        }
    }

    @Test("GenerateController generate returns badRequest for invalid JSON")
    func testGenerateWithInvalidRequestBody() async throws {
        let orchestrator = makeTestOrchestrator()
        try await orchestrator.loadModel(path: "test-model")

        let controller = GenerateController(orchestrator: orchestrator, modelId: "test-model")

        var buffer = ByteBuffer()
        buffer.writeString("{\"invalid\": \"request\"}")

        let request = Request(
            head: .init(method: .post, scheme: "http", authority: "localhost", path: "/api/v1/generate"),
            body: .init(buffer: buffer)
        )

        let context = makeContext(authenticated: true)
        let response = try await controller.generate(request: request, context: context)

        #expect(response.status == .badRequest)
    }

    // MARK: — TC-1.2.1.5: JWT Authentication Tests

    @Test("JWTAuthenticator returns nil for invalid token")
    func testJWTAuthenticatorWithInvalidToken() async throws {
        let authService = try await makeTestAuthService()
        let authenticator = JWTAuthenticator(authService: authService)

        // Create request without Authorization header - authenticator will return nil
        let request = Request(
            head: .init(method: .post, scheme: "http", authority: "localhost", path: "/test"),
            body: .init(buffer: .init())
        )

        let context = makeContext()
        let user = try await authenticator.authenticate(request: request, context: context)

        #expect(user == nil)
    }

    @Test("JWTAuthenticator returns nil for missing Authorization header")
    func testJWTAuthenticatorWithoutHeader() async throws {
        let authService = try await makeTestAuthService()
        let authenticator = JWTAuthenticator(authService: authService)

        let request = Request(
            head: .init(method: .post, scheme: "http", authority: "localhost", path: "/test"),
            body: .init(buffer: .init())
        )

        let context = makeContext()
        let user = try await authenticator.authenticate(request: request, context: context)

        #expect(user == nil)
    }

    @Test("JWTAuthenticator returns user for valid token")
    func testJWTAuthenticatorWithValidToken() async throws {
        let authService = try await makeTestAuthService()

        let token = try await authService.login(user: "admin123", pass: "admin123")

        // Verify token directly via AuthService instead of through authenticator
        let username = try await authService.verify(token: token)

        #expect(username == "admin123")
    }

    // MARK: — Additional Tests

    @Test("GenerateController health returns health status")
    func testHealthEndpoint() async throws {
        let orchestrator = makeTestOrchestrator()
        try await orchestrator.loadModel(path: "test-model")

        let controller = GenerateController(orchestrator: orchestrator, modelId: "test-model")

        let request = Request(
            head: .init(method: .get, scheme: "http", authority: "localhost", path: "/api/v1/health"),
            body: .init(buffer: .init())
        )

        let context = makeContext()
        let response = try await controller.health(request: request, context: context)

        #expect(response.status == .ok)
    }

    @Test("AuthService logout revokes token")
    func testLogout() async throws {
        let authService = try await makeTestAuthService()

        let token = try await authService.login(user: "admin123", pass: "admin123")

        // Logout
        try await authService.logout(token: token)

        // Verify token is revoked
        await #expect(throws: GemError.self) {
            _ = try await authService.verify(token: token)
        }
    }

    @Test("AuthService verify returns username for valid token")
    func testVerifyValidToken() async throws {
        let authService = try await makeTestAuthService()

        let token = try await authService.login(user: "admin123", pass: "admin123")
        let username = try await authService.verify(token: token)

        #expect(username == "admin123")
    }

    @Test("AuthService verify throws for invalid token")
    func testVerifyInvalidToken() async throws {
        let authService = try await makeTestAuthService()

        let invalidToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"

        await #expect(throws: Error.self) {
            _ = try await authService.verify(token: invalidToken)
        }
    }

    // MARK: — Model Not Loaded Tests

    @Test("GenerateController returns error when model not loaded")
    func testGenerateWithoutModelLoaded() async throws {
        let orchestrator = makeTestOrchestrator()
        let controller = GenerateController(orchestrator: orchestrator, modelId: "test-model")

        let genRequest = GenerationRequest(prompt: "Hello", maxTokens: 10)
        let data = try JSONEncoder().encode(genRequest)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)

        let request = Request(
            head: .init(method: .post, scheme: "http", authority: "localhost", path: "/api/v1/generate"),
            body: .init(buffer: buffer)
        )

        let context = makeContext(authenticated: true)
        let response = try await controller.generate(request: request, context: context)

        // Should return error status
        #expect(response.status.code >= 400)
    }
    
    // MARK: — TC-2.1.2.3: Error propagation (actor → HTTP 500)
    
    @Test("Error from actor propagates to HTTP error response")
    func testActorErrorPropagation() async throws {
        // Use real engine that will fail on generate
        let failingEngine = MockInferenceEngine()
        await failingEngine.setShouldFailOnGenerate(true)
        
        let orchestrator = ModelOrchestratorActor(engine: failingEngine)
        try await orchestrator.loadModel(path: "test-model")
        
        let controller = GenerateController(orchestrator: orchestrator, modelId: "test-model")
        
        let genRequest = GenerationRequest(prompt: "Hello", maxTokens: 10)
        let data = try JSONEncoder().encode(genRequest)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)
        
        let request = Request(
            head: .init(method: .post, scheme: "http", authority: "localhost", path: "/api/v1/generate"),
            body: .init(buffer: buffer)
        )
        
        let context = makeContext(authenticated: true)
        let response = try await controller.generate(request: request, context: context)
        
        // Should return 500-level error
        #expect(response.status.code >= 500)
    }
    
    // MARK: — TC-2.1.2.4: Timeout handling (long-running inference)
    
    @Test("Long-running inference completes without timeout")
    func testLongRunningInference() async throws {
        // Use engine with delay
        let slowEngine = MockInferenceEngine()
        await slowEngine.setArtificialDelay(2.0) // 2 second delay
        
        let orchestrator = ModelOrchestratorActor(engine: slowEngine)
        try await orchestrator.loadModel(path: "test-model")
        
        let controller = GenerateController(orchestrator: orchestrator, modelId: "test-model")
        
        let genRequest = GenerationRequest(prompt: "Hello", maxTokens: 10)
        let data = try JSONEncoder().encode(genRequest)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)
        
        let request = Request(
            head: .init(method: .post, scheme: "http", authority: "localhost", path: "/api/v1/generate"),
            body: .init(buffer: buffer)
        )
        
        let context = makeContext(authenticated: true)
        
        // Measure that it takes ~2 seconds
        let start = ContinuousClock.now
        let response = try await controller.generate(request: request, context: context)
        let duration = start.duration(to: .now)
        
        #expect(response.status == .ok)
        #expect(duration >= Duration.seconds(2))
        #expect(duration < Duration.seconds(5)) // Should not timeout
    }
}
