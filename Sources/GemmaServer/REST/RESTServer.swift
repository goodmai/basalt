import Foundation
import Hummingbird
import Logging

// MARK: — Hummingbird 2.x REST Server

/// Настраивает и запускает HTTP-сервер на указанном порту.
/// Возвращает управление только при завершении (Ctrl-C / SIGTERM).
struct RESTServer: Sendable {

    let orchestrator: ModelOrchestratorActor
    let config: ServerConfig

    func run() async throws {
        let authService = try AuthService(dbPath: config.dbPath, jwtSecret: config.jwtSecret)
        
        let generateController = GenerateController(
            orchestrator: orchestrator,
            modelId: config.modelId ?? config.modelPath.split(separator: "/").last.map(String.init)
        )
        let authController = AuthController(authService: authService)
        let rateLimiter = RateLimiter(maxRequests: 50, window: 60)

        let router = buildRouter(generateController: generateController, authController: authController, rateLimiter: rateLimiter)

        var logger = Logger(label: "GemmaServer")
        logger.logLevel = .info

        let app = Application(
            router: router,
            configuration: .init(address: .hostname(config.host, port: config.restPort)),
            logger: logger
        )

        log("REST (A2A) listening on http://\(config.host):\(config.restPort)")
        try await app.runService()
    }

    // MARK: — Router

    private func buildRouter(generateController: GenerateController, authController: AuthController, rateLimiter: RateLimiter) -> Router<GemmaRequestContext> {
        let router = Router(context: GemmaRequestContext.self)
        
        router.add(middleware: LogRequestsMiddleware(.info))

        // Public routes
        let v1 = router.group("/api/v1")
        v1.get("/health", use: generateController.health)
        
        let auth = v1.group("/auth")
        auth.post("/login", use: authController.login)
        
        // Protected routes
        let protected = v1.group()
        protected.add(middleware: JWTAuthenticator(authService: authController.authService))
        protected.add(middleware: RateLimitMiddleware(rateLimiter: rateLimiter))
        
        protected.post("/generate", use: generateController.generate)
        protected.post("/generate/stream", use: generateController.generateStream)
        protected.post("/auth/logout", use: authController.logout)

        // Root health — convenient for load balancers
        router.get("/") { _, _ -> Response in
            var b = ByteBuffer()
            b.writeString("GemmaServer \(HealthResponse.version)")
            return Response(
                status: .ok,
                headers: [.contentType: "text/plain"],
                body: ResponseBody(byteBuffer: b)
            )
        }

        return router
    }

    private func log(_ message: String) {
        fputs("[REST] \(message)\n", stderr)
    }
}
