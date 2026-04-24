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
            modelId: config.modelPath.split(separator: "/").last.map(String.init)
        )
        let authController = AuthController(authService: authService)

        let router = buildRouter(generateController: generateController, authController: authController)

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

    private func buildRouter(generateController: GenerateController, authController: AuthController) -> Router<BasicRequestContext> {
        let router = Router(context: BasicRequestContext.self)
        
        router.add(middleware: LogRequestsMiddleware(.info))

        // v1 API — Hummingbird 2.x RouterGroup (non-closure form)
        let v1 = router.group("/api/v1")
        v1.post("/generate", use: generateController.generate)
        v1.get("/health",    use: generateController.health)
        
        let auth = v1.group("/auth")
        auth.post("/login", use: authController.login)
        auth.post("/logout", use: authController.logout)

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
