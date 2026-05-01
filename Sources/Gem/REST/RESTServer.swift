import Foundation
import Hummingbird
import HummingbirdWebSocket
import Logging

struct DebugLoggingMiddleware<Context: RequestContext>: RouterMiddleware {
    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        fputs("🟢 [DEBUG] \(request.method) \(request.uri)\n", stderr)
        fflush(stderr)
        let response = try await next(request, context)
        fputs("🔴 [DEBUG] \(request.method) \(request.uri) -> \(response.status.code)\n", stderr)
        fflush(stderr)
        return response
    }
}

// MARK: — Hummingbird 2.x REST + WebSocket Server

/// Настраивает и запускает HTTP-сервер на указанном порту.
/// Возвращает управление только при завершении (Ctrl-C / SIGTERM).
struct RESTServer: Sendable {

    let orchestrator: ModelOrchestratorActor
    let config: ServerConfig

    func run() async throws {
        let authService = try AuthService(dbPath: config.dbPath, jwtSecret: config.jwtSecret)
        
        // Seed admin user from environment if database is empty
        if let adminUser = ProcessInfo.processInfo.environment["GEMM_ADMIN_USER"],
           let adminPass = ProcessInfo.processInfo.environment["GEMM_ADMIN_PASSWORD"] {
            if try await !authService.hasUsers() {
                try await authService.createUser(username: adminUser, password: adminPass)
                log("Admin user '\(adminUser)' created from environment.")
            }
        }
        
        let generateController = GenerateController(
            orchestrator: orchestrator,
            modelId: config.modelId ?? config.modelPath.split(separator: "/").last.map(String.init)
        )
        let openAIController = OpenAIController(
            orchestrator: orchestrator,
            modelId: config.modelId ?? config.modelPath.split(separator: "/").last.map(String.init) ?? "gemm"
        )
        let anthropicController = AnthropicController(
            orchestrator: orchestrator,
            modelId: config.modelId ?? config.modelPath.split(separator: "/").last.map(String.init) ?? "gemm"
        )
        let authController = AuthController(authService: authService)
        let rateLimiter = RateLimiter(maxRequests: 50, window: 60)

        let router = buildRouter(generateController: generateController, authController: authController, openAIController: openAIController, anthropicController: anthropicController, rateLimiter: rateLimiter)

        // Build WebSocket router
        let wsController = WebSocketController(orchestrator: orchestrator, authService: authService)
        let wsRouter = buildWSRouter(wsController: wsController, authService: authService)

        var logger = Logger(label: "Gem")
        logger.logLevel = .info

        let app = Application(
            router: router,
            server: .http1WebSocketUpgrade(webSocketRouter: wsRouter),
            configuration: .init(address: .hostname(config.host, port: config.restPort)),
            logger: logger
        )

        log("REST (A2A) listening on http://\(config.host):\(config.restPort)")
        log("WebSocket endpoint: ws://\(config.host):\(config.restPort)/ws/generate")
        log("Swagger UI available at http://\(config.host):\(config.restPort)/swagger")
        try await app.runService()
    }

    // MARK: — WebSocket Router

    private func buildWSRouter(wsController: WebSocketController, authService: AuthService) -> Router<BasicWebSocketRequestContext> {
        let wsRouter = Router(context: BasicWebSocketRequestContext.self)

        wsRouter.ws("/ws/generate") { request, _ in
            // Authenticate via query parameter
            let uri = request.uri.string
            if let queryStart = uri.firstIndex(of: "?") {
                let queryString = String(uri[uri.index(after: queryStart)...])
                let params = queryString.split(separator: "&").reduce(into: [String: String]()) { dict, pair in
                    let parts = pair.split(separator: "=", maxSplits: 1)
                    if parts.count == 2 {
                        dict[String(parts[0])] = String(parts[1])
                    }
                }
                if let token = params["token"] {
                    do {
                        _ = try await authService.verify(token: token)
                        fputs("🔌 [WS] Token verified, upgrading connection\n", stderr)
                        fflush(stderr)
                        return .upgrade([:])
                    } catch {
                        fputs("🔌 [WS] Token verification failed: \(error)\n", stderr)
                        fflush(stderr)
                        return .dontUpgrade
                    }
                }
            }
            fputs("🔌 [WS] No token in query, rejecting upgrade\n", stderr)
            fflush(stderr)
            return .dontUpgrade
        } onUpgrade: { inbound, outbound, _ in
            await wsController.handle(inbound, outbound, WebSocketContext())
        }

        return wsRouter
    }

    // MARK: — HTTP Router

    private func buildRouter(generateController: GenerateController, authController: AuthController, openAIController: OpenAIController, anthropicController: AnthropicController, rateLimiter: RateLimiter) -> Router<GemmaRequestContext> {
        let router = Router(context: GemmaRequestContext.self)

        router.add(middleware: LogRequestsMiddleware(.info))
        router.add(middleware: DebugLoggingMiddleware())

        // Public routes
        let v1 = router.group("/api/v1")
        v1.get("/health", use: generateController.health)

        let auth = v1.group("/auth")
        auth.post("/login", use: authController.login)

        // OpenAI-compatible + Anthropic-compatible routes (no auth)
        // Claude Code local mode:
        //   export ANTHROPIC_API_KEY=local
        //   export ANTHROPIC_BASE_URL=http://localhost:8080
        //   claude
        let oai = router.group("/v1")
        oai.get("/models", use: openAIController.listModels)
        oai.post("/chat/completions", use: openAIController.chatCompletions)
        oai.post("/generate", use: openAIController.generateNoAuth)
        oai.post("/messages", use: anthropicController.messages)   // Anthropic Messages API
        
        // Protected routes
        let protected = v1.group()
        protected.add(middleware: JWTAuthenticator(authService: authController.authService))
        protected.add(middleware: RateLimitMiddleware(rateLimiter: rateLimiter))
        
        protected.post("/generate", use: generateController.generate)
        protected.post("/generate/stream", use: generateController.generateStream)
        protected.post("/auth/logout", use: authController.logout)

        // Swagger routes
        router.get("/swagger") { _, _ -> Response in
            let html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>Swagger UI</title>
              <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
            </head>
            <body>
            <div id="swagger-ui"></div>
            <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js" crossorigin></script>
            <script>
              window.onload = () => {
                window.ui = SwaggerUIBundle({
                  url: '/api/v1/openapi.json',
                  dom_id: '#swagger-ui',
                });
              };
            </script>
            </body>
            </html>
            """
            var b = ByteBuffer()
            b.writeString(html)
            return Response(status: .ok, headers: [.contentType: "text/html"], body: ResponseBody(byteBuffer: b))
        }

        router.get("/api/v1/openapi.json") { _, _ -> Response in
            let spec = """
            {
              "openapi": "3.0.0",
              "info": {
                "title": "Gemm REST API",
                "version": "1.0.0"
              },
              "paths": {
                "/api/v1/auth/login": {
                  "post": {
                    "summary": "Login",
                    "requestBody": {
                      "required": true,
                      "content": {
                        "application/json": {
                          "schema": {
                            "type": "object",
                            "properties": {
                              "username": { "type": "string" },
                              "password": { "type": "string" }
                            }
                          }
                        }
                      }
                    },
                    "responses": {
                      "200": { "description": "OK" },
                      "401": { "description": "Unauthorized" }
                    }
                  }
                },
                "/api/v1/generate": {
                  "post": {
                    "summary": "Generate text",
                    "security": [ { "BearerAuth": [] } ],
                    "requestBody": {
                      "required": true,
                      "content": {
                        "application/json": {
                          "schema": {
                            "type": "object",
                            "properties": {
                              "prompt": { "type": "string" },
                              "maxTokens": { "type": "integer" }
                            }
                          }
                        }
                      }
                    },
                    "responses": {
                      "200": { "description": "OK" }
                    }
                  }
                },
                "/api/v1/generate/stream": {
                  "post": {
                    "summary": "Generate text stream (SSE)",
                    "security": [ { "BearerAuth": [] } ],
                    "requestBody": {
                      "required": true,
                      "content": {
                        "application/json": {
                          "schema": {
                            "type": "object",
                            "properties": {
                              "prompt": { "type": "string" },
                              "maxTokens": { "type": "integer" }
                            }
                          }
                        }
                      }
                    },
                    "responses": {
                      "200": { "description": "SSE stream" }
                    }
                  }
                },
                "/ws/generate": {
                  "get": {
                    "summary": "WebSocket streaming endpoint",
                    "description": "Upgrade to WebSocket. Pass JWT token as query param: /ws/generate?token=<JWT>. Send JSON {prompt, maxTokens} to generate.",
                    "parameters": [
                      {
                        "name": "token",
                        "in": "query",
                        "required": true,
                        "schema": { "type": "string" }
                      }
                    ],
                    "responses": {
                      "101": { "description": "Switching Protocols (WebSocket upgrade)" }
                    }
                  }
                }
              },
              "components": {
                "securitySchemes": {
                  "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                  }
                }
              }
            }
            """
            var b = ByteBuffer()
            b.writeString(spec)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: b))
        }

        // Root health — convenient for load balancers
        router.get("/") { _, _ -> Response in
            var b = ByteBuffer()
            b.writeString("Gem \(HealthResponse.version)")
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

/// Simple context type for the WebSocket handler.
struct WebSocketContext: Sendable {}
