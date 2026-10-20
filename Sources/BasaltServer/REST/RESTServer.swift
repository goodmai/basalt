import Foundation
import Hummingbird
import Logging
import BasaltCore

// MARK: — Hummingbird 2.x REST Server (plain HTTP/1.1 + SSE)

/// Configures and runs the HTTP server on the configured port.
/// Blocks until shutdown (Ctrl-C / SIGTERM).
package struct RESTServer: Sendable {

    let orchestrator: ModelOrchestratorActor
    let config: ServerConfig

    package init(orchestrator: ModelOrchestratorActor, config: ServerConfig) {
        self.orchestrator = orchestrator
        self.config = config
    }

    package func run() async throws {
        let router = makeRouter()

        // Suppress Hummingbird's own verbose logging; our GemLogger handles app-level logs.
        var logger = Logger(label: "Gem")
        logger.logLevel = .warning

        let app = Application(
            router: router,
            server: .http1(),
            configuration: .init(address: .hostname(config.host, port: config.restPort)),
            logger: logger
        )

        log("Listening on http://\(config.host):\(config.restPort)")
        if let key = config.apiKey, !key.isEmpty {
            log("API key protection: ENABLED (bearer token required)")
        } else {
            log("API key protection: DISABLED (loopback open access)")
        }
        if let limit = config.rateLimitPerMinute {
            log("Rate limit: \(limit) requests/minute (429 beyond it)")
        } else {
            log("Rate limit: DISABLED")
        }
        log("Browser origins: loopback on port \(config.restPort)" + config.allowedOrigins.map { ", \($0)" }.joined())
        log("Swagger: http://\(config.host):\(config.restPort)/swagger")
        try await app.runService()
    }

    // MARK: — Router

    /// The fully wired router, auth middleware included — also what the perimeter tests drive.
    func makeRouter() -> Router<BasicRequestContext> {
        let defaultId = config.modelId ?? config.modelPath.split(separator: "/").last.map(String.init) ?? "basalt"
        return buildRouter(
            generateController:  GenerateController(orchestrator: orchestrator, modelId: defaultId),
            openAIController:    OpenAIController(orchestrator: orchestrator, modelId: defaultId),
            anthropicController: AnthropicController(orchestrator: orchestrator, modelId: defaultId),
            modelsController:    ModelsController(orchestrator: orchestrator, defaultModelId: defaultId)
        )
    }

    private func buildRouter(
        generateController:  GenerateController,
        openAIController:    OpenAIController,
        anthropicController: AnthropicController,
        modelsController:    ModelsController
    ) -> Router<BasicRequestContext> {
        let router = Router(context: BasicRequestContext.self)

        // Minimal request logging (method + path only, goes to stderr via GemLogger)
        router.add(middleware: LogRequestsMiddleware(.info))
        router.add(middleware: RequestGuardMiddleware(bindHost: config.host, bindPort: config.restPort, allowedOrigins: config.allowedOrigins))

        if let apiKey = config.apiKey, !apiKey.isEmpty {
            router.add(middleware: APIKeyAuthenticator(apiKey: apiKey))
        }
        // After the authenticator: a request without the key never spends the budget.
        if let limit = config.rateLimitPerMinute {
            router.add(middleware: RateLimitMiddleware(requestsPerMinute: limit))
        }
        router.add(middleware: JSONNestingLimitMiddleware())

        // ── Native API ──────────────────────────────────────────────────────────
        let v1 = router.group("/api/v1")
        v1.get("/health",          use: generateController.health)
        v1.get("/stats",           use: generateController.stats)
        v1.post("/generate",       use: generateController.generate)
        v1.post("/generate/stream", use: generateController.generateStream)

        // ── OpenAI + Anthropic compatible ───────────────────────────────────────
        // Claude Code: ANTHROPIC_BASE_URL=http://localhost:8080  ANTHROPIC_AUTH_TOKEN=local
        let oai = router.group("/v1")

        // Model management
        oai.get("/models",         use: modelsController.listModels)   // OpenAI list format
        oai.get("/models/current", use: modelsController.currentModel) // readiness + size
        oai.post("/models/load",   use: modelsController.loadModel)    // hot-swap at runtime

        // Generation
        oai.post("/chat/completions", use: openAIController.chatCompletions) // OpenAI SSE
        oai.post("/responses",        use: openAIController.responses)       // OpenAI Responses API (Codex/Cursor)
        oai.post("/generate",         use: openAIController.generateRaw)  // raw prompt
        oai.post("/messages",         use: anthropicController.messages)     // Anthropic SSE

        // ── Swagger UI ──────────────────────────────────────────────────────────
        router.get("/swagger") { _, _ -> Response in
            let html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>Basalt API</title>
              <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" integrity="sha384-+yyzNgM3K92sROwsXxYCxaiLWxWJ0G+v/9A+qIZ2rgefKgkdcmJI+L601cqPD/Ut" crossorigin="anonymous" />
            </head>
            <body>
            <div id="swagger-ui"></div>
            <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js" integrity="sha384-qn5tagrAjZi8cSmvZ+k3zk4+eDEEUcP9myuR2J6V+/H6rne++v6ChO7EeHAEzqxQ" crossorigin="anonymous"></script>
            <script>
              window.onload = () => {
                window.ui = SwaggerUIBundle({ url: '/api/v1/openapi.json', dom_id: '#swagger-ui' });
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
              "info": { "title": "Basalt REST API", "version": "1.0.0",
                        "description": "Local LLM inference — open on loopback; an API key (Bearer or x-api-key) is required when bound off loopback" },
              "paths": {
                "/api/v1/health":          { "get":  { "summary": "Health check" } },
                "/api/v1/generate":        { "post": { "summary": "Raw text generation" } },
                "/api/v1/generate/stream": { "post": { "summary": "SSE streaming generation" } },
                "/v1/models":              { "get":  { "summary": "List cached models (OpenAI format)" } },
                "/api/v1/stats":           { "get":  { "summary": "Telemetry of the serving process (same snapshot as serve --stats and MCP status)" } },
                "/v1/models/current":      { "get":  { "summary": "Currently loaded model + status" } },
                "/v1/models/load":         { "post": { "summary": "Hot-swap model at runtime" } },
                "/v1/chat/completions":    { "post": { "summary": "OpenAI-compatible chat completions" } },
                "/v1/responses":           { "post": { "summary": "OpenAI Responses API (Codex/Cursor)" } },
                "/v1/messages":            { "post": { "summary": "Anthropic-compatible messages API" } }
              }
            }
            """
            var b = ByteBuffer()
            b.writeString(spec)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: b))
        }

        // Root — useful for health checks / load balancers
        router.get("/") { _, _ -> Response in
            var b = ByteBuffer()
            b.writeString("Basalt \(HealthResponse.version) — http://localhost:\(config.restPort)/swagger")
            return Response(status: .ok, headers: [.contentType: "text/plain"], body: ResponseBody(byteBuffer: b))
        }

        return router
    }

    private func log(_ message: String) {
        fputs("[REST] \(message)\n", stderr)
    }
}
