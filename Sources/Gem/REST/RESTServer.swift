import Foundation
import Hummingbird
import Logging

// MARK: — Hummingbird 2.x REST Server (plain HTTP/1.1, no WebSocket)

/// Configures and runs the HTTP server on the configured port.
/// Blocks until shutdown (Ctrl-C / SIGTERM).
struct RESTServer: Sendable {

    let orchestrator: ModelOrchestratorActor
    let config: ServerConfig

    func run() async throws {
        let defaultId = config.modelId ?? config.modelPath.split(separator: "/").last.map(String.init) ?? "gemm"

        let generateController  = GenerateController(orchestrator: orchestrator, modelId: defaultId)
        let openAIController    = OpenAIController(orchestrator: orchestrator, modelId: defaultId)
        let anthropicController = AnthropicController(orchestrator: orchestrator, modelId: defaultId)
        let modelsController    = ModelsController(orchestrator: orchestrator, defaultModelId: defaultId)

        let router = buildRouter(
            generateController:  generateController,
            openAIController:    openAIController,
            anthropicController: anthropicController,
            modelsController:    modelsController
        )

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
        log("Swagger: http://\(config.host):\(config.restPort)/swagger")
        try await app.runService()
    }

    // MARK: — Router

    private func buildRouter(
        generateController:  GenerateController,
        openAIController:    OpenAIController,
        anthropicController: AnthropicController,
        modelsController:    ModelsController
    ) -> Router<BasicRequestContext> {
        let router = Router(context: BasicRequestContext.self)

        // Minimal request logging (method + path only, goes to stderr via GemLogger)
        router.add(middleware: LogRequestsMiddleware(.info))

        // ── Native API ──────────────────────────────────────────────────────────
        let v1 = router.group("/api/v1")
        v1.get("/health",          use: generateController.health)
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
        oai.post("/generate",         use: openAIController.generateNoAuth)  // raw prompt
        oai.post("/messages",         use: anthropicController.messages)     // Anthropic SSE

        // ── Swagger UI ──────────────────────────────────────────────────────────
        router.get("/swagger") { _, _ -> Response in
            let html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>Gemm API</title>
              <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
            </head>
            <body>
            <div id="swagger-ui"></div>
            <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js" crossorigin></script>
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
              "info": { "title": "Gemm REST API", "version": "1.0.0",
                        "description": "Local LLM inference — no auth required" },
              "paths": {
                "/api/v1/health":          { "get":  { "summary": "Health check" } },
                "/api/v1/generate":        { "post": { "summary": "Raw text generation" } },
                "/api/v1/generate/stream": { "post": { "summary": "SSE streaming generation" } },
                "/v1/models":              { "get":  { "summary": "List cached models (OpenAI format)" } },
                "/v1/models/current":      { "get":  { "summary": "Currently loaded model + status" } },
                "/v1/models/load":         { "post": { "summary": "Hot-swap model at runtime" } },
                "/v1/chat/completions":    { "post": { "summary": "OpenAI-compatible chat completions" } },
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
            b.writeString("Gemm \(HealthResponse.version) — http://localhost:\(config.restPort)/swagger")
            return Response(status: .ok, headers: [.contentType: "text/plain"], body: ResponseBody(byteBuffer: b))
        }

        return router
    }

    private func log(_ message: String) {
        fputs("[REST] \(message)\n", stderr)
    }
}
