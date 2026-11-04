import Foundation
import Hummingbird
import ServiceLifecycle
import Logging
import BasaltCore

package struct RESTServer: Sendable {

    let orchestrator: ModelOrchestratorActor
    let config: ServerConfig
    let embeddings: (any EmbeddingEngine)?

    let shutdownDeadline: Duration
    let handleSignals: Bool
    let rateClock: @Sendable () -> Double

    package init(
        orchestrator: ModelOrchestratorActor,
        config: ServerConfig,
        embeddings: (any EmbeddingEngine)? = nil,
        shutdownDeadline: Duration = .seconds(5),
        handleSignals: Bool = true,
        rateClock: @escaping @Sendable () -> Double = { ProcessInfo.processInfo.systemUptime }
    ) {
        self.orchestrator = orchestrator
        self.config = config
        self.embeddings = embeddings
        self.shutdownDeadline = shutdownDeadline
        self.handleSignals = handleSignals
        self.rateClock = rateClock
    }

    package func run() async throws {
        let router = makeRouter()

        var logger = Logger(label: "Basalt")
        logger.logLevel = .warning

        let app = Application(
            router: router,
            server: .http1(),
            configuration: ApplicationConfiguration(address: .hostname(config.host, port: config.restPort)),
            logger: logger
        )

        logStartup()
        let drain = GenerationDrainService(orchestrator: orchestrator, deadline: shutdownDeadline)
        let group = ServiceGroup(configuration: ServiceGroupConfiguration(
            services: [
                ServiceGroupConfiguration.ServiceConfiguration(service: app),
                ServiceGroupConfiguration.ServiceConfiguration(service: drain, successTerminationBehavior: .gracefullyShutdownGroup),
            ],
            gracefulShutdownSignals: handleSignals ? [.sigterm, .sigint] : [],
            logger: logger
        ))
        try await group.run()
    }

    private func logStartup() {
        log("Listening on http://\(config.host):\(config.restPort)")
        if let key = config.apiKey, !key.isEmpty {
            log("API key protection: ENABLED (bearer token required)")
        } else {
            log("API key protection: DISABLED (loopback open access)")
        }
        if let limit = config.rateLimitPerMinute {
            let burstStr = config.rateBurst.map { ", burst \($0)" } ?? ""
            log("Rate limit: \(limit) requests/minute\(burstStr) (429 beyond it)")
        } else {
            log("Rate limit: DISABLED")
        }
        log("Browser origins: loopback on port \(config.restPort)" + config.allowedOrigins.map { ", \($0)" }.joined())
        log("Swagger: http://\(config.host):\(config.restPort)/swagger")
    }

    func makeRouter() -> Router<BasicRequestContext> {
        let defaultId = config.modelId ?? config.modelPath.split(separator: "/").last.map { String($0) } ?? "basalt"
        let router = Router(context: BasicRequestContext.self)
        addMiddleware(to: router)
        addNativeRoutes(to: router.group("/api/v1"), GenerateController(orchestrator: orchestrator, modelId: defaultId))
        addModelRoutes(
            to: router.group("/v1"),
            ModelsController(orchestrator: orchestrator, defaultModelId: defaultId, embeddingModels: embeddings?.models ?? []))
        addGenerationRoutes(to: router.group("/v1"), defaultId: defaultId)
        addDocumentationRoutes(to: router)
        return router
    }

    private func addMiddleware(to router: Router<BasicRequestContext>) {
        router.add(middleware: LogRequestsMiddleware(.info))
        router.add(middleware: RequestGuardMiddleware(bindHost: config.host, bindPort: config.restPort, allowedOrigins: config.allowedOrigins))
        if !(config.apiKey ?? "").isEmpty || !config.virtualKeys.isEmpty {
            router.add(middleware: APIKeyAuthenticator(apiKey: config.apiKey ?? "", virtualKeys: config.virtualKeys))
        }
        if let limit = config.rateLimitPerMinute {
            router.add(middleware: RateLimitMiddleware(requestsPerMinute: limit, burst: config.rateBurst, clock: rateClock))
        }
        router.add(middleware: JSONNestingLimitMiddleware())
        router.add(middleware: RetryAfterMiddleware())
    }

    private func addNativeRoutes(to v1: RouterGroup<BasicRequestContext>, _ controller: GenerateController) {
        v1.get("/health") { try await controller.health(request: $0, context: $1) }
        v1.get("/stats") { try await controller.stats(request: $0, context: $1) }
        v1.post("/generate", use: controller.generate)
        v1.post("/generate/stream", use: controller.generateStream)
    }

    private func addModelRoutes(to oai: RouterGroup<BasicRequestContext>, _ controller: ModelsController) {
        oai.get("/models") { try await controller.listModels(request: $0, context: $1) }
        oai.get("/models/current") { try await controller.currentModel(request: $0, context: $1) }
        oai.get("/models/preview") { try await controller.previewModel(request: $0, context: $1) }
        oai.post("/models/load", use: controller.loadModel)
    }

    private func addGenerationRoutes(to oai: RouterGroup<BasicRequestContext>, defaultId: String) {
        let openAI = OpenAIController(orchestrator: orchestrator, modelId: defaultId)
        let embedding = EmbeddingsController(engine: embeddings, gate: orchestrator.gate)
        oai.post("/chat/completions", use: openAI.chatCompletions)
        oai.post("/responses", use: openAI.responses)
        oai.post("/generate", use: openAI.generateRaw)
        oai.post("/messages", use: AnthropicController(orchestrator: orchestrator, modelId: defaultId).messages)
        oai.post("/embeddings", use: embedding.embeddings)
        oai.post("/rerank", use: embedding.rerank)
    }

    private func addDocumentationRoutes(to router: Router<BasicRequestContext>) {
        let restPort = config.restPort
        router.get("/swagger") { _, _ in Self.response(Self.swaggerHTML, contentType: "text/html") }
        router.get("/api/v1/openapi.json") { _, _ in Self.response(Self.openAPISpec, contentType: "application/json") }
        router.get("/") { _, _ in
            Self.response("Basalt \(HealthResponse.version) — http://localhost:\(restPort)/swagger", contentType: "text/plain")
        }
    }

    private static func response(_ text: String, contentType: String) -> Response {
        var b = ByteBuffer()
        b.writeString(text)
        return Response(status: .ok, headers: [.contentType: contentType], body: ResponseBody(byteBuffer: b))
    }

    private func log(_ message: String) {
        fputs("[REST] \(message)\n", stderr)
    }
}

extension RESTServer {
    static let swaggerHTML = """
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

    static let openAPISpec = """
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
        "/v1/models/preview":      { "get":  { "summary": "Memory a model would need, without loading it (?model_id=&context=)" } },
        "/v1/models/load":         { "post": { "summary": "Hot-swap model at runtime" } },
        "/v1/chat/completions":    { "post": { "summary": "OpenAI-compatible chat completions" } },
        "/v1/responses":           { "post": { "summary": "OpenAI Responses API (Codex/Cursor)" } },
        "/v1/messages":            { "post": { "summary": "Anthropic-compatible messages API" } },
        "/v1/embeddings":          { "post": { "summary": "OpenAI-compatible embeddings (serve --embedding-model)" } },
        "/v1/rerank":              { "post": { "summary": "Cohere/Jina-compatible rerank (serve --rerank-model)" } }
      }
    }
    """
}
