import Foundation
import Hummingbird
import ServiceLifecycle
import Logging
import BasaltCore

package struct RESTServer: Sendable {

    let orchestrator: ModelOrchestratorActor
    let config: ServerConfig
    let embeddings: (any EmbeddingEngine)?
    private(set) var decisions: (any DecisionEngine)?
    private(set) var pipeline = PipelineStages()
    private(set) var pipelines: PipelineRegistry?
    private(set) var compose: ComposeControl?
    private(set) var onReady: (@Sendable (Int) -> Void)?

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
        self.decisions = nil
        self.shutdownDeadline = shutdownDeadline
        self.handleSignals = handleSignals
        self.rateClock = rateClock
    }

    package func withDecisions(_ engine: (any DecisionEngine)?) -> RESTServer {
        var server = self
        server.decisions = GatedDecisionEngine.serving(engine, gate: orchestrator.gate)
        return server
    }

    package func withReadyHandler(_ handler: (@Sendable (Int) -> Void)?) -> RESTServer {
        var server = self
        server.onReady = handler
        return server
    }

    package static func readyLine(port: Int, pid: Int32 = ProcessInfo.processInfo.processIdentifier) -> String {
        let fields: [String: Any] = ["port": port, "pid": Int(pid), "version": HealthResponse.version]
        let json = (try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys])).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        return "basalt-ready \(json)"
    }

    package func withPipeline(_ stages: PipelineStages) -> RESTServer {
        var server = self
        server.pipeline = stages
        return server
    }

    package func withPipelines(_ registry: PipelineRegistry?, compose: ComposeControl?) -> RESTServer {
        var server = self
        (server.pipelines, server.compose) = (registry, compose)
        return server
    }

    package func run() async throws {
        let router = makeRouter()

        var logger = Logger(label: "Basalt")
        logger.logLevel = .warning

        let app = Application(
            router: router,
            server: .http1(),
            configuration: ApplicationConfiguration(address: .hostname(config.host, port: config.restPort)),
            onServerRunning: { [onReady] channel in
                if let port = channel.localAddress?.port { onReady?(port) }
            },
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
        router.add(middleware: JSONNestingLimitMiddleware(zeroCopyBody: config.experimentalZeroCopyBody))
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
        let openAI = OpenAIController(orchestrator: orchestrator, modelId: defaultId, zeroCopyBody: config.experimentalZeroCopyBody)
        let embedding = EmbeddingsController(engine: embeddings, gate: orchestrator.gate)
        oai.post("/chat/completions", use: openAI.chatCompletions)
        oai.post("/responses", use: openAI.responses)
        oai.post("/generate", use: openAI.generateRaw)
        oai.post("/score", use: openAI.score)
        oai.post("/messages", use: AnthropicController(orchestrator: orchestrator, modelId: defaultId, zeroCopyBody: config.experimentalZeroCopyBody).messages)
        oai.post("/embeddings", use: embedding.embeddings)
        oai.post("/rerank", use: embedding.rerank)
        oai.post("/systemone", use: SystemOneController(engine: decisions, log: orchestrator.sessionLog).decide)
        oai.post("/audio/transcriptions", use: TranscriptionsController(orchestrator: orchestrator).transcriptions)
        oai.post("/ocr", use: OCRController(orchestrator: orchestrator).ocr)
        let registry = pipelines ?? PipelineRegistry([ComposeSpec.defaultPipeline: PipelineSystem(decisions: decisions, stages: pipeline)])
        let pipelineController = PipelineController(registry: registry, chat: orchestrator)
        oai.post("/pipeline", use: pipelineController.pipeline)
        oai.post("/pipelines/{name}", use: pipelineController.pipeline)
        if let compose { ComposeController(control: compose).add(to: oai) }
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
        "/v1/models/preview":      { "get":  { "summary": "Memory a model would need, without loading it (?model_id=&context=), or the compute plan of a pipeline preset (?pipeline_preset=document-qa|voice-agent|support-triage&pipeline_memory=auto|resident|sequential)" } },
        "/v1/models/load":         { "post": { "summary": "Hot-swap model at runtime" } },
        "/v1/chat/completions":    { "post": { "summary": "OpenAI-compatible chat completions" } },
        "/v1/responses":           { "post": { "summary": "OpenAI Responses API (Codex/Cursor)" } },
        "/v1/messages":            { "post": { "summary": "Anthropic-compatible messages API" } },
        "/v1/score":               { "post": { "summary": "Top-k logprobs of supplied completion token ids after chat messages, in one forward pass (teacher for on-policy distill)" } },
        "/v1/embeddings":          { "post": { "summary": "OpenAI-compatible embeddings (serve --embedding-model)" } },
        "/v1/systemone":           { "post": { "summary": "Single-pass typed decisions: noul, choice, score (serve --decision-model; with cloud:<model> OpenRouter answers first and questions below --decision-escalate-below go to --decision-escalate, listed per stage in usage.stages)" } },
        "/v1/rerank":              { "post": { "summary": "Cohere/Jina-compatible rerank (serve --rerank-model)" } },
        "/v1/audio/transcriptions": { "post": { "summary": "OpenAI-compatible speech to text (multipart: file, model, language, response_format, temperature)" } },
        "/v1/ocr":                 { "post": { "summary": "Document OCR with GLM-OCR, MinerU2.5, PaddleOCR-VL or DeepSeek-OCR (JSON: file as base64 PDF or image, pages such as 1-3, task document|text|table|formula|chart, format markdown|json, model, resolution tiny|small|base|large|gundam for DeepSeek-OCR)" } },
        "/v1/pipeline":            { "post": { "summary": "Perceive (OCR of pdf/image attachments with serve --pipeline-ocr-model or Apple Vision, transcription of audio with serve --pipeline-audio-model, the chat model's audio encoder or SpeechAnalyzer (with --experimental-speech-analyzer)) → decide (System One on typed questions) → answer (the chat model, only for abstained questions and a task); returns perceived text, answers, the System 2 reply and a route receipt (swap_ms when serve --pipeline-memory loads models per stage; --pipeline-preset fills the models)" } },
        "/v1/pipelines/{name}":    { "post": { "summary": "The same request as /v1/pipeline, for a named pipeline of the compose file served with basalt compose up" } },
        "/v1/compose":             { "get":  { "summary": "Status of the compose system: pipelines, and per model whether it is loaded, the last use and idle unload of an on-demand model, and cloud spend today (basalt compose ps)" } },
        "/v1/compose/apply":       { "post": { "summary": "Apply a changed compose file to the running system ({\"yaml\": ...}): loads added models, reloads changed ones, unloads removed ones and swaps the pipelines; refuses changes that need a restart (basalt compose up on a running system)" } }
      }
    }
    """
}
