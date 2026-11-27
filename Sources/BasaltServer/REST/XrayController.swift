import BasaltCore
import Foundation
import HTTPTypes
import Hummingbird

public struct FrontierTelemetryResponse: Sendable, Codable {
    public var loaded: Bool
    public var modelId: String
    public var servingSLO: ServingSLOSnapshot?
    public var experts: ExpertTelemetry?
    public var stateHealth: RecurrentStateHealthSnapshot?
    public var reasoning: ReasoningTelemetrySnapshot?

    public init(
        loaded: Bool,
        modelId: String,
        servingSLO: ServingSLOSnapshot? = nil,
        experts: ExpertTelemetry? = nil,
        stateHealth: RecurrentStateHealthSnapshot? = nil,
        reasoning: ReasoningTelemetrySnapshot? = nil
    ) {
        self.loaded = loaded
        self.modelId = modelId
        self.servingSLO = servingSLO
        self.experts = experts
        self.stateHealth = stateHealth
        self.reasoning = reasoning
    }
}

struct XrayController: Sendable {
    let orchestrator: any ModelOrchestrating
    var researchTools: Bool = false
    var toolFilter: MCPToolFilter = .standard

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.get("/xray/inspect", use: inspect)
        group.post("/xray/depth", use: depth)
        group.post("/xray/outliers", use: outliers)
        group.post("/xray/causal", use: causal)
        group.get("/telemetry/frontier", use: frontierTelemetry)
    }

    @Sendable
    func inspect(request: Request, context: BasicRequestContext) async throws -> Response {
        do {
            let anatomy = try await orchestrator.research { model throws(BasaltError) in
                try await model.anatomy()
            }
            return Self.jsonResponse(try JSONEncoder().encode(anatomy))
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/v1/xray/inspect")
        }
    }

    @Sendable
    func depth(request: Request, context: BasicRequestContext) async throws -> Response {
        guard toolFilter.allows("depth_diagnostics", researchTools: researchTools) else {
            return Self.researchDisabledResponse(tool: "depth_diagnostics")
        }
        return try await postJSON(request, path: "/v1/xray/depth") { (dto: DepthDiagnosticsRequest) in
            try await orchestrator.research { model throws(BasaltError) in
                try await ResearchCall.lab(model).runDepthDiagnostics(dto)
            }
        }
    }

    @Sendable
    func outliers(request: Request, context: BasicRequestContext) async throws -> Response {
        guard toolFilter.allows("outlier_profile", researchTools: researchTools) else {
            return Self.researchDisabledResponse(tool: "outlier_profile")
        }
        return try await postJSON(request, path: "/v1/xray/outliers") { (dto: OutlierProfileRequest) in
            try await orchestrator.research { model throws(BasaltError) in
                try await ResearchCall.lab(model).runOutlierProfile(dto)
            }
        }
    }

    @Sendable
    func causal(request: Request, context: BasicRequestContext) async throws -> Response {
        guard toolFilter.allows("causal_sweep", researchTools: researchTools) else {
            return Self.researchDisabledResponse(tool: "causal_sweep")
        }
        return try await postJSON(request, path: "/v1/xray/causal") { (dto: CausalSweepRequest) in
            try await orchestrator.research { model throws(BasaltError) in
                try await ResearchCall.lab(model).runCausalSweep(dto)
            }
        }
    }

    @Sendable
    func frontierTelemetry(request: Request, context: BasicRequestContext) async throws -> Response {
        let snapshot = await orchestrator.telemetry.snapshot()
        let payload = FrontierTelemetryResponse(
            loaded: await orchestrator.isLoaded,
            modelId: await orchestrator.currentModelId,
            servingSLO: snapshot.servingSLO,
            experts: snapshot.experts,
            stateHealth: snapshot.stateHealth,
            reasoning: snapshot.reasoning
        )
        return Self.jsonResponse(try JSONEncoder().encode(payload))
    }

    private func postJSON<Req: Decodable & Sendable, Res: Encodable & Sendable>(
        _ request: Request,
        path: String,
        run: (Req) async throws -> Res
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: ChatPipeline.textBodyLimit)
        let data = Data(buffer.readableBytesView)
        let dto: Req
        do {
            dto = try JSONDecoder().decode(Req.self, from: data)
        } catch {
            return ProviderErrorResponse.response(
                status: .badRequest,
                message: "Invalid \(path) body: \(error.localizedDescription)",
                error: error,
                shape: .openAI
            )
        }
        do {
            let result = try await run(dto)
            return Self.jsonResponse(try JSONEncoder().encode(result))
        } catch {
            return ProviderErrorResponse.response(for: error, path: path)
        }
    }

    private static func researchDisabledResponse(tool: String) -> Response {
        ProviderErrorResponse.response(
            status: .forbidden,
            message: "\(tool) is disabled; start the server with serve --research-tools (or --mcp-enable \(tool))",
            shape: .openAI
        )
    }

    private static func jsonResponse(_ data: Data) -> Response {
        var headers = HTTPFields()
        headers[.contentType] = "application/json; charset=utf-8"
        return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
    }
}
