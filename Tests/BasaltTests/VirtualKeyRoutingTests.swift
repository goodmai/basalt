import Testing
import Foundation
import Hummingbird
import HTTPTypes
import Logging
import NIOCore
import HummingbirdTesting
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltCLI

struct KeySource: RequestContextSource {
    var logger: Logger = Logger(label: "test")
}

struct KeyContext: RequestContext {
    typealias Source = KeySource
    var coreContext: CoreRequestContextStorage
    init(source: KeySource = KeySource()) { self.coreContext = .init(source: source) }
}

@Suite("Virtual keys and session routing (#34)")
struct VirtualKeyRoutingTests {
    static let team = VirtualKey(key: "vk-team-a-0123456789", id: "team-a", maxBudget: 2)
    static let budgetHeader = APIKeyAuthenticator<KeyContext>.budgetKeyHeader

    func pass(
        _ auth: APIKeyAuthenticator<KeyContext>, path: String = "/v1/chat/completions", headers: HTTPFields
    ) async throws -> (status: HTTPResponse.Status, budgetKey: String?) {
        let request = Request(
            head: .init(method: .post, scheme: nil, authority: nil, path: path, headerFields: headers),
            body: .init(buffer: ByteBuffer()))
        let seen = Seen()
        let response = try await auth.handle(request, context: KeyContext()) { forwarded, _ in
            await seen.set(forwarded.headers[Self.budgetHeader])
            return Response(status: .ok)
        }
        return (response.status, await seen.value)
    }

    actor Seen {
        var value: String?
        func set(_ value: String?) { self.value = value }
    }

    func bearer(_ token: String, spoof: String? = nil) -> HTTPFields {
        var headers = HTTPFields()
        headers[.authorization] = "Bearer \(token)"
        if let spoof { headers[Self.budgetHeader] = spoof }
        return headers
    }

    @Test("A virtual key is accepted and tags the request with its id; the main key tags nothing")
    func virtualKeyTagsRequest() async throws {
        let auth = APIKeyAuthenticator<KeyContext>(apiKey: "main-secret", virtualKeys: [Self.team])
        #expect(try await pass(auth, headers: bearer("vk-team-a-0123456789")) == (.ok, "team-a"))
        var xKey = HTTPFields()
        xKey[HTTPField.Name("x-api-key")!] = "vk-team-a-0123456789"
        #expect(try await pass(auth, headers: xKey) == (.ok, "team-a"))
        #expect(try await pass(auth, headers: bearer("main-secret")) == (.ok, nil))
        #expect(try await pass(auth, headers: bearer("vk-team-a-wrong")).status == .unauthorized)
    }

    @Test("A client cannot pick a budget key by sending the header itself")
    func spoofedBudgetHeaderIsStripped() async throws {
        let auth = APIKeyAuthenticator<KeyContext>(apiKey: "main-secret", virtualKeys: [Self.team])
        #expect(try await pass(auth, headers: bearer("main-secret", spoof: "team-a")) == (.ok, nil))
        #expect(try await pass(auth, headers: bearer("vk-team-a-0123456789", spoof: "other")) == (.ok, "team-a"))
        #expect(try await pass(auth, path: "/api/v1/health", headers: bearer("x", spoof: "team-a")) == (.ok, nil))
    }

    @Test("With only virtual keys configured, an empty main key never matches")
    func noMainKey() async throws {
        let auth = APIKeyAuthenticator<KeyContext>(apiKey: "", virtualKeys: [Self.team])
        var empty = HTTPFields()
        empty[HTTPField.Name("x-api-key")!] = ""
        #expect(try await pass(auth, headers: empty).status == .unauthorized)
        #expect(try await pass(auth, headers: bearer("vk-team-a-0123456789")) == (.ok, "team-a"))
    }

    @Test("Controllers carry x-session-id and the authenticated key into the generation request")
    func routedByHeaders() {
        var headers = HTTPFields()
        headers[GenerationRequest.sessionHeader] = "agent-7"
        headers[Self.budgetHeader] = "team-a"
        let routed = GenerationRequest(prompt: "p").routed(by: headers)
        #expect(routed.sessionId == "agent-7")
        #expect(routed.budgetKey == "team-a")
        #expect(GenerationRequest(prompt: "p").routed(by: [:]).sessionId == nil)
    }

    @Test("A budgetKey in the JSON body is ignored: only the authenticator sets it")
    func budgetKeyNotDecodable() throws {
        let request = try JSONDecoder().decode(
            GenerationRequest.self, from: Data(#"{"prompt": "p", "budgetKey": "team-a", "sessionId": "s"}"#.utf8))
        #expect(request.budgetKey == nil)
        #expect(request.sessionId == nil)
    }

    @Test("virtual_keys in the config: parsed, validated, unique")
    func configParsing() throws {
        func network(_ json: String) throws -> [VirtualKey] {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("vk-\(UUID().uuidString).json")
            try Data(json.utf8).write(to: url)
            return try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: nil, configFile: url.path).virtualKeys
        }
        #expect(try network(#"{"virtual_keys": [{"key": "vk-team-a-0123456789", "id": "team-a", "max_budget": 2}, {"key": "vk-team-b-0123456789", "id": "team-b"}]}"#)
            == [Self.team, VirtualKey(key: "vk-team-b-0123456789", id: "team-b")])
        #expect(try network(#"{"port": 8080}"#).isEmpty)
        for bad in [
            #"{"virtual_keys": [{"key": "short", "id": "a"}]}"#,
            #"{"virtual_keys": [{"key": "vk-team-a-0123456789"}]}"#,
            #"{"virtual_keys": [{"key": "vk-team-a-0123456789", "id": "a", "max_budget": -1}]}"#,
            #"{"virtual_keys": [{"key": "vk-team-a-0123456789", "id": "a", "max_budget": true}]}"#,
            #"{"virtual_keys": [{"key": "vk-team-a-0123456789", "id": "a"}, {"key": "vk-team-b-0123456789", "id": "a"}]}"#,
        ] {
            #expect(throws: (any Error).self) { try network(bad) }
        }
    }

    @Test("Key and session budgets from the config stop the cloud for that key or session only")
    func budgetsFromRequests() async throws {
        let costURL = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: costURL) }
        var settings = ServeCommand.CloudFallbackSettings(
            model: "openai/gpt-4-turbo", baseURL: URL(string: "http://127.0.0.1:9/v1")!, apiKey: "", localContextLimit: nil)
        settings.sessionBudget = 0
        let router = try await ServeCommand.cloudFallbackRouter(
            settings, orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            virtualKeys: [VirtualKey(key: "vk-team-a-0123456789", id: "team-a", maxBudget: 0)], costURL: costURL)
        let plain = GenerationRequest(prompt: "hi")
        #expect(await router.assess(plain) != nil)
        #expect(await router.assess(plain.withRouting(sessionId: nil, budgetKey: "team-a")) == nil)
        #expect(await router.assess(plain.withRouting(sessionId: nil, budgetKey: "team-b")) != nil)
        #expect(await router.assess(plain.withRouting(sessionId: "s1", budgetKey: nil)) == nil)
    }

    @Test("router_params turn on the capability reason; a threshold without parameters is an error")
    func routerSettings() async throws {
        let params = FileManager.default.temporaryDirectory.appendingPathComponent("bt-\(UUID().uuidString).json")
        try Data(#"{"strongEmbedding":[2],"weakEmbedding":[0],"weights":[1],"projection":[[1,-1]]}"#.utf8).write(to: params)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("cfg-\(UUID().uuidString).json")
        try Data(#"{"cloud_fallback": {"base_url": "http://127.0.0.1:9/v1", "model": "big", "router_params": "\#(params.path)", "cost_threshold": 0.5, "cache_stickiness": 0}}"#.utf8).write(to: file)
        let settings = try #require(try ServeCommand.resolveCloudFallback(
            cliModel: nil, cliBaseURL: nil, cliDisabled: false, cliLocalContextLimit: nil, configFile: file.path,
            environment: [:], savedOpenRouterKey: { nil }))
        #expect(settings.routerParams == params)
        let costURL = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: costURL) }
        let router = try await ServeCommand.cloudFallbackRouter(
            settings, orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()), costURL: costURL)
        let hard = try #require(await router.assess(GenerationRequest(prompt: BradleyTerryRouterTests.hardWord)))
        let easy = try #require(await router.assess(GenerationRequest(prompt: BradleyTerryRouterTests.easyWord)))
        #expect(hard.capabilityGap > 0.8)
        #expect(easy.capabilityGap == 0)

        #expect(throws: (any Error).self) {
            try ServeCommand.resolveCloudFallback(
                cliModel: "m", cliBaseURL: "http://127.0.0.1:9/v1", cliDisabled: false, cliLocalContextLimit: nil,
                cliCostThreshold: 0.4, configFile: nil, environment: [:], savedOpenRouterKey: { nil })
        }
    }

    @Test("Client-chosen session ids cannot grow the session tables without bound")
    func sessionTablesAreBounded() async throws {
        let costURL = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: costURL) }
        let tracker = try CostTracker(storageURL: costURL, budget: .init(dailyLimit: 1e9, monthlyLimit: 1e9))
        for i in 0...CostTracker.maxTrackedSessions {
            try await tracker.recordUsage(model: "m", cost: 0.001, scopes: [.session("s\(i)")])
        }
        #expect(await tracker.trackedSessionCount <= CostTracker.maxTrackedSessions)
    }

    @Test("Through the real router a virtual key's request keeps its body and reaches the engine tagged with the key and session")
    func endToEnd() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock", virtualKeys: [Self.team]))
        let app = Application(router: server.makeRouter())
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "application/json", .authorization: "Bearer vk-team-a-0123456789",
                          GenerationRequest.sessionHeader: "agent-7", Self.budgetHeader: "spoofed"],
                body: ByteBuffer(string: #"{"messages": [{"role": "user", "content": "hello there"}]}"#)
            ) { res in
                #expect(res.status == .ok)
            }
            try await client.execute(
                uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"messages": [{"role": "user", "content": "x"}]}"#)
            ) { res in
                #expect(res.status == .unauthorized)
            }
        }
        let sent = try #require(await engine.getRequests().first)
        #expect(sent.messages?.first?.text == "hello there")
        #expect(sent.budgetKey == "team-a")
        #expect(sent.sessionId == "agent-7")
    }
}
