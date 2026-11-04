import Testing
import Foundation
import ArgumentParser
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltCLI

@Suite("Cloud fallback settings (#22)")
struct CloudFallbackSettingsTests {
    func config(_ json: String) throws -> String {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-\(UUID().uuidString).json")
        try Data(json.utf8).write(to: url)
        return url.path
    }

    func resolve(
        model: String? = nil, baseURL: String? = nil, disabled: Bool = false, limit: Int? = nil, file: String? = nil,
        env: [String: String] = [:], saved: String? = nil
    ) throws -> ServeCommand.CloudFallbackSettings? {
        try ServeCommand.resolveCloudFallback(
            .init(model: model, baseURL: baseURL, localContextLimit: limit), disabled: disabled, configFile: file,
            environment: env, savedOpenRouterKey: { saved })
    }

    @Test("Off unless a model is configured")
    func offByDefault() throws {
        #expect(try resolve() == nil)
        #expect(try resolve(file: config(#"{"port": 8080}"#)) == nil)
    }

    @Test("A self-hosted OpenAI-compatible endpoint from the config needs no key, and the OpenRouter key never goes there")
    func selfHosted() throws {
        let file = try config(#"{"cloud_fallback": {"enabled": true, "base_url": "http://192.168.1.20:8000/v1", "model": "qwen3-235b", "local_context_limit": 32768}}"#)
        let settings = try #require(try resolve(file: file, env: ["OPENROUTER_API_KEY": "sk-or"], saved: "sk-saved"))
        #expect(settings == .init(model: "qwen3-235b", baseURL: URL(string: "http://192.168.1.20:8000/v1")!, apiKey: "", localContextLimit: 32768))
        #expect(throws: Never.self) {
            _ = try OpenRouterClient(config: .init(apiKey: "", baseURL: settings.baseURL, timeout: 1, maxRetries: 0))
        }
    }

    @Test("A key for a self-hosted endpoint comes from the variable named by api_key_env, or api_key")
    func selfHostedKey() throws {
        let file = try config(#"{"cloud_fallback": {"base_url": "https://llm.lan/v1", "model": "m", "api_key_env": "LAN_KEY"}}"#)
        #expect(try resolve(file: file, env: ["LAN_KEY": "lan"])?.apiKey == "lan")
        let inline = try config(#"{"cloud_fallback": {"base_url": "https://llm.lan/v1", "model": "m", "api_key": "inline"}}"#)
        #expect(try resolve(file: inline, env: ["LAN_KEY": "lan"])?.apiKey == "inline")
    }

    @Test("OpenRouter is the default endpoint and requires its key")
    func openRouter() throws {
        #expect(try resolve(model: "anthropic/claude-sonnet-4", env: ["OPENROUTER_API_KEY": "sk-or"])
            == .init(model: "anthropic/claude-sonnet-4", baseURL: URL(string: "https://openrouter.ai/api/v1")!, apiKey: "sk-or", localContextLimit: nil))
        #expect(try resolve(model: "m", saved: "sk-saved")?.apiKey == "sk-saved")
        #expect(throws: ValidationError.self) { try resolve(model: "m") }
        #expect(throws: BasaltError.self) {
            _ = try OpenRouterClient(config: .init(apiKey: "", baseURL: URL(string: "https://openrouter.ai/api/v1")!, timeout: 1, maxRetries: 0))
        }
    }

    @Test("enabled: false switches it off; --cloud-fallback turns it back on; --no-cloud-fallback wins over both")
    func toggles() throws {
        let file = try config(#"{"cloud_fallback": {"enabled": false, "base_url": "http://127.0.0.1:1234/v1", "model": "local-big"}}"#)
        #expect(try resolve(file: file) == nil)
        #expect(try resolve(model: "other", file: file)?.model == "other")
        let on = try config(#"{"cloud_fallback": {"base_url": "http://127.0.0.1:1234/v1", "model": "local-big"}}"#)
        #expect(try resolve(file: on)?.model == "local-big")
        #expect(try resolve(disabled: true, file: on) == nil)
        #expect(try resolve(baseURL: "http://10.0.0.5:8000/v1", file: on)?.baseURL.host == "10.0.0.5")
    }

    @Test("Bad values are errors, never a silent fallback")
    func badValues() throws {
        #expect(throws: ValidationError.self) { try resolve(model: "m", baseURL: "ftp://x/v1") }
        #expect(throws: ValidationError.self) { try resolve(model: "m", baseURL: "http://x/v1", limit: 0) }
        #expect(throws: ValidationError.self) { try resolve(file: config(#"{"cloud_fallback": {"model": 3}}"#)) }
        #expect(throws: ValidationError.self) { try resolve(file: config(#"{"cloud_fallback": "on"}"#)) }
    }

    @Test("The routing outcome bias is kept next to the cost file only with router params, and read back on the next start")
    func outcomeBiasPersistence() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("bias-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let params = dir.appendingPathComponent("bt.json")
        try Data(#"{"strongEmbedding":[2],"weakEmbedding":[0],"weights":[1],"projection":[[1,-1]]}"#.utf8).write(to: params)
        let costURL = dir.appendingPathComponent("cost.json")
        let biasURL = CloudCommand.routerBiasURL(nextTo: costURL)
        let request = GenerationRequest(prompt: "p", tools: [ToolDefinition(name: "t")])
        let failing = BradleyTerryRouterTests.response(.repetition)
        let base = ServeCommand.CloudFallbackSettings(model: "big", baseURL: URL(string: "http://127.0.0.1:9/v1")!, apiKey: "", localContextLimit: nil)
        var routed = base
        routed.routerParams = params
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())

        let plain = try await ServeCommand.cloudFallbackRouter(base, orchestrator: orchestrator, costURL: costURL)
        await plain.recordLocalOutcome(request, response: failing, localModel: "qwen@abc")
        #expect(!FileManager.default.fileExists(atPath: biasURL.path))
        #expect(await plain.routingBias == nil)

        let first = try await ServeCommand.cloudFallbackRouter(routed, orchestrator: orchestrator, costURL: costURL)
        await first.recordLocalOutcome(request, response: failing, localModel: "qwen@abc")
        #expect(FileManager.default.fileExists(atPath: biasURL.path))
        let restarted = try await ServeCommand.cloudFallbackRouter(routed, orchestrator: orchestrator, costURL: costURL)
        let beta = try #require(await restarted.routingBias?["qwen@abc"]?["toolUse"])
        #expect(beta > 0.9 * RoutingOutcomeBias.eta && beta <= RoutingOutcomeBias.eta)
    }
}
