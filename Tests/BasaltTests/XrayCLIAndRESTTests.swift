import ArgumentParser
import Foundation
import Hummingbird
import HummingbirdTesting
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("basalt xray CLI parser, formatter, and REST /v1/xray/* + /v1/telemetry/frontier (#877)", .serialized)
struct XrayCLIAndRESTTests {
    @Test("XrayCommand parses inspect, depth, outliers, and causal subcommands and rejects invalid --layers")
    func cliParsingAndLayerValidation() throws {
        let inspect = try #require(try XrayCommand.parseAsRoot(["inspect", "org/model", "--json"]) as? XrayCommand.Inspect)
        #expect(inspect.model == "org/model" && inspect.json)

        let depth = try #require(try XrayCommand.parseAsRoot(["depth", "org/model", "--prompt", "hello", "--max-seq-len", "128", "--layers", "0,2"]) as? XrayCommand.Depth)
        #expect(depth.prompt == "hello" && depth.maxSeqLen == 128)
        #expect(try XrayFormatter.parseLayers(depth.layers) == [0, 2])

        let outliers = try #require(try XrayCommand.parseAsRoot(["outliers", "org/model", "--prompt", "hi", "--sink-count", "2"]) as? XrayCommand.Outliers)
        #expect(outliers.sinkCount == 2 && !outliers.json)

        let causal = try #require(try XrayCommand.parseAsRoot([
            "causal", "org/model", "--clean", "a", "--corrupted", "b", "--target-id", "1", "--foil-id", "2", "--no-patching",
        ]) as? XrayCommand.Causal)
        #expect(causal.clean == "a" && causal.corrupted == "b" && causal.targetID == 1 && causal.foilID == 2 && causal.noPatching)

        #expect(try XrayFormatter.parseLayers(nil) == nil)
        #expect(throws: ValidationError.self) { _ = try XrayFormatter.parseLayers("0,-1") }
        #expect(throws: ValidationError.self) { _ = try XrayFormatter.parseLayers("abc") }
    }

    @Test("XrayFormatter renders human tables and JSON from live TinyModel.engine(.qwen3) reports")
    func formatterRendersLiveReports() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let anatomy = try await engine.anatomy()
        let tensors = try await engine.tensors(matching: nil)
        let inspectText = XrayFormatter.formatInspect(anatomy, tensors: tensors)
        #expect(inspectText.contains("Model X-Ray Inspect") && inspectText.contains("tensors"))
        let inspectJSON = try XrayFormatter.json(XrayInspectSnapshot(anatomy: anatomy, tensors: tensors))
        #expect(inspectJSON.contains("\"architecture\""))

        let depthReport = try await engine.runDepthDiagnostics(DepthDiagnosticsRequest(prompt: "hello world", maxSequenceLength: 32))
        let depthText = XrayFormatter.formatDepth(depthReport)
        #expect(depthText.contains("Activation Depth Diagnostics") && depthText.contains("softcap_sat"))

        let outlierReport = try await engine.runOutlierProfile(OutlierProfileRequest(prompts: ["hello world"], sinkTokenCount: 2))
        #expect(XrayFormatter.formatOutliers(outlierReport).contains("Outlier & Attention Sink Profile"))

        let causalReport = try await engine.runCausalSweep(CausalSweepRequest(
            cleanPrompt: "hello world", corruptedPrompt: "goodbye moon", targetTokenID: 1, foilTokenID: 2, includePatching: false
        ))
        #expect(XrayFormatter.formatCausal(causalReport).contains("Causal Sweep"))
    }

    @Test("REST /v1/xray/{depth,outliers,causal} require --research-tools and honor MCPToolFilter")
    func restXrayResearchGating() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let orchestrator = ModelOrchestratorActor(engine: engine)
        let defaultApp = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "tiny-qwen3")).makeRouter())
        try await defaultApp.test(.router) { client in
            try await client.execute(uri: "/v1/xray/inspect", method: .get) { #expect($0.status == .ok) }
            for path in ["/v1/xray/depth", "/v1/xray/outliers", "/v1/xray/causal"] {
                try await client.execute(uri: path, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "{\"prompt\":\"hello\"}")) {
                    #expect($0.status == .forbidden)
                }
            }
        }
        let filteredServer = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "tiny-qwen3"))
            .withResearchTools(true, filter: try MCPToolFilter(disable: ["causal_sweep"]))
        let filteredApp = Application(router: filteredServer.makeRouter())
        try await filteredApp.test(.router) { client in
            try await client.execute(uri: "/v1/xray/depth", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "{\"prompt\":\"hello world\"}")) {
                #expect($0.status == .ok)
            }
            try await client.execute(uri: "/v1/xray/causal", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "{\"cleanPrompt\":\"a\",\"corruptedPrompt\":\"b\",\"targetTokenID\":1,\"foilTokenID\":2}")) {
                #expect($0.status == .forbidden)
            }
        }
    }

    @Test("REST /v1/xray/depth requires admin key and rejects virtual-key tenants")
    func restXrayAuthAndVirtualKeys() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let orchestrator = ModelOrchestratorActor(engine: engine)
        let authConfig = try ServerConfig(
            modelPath: "tiny-qwen3",
            apiKey: "admin-secret",
            access: .init(virtualKeys: [VirtualKey(key: "tenant-secret", id: "tenant")])
        )
        let authApp = Application(router: RESTServer(orchestrator: orchestrator, config: authConfig).withResearchTools(true).makeRouter())
        try await authApp.test(.router) { client in
            try await client.execute(uri: "/v1/xray/depth", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "{\"prompt\":\"hello\"}")) {
                #expect($0.status == .unauthorized)
            }
            try await client.execute(uri: "/v1/xray/depth", method: .post, headers: [.contentType: "application/json", .authorization: "Bearer tenant-secret"], body: ByteBuffer(string: "{\"prompt\":\"hello\"}")) {
                #expect($0.status == .forbidden)
            }
            try await client.execute(uri: "/v1/xray/depth", method: .post, headers: [.contentType: "application/json", .authorization: "Bearer admin-secret"], body: ByteBuffer(string: "{\"prompt\":\"hello\"}")) {
                #expect($0.status == .ok)
            }
        }
    }

    @Test("REST /v1/xray/* matches direct engine output and accepts minimal JSON")
    func restXrayEndpointsParity() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let app = Application(router: RESTServer(orchestrator: ModelOrchestratorActor(engine: engine), config: try ServerConfig(modelPath: "tiny-qwen3")).withResearchTools(true).makeRouter())
        let directDepth = try await engine.runDepthDiagnostics(DepthDiagnosticsRequest(prompt: "hello world"))
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/xray/depth", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "{\"prompt\":\"hello world\"}")) { res in
                #expect(res.status == .ok)
                let report = try JSONDecoder().decode(DepthDiagnosticsReport.self, from: Data(res.body.readableBytesView))
                #expect(report.layerCount == directDepth.layerCount && abs(report.maxSMaxOverall - directDepth.maxSMaxOverall) < 1e-6)
            }
            let outlierBody = try JSONEncoder().encode(OutlierProfileRequest(prompts: ["hello world"], sinkTokenCount: 2))
            try await client.execute(uri: "/v1/xray/outliers", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(bytes: outlierBody)) { res in
                #expect(res.status == .ok)
                let report = try JSONDecoder().decode(OutlierProfileReport.self, from: Data(res.body.readableBytesView))
                #expect(report.layerCount > 0 && !report.layers.isEmpty)
            }
            let causalBody = try JSONEncoder().encode(CausalSweepRequest(cleanPrompt: "hello world", corruptedPrompt: "goodbye moon", targetTokenID: 1, foilTokenID: 2, includePatching: false))
            try await client.execute(uri: "/v1/xray/causal", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(bytes: causalBody)) { res in
                #expect(res.status == .ok)
                let report = try JSONDecoder().decode(CausalSweepReport.self, from: Data(res.body.readableBytesView))
                #expect(report.conservationError < 1e-3)
            }
        }
    }

    @Test("REST /v1/xray/* rejects malformed bodies, invalid bounds, and unloaded engines")
    func restXrayValidationAndUnloaded() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let app = Application(router: RESTServer(orchestrator: ModelOrchestratorActor(engine: engine), config: try ServerConfig(modelPath: "tiny-qwen3")).withResearchTools(true).makeRouter())
        try await app.test(.router) { client in
            for badBody in ["{not valid json", "{\"prompt\":\"hello\",\"maxSequenceLength\":0}", "{\"prompt\":\"hello\",\"layers\":[99]}"] {
                try await client.execute(uri: "/v1/xray/depth", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: badBody)) {
                    #expect($0.status == .badRequest)
                }
            }
            let tooMany = try JSONEncoder().encode(OutlierProfileRequest(prompts: Array(repeating: "hi", count: 33)))
            try await client.execute(uri: "/v1/xray/outliers", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(bytes: tooMany)) {
                #expect($0.status == .badRequest)
            }
        }
        let unloadedApp = Application(router: RESTServer(orchestrator: ModelOrchestratorActor(engine: MLXInferenceEngine(.standard)), config: try ServerConfig(modelPath: "none")).withResearchTools(true).makeRouter())
        try await unloadedApp.test(.router) { client in
            try await client.execute(uri: "/v1/xray/inspect", method: .get) { #expect($0.status == .serviceUnavailable) }
        }
    }

    @Test("GET /v1/telemetry/frontier returns populated recurrent state_health after TinyModel.gdn2 generation")
    func restFrontierTelemetryLiveGDN2() async throws {
        let tracker = TelemetryTracker()
        let engine = MLXInferenceEngine(.standard, telemetry: tracker)
        await engine.attach(try TinyModel.container(.gdn2))
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: tracker)
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "hello world", maxTokens: 3, temperature: 0))
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "tiny-gdn2"))
        let app = Application(router: server.makeRouter())

        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/telemetry/frontier", method: .get) { res in
                #expect(res.status == .ok)
                let payload = try JSONDecoder().decode(FrontierTelemetryResponse.self, from: Data(res.body.readableBytesView))
                #expect(payload.loaded)
                let health = try #require(payload.stateHealth)
                #expect(!health.layers.isEmpty)
                #expect(health.layers.allSatisfy { $0.nonFiniteCount == 0 })
            }
        }
    }
}
