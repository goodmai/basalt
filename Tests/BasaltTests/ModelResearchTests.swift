import Foundation
import Testing
import MLXNN
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Model X-ray MCP tools: arguments, schemas, gating and anatomy (#328)")
struct ModelResearchTests {
    static func arguments(_ tool: ResearchTool) -> [String: Any] {
        switch tool {
        case .modelDescribe, .overlaysList: [:]
        case .modelTensors: ["filter": "layers.1"]
        case .modelTensor: ["name": "model.norm.weight", "slice": [[0, 4]], "maxValues": 8]
        case .kvInspect: ["namespace": "chat"]
        case .traceRun: ["prompt": "hi", "position": 1, "layers": [0, 2], "topK": 5, "attention": false]
        case .intervene: ["kind": "steer", "layer": 1, "vector": [0.5, -1.0], "scale": 2.0]
        case .weightsDelta: ["tensor": "model.norm.weight", "values": [0.25]]
        case .overlayUndo: ["id": 3]
        case .experimentCompare: ["prompts": ["a", "b"], "overlays": [3], "maxTokens": 8]
        case .adapterApply: ["module": "model.layers.0.mlp.down_proj", "input": [1.0], "output": [2.0], "scale": 0.5, "gate": ["heldOut": ["h"], "maxKL": 0.25, "target": [["prompt": "2 + 2 =", "expected": "4"]]]]
        case .gateCheck: ["heldOut": ["h"], "maxPerplexityRise": 0.5]
        case .tensorPack: ["tensor": "w", "format": "affine", "bits": 3, "groupSize": 32]
        case .tensorUnpack: ["tensor": "w", "format": "float16"]
        case .trainExperts: ["count": 40, "seed": 7, "layers": [2, 3], "rank": 4, "iterations": 20, "heldOut": ["Name a colour."]]
        case .steerSearch: LabSteerTests.searchArguments
        case .steerCommit: LabSteerTests.commitArguments
        }
    }

    static func expected(_ tool: ResearchTool) -> ResearchCall {
        switch tool {
        case .modelDescribe: .describe
        case .modelTensors: .tensors(filter: "layers.1")
        case .modelTensor: .tensor(TensorRequest(name: "model.norm.weight", slice: [0 ..< 4], maxValues: 8))
        case .kvInspect: .promptCache(namespace: "chat")
        case .traceRun: .trace(Self.trace)
        case .intervene: .intervene(.steer(layer: 1, vector: [0.5, -1], scale: 2))
        case .weightsDelta: .delta(WeightDelta(tensor: "model.norm.weight", values: [0.25]))
        case .overlaysList: .overlays
        case .overlayUndo: .undo(3)
        case .experimentCompare: .compare(CompareRequest(prompts: ["a", "b"], overlays: [3], maxTokens: 8))
        case .adapterApply: .adapter(.rankOne(module: "model.layers.0.mlp.down_proj", input: [1], output: [2], scale: 0.5), gate: Self.gate)
        case .gateCheck: .gate(GateRequest(heldOut: ["h"], maxPerplexityRise: 0.5))
        case .tensorPack: .pack(PackRequest(tensor: "w", format: .affine(bits: 3, groupSize: 32)))
        case .tensorUnpack: .pack(PackRequest(tensor: "w", format: .float16))
        case .trainExperts: .train(Self.train)
        case .steerSearch: .contrast(LabSteerTests.searchRequest)
        case .steerCommit: .commitSteering(LabSteerTests.commitRequest)
        }
    }

    static var gate: GateRequest {
        var request = GateRequest(heldOut: ["h"], maxKL: 0.25)
        request.target = [GateTarget(prompt: "2 + 2 =", expected: "4")]
        return request
    }

    static var train: TrainRequest {
        var request = TrainRequest(tasks: .arithmetic(count: 40, seed: 7), gate: GateRequest(heldOut: ["Name a colour."]))
        (request.layers, request.rank, request.iterations) = ([2, 3], 4, 20)
        return request
    }

    static var trace: TraceRequest {
        var request = TraceRequest(prompt: "hi", position: 1, layers: [0, 2], topK: 5)
        request.attention = false
        return request
    }

    @Test("Every research tool resolves by name and parses its arguments into the contract's request", arguments: ResearchTool.allCases)
    func parses(_ tool: ResearchTool) throws {
        #expect(MCPTool(name: tool.rawValue) == .research(tool))
        #expect(try ResearchCall(tool, Self.arguments(tool)) == Self.expected(tool))
    }

    @Test("Omitted optional arguments take the contract's defaults")
    func defaults() throws {
        #expect(try TraceRequest(mcp: ["prompt": "x"]) == TraceRequest(prompt: "x"))
        #expect(try TensorRequest(mcp: ["name": "w"]) == TensorRequest(name: "w", maxValues: 256))
        #expect(try CompareRequest(mcp: ["prompts": ["p"]]) == CompareRequest(prompts: ["p"], overlays: [], maxTokens: 32))
        #expect(try WeightDelta(mcp: ["tensor": "w", "scale": NSNull()]) == WeightDelta(tensor: "w", scale: 1))
        #expect(try Intervention(mcp: ["kind": "ablate_expert", "layer": 2, "expert": 5]) == .ablateExpert(layer: 2, expert: 5))
        #expect(try Intervention(mcp: ["kind": "ablate_head", "layer": 0, "head": 1]) == .ablateHead(layer: 0, head: 1))
    }

    @Test("Malformed research arguments are -32602 naming the argument")
    func rejects() {
        let cases: [(ResearchTool, [String: Any], String)] = [
            (.modelTensor, [:], "Missing required argument: name"),
            (.modelTensor, ["name": "w", "slice": [[3, 1]]], "slice must be"),
            (.modelTensor, ["name": "w", "maxValues": 5000], "maxValues must be an integer in 0...4096"),
            (.modelTensors, ["filter": 7], "filter must be a string"),
            (.traceRun, ["prompt": "x", "topK": true], "topK must be an integer"),
            (.traceRun, ["prompt": "x", "attention": "yes"], "attention must be a boolean"),
            (.traceRun, ["prompt": 3], "prompt must be a string"),
            (.traceRun, ["prompt": "x", "layers": [-1]], "layers must be"),
            (.intervene, ["kind": "melt", "layer": 0], "kind must be one of: steer, ablate_head, ablate_expert"),
            (.intervene, ["kind": "steer", "layer": 0], "Missing required argument: vector"),
            (.intervene, ["kind": "ablate_head", "layer": -1, "head": 0], "layer must be an integer in 0..."),
            (.overlayUndo, ["id": 1.5], "id must be an integer"),
            (.experimentCompare, ["prompts": [String]()], "prompts must be an array of 1...16 strings"),
            (.experimentCompare, ["prompts": ["p"], "maxTokens": 0], "maxTokens must be an integer in 1...512"),
            (.weightsDelta, ["tensor": "w", "values": [Double]()], "values must be a non-empty array of numbers"),
            (.trainExperts, ["count": 40], "heldOut must be a non-empty array"),
            (.trainExperts, ["heldOut": ["p"], "count": 1], "count must be an integer in 2...10000"),
        ]
        for (tool, args, fragment) in cases {
            let failure = #expect(throws: MCPFailure.self) { try ResearchCall(tool, args) }
            #expect(failure?.code == -32602, "\(tool)")
            #expect(failure?.message.contains(fragment) == true, "\(tool): \(failure?.message ?? "no error")")
        }
    }

    @Test("Every research tool has an object schema whose properties cover its arguments and requirements", arguments: ResearchTool.allCases)
    func schema(_ tool: ResearchTool) throws {
        let schema = try #require(tool.schema.rawValue as? [String: Any])
        let properties = try #require(schema["properties"] as? [String: Any])
        #expect(schema["type"] as? String == "object")
        #expect(Set(Self.arguments(tool).keys).isSubset(of: properties.keys))
        #expect(Set(schema["required"] as? [String] ?? []).isSubset(of: properties.keys))
        #expect(!tool.summary.isEmpty)
    }

    @Test("Experiment tools are listed only with --research-tools; the read-only X-ray tools always are")
    func catalogGating() {
        let experiments = Set(ResearchTool.allCases.filter(\.isExperiment).map(\.rawValue))
        let readOnly = Set(ResearchTool.allCases.filter { !$0.isExperiment }.map(\.rawValue))
        let off = Set(MCPToolCatalog.catalog(researchTools: false).tools.map(\.name))
        let on = Set(MCPToolCatalog.catalog(researchTools: true).tools.map(\.name))
        #expect(readOnly == ["model_describe", "model_tensors", "model_tensor", "kv_inspect", "trace_run"])
        #expect(readOnly.isSubset(of: off) && off.isDisjoint(with: experiments))
        #expect(experiments.union(readOnly).isSubset(of: on) && on.contains("basalt_generate"))
    }

    @Test("Without --research-tools an experiment call is refused before it reaches the orchestrator")
    func callGating() async throws {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "/mock")
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: engine)) { capturer.capture($0) }
        let gated = MCPServer(orchestrator: ModelOrchestratorActor(engine: engine)) { capturer.capture($0) }.withResearchTools(true)
        let call = #"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"overlays_list"}}"#
        await server.dispatch(#"{"jsonrpc":"2.0","id":0,"method":"tools/list"}"#)
        await server.dispatch(call)
        await gated.dispatch(call)
        let lines = capturer.lines
        #expect(lines[0].contains("model_describe") && !lines[0].contains("overlays_list"))
        #expect(lines[1].contains("-32601") && lines[1].contains("--research-tools"))
        #expect(lines[2].contains("-32602") && lines[2].contains("no research tools"))
    }

    @Test("A research call with no model loaded is modelNotLoaded")
    func notLoaded() async {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())
        await #expect(throws: BasaltError.self) { try await orchestrator.research { model throws(BasaltError) in try await model.anatomy() } }
        let failure = MCPFailure(BasaltError.modelNotLoaded)
        #expect(failure.code == -32011 && BasaltError.modelNotLoaded.httpStatus == 503)
    }

    static func engine(_ family: TinyModel.Family) async throws -> MLXInferenceEngine {
        let directory = FileManager.default.temporaryDirectory.appending(path: "anatomy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(family.config.utf8).write(to: directory.appending(path: "config.json"))
        let engine = MLXInferenceEngine()
        await engine.attach(try TinyModel.container(family), modelDescription: directory.path)
        return engine
    }

    @Test("Hybrid Qwen3.5: linear-attention layers on MambaCache, full attention every second layer on KVCacheSimple")
    func hybridAnatomy() async throws {
        let anatomy = try await Self.engine(.qwen35).anatomy()
        #expect(anatomy.modelType == "qwen3_5_text")
        #expect(anatomy.layers.map(\.kind) == ["linear attention", "full attention", "linear attention", "full attention"])
        #expect(anatomy.layers.map(\.cache) == ["MambaCache", "KVCacheSimple", "MambaCache", "KVCacheSimple"])
        var fullAttention = LayerAnatomy(index: 1, kind: "full attention", cache: "KVCacheSimple")
        (fullAttention.heads, fullAttention.kvHeads, fullAttention.headDim) = (4, 2, 16)
        #expect(anatomy.layers[1] == fullAttention)
        #expect(anatomy.layers[0].heads == 4 && anatomy.layers[0].kvHeads == 2 && anatomy.layers[0].headDim == 16)
        #expect(anatomy.layers.allSatisfy { $0.experts == nil && $0.window == nil })
        #expect(anatomy.vocabulary == 256 && anatomy.hiddenSize == 64)
        #expect(anatomy.quantization == "None (FP16/BF16 Base)")
        #expect(anatomy.parameterCount > 0 && anatomy.bytes.rawValue >= Int64(anatomy.parameterCount))
    }

    @Test("MoE LFM2: conv and attention layers, a dense first layer, 4 experts with 2 per token after it")
    func moeAnatomy() async throws {
        let anatomy = try await Self.engine(.lfm2Moe).anatomy()
        #expect(anatomy.modelType == "lfm2_moe")
        #expect(anatomy.layers.map(\.kind) == ["conv", "full attention", "conv", "full attention"])
        #expect(anatomy.layers.map(\.experts) == [nil, 4, 4, 4])
        #expect(anatomy.layers.map(\.expertsPerToken) == [nil, 2, 2, 2])
        #expect(anatomy.layers.map(\.cache) == ["MambaCache", "KVCacheSimple", "MambaCache", "KVCacheSimple"])
        #expect(anatomy.layers[3].heads == 4 && anatomy.layers[3].kvHeads == 2 && anatomy.layers[3].headDim == 16)
        #expect(anatomy.layers[0].heads == nil)
    }

    @Test("A quantized model counts the parameters it was quantized from, not its packed words")
    func quantizedCount() throws {
        let model = try TinyModel.make(.lfm2Moe)
        let dense = AnatomyScan(model: model, caches: [])
        quantize(model: model, groupSize: 32, bits: 8)
        let quantized = AnatomyScan(model: model, caches: [])
        #expect(quantized.parameterCount == dense.parameterCount)
        #expect(quantized.bytes < dense.bytes)
        #expect(quantized.layers.map(\.experts) == [nil, 4, 4, 4])
    }

    @Test("model_describe goes through the orchestrator's gate and leaves it idle")
    func describeThroughServer() async throws {
        let orchestrator = ModelOrchestratorActor(engine: try await Self.engine(.qwen35))
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orchestrator) { capturer.capture($0) }
        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"model_describe"}}"#)
        #expect(capturer.lines.first?.contains("linear attention") == true)
        #expect(await orchestrator.gate.inFlight == 0)
    }
}
