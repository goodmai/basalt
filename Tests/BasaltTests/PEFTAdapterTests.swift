import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("PEFT LoRA adapters: harness-hub layout loads, maps VLM-style keys, and matches W + s·B·A (#675 P0, #228)", .serialized)
struct PEFTAdapterTests {
    static let targets: [TinyModel.Family: [String]] = [
        .qwen35: ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"],
        .lfm2Moe: ["conv.in_proj", "self_attn.q_proj", "feed_forward.w1"],
    ]
    static let rank = 4

    struct Pair {
        let path: String
        let a: MLXArray
        let b: MLXArray
    }

    static func pairs(_ model: any LanguageModel, family: TinyModel.Family) -> [Pair] {
        let suffixes = targets[family] ?? []
        let linears = model.namedModules().compactMap { path, module in (module as? Linear).map { (path, $0) } }
            .filter { path, _ in suffixes.contains { path.hasSuffix(".\($0)") } }
        return withRandomState(MLXRandom.RandomState(seed: 675)) {
            linears.map { path, linear in
                let (outputs, inputs) = linear.shape
                return Pair(path: path, a: MLXRandom.normal([rank, inputs]) * 0.05, b: MLXRandom.normal([outputs, rank]) * 0.05)
            }
        }
    }

    static func peftKey(_ path: String) -> String {
        "base_model.model." + path.replacingOccurrences(of: "model.layers.", with: "model.language_model.layers.")
    }

    static func directory(_ pairs: [Pair], config: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "peft-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(config.utf8).write(to: directory.appending(component: "adapter_config.json"))
        var arrays = [String: MLXArray]()
        for pair in pairs {
            arrays[peftKey(pair.path) + ".lora_A.weight"] = pair.a
            arrays[peftKey(pair.path) + ".lora_B.weight"] = pair.b
        }
        try save(arrays: arrays, url: directory.appending(component: PEFTLoRAReader.weightsFile))
        return directory
    }

    static func config(alpha: Int = 8, extra: String = "") -> String {
        #"{"peft_type": "LORA", "r": \#(rank), "lora_alpha": \#(alpha), "use_rslora": false, "use_dora": false, "modules_to_save": null, "rank_pattern": {}, "alpha_pattern": {}, "target_modules": ["q_proj"]\#(extra)}"#
    }

    static var tokens: MLXArray { MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init))[.newAxis] }

    static func merged(_ family: TinyModel.Family, _ pairs: [Pair], scale: Float) throws -> any LanguageModel {
        let model = try TinyModel.make(family)
        let weights = pairs.map { pair -> (String, MLXArray) in
            let linear = model.namedModules().first { $0.0 == pair.path }!.1 as! Linear
            return ("\(pair.path).weight", linear.weight + scale * matmul(pair.b, pair.a))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        return model
    }

    @Test("A PEFT adapter with VLM-style keys gives the logits of the dense W + s·B·A model within 1e-4 and the same greedy ids, on the hybrid and the conv fixture",
          arguments: [TinyModel.Family.qwen35, .lfm2Moe])
    func matchesMergedWeights(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let pairs = Self.pairs(model, family: family)
        #expect(pairs.count >= 3)
        let directory = try Self.directory(pairs, config: Self.config())
        defer { try? FileManager.default.removeItem(at: directory) }
        let wrapped = try MLXInferenceEngine.attachAnyAdapterDirectory(directory, to: model)
        #expect(Set(wrapped) == Set(pairs.map(\.path)))
        let adapted = model(Self.tokens, cache: nil)
        let merged = try Self.merged(family, pairs, scale: 2)
        let expected = merged(Self.tokens, cache: nil)
        #expect(abs(adapted - expected).max().item(Float.self) < 1e-4)
        #expect(argMax(adapted, axis: -1).asArray(Int32.self) == argMax(expected, axis: -1).asArray(Int32.self))
    }

    @Test("The same adapter written in mlx-lm layout gives bit-identical logits")
    func matchesMLXLayout() throws {
        let peftModel = try TinyModel.make(.qwen35)
        let pairs = Self.pairs(peftModel, family: .qwen35)
        let peft = try Self.directory(pairs, config: Self.config())
        defer { try? FileManager.default.removeItem(at: peft) }
        _ = try MLXInferenceEngine.attachAnyAdapterDirectory(peft, to: peftModel)

        let mlx = FileManager.default.temporaryDirectory.appending(path: "mlxlm-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: mlx, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: mlx) }
        let keys = Set(pairs.compactMap { PEFTLoRAReader.layerRelative($0.path) }).sorted()
        let config = #"{"fine_tune_type": "lora", "num_layers": 1024, "lora_parameters": {"rank": \#(Self.rank), "scale": 2.0, "keys": \#(String(decoding: try JSONEncoder().encode(keys), as: UTF8.self))}}"#
        try Data(config.utf8).write(to: mlx.appending(component: "adapter_config.json"))
        var arrays = [String: MLXArray]()
        for pair in pairs { (arrays["\(pair.path).lora_a"], arrays["\(pair.path).lora_b"]) = (pair.a.T, pair.b.T) }
        try save(arrays: arrays, url: mlx.appending(component: "adapters.safetensors"))
        let mlxModel = try TinyModel.make(.qwen35)
        _ = try MLXInferenceEngine.attachAnyAdapterDirectory(mlx, to: mlxModel)
        #expect(abs(peftModel(Self.tokens, cache: nil) - mlxModel(Self.tokens, cache: nil)).max().item(Float.self) == 0)
    }

    @Test("rsLoRA scales by alpha over the square root of r")
    func rsLoRAScale() throws {
        let configuration = try JSONDecoder().decode(PEFTLoRAConfiguration.self, from: Data(Self.config(alpha: 8, extra: #", "use_rslora": true"#).replacingOccurrences(of: #""use_rslora": false, "#, with: "").utf8))
        #expect(configuration.scale == 4)
        #expect(configuration.refusal == nil)
    }

    @Test("DoRA, modules_to_save, rank_pattern, a base-changing init, a zero alpha, a key with no layer, a lone lora_A and a wrong rank are refused with the reason", arguments: [
        "dora", "save", "pattern", "pissa", "alpha", "unmatched", "lone", "rank",
    ])
    func refusals(case name: String) throws {
        let model = try TinyModel.make(.qwen35)
        var pairs = Self.pairs(model, family: .qwen35)
        var config = Self.config()
        switch name {
        case "dora": config = config.replacingOccurrences(of: #""use_dora": false"#, with: #""use_dora": true"#)
        case "save": config = config.replacingOccurrences(of: #""modules_to_save": null"#, with: #""modules_to_save": ["lm_head"]"#)
        case "pattern": config = config.replacingOccurrences(of: #""rank_pattern": {}"#, with: #""rank_pattern": {"q_proj": 8}"#)
        case "pissa": config = Self.config(extra: #", "init_lora_weights": "pissa_niter_4""#)
        case "alpha": config = Self.config(alpha: 0)
        case "unmatched": pairs[0] = Pair(path: "model.layers.99.mlp.down_proj", a: pairs[0].a, b: pairs[0].b)
        case "rank": pairs[0] = Pair(path: pairs[0].path, a: MLXArray.zeros([Self.rank + 1, pairs[0].a.dim(1)]), b: MLXArray.zeros([pairs[0].b.dim(0), Self.rank + 1]))
        default: break
        }
        var directory = try Self.directory(pairs, config: config)
        if name == "lone" {
            var arrays = try MLX.loadArrays(url: directory.appending(component: PEFTLoRAReader.weightsFile))
            arrays.removeValue(forKey: Self.peftKey(pairs[0].path) + ".lora_B.weight")
            try FileManager.default.removeItem(at: directory)
            directory = try Self.directory([], config: config)
            try save(arrays: arrays, url: directory.appending(component: PEFTLoRAReader.weightsFile))
        }
        defer { try? FileManager.default.removeItem(at: directory) }
        let expected = ["dora": "use_dora", "save": "modules_to_save", "pattern": "rank_pattern", "unmatched": "belongs to no linear layer", "lone": "only one of lora_A and lora_B", "rank": "has shape",
            "pissa": "init_lora_weights", "alpha": "must be positive"][name]!
        #expect {
            try MLXInferenceEngine.attachAnyAdapterDirectory(directory, to: model)
        } throws: { error in
            "\(error)".contains(expected)
        }
        #expect(MLXInferenceEngine.loraModules(of: model).isEmpty)
    }

    @Test("Served through the orchestrator by name, a PEFT adapter answers like the merged model and the base comes back")
    func servedByName() async throws {
        let engine = try await TinyModel.engine(.qwen35)
        let pairs = Self.pairs(try TinyModel.make(.qwen35), family: .qwen35)
        let directory = try Self.directory(pairs, config: Self.config())
        defer { try? FileManager.default.removeItem(at: directory) }
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadAdapters(["pi": directory])
        let base = try await orchestrator.generate(request: greedy(Conversation.firstTurn, key: nil))
        let adapted = try await orchestrator.generate(request: greedy(Conversation.firstTurn, key: nil).withAdapter("pi"))
        let mergedEngine = MLXInferenceEngine()
        await mergedEngine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-merged"), model: try Self.merged(.qwen35, pairs, scale: 2), processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
        let merged = try await mergedEngine.generate(request: greedy(Conversation.firstTurn, key: nil))
        #expect(adapted.generatedText == merged.generatedText)
        #expect(adapted.generatedText != base.generatedText)
        #expect(try await orchestrator.generate(request: greedy(Conversation.firstTurn, key: nil)).generatedText == base.generatedText)
    }

    @Test("LoRA weights take the layer's float type: the weight's for a dense layer, the affine scales' for a quantized one, and bfloat16 where the scales are uint8 (mxfp4, mxfp8, nvfp4)")
    func computeTypes() {
        #expect(PEFTLoRAReader.computeType(of: Linear(64, 32)) == .float32)
        let affine = QuantizedLinear(weight: MLXArray.zeros([32, 8], dtype: .uint32), bias: nil, scales: MLXArray.zeros([32, 2], dtype: .float16), biases: MLXArray.zeros([32, 2], dtype: .float16), groupSize: 32, bits: 4)
        #expect(PEFTLoRAReader.computeType(of: affine) == .float16)
        let microscaled = QuantizedLinear(weight: MLXArray.zeros([32, 8], dtype: .uint32), bias: nil, scales: MLXArray.zeros([32, 2], dtype: .uint8), biases: nil, groupSize: 32, bits: 4, mode: .mxfp4)
        #expect(PEFTLoRAReader.computeType(of: microscaled) == .bfloat16)
    }

    @Test("A config without lora_alpha and an unreadable weights file give different reasons")
    func separateReasons() throws {
        let model = try TinyModel.make(.qwen35)
        let directory = try Self.directory(Self.pairs(model, family: .qwen35), config: #"{"peft_type": "LORA", "r": 4}"#)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: PEFTLoRAReader.Refused.self) { try PEFTLoRAReader.container(at: directory, for: model) }
        do throws(PEFTLoRAReader.Refused) { _ = try PEFTLoRAReader.container(at: directory, for: model) } catch { #expect(error.reason.contains("adapter_config.json")) }
        try Data(Self.config().utf8).write(to: directory.appending(component: "adapter_config.json"))
        try Data("x".utf8).write(to: directory.appending(component: PEFTLoRAReader.weightsFile))
        do throws(PEFTLoRAReader.Refused) { _ = try PEFTLoRAReader.container(at: directory, for: model) } catch { #expect(error.reason.contains(PEFTLoRAReader.weightsFile)) }
    }
}
