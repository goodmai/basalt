import Foundation
import Testing
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
@testable import BasaltInference

enum TinyPrism {
    static let rotated = ["lm_head", "model.embed_tokens", "model.layers.0.mlp.down_proj", "model.layers.1.self_attn.q_proj"]

    static func config(modules: [String] = rotated, schema: Int = 2) -> String {
        let entries = modules.map { path in
            #"{"path": "\#(path)", "block": 64, "embedding": \#(path.hasSuffix("embed_tokens")), "dtype": "float16"}"#
        }.joined(separator: ", ")
        return """
            {"model_type": "prism_hadamard_qwen35", "schema_version": \(schema), "base_model_type": "qwen3_5",
             "tensor_namespace": "mlx-vlm-qwen3_5", "gdn_activation_layout": "grouped",
             "quantization": {"bits": 8, "group_size": 32, "mode": "affine"},
             "modules": [\(entries)], "text_config": \(TinyHybrid.config)}
            """
    }

    static func base() throws -> MLXLLM.Qwen35Model {
        let model = try withRandomState(MLXRandom.RandomState(seed: 7)) {
            MLXLLM.Qwen35Model(try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: Data(config().utf8)))
        }
        eval(model)
        return model
    }

    static func packed(from base: MLXLLM.Qwen35Model) throws -> (model: MLXLLM.Qwen35Model, signs: [String: MLXArray]) {
        let model = try PrismHadamard.model(configuration: Data(config().utf8))
        var parameters = Dictionary(uniqueKeysWithValues: base.parameters().flattened())
        var signs: [String: MLXArray] = [:]
        for (index, path) in rotated.map({ "language_model." + $0 }).enumerated() {
            let weight = try #require(parameters[path + ".weight"])
            let sign = withRandomState(MLXRandom.RandomState(seed: UInt64(index + 1))) {
                MLXRandom.bernoulli(0.5, [weight.dim(1)]).asType(.float32) * 2 - 1
            }
            let rotation = try #require(SignedBlockHadamard(blockSize: 64, width: weight.dim(1)))
            let (wq, scales, biases) = MLX.quantized(rotation.forward(weight.asType(.float32), signs: sign), groupSize: 32, bits: 8)
            parameters[path + ".weight"] = wq
            parameters[path + ".scales"] = scales.asType(.float16)
            parameters[path + ".biases"] = biases?.asType(.float16)
            parameters[path + ".signs"] = sign
            signs[path] = sign
        }
        try model.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        eval(model)
        return (model, signs)
    }
}

@Suite("Hadamard-rotated ternary checkpoints (Bonsai 2, prism_hadamard_qwen35)")
struct PrismHadamardTests {
    @Test("The signed block rotation is undone by its inverse")
    func rotationRoundTrip() throws {
        let rotation = try #require(SignedBlockHadamard(blockSize: 64, width: 128))
        let x = MLXRandom.normal([3, 128])
        let signs = MLXRandom.bernoulli(0.5, [128]).asType(.float32) * 2 - 1
        #expect(abs(rotation.inverse(rotation.forward(x, signs: signs), signs: signs) - x).max().item(Float.self) < 1e-4)
        #expect(SignedBlockHadamard(blockSize: 48, width: 96) == nil)
        #expect(SignedBlockHadamard(blockSize: 64, width: 96) == nil)
    }

    @Test("A model loaded from rotated weights computes the same logits as the unrotated model")
    func rotatedModelMatchesBase() throws {
        let base = try TinyPrism.base()
        let (packed, _) = try TinyPrism.packed(from: base)
        let leaves = Dictionary(uniqueKeysWithValues: packed.leafModules().flattened())
        #expect(leaves["language_model.lm_head"] is HadamardQuantizedLinear)
        #expect(leaves["language_model.model.embed_tokens"] is HadamardQuantizedEmbedding)
        #expect(leaves["language_model.model.layers.0.mlp.down_proj"] is HadamardQuantizedLinear)
        #expect(!(leaves["language_model.model.layers.0.mlp.up_proj"] is HadamardQuantizedLinear))

        let tokens = MLXArray([3, 17, 42, 99, 5, 200, 7, 64].map(Int32.init))[.newAxis]
        let expected = base(tokens, cache: nil)
        let actual = packed(tokens, cache: nil)
        let spread = abs(expected).max().item(Float.self)
        #expect(abs(actual - expected).max().item(Float.self) < 0.02 * spread)
        #expect(argMax(actual, axis: -1).asArray(Int32.self) == argMax(expected, axis: -1).asArray(Int32.self))
    }

    @Test("After loading, floating weights outside the rotated modules run in float16, A_log and the signs stay float32")
    func preparedDTypes() throws {
        let (packed, _) = try TinyPrism.packed(from: try TinyPrism.base())
        try PrismHadamard.prepareLoaded(packed)
        let parameters = Dictionary(uniqueKeysWithValues: packed.parameters().flattened())
        #expect(parameters["language_model.model.norm.weight"]?.dtype == .float16)
        #expect(parameters["language_model.model.layers.0.mlp.up_proj.weight"]?.dtype == .float16)
        #expect(parameters["language_model.model.layers.0.linear_attn.A_log"]?.dtype == .float32)
        #expect(parameters["language_model.lm_head.signs"]?.dtype == .float32)
        #expect(parameters["language_model.lm_head.scales"]?.dtype == .float16)
        let logits = packed(MLXArray([1, 2, 3].map(Int32.init))[.newAxis], cache: nil)
        #expect(logits.dtype == .float16)
    }

    @Test("Signs other than ±1 are refused at load")
    func invalidSigns() throws {
        let (packed, signs) = try TinyPrism.packed(from: try TinyPrism.base())
        let path = "language_model.model.layers.1.self_attn.q_proj"
        let broken = try #require(signs[path]) * 2
        try packed.update(parameters: ModuleParameters.unflattened([(path + ".signs", broken)]), verify: [.noUnusedKeys])
        #expect(throws: PrismHadamardError.invalidSigns(path)) { try PrismHadamard.prepareLoaded(packed) }
    }

    @Test("A model without rotated modules is left alone by the post-load step")
    func plainModelUntouched() throws {
        let base = try TinyPrism.base()
        try PrismHadamard.prepareLoaded(base)
        #expect(Dictionary(uniqueKeysWithValues: base.parameters().flattened())["language_model.model.norm.weight"]?.dtype == .float32)
    }

    @Test("Manifests this loader cannot honor are refused")
    func refusedManifests() {
        #expect(throws: PrismHadamardError.unsupported("schema_version 1, expected 2")) {
            try PrismHadamard.model(configuration: Data(TinyPrism.config(schema: 1).utf8))
        }
        #expect(throws: PrismHadamardError.module("language_model.model.layers.9.mlp.up_proj", "not in the model")) {
            try PrismHadamard.model(configuration: Data(TinyPrism.config(modules: ["model.layers.9.mlp.up_proj"]).utf8))
        }
        #expect(throws: PrismHadamardError.unsupported("a module is listed twice")) {
            try PrismHadamard.model(configuration: Data(TinyPrism.config(modules: ["lm_head", "lm_head"]).utf8))
        }
    }

    @Test("The model type is registered with the LLM factory")
    func registered() async {
        await MLXInferenceEngine.registerAliases()
        #expect(await LLMTypeRegistry.shared.contains(PrismHadamard.modelType))
    }
}

@Suite("MTP head sidecar (mtp.safetensors)")
struct MTPSidecarTests {
    @Test("Bits and group size come from the packed fc shapes")
    func inferredQuantization() throws {
        let weights = ["mtp.fc.weight": MLXArray.zeros([64, 16], dtype: .uint32), "mtp.fc.scales": MLXArray.zeros([64, 2], dtype: .float16)]
        let quantization = try #require(try MTPSidecar.quantization(of: weights, hiddenSize: 64))
        #expect(quantization.groupSize == 64 && quantization.bits == 4)
        #expect(try MTPSidecar.quantization(of: ["mtp.fc.weight": MLXArray.zeros([64, 128])], hiddenSize: 64) == nil)
        #expect(throws: MTPSidecarError.self) {
            try MTPSidecar.quantization(of: ["mtp.fc.weight": MLXArray.zeros([64, 7], dtype: .uint32), "mtp.fc.scales": MLXArray.zeros([64, 2])], hiddenSize: 64)
        }
    }

    @Test("The sidecar named by mlx_lm_extra_tensors wins, then mtp.safetensors, then mtp/weights.safetensors")
    func sidecarLookup() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "mtp-sidecar-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory.appending(path: "mtp"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{}".utf8).write(to: directory.appending(path: "config.json"))
        #expect(MTPSidecar.file(in: directory) == nil)
        try Data().write(to: directory.appending(path: "mtp/weights.safetensors"))
        #expect(MTPSidecar.file(in: directory)?.lastPathComponent == "weights.safetensors")
        try Data().write(to: directory.appending(path: "mtp.safetensors"))
        #expect(MTPSidecar.file(in: directory)?.lastPathComponent == "mtp.safetensors")
        try Data(#"{"mlx_lm_extra_tensors": {"mtp_file": "head.safetensors"}}"#.utf8).write(to: directory.appending(path: "config.json"))
        try Data().write(to: directory.appending(path: "head.safetensors"))
        #expect(MTPSidecar.file(in: directory)?.lastPathComponent == "head.safetensors")
    }

    @Test("At T > 0 the MTP iterator keeps speculating and samples the same distribution as plain decoding")
    func exactAtTemperature() throws {
        let target = try TinyPrism.base()
        let configuration = try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: Data(TinyPrism.config().utf8))
        let drafter = withRandomState(MLXRandom.RandomState(seed: 11)) { GreedyDraftingMTPDrafter(Qwen35MTPDraftModel(configuration)) }
        eval(drafter)
        let prompt = LMInput(tokens: MLXArray([5, 9, 5, 9, 5, 9, 5].map(Int32.init)))
        let parameters = GenerateParameters(maxTokens: 3, temperature: 0.03)
        let runs = 300
        var speculative: [[Int]: Int] = [:]
        var plain: [[Int]: Int] = [:]
        var control: [[Int]: Int] = [:]
        var proposed = 0
        for run in 0 ..< runs {
            MLXRandom.seed(UInt64(run))
            var mtp = try MTPSpeculativeTokenIterator(input: prompt, mainModel: target, drafter: drafter, parameters: parameters, blockSize: 2)
            var tokens: [Int] = []
            while let token = mtp.next() { tokens.append(token) }
            proposed += mtp.proposedCount
            #expect(mtp.passthroughReason == nil)
            speculative[tokens, default: 0] += 1
            MLXRandom.seed(UInt64(50_000 + run))
            plain[Array(IteratorSequence(try TokenIterator(input: prompt, model: target, parameters: parameters))), default: 0] += 1
            MLXRandom.seed(UInt64(90_000 + run))
            control[Array(IteratorSequence(try TokenIterator(input: prompt, model: target, parameters: parameters))), default: 0] += 1
        }
        #expect(proposed >= runs)
        func distance(_ a: [[Int]: Int], _ b: [[Int]: Int]) -> Double {
            Set(a.keys).union(b.keys).map { abs(Double(a[$0, default: 0] - b[$0, default: 0])) }.reduce(0, +) / Double(2 * runs)
        }
        let noise = distance(plain, control)
        #expect(Set(plain.keys).count > 1, "the continuation must be random for the comparison to mean anything")
        #expect(distance(speculative, plain) < noise + 0.08, "total variation \(distance(speculative, plain)) against \(noise) between two plain samples")
    }

    @Test("Only Qwen3.5-family targets get a drafter")
    func nonQwenTarget() throws {
        let model = try TinyHybrid.make()
        #expect(try MTPSidecar.load(directory: FileManager.default.temporaryDirectory, target: model) == nil)
    }
}
