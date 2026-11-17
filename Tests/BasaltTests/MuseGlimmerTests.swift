import Foundation
import Testing
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import MLXVLM
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

private struct OnyxStubTokenizer: MLXLMCommon.Tokenizer, Sendable {
    let start = 250
    let message = 251
    let eom = 252
    let eot = 253
    let endOfText = 254

    private let vocab: [Int: String]
    private let reverse: [String: Int]

    init(extra: [Int: String] = [:]) {
        var map: [Int: String] = [
            250: "<|start|>",
            251: "<|message|>",
            252: "<|eom|>",
            253: "<|eot|>",
            254: "<|end_of_text|>",
            10: " to=self",
            11: "thinking step 1. ",
            12: "thinking step 2.",
            13: "assistant to=user",
            14: "Final answer.",
            15: "assistant to=weather.get",
            16: "<atem:function_calls><atem:invoke name=\"weather.get\"><atem:parameter name=\"days\">3</atem:parameter></atem:invoke></atem:function_calls>",
        ]
        for (k, v) in extra { map[k] = v }
        self.vocab = map
        self.reverse = Dictionary(uniqueKeysWithValues: map.map { ($0.value, $0.key) })
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let exact = reverse[text] { return [exact] }
        return text.utf8.map { Int($0 % 200) + 1 }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { id in
            if skipSpecialTokens && id >= 250 { return nil }
            return vocab[id] ?? UnicodeScalar(max(32, min(126, id))).map(String.init) ?? " "
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { reverse[token] }
    func convertIdToToken(_ id: Int) -> String? { vocab[id] }
    var bosToken: String? { nil }
    var eosToken: String? { "<|eot|>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [1, 2, 3]
    }
}

private struct FixedTokenIterator: TokenIteratorProtocol {
    private var tokens: IndexingIterator<[Int]>
    let maxTokens: Int? = nil
    private(set) var tokenCount = 0
    let promptPrefillTime: TimeInterval = 0.001

    init(_ tokens: [Int]) {
        self.tokens = tokens.makeIterator()
    }

    mutating func next() -> Int? {
        guard let token = tokens.next() else { return nil }
        tokenCount += 1
        return token
    }
}

@Suite("Meta Muse Glimmer 30B: text/VLM parity, config, profile and capabilities (#533)", .serialized)
struct MuseGlimmerModelTests {
    @Test("MuseGlimmerVLM and MuseGlimmerTextModel match MLXVLM.MuseGlimmer on multimodal image+text and text-only KV-cached decode (#228)")
    func textAndVLMModelsMatchStockVLMLogitsAndGreedyDecode() throws {
        let vlmData = Data(TinyModel.Family.museGlimmer.config.utf8)
        let vlmConfig = try JSONDecoder.json5().decode(MuseGlimmerConfiguration.self, from: vlmData)
        let textConfig = try MuseGlimmerTextConfiguration.decodeFlexible(from: vlmData)

        let vlm = withRandomState(MLXRandom.RandomState(seed: 533)) {
            let model = MuseGlimmer(vlmConfig)
            let centered = model.parameters().flattened().filter { $0.0.hasSuffix("_layernorm.weight") }
            model.update(parameters: ModuleParameters.unflattened(centered.map { ($0.0, MLXRandom.normal($0.1.shape) * 0.25) }))
            return model
        }
        try vlm.prepare()
        eval(vlm)

        let rawWeights = Dictionary(uniqueKeysWithValues: vlm.parameters().flattened())
        let basaltVLM = MuseGlimmerVLM(vlmConfig)
        try basaltVLM.update(parameters: ModuleParameters.unflattened(basaltVLM.sanitize(weights: rawWeights)), verify: [.all])
        try basaltVLM.prepare()
        eval(basaltVLM)

        let textModel = MuseGlimmerTextModel(textConfig)
        let sanitized = textModel.sanitize(weights: rawWeights)
        try textModel.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        try textModel.prepare()
        eval(textModel)

        let patchDim = vlmConfig.visionConfiguration.patchTemporal * 3 * vlmConfig.visionConfiguration.patchSize * vlmConfig.visionConfiguration.patchSize
        let pixels = withRandomState(MLXRandom.RandomState(seed: 77)) { MLXRandom.normal([16, patchDim]) }
        let mmTokens = MLXArray([Int32(3), 254, 254, 254, 254, 17, 42]).reshaped(1, 7)
        let mmInput = LMInput(text: .init(tokens: mmTokens), image: .init(pixels: pixels, frames: [THW(1, 4, 4)]))
        let stockMM = try vlm.prepare(mmInput, cache: vlm.newCache(parameters: nil), state: nil, prefill: .init())
        let basaltMM = try basaltVLM.prepare(mmInput, cache: basaltVLM.newCache(parameters: nil), state: nil, prefill: .init())
        if case .logits(let sOut) = stockMM, case .logits(let bOut) = basaltMM {
            let mmDiff = abs(sOut.logits.asType(.float32) - bOut.logits.asType(.float32)).max().item(Float.self)
            #expect(mmDiff < 1e-5, "multimodal logit drift \(mmDiff) exceeded 1e-5")
        } else {
            Issue.record("expected .logits PrepareResult from both VLMs")
        }

        let promptIds: [Int32] = [3, 17, 42, 88, 19, 7, 105, 64, 12, 91]
        let input = LMInput.Text(tokens: MLXArray(promptIds).reshaped(1, promptIds.count))
        let vlmLogits = vlm(input, cache: nil, state: nil).logits.asType(.float32)
        let textLogits = textModel(input, cache: nil, state: nil).logits.asType(.float32)
        let basaltVLMLogits = basaltVLM(input, cache: nil, state: nil).logits.asType(.float32)
        #expect(abs(vlmLogits - textLogits).max().item(Float.self) < 1e-5)
        #expect(abs(vlmLogits - basaltVLMLogits).max().item(Float.self) < 1e-5)

        let vlmIter = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptIds)),
            model: vlm,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0)
        )
        let textIter = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptIds)),
            model: textModel,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0)
        )
        let basaltVLMIter = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptIds)),
            model: basaltVLM,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0)
        )
        let expectedTokens = Array(vlmIter)
        #expect(Array(textIter) == expectedTokens)
        #expect(Array(basaltVLMIter) == expectedTokens)
    }

    @Test("MuseGlimmerTextConfiguration validates invariants, supports tied embeddings, and sanitizes HF/MLX prefixes")
    func configurationValidationAndSanitizer() throws {
        let tiedJSON = TinyModel.Family.museGlimmerText.config.replacingOccurrences(
            of: "\"tie_word_embeddings\": false",
            with: "\"tie_word_embeddings\": true"
        )
        let tiedConfig = try MuseGlimmerTextConfiguration.decodeFlexible(from: Data(tiedJSON.utf8))
        let tiedModel = MuseGlimmerTextModel(tiedConfig)
        #expect(tiedModel.toolCallFormat == .atem)
        #expect(tiedModel.reasoningConfig?.startDelimiter == "to=self<|message|>")
        #expect(tiedModel.reasoningConfig?.endDelimiter == "<|eom|>")

        let dirty: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([256, 64]),
            "language_model.lm_head.weight": MLXArray.zeros([256, 64]),
            "vision_tower.patch_embedder.patch_embedding.weight": MLXArray.zeros([16, 16]),
            "vision_adapter.fc1.weight": MLXArray.zeros([16, 16]),
            "vision_projection.weight": MLXArray.zeros([64, 16]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([8]),
        ]
        let cleaned = tiedModel.sanitize(weights: dirty)
        #expect(cleaned["model.embed_tokens.weight"] != nil)
        #expect(cleaned["lm_head.weight"] == nil)
        #expect(cleaned.keys.allSatisfy { !$0.contains("vision") && !$0.contains("inv_freq") })

        let badHeads = TinyModel.Family.museGlimmerText.config.replacingOccurrences(
            of: "\"num_key_value_heads\": 2",
            with: "\"num_key_value_heads\": 3"
        )
        #expect(throws: BasaltError.self) {
            _ = try MuseGlimmerTextConfiguration.decodeFlexible(from: Data(badHeads.utf8))
        }

        let badLayerType = TinyModel.Family.museGlimmerText.config.replacingOccurrences(
            of: "\"full_attention\"",
            with: "\"linear_attention\""
        )
        #expect(throws: BasaltError.self) {
            _ = try MuseGlimmerTextConfiguration.decodeFlexible(from: Data(badLayerType.utf8))
        }
    }

    @Test("MuseGlimmerProfile, ModelCapabilities, KVCacheGeometry, PolarGeometry, ModelDatabase and ComposeProfileCatalog")
    func profileCapabilitiesGeometryAndCatalog() throws {
        let profile = ModelProfileRegistry.profile(forModelType: "muse_glimmer")
        #expect(profile is MuseGlimmerProfile)
        #expect(ModelProfileRegistry.profile(forModelType: "muse_glimmer_text") is MuseGlimmerProfile)
        #expect(profile.templateContext(for: .modelDefault).isEmpty)
        #expect(profile.templateContext(for: .disabled)["reasoning_strength"] as? String == "low")
        #expect(profile.templateContext(for: .effort("minimal"))["reasoning_strength"] as? String == "low")
        #expect(profile.templateContext(for: .effort("xhigh"))["reasoning_strength"] as? String == "xhigh")
        #expect(profile.templateContext(for: .effort("bogus")).isEmpty)
        #expect(profile.defaultMaxTokens == 16_384)

        let caps = try #require(ModelCapabilities.of(modelType: "muse_glimmer"))
        #expect(caps.allows(.kvQuantization))
        #expect(caps.allows(.kvZones))
        #expect(caps.allows(.batchedDecode))
        #expect(caps.allows(.speculativeLookup))
        #expect(caps.allows(.quest))
        #expect(!caps.allows(.mtp))
        #expect(!caps.allows(.cacheBlend))

        let prodJSON: [String: Any] = [
            "model_type": "muse_glimmer",
            "text_config": [
                "model_type": "muse_glimmer_text",
                "hidden_size": 6656,
                "num_hidden_layers": 52,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "sliding_window": 2048,
                "layer_types": (0 ..< 52).map { ($0 + 1) % 4 == 0 ? "full_attention" : "sliding_attention" },
            ] as [String: Any],
        ]
        #expect(PolarGeometry.rotatedDims(config: prodJSON) == nil)
        let geom = try #require(KVCacheGeometry(config: prodJSON))
        #expect(geom.window == 2048)
        #expect(geom.quantizedGroupSize == 64)
        #expect(geom.elementsPerToken == Double(13 * 2 * 256))

        #expect(ModelDatabase.find(id: "mlx-community/Muse-Glimmer-30B-4bit")?.family == .muse)
        #expect(ModelDatabase.models(for: .muse).count == 3)
        #expect(ComposeProfileCatalog.profiles.contains { $0.primaryModel.contains("Muse-Glimmer-30B") })
    }
}

@Suite("Meta Muse Glimmer 30B: Onyx reasoning tap and GGUF import (#533)", .serialized)
struct MuseGlimmerOnyxAndGGUFTests {
    @Test("OnyxReasoningTap streams to=self private reasoning before to=user response and ATEM tool calls")
    func onyxReasoningTapAndRelayEndToEnd() async throws {
        let tokenizer = OnyxStubTokenizer()
        let model = try TinyModel.make(.museGlimmerText)
        let context = ModelContext(
            configuration: ModelConfiguration(id: "tiny-muse-glimmer"),
            model: model,
            processor: ByteInputProcessor(),
            tokenizer: tokenizer
        )
        let stopIds = MLXInferenceEngine.stopTokenIds(context)
        #expect(stopIds.contains(tokenizer.eot))
        #expect(stopIds.contains(tokenizer.endOfText))

        let tools: [ToolSpec] = [[
            "type": "function",
            "function": [
                "name": "weather.get",
                "description": "Weather",
                "parameters": [
                    "type": "object",
                    "properties": ["days": ["type": "integer"] as [String: any Sendable]] as [String: any Sendable],
                    "required": ["days"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]

        let streamTokens = [
            10, tokenizer.message, 11, 12, tokenizer.eom,
            tokenizer.start, 15, tokenizer.message, 16, tokenizer.eom,
            tokenizer.start, 13, tokenizer.message, 14, tokenizer.eot,
        ]
        let (genStream, task, prefill, onyxBox) = MLXInferenceEngine.handOffWithOnyx(
            promptTokenCount: 3,
            context: context,
            iterator: FixedTokenIterator(streamTokens),
            tools: tools,
            toolCallPolicy: .init()
        )
        let box = try #require(onyxBox)
        let container = ModelContainer(context: context)
        let engine = MLXInferenceEngine()
        await engine.attach(container, profile: MuseGlimmerProfile())

        let request = GenerationRequest(
            prompt: "Weather?",
            maxTokens: 32,
            tools: [ToolDefinition(name: "weather.get", description: "Weather", parametersJSON: #"{"type":"object","properties":{"days":{"type":"integer"}},"required":["days"]}"#)]
        )
        let relay = GenerationRelay(GenerationRelay.Setup(
            request: request,
            promptTokens: 3,
            onyxReasoning: box
        ))
        let inputs = MLXInferenceEngine.RelayInputs(
            stream: genStream,
            task: task,
            prefillSeconds: prefill,
            promptTokens: [1, 2, 3],
            namespace: nil,
            leaseID: 0,
            epoch: ModelEpoch(rawValue: 0),
            constraint: nil,
            container: container,
            faults: FaultPlan()
        )
        let outStream = AsyncStream<StreamChunk> { continuation in
            Task {
                await engine.relay(relay, inputs, into: continuation)
            }
        }

        var reasoning = ""
        var answer = ""
        var calls: [ToolCallOutput] = []
        var meta: GenerationResponse?
        for await chunk in outStream {
            switch chunk {
            case .reasoning(let r): reasoning += r
            case .text(let t): answer += t
            case .toolCall(let c): calls.append(c)
            case .metadata(let m): meta = m
            }
        }
        #expect(reasoning == "thinking step 1. thinking step 2.")
        #expect(answer == "Final answer.")
        #expect(calls.count == 1)
        #expect(calls.first?.name == "weather.get")
        #expect((try #require(meta).reasoningTokens ?? 0) > 0)
    }

    @Test("GGUFMuseGlimmerLayout de-centers 1+w block norms, skips synthetic qk_norm, strips vision_config and matches dense greedy decode")
    func ggufImportMatchesDenseReference() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "gguf-muse-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let template = folder.appending(path: "template")
        try FileManager.default.createDirectory(at: template, withIntermediateDirectories: true)
        try Data(TinyModel.Family.museGlimmer.config.utf8).write(to: template.appending(path: "config.json"))
        try Data("{}".utf8).write(to: template.appending(path: "tokenizer_config.json"))

        let layout = try #require(try GGUFImporter.layout(template: template) as? GGUFMuseGlimmerLayout)
        let reference = try #require(try TinyModel.make(.museGlimmerText) as? MuseGlimmerTextModel)
        reference.update(parameters: reference.parameters().mapValues { $0.asType(.bfloat16) })
        eval(reference)
        let refMap = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())

        let emb = try #require(refMap["model.embed_tokens.weight"])
        let outNorm = try #require(refMap["model.norm.weight"])
        let head = try #require(refMap["lm_head.weight"])
        var tensors: [GGUFImportTests.Tensor] = [
            .init(
                name: "token_embd.weight",
                type: .f32,
                dims: [64, 256],
                bytes: GGUFImportTests.floats(emb.asType(.float32).asArray(Float.self))
            ),
            .init(
                name: "output_norm.weight",
                type: .f32,
                dims: [64],
                bytes: GGUFImportTests.floats(outNorm.asType(.float32).asArray(Float.self))
            ),
            .init(
                name: "output.weight",
                type: .f32,
                dims: [64, 256],
                bytes: GGUFImportTests.floats(head.asType(.float32).asArray(Float.self))
            ),
        ]

        let blockPairs: [(String, String, Bool)] = [
            ("attn_norm.weight", "input_layernorm.weight", true),
            ("post_attention_norm.weight", "post_attention_layernorm.weight", true),
            ("ffn_norm.weight", "pre_feedforward_layernorm.weight", true),
            ("post_ffw_norm.weight", "post_feedforward_layernorm.weight", true),
            ("attn_q.weight", "self_attn.q_proj.weight", false),
            ("attn_k.weight", "self_attn.k_proj.weight", false),
            ("attn_v.weight", "self_attn.v_proj.weight", false),
            ("attn_output.weight", "self_attn.o_proj.weight", false),
            ("attn_gate.weight", "self_attn.gate_proj.weight", false),
            ("ffn_gate.weight", "mlp.gate_proj.weight", false),
            ("ffn_up.weight", "mlp.up_proj.weight", false),
            ("ffn_down.weight", "mlp.down_proj.weight", false),
        ]

        for layer in 0 ..< 4 {
            for (ggufName, targetSuffix, centered) in blockPairs {
                let arr = try #require(refMap["model.layers.\(layer).\(targetSuffix)"])
                let shape = arr.shape
                let dims = shape.count == 1 ? shape : [shape[1], shape[0]]
                var vals = arr.asType(.float32).asArray(Float.self)
                if centered { vals = vals.map { $0 + 1.0 } }
                tensors.append(.init(name: "blk.\(layer).\(ggufName)", type: .f32, dims: dims, bytes: GGUFImportTests.floats(vals)))
            }
            tensors.append(.init(name: "blk.\(layer).attn_q_norm.weight", type: .f32, dims: [16], bytes: GGUFImportTests.floats(Array(repeating: 3.87, count: 16))))
            tensors.append(.init(name: "blk.\(layer).attn_k_norm.weight", type: .f32, dims: [16], bytes: GGUFImportTests.floats(Array(repeating: 3.87, count: 16))))
        }

        let ggufURL = folder.appending(path: "muse.gguf")
        try GGUFImportTests.gguf(tensors).write(to: ggufURL)
        let ggufFile = try GGUFFile(url: ggufURL)
        let outDir = folder.appending(path: "imported")
        let summary = try GGUFImporter(file: ggufFile, layout: layout, grid: .native).run(output: outDir, template: template)
        #expect(summary.skipped.count == 8)

        let importedConfigData = try Data(contentsOf: outDir.appending(path: "config.json"))
        let importedJSON = try #require(JSONSerialization.jsonObject(with: importedConfigData) as? [String: Any])
        #expect(importedJSON["model_type"] as? String == MuseGlimmerTextModel.modelType)
        #expect(importedJSON["vision_config"] == nil)

        await MLXInferenceEngine.registerAliases()
        let rawLoaded = try await LLMTypeRegistry.shared.createModel(
            configuration: importedConfigData,
            modelType: MuseGlimmerTextModel.modelType
        )
        let loaded = try #require(rawLoaded as? MuseGlimmerTextModel)
        var importedWeights: [String: MLXArray] = [:]
        for url in try FileManager.default.contentsOfDirectory(at: outDir, includingPropertiesForKeys: nil) where url.pathExtension == "safetensors" {
            importedWeights.merge(try loadArrays(url: url)) { $1 }
        }
        try loaded.update(parameters: ModuleParameters.unflattened(loaded.sanitize(weights: importedWeights)), verify: [.all])
        try loaded.prepare()
        eval(loaded)

        let promptIds: [Int32] = [5, 19, 33, 71, 14]
        let refIter = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptIds)),
            model: reference,
            parameters: GenerateParameters(maxTokens: 12, temperature: 0)
        )
        let impIter = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptIds)),
            model: loaded,
            parameters: GenerateParameters(maxTokens: 12, temperature: 0)
        )
        #expect(Array(impIter) == Array(refIter))
    }
}
