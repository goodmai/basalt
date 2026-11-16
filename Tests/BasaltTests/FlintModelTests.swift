import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Recursive Flint architecture: configuration invariants, learned attention sinks, biased sigmoid MoE kernel, checkpoint sanitization, SSD expert streaming, GGUF layout and agentic tool repair (#518)", .serialized)
struct FlintModelTests {
    static let officialConfigJSON = """
        {
          "architectures": ["FlintForCausalLM"],
          "model_type": "flint",
          "vocab_size": 150144,
          "hidden_size": 2048,
          "intermediate_size": 6144,
          "num_hidden_layers": 49,
          "num_attention_heads": 32,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "attention_bias": false,
          "use_sliding_window": true,
          "sliding_window": 513,
          "layer_types": [
            "full_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention", "sliding_attention", "sliding_attention", "full_attention"
          ],
          "no_rope_layer_indices": [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48],
          "mlp_only_layers": [0, 1],
          "decoder_sparse_step": 1,
          "num_experts": 256,
          "num_experts_per_tok": 6,
          "moe_intermediate_size": 512,
          "shared_expert_intermediate_size": 1024,
          "router_score_function": "sigmoid",
          "norm_topk_prob": true,
          "softmax_type": "learnable",
          "tie_word_embeddings": false,
          "max_position_embeddings": 262144
        }
        """

    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let delta = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(delta)
        return delta.item(Float.self)
    }

    @Test("Official 49-layer RecursiveSI/flint configuration decodes exact hybrid layer schedule, NoPE indices, dense prefix layers and rejects malformed invariants")
    func configurationAndGeometry() throws {
        let data = Data(Self.officialConfigJSON.utf8)
        let config = try JSONDecoder.json5().decode(FlintConfiguration.self, from: data).validated()
        #expect(config.modelType == "flint" && config.hiddenLayers == 49 && config.vocabularySize == 150_144)
        #expect(config.effectiveSlidingWindow == 513 && config.usesLearnedAttentionSinks)
        let slidingLayers = (0 ..< 49).filter { config.usesSlidingWindow(layer: $0) }
        let nopeLayers = (0 ..< 49).filter { config.skipsRoPE(layer: $0) }
        let moeLayers = (0 ..< 49).filter { config.isSparseMoE(layer: $0) }
        #expect(slidingLayers.count == 35 && nopeLayers.count == 14 && moeLayers.count == 47)
        #expect(!config.isSparseMoE(layer: 0) && !config.isSparseMoE(layer: 1) && config.isSparseMoE(layer: 2))
        let roundTrip = try JSONDecoder().decode(FlintConfiguration.self, from: JSONEncoder().encode(config))
        #expect(roundTrip == config)

        let raw = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let geometry = try #require(KVCacheGeometry(config: raw))
        let hybridBytes = geometry.bytes(tokens: TokenCount(262_144)).rawValue
        let fullBytes = Int64(KVCacheGeometry.kvBytesPerToken(layers: 49, kvHeads: 4, headDim: 128, bytesPerElement: 2) * 262_144)
        #expect(Double(hybridBytes) < Double(fullBytes) * 0.30)
        let lines = architectureLines(config: raw).joined(separator: " | ")
        #expect(lines.contains("Experts: 256, top-6, 35/49 sliding"))

        var brokenLayers = config
        brokenLayers.layerTypes.removeLast()
        #expect(throws: BasaltError.self) { _ = try brokenLayers.validated() }
        var brokenTopK = config
        brokenTopK.numExpertsPerToken = 300
        #expect(throws: BasaltError.self) { _ = try brokenTopK.validated() }
        var brokenHeads = config
        brokenHeads.kvHeads = 5
        #expect(throws: BasaltError.self) { _ = try brokenHeads.validated() }
    }

    @Test("FlintAttention matches the explicit FP32 attention-sink equation on both NoPE full-attention and RoPE sliding-window layers (#228)")
    func attentionSinkAndNoPEParity() throws {
        let config = try JSONDecoder.json5().decode(FlintConfiguration.self, from: Data(TinyModel.Family.flint.config.utf8)).validated()
        withRandomState(MLXRandom.RandomState(seed: 5180)) {
            let fullAttn = FlintAttention(config, layerIdx: 0)
            let slidingAttn = FlintAttention(config, layerIdx: 1)
            fullAttn.update(parameters: ModuleParameters.unflattened([("softmax_offset", MLXArray([Float(0.8), -0.4, 1.2, 0.1]))]))
            slidingAttn.update(parameters: ModuleParameters.unflattened([("softmax_offset", MLXArray([Float(-0.3), 0.9, 0.4, -0.7]))]))
            let x = MLXRandom.normal([2, 10, 64])
            eval(fullAttn, slidingAttn, x)

            let fullMask = createCausalMask(n: 10, offset: 0, windowSize: nil)
            let fullFast = fullAttn(x, mask: .array(fullMask), cache: nil)
            let q0 = fullAttn.qNorm(fullAttn.wq(x).reshaped(2, 10, 4, 16)).transposed(0, 2, 1, 3)
            let k0 = fullAttn.kNorm(fullAttn.wk(x).reshaped(2, 10, 2, 16)).transposed(0, 2, 1, 3)
            let v0 = fullAttn.wv(x).reshaped(2, 10, 2, 16).transposed(0, 2, 1, 3)
            let fullRef = fullAttn.wo(
                FlintAttention.explicitAttentionWithSinks(
                    queries: q0, keys: k0, values: v0, scale: fullAttn.scale,
                    booleanMask: fullMask, sinks: fullAttn.softmaxOffset
                ).transposed(0, 2, 1, 3).reshaped(2, 10, -1)
            )
            let withoutSinks = fullAttn.wo(
                FlintAttention.explicitAttentionWithSinks(
                    queries: q0, keys: k0, values: v0, scale: fullAttn.scale,
                    booleanMask: fullMask, sinks: nil
                ).transposed(0, 2, 1, 3).reshaped(2, 10, -1)
            )
            #expect(Self.maxAbsDiff(fullFast, fullRef) < 1e-4)
            #expect(Self.maxAbsDiff(fullFast, withoutSinks) > 1e-3)

            let winMask = createCausalMask(n: 10, offset: 0, windowSize: 5)
            let slidingFast = slidingAttn(x, mask: .array(winMask), cache: nil)
            let q1 = applyRotaryPosition(slidingAttn.rope, to: slidingAttn.qNorm(slidingAttn.wq(x).reshaped(2, 10, 4, 16)).transposed(0, 2, 1, 3), offset: nil)
            let k1 = applyRotaryPosition(slidingAttn.rope, to: slidingAttn.kNorm(slidingAttn.wk(x).reshaped(2, 10, 2, 16)).transposed(0, 2, 1, 3), offset: nil)
            let v1 = slidingAttn.wv(x).reshaped(2, 10, 2, 16).transposed(0, 2, 1, 3)
            let slidingRef = slidingAttn.wo(
                FlintAttention.explicitAttentionWithSinks(
                    queries: q1, keys: k1, values: v1, scale: slidingAttn.scale,
                    booleanMask: winMask, sinks: slidingAttn.softmaxOffset
                ).transposed(0, 2, 1, 3).reshaped(2, 10, -1)
            )
            #expect(Self.maxAbsDiff(slidingFast, slidingRef) < 1e-4)
        }
    }

    @Test("FlintRouterKernel Metal SIMD kernel matches FP32 reference sigmoid routing across 8 and 256 experts while applying expert_bias to selection only (#228)",
          arguments: [DType.float32, .float16, .bfloat16])
    func routerKernelParity(dtype: DType) {
        withRandomState(MLXRandom.RandomState(seed: 5181)) {
            for (rows, experts, topK) in [(7, 8, 2), (13, 256, 6)] {
                let logits = (MLXRandom.normal([rows, experts]) * 1.5).asType(dtype)
                let bias = (MLXRandom.normal([experts]) * 0.4).asType(.float32)
                let fast = FlintRouterKernel.route(logits, expertBias: bias, k: topK, normTopkProb: true, weightDType: .float32)
                let ref = FlintRouterKernel.referenceSigmoid(logits, expertBias: bias, k: topK, normTopkProb: true, weightDType: .float32)
                let unbiased = FlintRouterKernel.referenceSigmoid(logits, expertBias: nil, k: topK, normTopkProb: true, weightDType: .float32)
                eval(fast.indices, fast.weights, ref.indices, ref.weights, unbiased.indices)
                #expect(fast.indices.asArray(Int32.self) == ref.indices.asArray(Int32.self))
                #expect(fast.indices.asArray(Int32.self) != unbiased.indices.asArray(Int32.self))
                #expect(Self.maxAbsDiff(fast.weights, ref.weights) < 1e-5)
                let rowSums = sum(fast.weights, axis: -1)
                #expect(Self.maxAbsDiff(rowSums, MLXArray.ones([rows], dtype: .float32)) < 1e-5)
            }
        }
    }
}

extension FlintModelTests {
    @Test("Unstacked Hugging Face expert weights, gate.expert_bias and ignored MTP tensors sanitize to exact SwitchGLU parity")
    func unstackedSanitizationParity() throws {
        let rawConfig = TinyModel.Family.flint.config
        let config = try JSONDecoder.json5().decode(FlintConfiguration.self, from: Data(rawConfig.utf8)).validated()
        let stackedModel = try #require(try TinyModel.make(.flint) as? FlintModel)
        let stackedWeights = Dictionary(uniqueKeysWithValues: stackedModel.parameters().flattened())

        var unstacked: [String: MLXArray] = [:]
        for (key, value) in stackedWeights {
            if key.contains(".mlp.switch_mlp."), key.hasSuffix(".weight") {
                let parts = key.components(separatedBy: ".mlp.switch_mlp.")
                let proj = String(parts[1].dropLast(".weight".count))
                for idx in 0 ..< config.numExperts {
                    unstacked["\(parts[0]).mlp.experts.\(idx).\(proj).weight"] = value[idx]
                }
            } else if key.hasSuffix(".mlp.expert_bias") {
                let prefix = String(key.dropLast(".mlp.expert_bias".count))
                unstacked["\(prefix).mlp.gate.expert_bias"] = value
            } else {
                unstacked[key] = value
            }
        }
        unstacked["mtp.head.weight"] = MLXArray.ones([4, 4])
        unstacked["model.layers.1.self_attn.rotary_emb.inv_freq"] = MLXArray.ones([8])

        let candidate = FlintModel(config)
        let sanitized = candidate.sanitize(weights: unstacked)
        try candidate.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.noUnusedKeys, .shapeMismatch])
        eval(candidate)

        let prompt = MLXArray([Int32(4), 9, 15, 23, 31, 42, 7, 19, 3, 11]).reshaped(1, 10)
        #expect(Self.maxAbsDiff(stackedModel(prompt, cache: nil), candidate(prompt, cache: nil)) < 1e-5)
    }

    @Test("Checkpoints with language_model.-prefixed mixed 3-bit and 8-bit weights load identically in resident RAM and under SSD ExpertStreamingLoader")
    func languageModelPrefixedMixedBitAndStreamingParity() async throws {
        await MLXInferenceEngine.registerAliases()
        let dir = FileManager.default.temporaryDirectory.appending(path: "flint-stream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let base = try #require(try TinyModel.make(.flint) as? FlintModel)
        quantize(model: base) { path, _ in
            if path.contains(".switch_mlp.") { return (32, 3, .affine) }
            if path.contains(".shared_expert.") || path.contains(".self_attn.") { return (32, 8, .affine) }
            return nil
        }
        eval(base)
        let prefixed = Dictionary(uniqueKeysWithValues: base.parameters().flattened().map { ("language_model.\($0.0)", $0.1) })
        try save(arrays: prefixed, metadata: ["format": "mlx"], url: dir.appending(component: "model.safetensors"))

        var configDict = try #require(JSONSerialization.jsonObject(with: Data(TinyModel.Family.flint.config.utf8)) as? [String: Any])
        configDict["quantization"] = [
            "group_size": 32,
            "bits": 3,
            "mode": "affine",
            "language_model.model.layers.2.mlp.shared_expert.down_proj": ["group_size": 32, "bits": 8, "mode": "affine"],
        ]
        try JSONSerialization.data(withJSONObject: configDict, options: [.sortedKeys]).write(to: dir.appending(component: "config.json"))
        #expect(ExpertStreamingLoader.canStream(dir))

        let loader = ByteTokenizerLoader()
        let resident = try await LLMModelFactory.shared.loadContainer(from: dir, using: loader)
        let streamedOpt = try await ExpertStreamingLoader.loadContainer(
            from: dir, streaming: ExpertStreaming(cacheBytes: ByteCount(512)), tokenizerLoader: loader)
        let streamed = try #require(streamedOpt)

        let tokens = [5, 12, 19, 27, 33, 41, 8, 14, 22, 30]
        let residentOut = try await resident.perform { try CacheBlendQualityTests.greedy(tokens, count: 8, model: $0.model) }
        let streamedOut = try await streamed.perform { try CacheBlendQualityTests.greedy(tokens, count: 8, model: $0.model) }
        let missesOpt = await streamed.perform { ExpertStreamingLoader.store(in: $0.model)?.stats.misses }
        let misses = try #require(missesOpt)
        #expect(residentOut == streamedOut)
        #expect(misses > 0)
    }

    @Test("Quantized 8-bit and 4-bit KV caches preserve finite Flint attention-sink logits across prefill and decode",
          arguments: [8, 4])
    func quantizedKVCacheWithAttentionSinks(bits: Int) throws {
        let flintConfig = TinyModel.Family.flint.config.replacingOccurrences(of: "\"head_dim\": 16", with: "\"head_dim\": 64")
        let config = try JSONDecoder.json5().decode(FlintConfiguration.self, from: Data(flintConfig.utf8)).validated()
        let model = withRandomState(MLXRandom.RandomState(seed: 5182)) {
            let m = FlintModel(config)
            for layer in m.model.layers {
                layer.selfAttn.update(parameters: ModuleParameters.unflattened([("softmax_offset", MLXRandom.normal([config.attentionHeads]) * 0.5)]))
            }
            return m
        }
        eval(model)
        let prompt = MLXArray((0 ..< 16).map { Int32(($0 * 7 + 3) % 120) }).reshaped(1, 16)
        let nextTok = MLXArray([Int32(42)]).reshaped(1, 1)

        let denseCache = try model.newCache(parameters: nil)
        _ = model(prompt, cache: denseCache)
        let denseNext = model(nextTok, cache: denseCache)

        let qParams = GenerateParameters(kvBits: bits, kvGroupSize: 64, quantizedKVStart: 0)
        var qCache = try model.newCache(parameters: qParams)
        _ = model(prompt, cache: qCache)
        maybeQuantizeKVCache(cache: &qCache, kvBits: bits, kvGroupSize: 64, quantizedKVStart: 0)
        #expect(qCache.contains { $0 is QuantizedKVCacheProtocol })
        let quantNext = model(nextTok, cache: qCache)
        eval(denseNext, quantNext)
        #expect(!any(isNaN(quantNext)).item(Bool.self))
        #expect(Self.maxAbsDiff(denseNext, quantNext) < (bits == 8 ? 0.25 : 1.0))
    }

    @Test("GGUFFlintLayout maps attention sinks, stacked 3D expert tensors and shared experts, and agentic XML tool repair extracts structured calls")
    func ggufLayoutProfileAndAgenticToolRepair() throws {
        let layout = GGUFFlintLayout()
        let sinkPlan = try #require(layout.plan(GGUFTensor(name: "blk.2.attn_sinks.weight", type: .f32, dims: [4], offset: 0)))
        let expsPlan = try #require(layout.plan(GGUFTensor(name: "blk.2.ffn_gate_exps.weight", type: .q4K, dims: [64, 32, 8], offset: 0)))
        let shexpPlan = try #require(layout.plan(GGUFTensor(name: "blk.2.ffn_down_shexp.weight", type: .q6K, dims: [32, 64], offset: 0)))
        let biasPlan = try #require(layout.plan(GGUFTensor(name: "blk.2.exp_probs_b.bias", type: .f32, dims: [8], offset: 0)))
        #expect(sinkPlan.target == "model.layers.2.self_attn.softmax_offset" && sinkPlan.bits == nil)
        #expect(expsPlan.target == "model.layers.2.mlp.switch_mlp.gate_proj.weight" && expsPlan.bits == 4)
        #expect(shexpPlan.target == "model.layers.2.mlp.shared_expert.down_proj.weight" && shexpPlan.bits == 6)
        #expect(biasPlan.target == "model.layers.2.mlp.expert_bias" && biasPlan.bits == nil)

        let profile = ModelProfileRegistry.profile(forModelType: "flint")
        #expect(profile is FlintProfile && profile.defaultMaxTokens == 32_768)
        #expect(profile.thinkingMarkers.open == ["<think>"] && profile.thinkingMarkers.close == ["</think>"])
        #expect(profile.templateContext(for: .modelDefault).isEmpty)
        let flintModel = try #require(try TinyModel.make(.flint) as? FlintModel)
        #expect(flintModel.toolCallFormat == .glm4)

        let rawXML = "<tool_call>run_suite\n<arg_key>filter</arg_key><arg_value>FlintModelTests</arg_value><arg_key>count</arg_key><arg_value>4</arg_value></tool_call>"
        let repaired = try #require(ToolCallRepairer.repairToolCall(
            name: nil, callID: "call_flint", rawText: rawXML, truncated: false, declaredTools: ["run_suite"]))
        #expect(repaired.name == "run_suite")
        let args = try #require(JSONSerialization.jsonObject(with: Data(repaired.arguments.utf8)) as? [String: Any])
        #expect(args["filter"] as? String == "FlintModelTests" && args["count"] as? Int == 4)
    }
}
