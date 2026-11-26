import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct RWKV7Reference {
    private struct RecurrenceArgs {
        let r, v, w, a, kk, kEff: MLXArray
    }

    let weights: [String: MLXArray]
    let config: RWKV7Configuration

    init(_ model: RWKV7Model) {
        self.weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) })
        self.config = model.configuration
    }

    func forward(_ tokens: [Int32]) throws -> MLXArray {
        var h = try weight("model.embeddings.weight")[MLXArray(tokens)]
        var vFirst: MLXArray?
        for layer in 0 ..< config.numHiddenLayers {
            (h, vFirst) = try block(h, layer: layer, vFirst: vFirst)
        }
        let normed = try layerNorm(h, "model.norm", eps: config.normEps)
        let head = try weights["lm_head.weight"] ?? weight("model.embeddings.weight")
        return matmul(normed, head.T)[.newAxis]
    }

    private func block(_ input: MLXArray, layer: Int, vFirst: MLXArray?) throws -> (MLXArray, MLXArray) {
        let p = "model.layers.\(layer)."
        let base = config.normFirst && layer == 0 ? try layerNorm(input, p + "pre_norm", eps: config.normEps) : input
        let attnIn = try layerNorm(base, p + "attn_norm", eps: config.normEps)
        let (timeOut, nextVFirst) = try timeMix(attnIn, prefix: p + "attn.", layer: layer, vFirst: vFirst)
        let mid = base + timeOut
        let ffnIn = try layerNorm(mid, p + "ffn_norm", eps: config.normEps)
        let ffnOut = try channelMix(ffnIn, prefix: p + "ffn.")
        return (mid + ffnOut, nextVFirst)
    }

    private func timeMix(_ x: MLXArray, prefix p: String, layer: Int, vFirst: MLXArray?) throws -> (MLXArray, MLXArray) {
        let (t, c, h, d) = (x.dim(0), config.hiddenSize, config.numHeads, config.headDim)
        let dx = shiftDelta(x)
        let r = try linear(x + dx * weight(p + "x_r").reshaped(1, c), p + "r_proj")
        let k = try linear(x + dx * weight(p + "x_k").reshaped(1, c), p + "k_proj")
        let xv = x + dx * (try weight(p + "x_v").reshaped(1, c))
        let (v, resolvedVFirst) = try resolveValue(xv: xv, prefix: p, layer: layer, vFirst: vFirst)
        let xw = x + dx * (try weight(p + "x_w").reshaped(1, c))
        let wRaw = try linearBias(tanh(try linear(xw, p + "w_lora.lora_a")), p + "w_lora.lora_b")
        let xa = x + dx * (try weight(p + "x_a").reshaped(1, c))
        let a = sigmoid(try linearBias(try linear(xa, p + "a_lora.lora_a"), p + "a_lora.lora_b"))
        let xg = x + dx * (try weight(p + "x_g").reshaped(1, c))
        let g = try linear(sigmoid(try linear(xg, p + "g_lora.lora_a")), p + "g_lora.lora_b")
        let kScaled = (k * (try weight(p + "k_k").reshaped(1, c))).reshaped(t, h, d)
        let kk = kScaled / maximum(sqrt((kScaled * kScaled).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        let kEff = (k * (1 + (a - 1) * (try weight(p + "k_a").reshaped(1, c)))).reshaped(t, h, d)
        let rH = r.reshaped(t, h, d)
        let vH = v.reshaped(t, h, d)
        let wDecay = exp(Float(-0.6065306597126334) * sigmoid(wRaw)).reshaped(t, h, d)
        let args = RecurrenceArgs(r: rH, v: vH, w: wDecay, a: a.reshaped(t, h, d), kk: kk, kEff: kEff)
        let normed = try groupNorm(recurrent(args), p + "g_norm", eps: config.groupNormEps)
        let bonus = ((rH * kEff * (try weight(p + "r_k"))).sum(axis: -1, keepDims: true) * vH).reshaped(t, c)
        return (try linear((normed + bonus) * g, p + "o_proj"), resolvedVFirst)
    }

    private func resolveValue(xv: MLXArray, prefix p: String, layer: Int, vFirst: MLXArray?) throws -> (MLXArray, MLXArray) {
        let vRaw = try linear(xv, p + "v_proj")
        if layer == 0 {
            return (vRaw, vRaw)
        }
        let first = try #require(vFirst)
        let gate = sigmoid(try linearBias(try linear(xv, p + "v_lora.lora_a"), p + "v_lora.lora_b"))
        return (vRaw + (first - vRaw) * gate, first)
    }

    private func recurrent(_ args: RecurrenceArgs) -> MLXArray {
        let (t, h, d) = (args.r.dim(0), config.numHeads, config.headDim)
        var state = MLXArray.zeros([h, d, d], dtype: .float32)
        var out: [MLXArray] = []
        for step in 0 ..< t {
            let (rt, vt) = (args.r[step], args.v[step])
            let (wt, at, kkt, kefft) = (args.w[step], args.a[step], args.kk[step], args.kEff[step])
            let sa = matmul(state, kkt.expandedDimensions(axis: -1))
            state = state * wt.expandedDimensions(axis: -2)
                - sa * (at * kkt).expandedDimensions(axis: -2)
                + vt.expandedDimensions(axis: -1) * kefft.expandedDimensions(axis: -2)
            out.append(matmul(state, rt.expandedDimensions(axis: -1)).squeezed(axis: -1))
        }
        return MLX.stacked(out, axis: 0)
    }

    private func channelMix(_ x: MLXArray, prefix p: String) throws -> MLXArray {
        let xk = x + shiftDelta(x) * (try weight(p + "x_k").reshaped(1, config.hiddenSize))
        let h = maximum(try linear(xk, p + "key"), MLXArray(Float(0)))
        return try linear(h * h, p + "value")
    }

    private func shiftDelta(_ x: MLXArray) -> MLXArray {
        let zero = MLXArray.zeros([1, x.dim(1)], dtype: .float32)
        let prev = x.dim(0) == 1 ? zero : concatenated([zero, x[..<(x.dim(0) - 1), 0...]], axis: 0)
        return prev - x
    }

    private func weight(_ key: String) throws -> MLXArray {
        try #require(weights[key], "Missing reference weight \(key)")
    }

    private func linear(_ x: MLXArray, _ key: String) throws -> MLXArray {
        matmul(x, try weight(key + ".weight").T)
    }

    private func linearBias(_ x: MLXArray, _ key: String) throws -> MLXArray {
        try linear(x, key) + (try weight(key + ".bias"))
    }

    private func layerNorm(_ x: MLXArray, _ key: String, eps: Float) throws -> MLXArray {
        let centered = x - mean(x, axis: -1, keepDims: true)
        let out = centered * rsqrt(mean(centered * centered, axis: -1, keepDims: true) + eps) * (try weight(key + ".weight"))
        return weights[key + ".bias"].map { out + $0 } ?? out
    }

    private func groupNorm(_ x: MLXArray, _ key: String, eps: Float) throws -> MLXArray {
        let centered = x - mean(x, axis: -1, keepDims: true)
        let flat = (centered * rsqrt(mean(centered * centered, axis: -1, keepDims: true) + eps)).reshaped(x.dim(0), -1)
        return flat * (try weight(key + ".weight")) + (try weight(key + ".bias"))
    }
}

@Suite("RWKV-7 Goose generalized delta-rule SSM model, recurrent cache, checkpoint sanitization, and registration (#825)", .serialized)
struct RWKV7ModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21]

    static func model(adding extra: String = "") throws -> RWKV7Model {
        try #require(try TinyModel.make(.rwkv7, adding: extra) as? RWKV7Model)
    }

    @Test("Full forward matches the independent plain-math RWKV-7 reference within 1e-4 and produces identical argmax ids (#228)")
    func matchesPlainMathReference() throws {
        let model = try Self.model()
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let reference = try RWKV7Reference(model).forward(Self.tokens)
        #expect(MLX.abs(logits - reference).max().item(Float.self) < 1e-4)
        #expect(argMax(logits, axis: -1).asArray(Int32.self) == argMax(reference, axis: -1).asArray(Int32.self))
    }

    @Test("Full-sequence prefill without cache matches token-by-token cached decode within 1e-4 and populates recurrent state shapes")
    func prefillMatchesStepByStepDecode() throws {
        let model = try Self.model()
        let full = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let cache = model.newCache(parameters: nil)
        #expect(cache.count == model.configuration.numHiddenLayers)
        let stepped = concatenated(Self.tokens.map { model(MLXArray([$0])[.newAxis], cache: cache).asType(.float32) }, axis: 1)
        #expect(MLX.abs(stepped - full).max().item(Float.self) < 1e-4)
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(full, axis: -1).asArray(Int32.self))
        for layerCache in cache {
            let rwkvCache = try #require(layerCache as? RWKV7LayerCache)
            #expect(rwkvCache.offset == Self.tokens.count)
            #expect(rwkvCache.isTrimmable == false)
            #expect(rwkvCache.trim(2) == 0)
            #expect(rwkvCache.stateMatrix?.shape == [1, model.configuration.numHeads, model.configuration.headDim, model.configuration.headDim])
            #expect(rwkvCache.prevTimeX?.shape == [1, 1, model.configuration.hiddenSize])
            #expect(rwkvCache.prevChannelX?.shape == [1, 1, model.configuration.hiddenSize])
            let isNoneMask: Bool = if case .none = rwkvCache.makeMask(n: 4, windowSize: nil, returnArray: false) { true } else { false }
            #expect(isNoneMask)
        }
    }

    @Test("Multi-chunk prefill (5 + 3 tokens) matches single-pass 8-token prefill in both logits and final recurrent state")
    func chunkedPrefillMatchesSinglePass() throws {
        let model = try Self.model()
        let singleCache = model.newCache(parameters: nil)
        let full = model(MLXArray(Self.tokens)[.newAxis], cache: singleCache).asType(.float32)
        let chunkCache = model.newCache(parameters: nil)
        let first = model(MLXArray(Array(Self.tokens[..<5]))[.newAxis], cache: chunkCache).asType(.float32)
        let second = model(MLXArray(Array(Self.tokens[5...]))[.newAxis], cache: chunkCache).asType(.float32)
        let combined = concatenated([first, second], axis: 1)
        #expect(MLX.abs(combined - full).max().item(Float.self) < 1e-4)
        for (a, b) in zip(singleCache, chunkCache) {
            let (ca, cb) = (try #require(a as? RWKV7LayerCache), try #require(b as? RWKV7LayerCache))
            #expect(ca.offset == cb.offset && cb.offset == Self.tokens.count)
            #expect(MLX.abs(try #require(ca.stateMatrix) - (try #require(cb.stateMatrix))).max().item(Float.self) < 1e-4)
            #expect(MLX.abs(try #require(ca.prevTimeX) - (try #require(cb.prevTimeX))).max().item(Float.self) < 1e-4)
            #expect(MLX.abs(try #require(ca.prevChannelX) - (try #require(cb.prevChannelX))).max().item(Float.self) < 1e-4)
        }
    }

    @Test("Layer 0 produces v_first without v_lora and deeper layers depend on layer 0's v_first residual")
    func vFirstResidualPropagatesToDeeperLayers() throws {
        let model = try Self.model()
        #expect(model.model.layers[0].attn.vLora == nil)
        #expect(model.model.layers[1].attn.vLora != nil)
        #expect(model.model.layers[2].attn.vLora != nil)
        let x = withRandomState(MLXRandom.RandomState(seed: 101)) { MLXRandom.normal([1, 4, model.configuration.hiddenSize]) }
        let vFirstA = withRandomState(MLXRandom.RandomState(seed: 202)) { MLXRandom.normal([1, 4, model.configuration.hiddenSize]) }
        let vFirstB = withRandomState(MLXRandom.RandomState(seed: 303)) { MLXRandom.normal([1, 4, model.configuration.hiddenSize]) }
        let (outA, carriedA) = model.model.layers[1].attn(x, vFirst: vFirstA, cache: nil)
        let (outB, carriedB) = model.model.layers[1].attn(x, vFirst: vFirstB, cache: nil)
        #expect(MLX.abs(outA - outB).max().item(Float.self) > 1e-3)
        #expect(MLX.abs(carriedA - vFirstA).max().item(Float.self) == 0)
        #expect(MLX.abs(carriedB - vFirstB).max().item(Float.self) == 0)
    }

    @Test("Cache copy keeps state independent across branches, reset clears state cleanly, and withRandomState reproduces logits identically")
    func cacheLifecycleAndDeterministicReproducibility() throws {
        let model = try Self.model()
        let prefix = Array(Self.tokens[..<4])
        let cache = model.newCache(parameters: nil)
        _ = model(MLXArray(prefix)[.newAxis], cache: cache)
        let cloned = cache.map { $0.copy() }
        let branchA = model(MLXArray([Int32(91)])[.newAxis], cache: cache).asType(.float32)
        for layerCache in cache { (layerCache as? RWKV7LayerCache)?.reset() }
        #expect(cache.allSatisfy { $0.offset == 0 && $0.innerState().isEmpty })
        let branchB = model(MLXArray([Int32(44)])[.newAxis], cache: cloned).asType(.float32)
        let referenceB = model(MLXArray(prefix + [44])[.newAxis], cache: nil)[0..., 4..., 0...].asType(.float32)
        #expect(MLX.abs(branchB - referenceB).max().item(Float.self) < 1e-4)
        #expect(MLX.abs(branchA - branchB).max().item(Float.self) > 1e-4)

        let restored = cloned.map { source -> any KVCache in
            let target = RWKV7LayerCache()
            target.metaState = source.metaState
            target.state = source.state
            return target
        }
        let nextCloned = model(MLXArray([Int32(12)])[.newAxis], cache: cloned).asType(.float32)
        let nextRestored = model(MLXArray([Int32(12)])[.newAxis], cache: restored).asType(.float32)
        #expect(MLX.abs(nextCloned - nextRestored).max().item(Float.self) == 0)

        let replayed = model(MLXArray(prefix)[.newAxis], cache: cache).asType(.float32)
        let fresh = model(MLXArray(prefix)[.newAxis], cache: nil).asType(.float32)
        #expect(MLX.abs(replayed - fresh).max().item(Float.self) == 0)

        let data = Data(TinyModel.Family.rwkv7.config.utf8)
        let m1 = try withRandomState(MLXRandom.RandomState(seed: 999)) { try RWKV7Model(configurationData: data) }
        let m2 = try withRandomState(MLXRandom.RandomState(seed: 999)) { try RWKV7Model(configurationData: data) }
        let l1 = m1(MLXArray(Self.tokens)[.newAxis], cache: nil)
        let l2 = m2(MLXArray(Self.tokens)[.newAxis], cache: nil)
        #expect(MLX.abs(l1 - l2).max().item(Float.self) == 0)
    }
}

extension RWKV7ModelTests {
    @Test("Invalid configs are refused with DecodingError without division-by-zero traps", arguments: [
        #"{"vocab_size": 0, "hidden_size": 64, "num_hidden_layers": 2, "head_dim": 16, "num_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 0, "num_hidden_layers": 2, "head_dim": 16, "num_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 0, "head_dim": 16, "num_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "head_dim": 16, "num_heads": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "head_dim": 0, "num_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "head_dim": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 65, "num_hidden_layers": 2, "num_heads": 4}"#,
        #"{"vocab_size": 256, "hidden_size": 65, "num_hidden_layers": 2, "head_dim": 16}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "head_dim": 16, "decay_low_rank_dim": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "head_dim": 16, "intermediate_size": 0}"#,
        #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "head_dim": 16, "norm_eps": 0}"#,
    ])
    func invalidConfigRefused(_ json: String) {
        #expect(throws: DecodingError.self) {
            _ = try RWKV7Model(configurationData: Data(json.utf8))
        }
    }

    @Test("Config supports FLA and canonical RWKV-7 key aliases and default head_dim = 64")
    func configSupportsFLAAliasesAndDefaults() throws {
        let flaJSON = """
            {"model_type": "rwkv7", "vocab_size": 512, "hidden_size": 128, "num_hidden_layers": 2,
             "gate_low_rank_dim": 32, "hidden_ratio": 3.5, "fuse_cross_entropy": false, "norm_first": false}
            """
        let fla = try JSONDecoder.json5().decode(RWKV7Configuration.self, from: Data(flaJSON.utf8))
        #expect(fla.headDim == 64 && fla.numHeads == 2)
        #expect(fla.gLowRankDim == 32 && fla.intermediateSize == 448)
        #expect(!fla.fuseCrossEntropy && !fla.normFirst)

        let flaRealJSON = """
            {"model_type": "rwkv7", "vocab_size": 65536, "hidden_size": 768, "num_hidden_layers": 12,
             "head_dim": 64, "num_heads": 32, "decay_low_rank_dim": 64, "a_low_rank_dim": 64,
             "v_low_rank_dim": 32, "gate_low_rank_dim": 128, "intermediate_size": 3072, "norm_eps": 1e-5}
            """
        let flaReal = try JSONDecoder.json5().decode(RWKV7Configuration.self, from: Data(flaRealJSON.utf8))
        #expect(flaReal.headDim == 64 && flaReal.numHeads == 12)

        let canonicalJSON = """
            {"vocab_size": 256, "n_embd": 64, "n_layer": 2, "n_head": 4, "w_lora_dim": 12,
             "a_lora_dim": 10, "v_lora_dim": 8, "g_lora_dim": 14, "dim_ffn": 96, "layer_norm_epsilon": 1e-6}
            """
        let canonical = try JSONDecoder.json5().decode(RWKV7Configuration.self, from: Data(canonicalJSON.utf8))
        #expect(canonical.hiddenSize == 64 && canonical.numHiddenLayers == 2 && canonical.headDim == 16 && canonical.numHeads == 4)
        #expect(canonical.decayLowRankDim == 12 && canonical.aLowRankDim == 10 && canonical.vLowRankDim == 8 && canonical.gLowRankDim == 14)
        #expect(canonical.intermediateSize == 96 && canonical.normEps == 1e-6)
    }

    @Test("sanitize maps FLA Sequential LoRA, stacked x_x, BlinkDL canonical weights, and drops lm_head when tie_word_embeddings is true")
    func sanitizeHandlesHFAndCanonicalWeightsAndTiedEmbeddings() throws {
        let tied = try Self.model(adding: #""tie_word_embeddings": true, "#)
        #expect(tied.lmHead == nil)
        let tiedLogits = tied(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let tiedRef = try RWKV7Reference(tied).forward(Self.tokens)
        #expect(MLX.abs(tiedLogits - tiedRef).max().item(Float.self) < 1e-4)

        let c = tied.configuration.hiddenSize
        let raw: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([tied.configuration.vocabSize, c]),
            "lm_head.weight": MLXArray.ones([tied.configuration.vocabSize, c]),
            "model.layers.0.attn.x_x": MLXArray.ones([6, 1, 1, c]),
            "model.layers.0.attn.w_lora.lora.0.weight": MLXArray.ones([16, c]),
            "model.layers.0.attn.w_lora.lora.2.weight": MLXArray.ones([c, 16]),
            "model.layers.0.attn.w_lora.lora.2.bias": MLXArray.ones([c]),
            "model.layers.0.attn.v_lora.lora.0.weight": MLXArray.ones([16, c]),
            "model.layers.1.attn.a_lora.lora.1.weight": MLXArray.ones([c, 16]),
            "model.layers.1.attn.r_k": MLXArray.ones([c]),
            "model.layers.1.ffn.x_k": MLXArray.ones([c]),
            "blocks.2.att.w1": MLXArray.ones([c, 16]),
            "blocks.2.att.w2": MLXArray.ones([16, c]),
            "blocks.2.att.w0": MLXArray.ones([1, 1, c]),
            "blocks.2.ln1.weight": MLXArray.ones([c]),
            "ln_out.weight": MLXArray.ones([c]),
        ]
        let sanitized = tied.sanitize(weights: raw)
        #expect(sanitized["model.embeddings.weight"] != nil)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.layers.0.attn.v_lora.lora_a.weight"] == nil)
        #expect(sanitized["model.layers.0.attn.x_r"]?.shape == [1, 1, c])
        #expect(sanitized["model.layers.0.attn.x_g"]?.shape == [1, 1, c])
        #expect(sanitized["model.layers.0.attn.w_lora.lora_a.weight"]?.shape == [16, c])
        #expect(sanitized["model.layers.0.attn.w_lora.lora_b.weight"]?.shape == [c, 16])
        #expect(sanitized["model.layers.1.attn.a_lora.lora_b.weight"]?.shape == [c, 16])
        #expect(sanitized["model.layers.1.attn.r_k"]?.shape == [tied.configuration.numHeads, tied.configuration.headDim])
        #expect(sanitized["model.layers.1.ffn.x_k"]?.shape == [1, 1, c])
        #expect(sanitized["model.layers.2.attn.w_lora.lora_a.weight"]?.shape == [16, c])
        #expect(sanitized["model.layers.2.attn.w_lora.lora_b.weight"]?.shape == [c, 16])
        #expect(sanitized["model.layers.2.attn.w_lora.lora_b.bias"]?.shape == [c])
        #expect(sanitized["model.layers.2.attn_norm.weight"] != nil)
        #expect(sanitized["model.norm.weight"] != nil)
    }

    @Test("LLMTypeRegistry registers rwkv7, basalt models types reports it loadable, and ModelCapabilities classifies it as recurrent")
    func registryAndCapabilitiesSupportRWKV7() async throws {
        await MLXInferenceEngine.registerAliases()
        #expect(await LLMTypeRegistry.shared.contains("rwkv7"))
        let loadable = await MLXInferenceEngine.loadableModelTypes(among: ["rwkv7", "not_a_model"])
        #expect(loadable == ["rwkv7"])
        let created = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(TinyModel.Family.rwkv7.config.utf8),
            modelType: "rwkv7"
        )
        #expect(created is RWKV7Model)
        let capabilities = try #require(ModelCapabilities.of(modelType: "rwkv7"))
        #expect(capabilities.allows(.promptCacheReuse))
        #expect(capabilities.allows(.grammarJumpForward))
        #expect(!capabilities.allows(.kvQuantization))
        #expect(throws: BasaltError.self) {
            try ModelCapabilities.refuseUnverifiedKVBits(.q4, modelType: "rwkv7", path: "tiny-rwkv7")
        }
    }
}
