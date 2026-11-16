import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Ricursive Ripple-65B architecture: 80-layer configuration invariants, PeriNorm, Value Residual, FP32 gated SWA/NoPE attention, asymmetric clamped SwiGLU, GGUF layout and agentic profile (#520)", .serialized)
struct RicursiveModelTests {
    static let officialConfigJSON = """
        {
          "architectures": ["RicursiveForCausalLM"],
          "model_type": "ricursive",
          "vocab_size": 100352,
          "hidden_size": 5120,
          "intermediate_size": 40960,
          "num_hidden_layers": 80,
          "num_attention_heads": 80,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "max_position_embeddings": 2097152,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "sliding_window": 1024,
          "embed_scale": 71.55417527999327,
          "ffn_clamp_threshold": 20.0,
          "gate_full_precision": true,
          "use_gated_attention": true,
          "use_valres": true,
          "tie_word_embeddings": false
        }
        """

    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let delta = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(delta)
        return delta.item(Float.self)
    }

    static func fp32RMSNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        let xf = x.asType(.float32)
        let variance = mean(xf * xf, axis: -1, keepDims: true)
        return weight * (xf * rsqrt(variance + eps)).asType(x.dtype)
    }

    @Test("Official 80-layer Ripple-65B configuration derives 64 sliding and 16 full layers, 2M context KV savings and rejects invalid layer_types")
    func configurationAndGeometry() throws {
        let data = Data(Self.officialConfigJSON.utf8)
        let config = try JSONDecoder.json5().decode(RicursiveConfiguration.self, from: data).validated()
        #expect(config.modelType == "ricursive" && config.hiddenLayers == 80 && config.vocabularySize == 100_352)
        #expect(config.effectiveSlidingWindow == 1024 && config.useGatedAttention && config.useValres)
        #expect(abs((config.embedScale ?? 0) - Float(5120.0.squareRoot())) < 1e-4)
        let sliding = (0 ..< 80).filter { config.usesSlidingWindow(layer: $0) }
        let full = (0 ..< 80).filter { !config.usesSlidingWindow(layer: $0) }
        #expect(sliding.count == 64 && full.count == 16)
        #expect(!config.usesSlidingWindow(layer: 4) && !config.usesSlidingWindow(layer: 79) && config.usesSlidingWindow(layer: 0))

        var raw = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        raw["layer_types"] = config.layerTypes
        let geometry = try #require(KVCacheGeometry(config: raw))
        let hybridBytes = geometry.bytes(tokens: TokenCount(131_072)).rawValue
        let fullBytes = Int64(KVCacheGeometry.kvBytesPerToken(layers: 80, kvHeads: 8, headDim: 128, bytesPerElement: 2) * 131_072)
        #expect(Double(hybridBytes) < Double(fullBytes) * 0.25)

        var badKind = config
        badKind.layerTypes[0] = "linear_attention"
        #expect(throws: BasaltError.self) { _ = try badKind.validated() }
        var badCount = config
        badCount.layerTypes.removeLast()
        #expect(throws: BasaltError.self) { _ = try badCount.validated() }
    }

    @Test("RicursiveDecoderLayer matches explicit FP32 reference for PeriNorm, Value Residual lerp on V only, gated SWA/NoPE attention and asymmetric SwiGLU clamping (#228)")
    func decoderLayerAndAsymmetricClampParity() throws {
        let config = try JSONDecoder.json5().decode(RicursiveConfiguration.self, from: Data(TinyModel.Family.ricursive.config.utf8)).validated()
        withRandomState(MLXRandom.RandomState(seed: 5200)) {
            var clampedConfig = config
            clampedConfig.ffnClampThreshold = 0.25
            let slidingLayer = RicursiveDecoderLayer(clampedConfig, layerIdx: 0)
            let fullLayer = RicursiveDecoderLayer(clampedConfig, layerIdx: 4)
            slidingLayer.selfAttn.update(parameters: ModuleParameters.unflattened([("value_res_lambda", MLXArray([Float(-0.6)]))]))
            fullLayer.selfAttn.update(parameters: ModuleParameters.unflattened([("value_res_lambda", MLXArray([Float(0.9)]))]))

            let x = MLXRandom.normal([2, 10, 64]) * 1.5
            let valRes = MLXRandom.normal([2, 10, 64]) * 2.0
            eval(slidingLayer, fullLayer, x, valRes)

            for (idx, layer, windowSize) in [(0, slidingLayer, Int?.some(8)), (4, fullLayer, Int?.none)] {
                let maskArr = createCausalMask(n: 10, offset: 0, windowSize: windowSize)
                let fast = layer(x, valueResidual: valRes, mask: .array(maskArr), cache: nil)
                let ref = Self.referenceDecoderLayer(layer, x: x, valueResidual: valRes, booleanMask: maskArr, eps: clampedConfig.rmsNormEps, clamp: 0.25)
                let noValRes = Self.referenceDecoderLayer(layer, x: x, valueResidual: nil, booleanMask: maskArr, eps: clampedConfig.rmsNormEps, clamp: 0.25)
                #expect(Self.maxAbsDiff(fast, ref) < 1e-4, "layer \(idx)")
                #expect(Self.maxAbsDiff(fast, noValRes) > 1e-3, "layer \(idx)")
            }

            let mlp = RicursiveMLP(clampedConfig)
            let wideIn = MLXRandom.normal([2100, 64]) * 4.0
            eval(mlp, wideIn)
            let chunked = mlp(wideIn)
            let gRaw = mlp.gate(wideIn)
            let uRaw = mlp.up(wideIn)
            let expectedUnchunked = mlp.down(silu(minimum(gRaw, MLXArray(Float(0.25)))) * clip(uRaw, min: MLXArray(Float(-0.25)), max: MLXArray(Float(0.25))))
            let wrongSymmetricGate = mlp.down(silu(clip(gRaw, min: MLXArray(Float(-0.25)), max: MLXArray(Float(0.25)))) * clip(uRaw, min: MLXArray(Float(-0.25)), max: MLXArray(Float(0.25))))
            #expect(Self.maxAbsDiff(chunked, expectedUnchunked) < 1e-5)
            #expect(Self.maxAbsDiff(chunked, wrongSymmetricGate) > 1e-3)
        }
    }

    static func referenceDecoderLayer(
        _ layer: RicursiveDecoderLayer,
        x: MLXArray,
        valueResidual: MLXArray?,
        booleanMask: MLXArray,
        eps: Float,
        clamp: Float
    ) -> MLXArray {
        let normed = fp32RMSNorm(x, weight: layer.inputLayerNorm.weight, eps: eps)
        let attn = layer.selfAttn
        let (b, l) = (normed.dim(0), normed.dim(1))
        let vIn: MLXArray
        if let valueResidual, let lam = attn.valueResLambda {
            let blend = sigmoid(lam.asType(.float32))[0]
            vIn = valueResidual + blend * (normed - valueResidual)
        } else {
            vIn = normed
        }
        var q = fp32RMSNorm(attn.wq(normed).reshaped(b, l, attn.heads, attn.headDim), weight: attn.qNorm.weight, eps: eps).transposed(0, 2, 1, 3)
        var k = fp32RMSNorm(attn.wk(normed).reshaped(b, l, attn.kvHeads, attn.headDim), weight: attn.kNorm.weight, eps: eps).transposed(0, 2, 1, 3)
        let v = attn.wv(vIn).reshaped(b, l, attn.kvHeads, attn.headDim).transposed(0, 2, 1, 3)
        if attn.usesRotary, let rope = attn.rope {
            q = applyRotaryPosition(rope, to: q.asType(.float32), offset: nil).asType(x.dtype)
            k = applyRotaryPosition(rope, to: k.asType(.float32), offset: nil).asType(x.dtype)
        }
        var attnOut = FlintAttention.explicitAttentionWithSinks(
            queries: q, keys: k, values: v, scale: attn.scale, booleanMask: booleanMask, sinks: nil
        ).transposed(0, 2, 1, 3).reshaped(b, l, -1)
        if let wg = attn.wg {
            attnOut = attnOut * sigmoid(wg(normed).asType(.float32)).asType(attnOut.dtype)
        }
        let afterAttn = x + fp32RMSNorm(attn.wo(attnOut), weight: layer.postAttentionLayerNorm.weight, eps: eps)
        let ffNorm = fp32RMSNorm(afterAttn, weight: layer.preFeedforwardLayerNorm.weight, eps: eps)
        let g = minimum(layer.mlp.gate(ffNorm), MLXArray(clamp))
        let u = clip(layer.mlp.up(ffNorm), min: MLXArray(-clamp), max: MLXArray(clamp))
        let ffOut = layer.mlp.down(silu(g) * u)
        return afterAttn + fp32RMSNorm(ffOut, weight: layer.postFeedforwardLayerNorm.weight, eps: eps)
    }

    @Test("GGUFRicursiveLayout maps PeriNorm, attn_gate and attn_vres_lambda tensors, and RicursiveProfile configures <think> reasoning and JSON tools")
    func ggufLayoutAndProfile() throws {
        let layout = GGUFRicursiveLayout()
        let gatePlan = try #require(layout.plan(GGUFTensor(name: "blk.4.attn_gate.weight", type: .q4K, dims: [5120, 10240], offset: 0)))
        let vresPlan = try #require(layout.plan(GGUFTensor(name: "blk.4.attn_vres_lambda", type: .f32, dims: [1], offset: 0)))
        let postAttn = try #require(layout.plan(GGUFTensor(name: "blk.4.attn_post_norm.weight", type: .f32, dims: [5120], offset: 0)))
        let postFFN = try #require(layout.plan(GGUFTensor(name: "blk.4.ffn_post_norm.weight", type: .f32, dims: [5120], offset: 0)))
        #expect(gatePlan.target == "model.layers.4.self_attn.g_proj.weight" && gatePlan.bits == 4)
        #expect(vresPlan.target == "model.layers.4.self_attn.value_res_lambda" && vresPlan.bits == nil)
        #expect(postAttn.target == "model.layers.4.post_attention_layernorm.weight" && postAttn.bits == nil)
        #expect(postFFN.target == "model.layers.4.post_feedforward_layernorm.weight" && postFFN.bits == nil)

        let profile = ModelProfileRegistry.profile(forModelType: "ricursive")
        let aliasProfile = ModelProfileRegistry.profile(forModelType: "ripple")
        #expect(profile is RicursiveProfile && aliasProfile is RicursiveProfile)
        #expect(profile.defaultMaxTokens == 32_768 && profile.thinkingMarkers.open == ["<think>"])
        let model = try #require(try TinyModel.make(.ricursive) as? RicursiveModel)
        #expect(model.toolCallFormat == .json)
    }
}
