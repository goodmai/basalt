import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXEmbedders
import BasaltCore

enum LeJEPABackbone {
    struct Output {
        let hiddenStates: MLXArray?
        let pooledOutput: MLXArray?

        init(hiddenStates: MLXArray?, pooledOutput: MLXArray? = nil) {
            self.hiddenStates = hiddenStates
            self.pooledOutput = pooledOutput
        }

        init(_ output: EmbeddingModelOutput) {
            self.hiddenStates = output.hiddenStates
            self.pooledOutput = output.pooledOutput
        }
    }

    static func layers(of model: Module) -> [Module] {
        if let lora = model as? LoRAModel { return lora.loraLayers }
        for candidate in [model, childModule(of: model, key: "model"), childModule(of: model, key: "encoder")].compactMap({ $0 }) {
            for key in ["layers", "layer"] {
                if case .array(let items) = candidate.children()[key] {
                    let modules = items.compactMap { item -> Module? in
                        if case .value(let m) = item { return m }
                        return nil
                    }
                    if !modules.isEmpty { return modules }
                }
            }
        }
        return []
    }

    static func headlessBackbone(of model: Module) -> Module {
        childModule(of: model, key: "model") ?? model
    }

    static func makeQwen3Encoder(
        configData: Data, weights: [String: MLXArray] = [:]
    ) throws(BasaltError) -> MLXEmbedders::Qwen3Model {
        let model: MLXEmbedders::Qwen3Model
        do {
            let config = try JSONDecoder.json5().decode(MLXEmbedders::Qwen3Configuration.self, from: configData)
            model = MLXEmbedders::Qwen3Model(config)
        } catch {
            throw .modelConfigurationInvalid(path: "config.json", reason: error.localizedDescription)
        }
        if !weights.isEmpty {
            let sanitized = model.sanitize(weights: weights)
            model.update(parameters: ModuleParameters.unflattened(sanitized))
        }
        eval(model)
        return model
    }

    static func attentionMask(
        inputIds: MLXArray, attentionMask: MLXArray?, isCausal: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let length = inputIds.dim(1)
        if isCausal {
            if let attentionMask, length > 1 {
                let causal = createCausalMask(n: length, offset: 0)
                let keep = attentionMask.asType(.bool)[0..., .newAxis, .newAxis, 0...]
                return .array(causal & keep)
            }
            return length > 1 ? .causal : .none
        }
        if let attentionMask {
            let keep = attentionMask.asType(.bool)[0..., .newAxis, .newAxis, 0...]
            return .array(keep)
        }
        return .none
    }

    static func encode(
        model: Module, inputIds: MLXArray, attentionMask: MLXArray? = nil, isCausal: Bool = false
    ) -> Output {
        let inputs = inputIds.ndim == 1 ? inputIds.reshaped(1, -1) : inputIds
        let inner = headlessBackbone(of: model)
        if let embedTokens = childUnary(of: inner, key: "embed_tokens"),
           let norm = childUnary(of: inner, key: "norm") {
            let blocks = layers(of: inner)
            if !blocks.isEmpty, blocks.allSatisfy({ childModule(of: $0, key: "self_attn") != nil }) {
                var h = embedTokens(inputs)
                if let firstBlock = blocks.first, childUnary(of: firstBlock, key: "pre_feedforward_layernorm") != nil {
                    let scale = MLXArray(sqrt(Float(h.dim(-1))), dtype: .bfloat16)
                    h = h * scale.asType(h.dtype)
                }
                let mask = Self.attentionMask(inputIds: inputs, attentionMask: attentionMask, isCausal: isCausal)
                for block in blocks {
                    h = forwardBlock(block, h, mask: mask)
                }
                return Output(hiddenStates: norm(h), pooledOutput: nil)
            }
        }
        if let embedder = model as? any EmbeddingModel {
            return Output(embedder(
                inputs, positionIds: nil, tokenTypeIds: MLXArray.zeros(like: inputs), attentionMask: attentionMask))
        }
        return Output(hiddenStates: nil, pooledOutput: nil)
    }

    static func pool(
        _ output: Output, mask: MLXArray? = nil, strategy: Pooling.Strategy = .mean
    ) -> MLXArray {
        switch strategy {
        case .mean:
            guard let hiddenStates = output.hiddenStates else {
                return (output.pooledOutput ?? MLXArray.zeros([1, 1])).asType(.float32)
            }
            let h = hiddenStates.asType(.float32)
            let m = (mask ?? MLXArray.ones([h.dim(0), h.dim(1)], type: Float.self)).asType(.float32)
            let expanded = m.expandedDimensions(axes: [-1])
            let masked = MLX.where(expanded .> MLXArray(Float(0)), h * expanded, MLXArray(Float(0)))
            let denom = MLX.maximum(MLX.sum(m, axis: -1, keepDims: true), MLXArray(Float(1e-9)))
            return MLX.sum(masked, axis: 1) / denom
        case .cls, .first:
            if let pooled = output.pooledOutput {
                return pooled.asType(.float32)
            }
            guard let hiddenStates = output.hiddenStates else {
                return MLXArray.zeros([1, 1], type: Float.self)
            }
            return hiddenStates[0..., 0, 0...].asType(.float32)
        case .last:
            guard let hiddenStates = output.hiddenStates else {
                return (output.pooledOutput ?? MLXArray.zeros([1, 1])).asType(.float32)
            }
            let maskArray = mask ?? MLXArray.ones([hiddenStates.dim(0), hiddenStates.dim(1)], type: Int32.self)
            let tokenCounts = MLX.sum(maskArray.asType(.int32), axis: -1)
            let tokenIndices = MLX.maximum(tokenCounts - MLXArray(Int32(1)), MLXArray(Int32(0)))
            let indices = tokenIndices.expandedDimensions(axes: [1, 2])
            let gathered = MLX.takeAlong(hiddenStates, indices, axis: 1)
            return MLX.squeezed(gathered, axis: 1).asType(.float32)
        case .max:
            guard let hiddenStates = output.hiddenStates else {
                return (output.pooledOutput ?? MLXArray.zeros([1, 1])).asType(.float32)
            }
            let maskArray = (mask ?? MLXArray.ones([hiddenStates.dim(0), hiddenStates.dim(1)], type: Float.self)).asType(hiddenStates.dtype)
            return MLX.max(hiddenStates * maskArray.expandedDimensions(axes: [-1]), axis: 1).asType(.float32)
        case .none:
            return (output.pooledOutput ?? output.hiddenStates ?? MLXArray.zeros([1, 1])).asType(.float32)
        }
    }

    static func applyLoRA(_ container: LoRAContainer, to model: Module) throws(BasaltError) {
        let blocks = MLXInferenceEngine.loraBlocks(of: model)
        guard !blocks.isEmpty else {
            throw .invalidRequestStructure(details: "adapter: this embedding model has no transformer layers for LoRA")
        }
        let count = container.configuration.numLayers
        guard count > 0, count <= blocks.count else {
            throw .invalidRequestStructure(details: "adapter: num_layers \(count) exceeds model layer count \(blocks.count)")
        }
        let defaultKeys = Array(Set(blocks.flatMap { $0.namedModules() }.compactMap { $0.1 is Linear ? $0.0 : nil })).sorted()
        let keys = container.configuration.loraParameters.keys ?? defaultKeys
        let start = blocks.count - count
        let indices = Set(start ..< blocks.count)
        let params = LoRAConfiguration.LoRAParameters(
            rank: container.configuration.loraParameters.rank,
            scale: container.configuration.loraParameters.scale,
            dropout: container.configuration.loraParameters.dropout,
            keys: keys)
        MLXInferenceEngine.replaceLoRA(
            in: blocks, indices: indices, parameters: params, fineTuneType: container.configuration.fineTuneType)
        do {
            try model.update(parameters: container.parameters, verify: .noUnusedKeys)
            model.invalidateCompiledTraces()
        } catch {
            throw .modelConfigurationInvalid(path: "adapters.safetensors", reason: error.localizedDescription)
        }
    }
}

extension LeJEPABackbone {
    fileprivate static func childModule(of module: Module, key: String) -> Module? {
        if case .value(let child) = module.children()[key] { return child }
        return nil
    }

    fileprivate static func childUnary(of module: Module, key: String) -> (any UnaryLayer)? {
        childModule(of: module, key: key) as? any UnaryLayer
    }

    fileprivate static func forwardBlock(
        _ block: Module, _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard let inputNorm = childUnary(of: block, key: "input_layernorm"),
              let postAttnNorm = childUnary(of: block, key: "post_attention_layernorm"),
              let attn = childModule(of: block, key: "self_attn"),
              let mlpMod = childModule(of: block, key: "mlp") else {
            return x
        }
        let r = forwardAttention(attn, inputNorm(x), mask: mask)
        if let preFF = childUnary(of: block, key: "pre_feedforward_layernorm"),
           let postFF = childUnary(of: block, key: "post_feedforward_layernorm") {
            let h = x + postAttnNorm(r)
            return h + postFF(forwardMLP(mlpMod, preFF(h)))
        }
        let h = x + r
        return h + forwardMLP(mlpMod, postAttnNorm(h))
    }

    fileprivate static func forwardAttention(
        _ attn: Module, _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard let wq = childUnary(of: attn, key: "q_proj"),
              let wk = childUnary(of: attn, key: "k_proj"),
              let wv = childUnary(of: attn, key: "v_proj"),
              let wo = childUnary(of: attn, key: "o_proj") ?? childUnary(of: attn, key: "out_proj"),
              let rope = childModule(of: attn, key: "rope") as? any OffsetLayer else {
            return x
        }
        let (B, L) = (x.dim(0), x.dim(1))
        let qNorm = childUnary(of: attn, key: "q_norm") ?? childUnary(of: attn, key: "q_layernorm")
        let kNorm = childUnary(of: attn, key: "k_norm") ?? childUnary(of: attn, key: "k_layernorm")
        let headDim = resolveHeadDim(attn: attn, qNorm: qNorm)
        var q = wq(x).reshaped(B, L, -1, headDim)
        var k = wk(x).reshaped(B, L, -1, headDim)
        let v = wv(x).reshaped(B, L, -1, headDim).transposed(0, 2, 1, 3)
        if let qNorm { q = qNorm(q) }
        if let kNorm { k = kNorm(k) }
        q = rope(q.transposed(0, 2, 1, 3), offset: 0)
        k = rope(k.transposed(0, 2, 1, 3), offset: 0)
        let scale = pow(Float(headDim), -0.5)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(attended)
    }

    fileprivate static func resolveHeadDim(attn: Module, qNorm: Module?) -> Int {
        if let weight = qNorm?.parameters().flattened().first?.1 {
            return weight.dim(0)
        }
        if let rope = childModule(of: attn, key: "rope") {
            for child in Mirror(reflecting: rope).children where child.label == "dimensions" || child.label == "dims" {
                if let dims = child.value as? Int { return dims }
            }
        }
        return 64
    }

    fileprivate static func forwardMLP(_ mlp: Module, _ x: MLXArray) -> MLXArray {
        if let unary = mlp as? any UnaryLayer {
            return unary(x)
        }
        if let gate = childUnary(of: mlp, key: "gate_proj"),
           let up = childUnary(of: mlp, key: "up_proj"),
           let down = childUnary(of: mlp, key: "down_proj") {
            return down(silu(gate(x)) * up(x))
        }
        return x
    }
}
