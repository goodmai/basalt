import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import BasaltCore

package final class NemotronMTPAttentionMixer: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    package init(_ args: NemotronHConfiguration) {
        self.numHeads = args.numAttentionHeads
        self.numKVHeads = args.numKeyValueHeads
        let dim = args.headDim ?? (args.hiddenSize / args.numAttentionHeads)
        self.headDim = dim
        self.scale = 1.0 / Float(dim).squareRoot()
        self._wq.wrappedValue = Linear(args.hiddenSize, args.numAttentionHeads * dim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(args.hiddenSize, args.numKeyValueHeads * dim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(args.hiddenSize, args.numKeyValueHeads * dim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(args.numAttentionHeads * dim, args.hiddenSize, bias: args.attentionBias)
    }

    package func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let q = wq(x).reshaped(b, l, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = wk(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let out = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
            .transposed(0, 2, 1, 3)
            .reshaped(b, l, -1)
        return wo(out)
    }
}

package final class NemotronMTPAttentionBlock: Module {
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronMTPAttentionMixer

    package init(_ args: NemotronHConfiguration) {
        self._enorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        self._hnorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        self._ehProj.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        self._mixer.wrappedValue = NemotronMTPAttentionMixer(args)
    }

    package func callAsFunction(inputsEmbeds: MLXArray, hiddenStates: MLXArray, cache: KVCache?) -> MLXArray {
        let fused = concatenated([enorm(inputsEmbeds), hnorm(hiddenStates)], axis: -1)
        let h = ehProj(fused)
        let mask = makeAttentionMask(n: h.dim(1), cache: cache)
        return h + mixer(norm(h), mask: mask, cache: cache)
    }
}

package final class NemotronMTPMoEBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronLatentMoEMixer
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm

    package init(_ args: NemotronHConfiguration, moeLatentSize: Int?) {
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        self._mixer.wrappedValue = NemotronLatentMoEMixer(args, moeLatentSize: moeLatentSize)
        self._finalLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    }

    package func callAsFunction(_ h: MLXArray) -> MLXArray {
        let combined = h + mixer(norm(h))
        return finalLayerNorm(combined)
    }
}

package final class NemotronMTPPredictor: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    private let attnBlock: NemotronMTPAttentionBlock
    private let moeBlock: NemotronMTPMoEBlock

    package init(_ args: NemotronHConfiguration, moeLatentSize: Int? = nil) {
        let attn = NemotronMTPAttentionBlock(args)
        let moe = NemotronMTPMoEBlock(args, moeLatentSize: moeLatentSize)
        self.attnBlock = attn
        self.moeBlock = moe
        self._layers.wrappedValue = [attn, moe]
    }

    package func newCache() -> [any KVCache] {
        [KVCacheSimple()]
    }

    package func callAsFunction(inputsEmbeds: MLXArray, hiddenStates: MLXArray, cache: [any KVCache]) -> MLXArray {
        let first = attnBlock(inputsEmbeds: inputsEmbeds, hiddenStates: hiddenStates, cache: cache.first)
        return moeBlock(first)
    }
}

public final class NemotronHMTPDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: NemotronHConfiguration
    public let maximumBlockSize: Int? = 2
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true

    @ModuleInfo(key: "mtp") var mtp: NemotronMTPPredictor

    public init(_ configuration: NemotronHConfiguration, moeLatentSize: Int? = nil) {
        self.configuration = configuration
        self._mtp.wrappedValue = NemotronMTPPredictor(configuration, moeLatentSize: moeLatentSize)
    }

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache())
    }

    public func prepareDrafterState(
        target: any LanguageModel,
        input: MTPPrefillInput,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        guard let (embed, lmHead) = Self.targetHead(target) else { return }
        let prompt = input.promptTokens.ndim == 1 ? input.promptTokens[.newAxis] : input.promptTokens
        guard prompt.dim(-1) > 0 else { return }
        let rows = input.targetHidden[0..., ..<prompt.dim(-1), 0...]
        let hiddenRows = state.carryHidden.map { concatenated([$0, rows], axis: 1) } ?? rows
        let nextTokens = state.carryHidden == nil ? prompt[0..., 1...] : prompt
        let bonusCol = input.firstBonus.map { $0.reshaped(1, 1) }
        let shifted = bonusCol.map { concatenated([nextTokens, $0], axis: 1) } ?? nextTokens
        state.carryHidden = input.firstBonus == nil ? hiddenRows[0..., (-1)..., 0...] : nil
        guard shifted.dim(1) > 0 else { return }
        let hidden = hiddenRows[0..., ..<shifted.dim(1), 0...]
        let mtpHidden = mtp(inputsEmbeds: embed(shifted), hiddenStates: hidden, cache: state.cache)
        state.nextPosition += shifted.dim(1)
        guard input.firstBonus != nil else { return }
        let seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedHidden = seedHidden
        state.seedToken = Self.projectAndSample(seedHidden, embed: embed, lmHead: lmHead, sampler: sampler)
        state.proposalAppended = 0
    }

    public func draftBlock(
        target: any LanguageModel,
        input: MTPDraftInput,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(target: target, input: input, state: &state, sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        input: MTPDraftInput,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        guard let (embed, lmHead) = Self.targetHead(target) else {
            return MLXArray.zeros([1, max(1, input.blockSize - 1)], type: Int32.self)
        }
        if let seed = state.seedToken {
            state.seedToken = nil
            state.seedHidden = nil
            state.proposalAppended = 0
            return seed.reshaped(1, 1)
        }
        let col = input.lastToken.reshaped(1, 1)
        let mtpHidden = mtp(inputsEmbeds: embed(col), hiddenStates: input.lastHidden, cache: state.cache)
        state.proposalAppended = 1
        state.nextPosition += 1
        return Self.projectAndSample(mtpHidden[0..., (-1)..., 0...], embed: embed, lmHead: lmHead, sampler: sampler)
    }

    public func commitDrafterState(
        target: any LanguageModel,
        input: MTPCommitInput,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        guard let (embed, lmHead) = Self.targetHead(target) else { return }
        let accepted = input.acceptedCount
        let keep = min(accepted, state.proposalAppended)
        let trim = state.proposalAppended - keep
        if trim > 0 {
            _ = trimPromptCache(state.cache, numTokens: trim)
            state.nextPosition -= trim
        }
        var tokens: [MLXArray] = []
        var hiddens: [MLXArray] = []
        for idx in keep ..< accepted {
            tokens.append(input.draftTokens[0..., idx ..< (idx + 1)])
            hiddens.append(input.targetHidden[0..., idx ..< (idx + 1), 0...])
        }
        tokens.append(input.finalToken.reshaped(1, 1))
        hiddens.append(input.targetHidden[0..., accepted ..< (accepted + 1), 0...])
        let committedTok = concatenated(tokens, axis: 1)
        let committedHid = concatenated(hiddens, axis: 1)
        let mtpHidden = mtp(inputsEmbeds: embed(committedTok), hiddenStates: committedHid, cache: state.cache)
        state.nextPosition += committedTok.dim(1)
        let seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedHidden = seedHidden
        state.seedToken = Self.projectAndSample(seedHidden, embed: embed, lmHead: lmHead, sampler: sampler)
        state.proposalAppended = 0
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights.filter { $0.key.hasPrefix("mtp.") }
        let prefix = "mtp.layers.1.mixer"
        for (src, dst) in [("up_proj", "fc1"), ("down_proj", "fc2")] {
            for suffix in ["weight", "scales", "biases"] {
                guard out["\(prefix).experts.0.\(src).\(suffix)"] != nil else { continue }
                let slices = (0 ..< configuration.nRoutedExperts).compactMap { idx in
                    out.removeValue(forKey: "\(prefix).experts.\(idx).\(src).\(suffix)")
                }
                if slices.count == configuration.nRoutedExperts {
                    out["\(prefix).switch_mlp.\(dst).\(suffix)"] = MLX.stacked(slices)
                }
            }
        }
        return out
    }

    private static func projectAndSample(
        _ hidden: MLXArray, embed: Embedding, lmHead: Linear?, sampler: any LogitSampler
    ) -> MLXArray {
        let logits = (lmHead.map { $0(hidden) } ?? embed.asLinear(hidden))[0..., -1, 0...]
        return sampler.sample(logits: logits).reshaped(1, 1)
    }

    private static func targetHead(_ target: any LanguageModel) -> (Embedding, Linear?)? {
        if let lightning = target as? NemotronLightningModel {
            return lightning.targetEmbeddingAndHead
        }
        if let stock = target as? NemotronHModel,
           let backbone = stock.children()[unwrapping: "backbone"],
           let emb = backbone.children()[unwrapping: "embeddings"] as? Embedding
        {
            let head = stock.children()[unwrapping: "lm_head"] as? Linear
            return (emb, head)
        }
        return nil
    }
}
