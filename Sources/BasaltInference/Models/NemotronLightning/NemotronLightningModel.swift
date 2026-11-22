import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import BasaltCore

package final class RecordingRMSNorm: RMSNorm {
    var shouldRecord = false
    var lastOutput: MLXArray?

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x)
        if shouldRecord {
            lastOutput = y
        }
        return y
    }
}

package final class NemotronReLU2SwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    package init(inputDims: Int, hiddenDims: Int, numExperts: Int, bias: Bool = false) {
        self._fc1.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._fc2.wrappedValue = SwitchLinear(inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)
    }

    package func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let sorted = indices.size >= 64
        var idx = indices
        var invOrder = MLXArray()
        if sorted {
            (expanded, idx, invOrder) = gatherSort(x: expanded, indices: indices)
        }
        let h1 = fc1(expanded, idx, sortedIndices: sorted)
        let activated = square(maximum(h1, MLXArray(0, dtype: h1.dtype)))
        var out = fc2(activated, idx, sortedIndices: sorted)
        if sorted {
            out = scatterUnsort(x: out, invOrder: invOrder, shape: indices.shape)
        }
        return out.squeezed(axis: -2)
    }
}

package final class NemotronSimpleMLP: Module, UnaryLayer {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    package init(hiddenSize: Int, intermediateSize: Int, bias: Bool = false) {
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let u = upProj(x)
        return downProj(square(maximum(u, MLXArray(0, dtype: u.dtype))))
    }
}

package final class NemotronMoEGateModule: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    package init(_ config: NemotronHConfiguration) {
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self._weight.wrappedValue = MLXArray.zeros([config.nRoutedExperts, config.hiddenSize])
        self._correctionBias.wrappedValue = MLXArray.zeros([config.nRoutedExperts])
    }

    package func callAsFunction(_ x: MLXArray) -> FusedMoERouterOutput {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        if nGroup <= 1 {
            let routed = FlintRouterKernel.route(
                logits,
                expertBias: correctionBias,
                k: topK,
                normTopkProb: topK > 1 && normTopkProb,
                scoreFunction: "sigmoid",
                weightDType: .float32
            )
            return FusedMoERouterOutput(
                indices: routed.indices,
                weights: routed.weights * routedScalingFactor
            )
        }
        let origScores = sigmoid(logits)
        var scores = origScores + correctionBias.asType(.float32)
        let expertsPerGroup = weight.dim(0) / nGroup
        let unflat = scores.reshaped(scores.shape.dropLast() + [nGroup, expertsPerGroup])
        let top2 = MLX.top(unflat, k: min(2, expertsPerGroup), axis: -1)
        let groupScores = sum(top2, axis: -1, keepDims: true)
        let dropCount = nGroup - topkGroup
        if dropCount > 0 {
            let dropIdx = stopGradient(argPartition(groupScores, kth: dropCount - 1, axis: -2)[.ellipsis, ..<dropCount, 0...])
            let masked = putAlong(unflat, dropIdx, values: MLXArray(-Float.infinity), axis: -2)
            scores = masked.reshaped(scores.shape)
        }
        let inds = stopGradient(argSort(-scores, axis: -1)[.ellipsis, ..<topK]).asType(.int32)
        var selected = takeAlong(origScores, inds, axis: -1)
        if topK > 1 && normTopkProb {
            selected = selected / (sum(selected, axis: -1, keepDims: true) + MLXArray(Float(1e-20)))
        }
        return FusedMoERouterOutput(indices: inds, weights: selected * routedScalingFactor)
    }
}

package final class NemotronLatentMoEMixer: Module, UnaryLayer {
    let moeLatentSize: Int?
    @ModuleInfo(key: "gate") var gate: NemotronMoEGateModule
    @ModuleInfo(key: "fc1_latent_proj") var fc1LatentProj: Linear?
    @ModuleInfo(key: "fc2_latent_proj") var fc2LatentProj: Linear?
    @ModuleInfo(key: "switch_mlp") var switchMLP: NemotronReLU2SwitchMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronSimpleMLP?

    package init(_ config: NemotronHConfiguration, moeLatentSize: Int?) {
        self.moeLatentSize = moeLatentSize
        self._gate.wrappedValue = NemotronMoEGateModule(config)
        let expertDim = moeLatentSize ?? config.hiddenSize
        if let moeLatentSize {
            self._fc1LatentProj.wrappedValue = Linear(config.hiddenSize, moeLatentSize, bias: config.mlpBias)
            self._fc2LatentProj.wrappedValue = Linear(moeLatentSize, config.hiddenSize, bias: config.mlpBias)
        }
        self._switchMLP.wrappedValue = NemotronReLU2SwitchMLP(
            inputDims: expertDim,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            bias: config.useBias
        )
        if let shared = config.nSharedExperts, shared > 0, config.moeSharedExpertIntermediateSize > 0 {
            self._sharedExperts.wrappedValue = NemotronSimpleMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeSharedExpertIntermediateSize,
                bias: config.mlpBias
            )
        }
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residuals = x
        let routed = gate(x)
        let latentIn = fc1LatentProj.map { $0(x) } ?? x
        let expertOut = switchMLP(latentIn, routed.indices)
        var y = (expertOut.asType(.float32) * routed.weights.expandedDimensions(axis: -1))
            .sum(axis: -2)
            .asType(expertOut.dtype)
        if let fc2LatentProj {
            y = fc2LatentProj(y)
        }
        if let sharedExperts {
            y = y + sharedExperts(residuals)
        }
        return y
    }
}

public final class NemotronLightningModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public static let modelType = "nemotron_h"
    public static let aliasModelType = "nemotron_lightning"

    private struct Backing {
        let model: NemotronHModel
        let recorder: RecordingRMSNorm
    }

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: NemotronHConfiguration
    package let moeLatentSize: Int?

    private let backing: Backing

    @ModuleInfo(key: "backbone") var backbone: Module
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(configurationData data: Data) throws {
        let normalized = try Self.normalizeConfigurationData(data)
        let decoded = try JSONDecoder.json5().decode(NemotronHConfiguration.self, from: normalized.data)
        self.configuration = decoded
        self.moeLatentSize = normalized.moeLatentSize
        self.vocabularySize = decoded.vocabSize

        let base = NemotronHModel(decoded)
        self.kvHeads = base.kvHeads

        let topChildren = base.children()
        guard let backboneMod = topChildren[unwrapping: "backbone"] else {
            throw BasaltError.modelConfigurationInvalid(path: decoded.modelType, reason: "NemotronHModel missing backbone module")
        }
        let recorder = RecordingRMSNorm(dimensions: decoded.hiddenSize, eps: decoded.layerNormEpsilon)
        backboneMod.update(modules: ModuleChildren.unflattened([("norm_f", recorder)]))
        self.backing = Backing(model: base, recorder: recorder)
        self._backbone.wrappedValue = backboneMod
        if let head = topChildren[unwrapping: "lm_head"] as? Linear {
            self._lmHead.wrappedValue = head
        }
    }

    public var loraLayers: [Module] {
        backing.model.loraLayers
    }

    public var toolCallFormat: ToolCallFormat? {
        backing.model.toolCallFormat
    }

    public var castPredicate: ((String) -> Bool)? {
        backing.model.castPredicate
    }

    package func embedTokens(_ inputs: MLXArray) -> MLXArray {
        backing.model.embedTokens(inputs)
    }

    package var targetEmbeddingAndHead: (Embedding, Linear?)? {
        guard let emb = backbone.children()[unwrapping: "embeddings"] as? Embedding else { return nil }
        return (emb, lmHead)
    }

    private func syncBackingTopModules() {
        let current = backing.model.children()
        var updates: [(String, Module)] = []
        if current[unwrapping: "backbone"] !== backbone {
            updates.append(("backbone", backbone))
        }
        if let lmHead, current[unwrapping: "lm_head"] !== lmHead {
            updates.append(("lm_head", lmHead))
        }
        if !updates.isEmpty {
            backing.model.update(modules: ModuleChildren.unflattened(updates))
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        syncBackingTopModules()
        return backing.model(inputs, cache: cache)
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        syncBackingTopModules()
        let emit = state?[mtpEmitFlagKey] ?? false
        backing.recorder.shouldRecord = emit
        backing.recorder.lastOutput = nil
        defer {
            backing.recorder.shouldRecord = false
            backing.recorder.lastOutput = nil
        }
        let logits = backing.model(input.tokens, cache: cache)
        guard emit, let hidden = backing.recorder.lastOutput else {
            return LMOutput(logits: logits)
        }
        var outState = state ?? LMOutput.State()
        outState[mtpLastHiddenStatesKey] = hidden
        return LMOutput(logits: logits, state: outState)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try backing.model.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let stripped = weights.filter { !$0.key.hasPrefix("mtp.") }
        return backing.model.sanitize(weights: stripped)
    }

    static func normalizeConfigurationData(_ data: Data) throws -> (data: Data, moeLatentSize: Int?) {
        guard var json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return (data, nil)
        }
        let latent = (json["moe_latent_size"] as? Int).flatMap { $0 > 0 ? $0 : nil }
        if json["hybrid_override_pattern"] == nil,
           let blocks = (json["layers_block_type"] as? [String]) ?? (json["layer_types"] as? [String])
        {
            var pattern = ""
            pattern.reserveCapacity(blocks.count)
            for block in blocks {
                switch block {
                case "mamba", "linear_attention", "conv": pattern.append("M")
                case "attention", "full_attention": pattern.append("*")
                case "mlp": pattern.append("-")
                case "moe": pattern.append("E")
                default:
                    throw BasaltError.modelConfigurationInvalid(
                        path: "nemotron_h",
                        reason: "unsupported NemotronH block type '\(block)'"
                    )
                }
            }
            json["hybrid_override_pattern"] = pattern
            if json["num_hidden_layers"] == nil {
                json["num_hidden_layers"] = blocks.count
            }
            let rewritten = try JSONSerialization.data(withJSONObject: json)
            return (rewritten, latent)
        }
        return (data, latent)
    }
}
