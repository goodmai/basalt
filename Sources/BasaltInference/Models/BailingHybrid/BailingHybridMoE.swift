import Foundation
import MLX
import MLXLMCommon
import MLXNN

package enum BailingHybridRouter {
    package struct Config: Sendable, Equatable {
        package let topK: Int
        package let nGroup: Int
        package let topkGroup: Int
        package let normTopkProb: Bool
        package let routedScalingFactor: Float
        package let enableExpertBias: Bool

        package init(
            topK: Int,
            nGroup: Int,
            topkGroup: Int,
            normTopkProb: Bool,
            routedScalingFactor: Float,
            enableExpertBias: Bool
        ) {
            self.topK = topK
            self.nGroup = nGroup
            self.topkGroup = topkGroup
            self.normTopkProb = normTopkProb
            self.routedScalingFactor = routedScalingFactor
            self.enableExpertBias = enableExpertBias
        }

        package init(_ config: BailingHybridConfiguration) {
            self.init(
                topK: config.numExpertsPerTok,
                nGroup: config.nGroup,
                topkGroup: config.topkGroup,
                normTopkProb: config.normTopkProb,
                routedScalingFactor: config.routedScalingFactor,
                enableExpertBias: config.moeRouterEnableExpertBias
            )
        }
    }

    package static func route(
        logits: MLXArray,
        expertBias: MLXArray?,
        config: Config,
        expertMask: MLXArray? = nil
    ) -> (indices: MLXArray, weights: MLXArray) {
        var scores = sigmoid(logits.asType(.float32))
        var selection = config.enableExpertBias
            ? (expertBias.map { scores + $0.asType(.float32) } ?? scores)
            : scores
        if let expertMask {
            let keep = expertMask .> 0
            scores = MLX.where(keep, scores, MLXArray(Float(0), dtype: scores.dtype))
            selection = MLX.where(keep, selection, MLXArray(-Float.infinity, dtype: selection.dtype))
        }

        let dropGroups = config.nGroup - config.topkGroup
        if dropGroups > 0 {
            let numExperts = selection.dim(-1)
            let expertsPerGroup = numExperts / config.nGroup
            let prefixShape = Array(selection.shape.dropLast())
            let grouped = selection.reshaped(prefixShape + [config.nGroup, expertsPerGroup])
            let groupScores = MLX.top(grouped, k: min(2, expertsPerGroup), axis: -1).sum(axis: -1, keepDims: true)
            let dropIdx = stopGradient(
                argPartition(groupScores, kth: dropGroups - 1, axis: -2)[.ellipsis, ..<dropGroups, 0...]
            )
            let expandedDrop = broadcast(dropIdx, to: prefixShape + [dropGroups, expertsPerGroup])
            let masked = putAlong(
                grouped,
                expandedDrop,
                values: MLXArray(-Float.infinity, dtype: grouped.dtype),
                axis: -2
            )
            selection = masked.reshaped(selection.shape)
        }

        let indices = stopGradient(argSort(-selection, axis: -1)[.ellipsis, ..<config.topK]).asType(.int32)
        var weights = takeAlong(scores, indices, axis: -1)
        if config.topK > 1, config.normTopkProb {
            let denominator = weights.sum(axis: -1, keepDims: true) + MLXArray(Float(1e-20))
            weights = weights / denominator
        }
        weights = weights * config.routedScalingFactor
        return (indices, weights)
    }
}

package final class BailingHybridMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    package init(hiddenSize: Int, intermediateSize: Int, bias: Bool = false) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

package final class BailingHybridGate: Module {
    let routerConfig: BailingHybridRouter.Config

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray?

    package init(_ config: BailingHybridConfiguration) {
        self.routerConfig = BailingHybridRouter.Config(config)
        self._weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize], dtype: .float32)
        if config.moeRouterEnableExpertBias {
            self._expertBias.wrappedValue = MLXArray.zeros([config.numExperts], dtype: .float32)
        }
    }

    package func callAsFunction(
        _ x: MLXArray,
        layerIdx: Int = 0,
        probe: (any ForwardProbe)? = nil
    ) -> (indices: MLXArray, weights: MLXArray) {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        probe?.routerLogits(layer: layerIdx, logits)
        let routed = BailingHybridRouter.route(
            logits: logits,
            expertBias: expertBias,
            config: routerConfig,
            expertMask: probe?.expertMask(layer: layerIdx)
        )
        probe?.routerTopK(layer: layerIdx, indices: routed.indices, gates: routed.weights)
        return routed
    }
}

package final class BailingHybridSparseMoE: Module, UnaryLayer {
    let layerIdx: Int

    @ModuleInfo(key: "gate") var gate: BailingHybridGate
    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingHybridMLP?

    package init(_ config: BailingHybridConfiguration, layerIdx: Int = 0) {
        self.layerIdx = layerIdx
        self._gate.wrappedValue = BailingHybridGate(config)
        self._experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            bias: config.useBias
        )
        if config.numSharedExperts > 0 {
            self._sharedExperts.wrappedValue = BailingHybridMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.sharedExpertIntermediateSize,
                bias: config.useBias
            )
        }
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, probe: nil)
    }

    package func callAsFunction(_ x: MLXArray, probe: (any ForwardProbe)?) -> MLXArray {
        let (indices, weights) = gate(x, layerIdx: layerIdx, probe: probe)
        let routed = experts(x, indices)
        var out = (routed.asType(.float32) * weights.expandedDimensions(axis: -1))
            .sum(axis: -2)
            .asType(x.dtype)
        if let sharedExperts {
            out = out + sharedExperts(x)
        }
        return out
    }
}
