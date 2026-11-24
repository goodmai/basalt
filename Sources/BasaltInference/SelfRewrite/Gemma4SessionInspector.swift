import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct EditEditableLayerSummary: Sendable, Codable, Equatable {
    public let path: String
    public let shape: [Int]
    public let format: String
    public let blastRadiusWarning: String?

    public init(path: String, shape: [Int], format: String, blastRadiusWarning: String?) {
        self.path = path
        self.shape = shape
        self.format = format
        self.blastRadiusWarning = blastRadiusWarning
    }
}

public struct EditModelSummary: Sendable, Codable, Equatable {
    public let modelType: String
    public let topology: Gemma4EditTopology?
    public let editableLayers: [EditEditableLayerSummary]
    public let totalParameterCount: Int

    public init(
        modelType: String,
        topology: Gemma4EditTopology?,
        editableLayers: [EditEditableLayerSummary],
        totalParameterCount: Int
    ) {
        self.modelType = modelType
        self.topology = topology
        self.editableLayers = editableLayers
        self.totalParameterCount = totalParameterCount
    }
}

public struct EditTensorInspection: Sendable, Codable, Equatable {
    public let info: TensorInfo
    public let stats: TensorStatsSummary
    public let blastRadiusWarning: String?
    public let forbiddenKind: EditForbiddenTargetKind?
    public let isScale: Bool

    public init(
        info: TensorInfo,
        stats: TensorStatsSummary,
        blastRadiusWarning: String?,
        forbiddenKind: EditForbiddenTargetKind?,
        isScale: Bool
    ) {
        self.info = info
        self.stats = stats
        self.blastRadiusWarning = blastRadiusWarning
        self.forbiddenKind = forbiddenKind
        self.isScale = isScale
    }
}

public struct ScalarRangeSummary: Sendable, Codable, Equatable {
    public let minValue: Float
    public let maxValue: Float
    public let mean: Float

    public init(minValue: Float, maxValue: Float, mean: Float) {
        self.minValue = minValue
        self.maxValue = maxValue
        self.mean = mean
    }
}

public struct MoERoutingDistribution: Sendable, Codable, Equatable {
    public let activeExpertCount: Int
    public let normalizedEntropy: Double
    public let topExperts: [Int]
    public let topExpertCounts: [Int]

    public init(
        activeExpertCount: Int,
        normalizedEntropy: Double,
        topExperts: [Int],
        topExpertCounts: [Int]
    ) {
        self.activeExpertCount = activeExpertCount
        self.normalizedEntropy = normalizedEntropy
        self.topExperts = topExperts
        self.topExpertCounts = topExpertCounts
    }
}

public struct MoERouterLayerSummary: Sendable, Codable, Equatable {
    public let layer: Int
    public let numExperts: Int
    public let tokensObserved: Int
    public let distribution: MoERoutingDistribution
    public let perExpertScale: ScalarRangeSummary?

    public init(
        layer: Int,
        numExperts: Int,
        tokensObserved: Int,
        distribution: MoERoutingDistribution,
        perExpertScale: ScalarRangeSummary?
    ) {
        self.layer = layer
        self.numExperts = numExperts
        self.tokensObserved = tokensObserved
        self.distribution = distribution
        self.perExpertScale = perExpertScale
    }
}

public enum Gemma4SessionInspector {
    public static func describeForEditing(
        model: Module,
        config: [String: Any]? = nil,
        scope: EditTargetScope = .textBackboneOnly
    ) -> EditModelSummary {
        let topology = config.flatMap { Gemma4EditTopology(config: $0) }
        var editable: [EditEditableLayerSummary] = []
        for (path, module) in model.namedModules() where !path.isEmpty {
            let linear: Linear? = if let lora = module as? LoRALayer {
                lora.reverted() as? Linear
            } else {
                module as? Linear
            }
            guard let base = linear, EditOpExecutor.supportsRankOne(base) else { continue }
            guard EditProgramValidator.forbiddenTargetKind(for: path, scope: scope) == nil else { continue }
            let (outputs, inputs) = base.shape
            editable.append(EditEditableLayerSummary(
                path: path,
                shape: [outputs, inputs],
                format: TensorPacker.describe(base),
                blastRadiusWarning: topology?.blastRadiusWarning(for: path)
            ))
        }
        editable.sort { $0.path < $1.path }
        return EditModelSummary(
            modelType: String(describing: type(of: model)),
            topology: topology,
            editableLayers: editable,
            totalParameterCount: model.logicalParameterCount
        )
    }

    public static func inspectTensor(
        _ name: String,
        in model: Module,
        topology: Gemma4EditTopology? = nil,
        scope: EditTargetScope = .textBackboneOnly
    ) throws -> EditTensorInspection {
        let report = try MLXInferenceEngine.report(TensorRequest(name: name, maxValues: 0), model: model)
        let stats = TensorStatsSummary(
            mean: report.mean,
            std: report.std,
            minValue: report.min,
            maxValue: report.max,
            l2Norm: report.l2
        )
        return EditTensorInspection(
            info: report.info,
            stats: stats,
            blastRadiusWarning: topology?.blastRadiusWarning(for: name),
            forbiddenKind: EditProgramValidator.forbiddenTargetKind(for: name, scope: scope),
            isScale: EditProgramValidator.isScaleTensor(name)
        )
    }

    public static func summarizeMoERouter(
        model: any LanguageModel,
        tokens: [Int32],
        layers: Set<Int>? = nil
    ) -> [MoERouterLayerSummary] {
        guard !tokens.isEmpty else { return [] }
        let sites = MoERWPipeline.discoverRouters(in: model)
        let probe = RouterHistogramProbe(layerFilter: layers)
        let input = MLXArray(tokens)[.newAxis]
        let out: MLXArray
        if let probable = model as? any BasaltProbableModel {
            out = probable.forwardWithProbe(input, cache: nil, probe: probe)
        } else {
            out = withForwardProbe(probe) { model(LMInput.Text(tokens: input), cache: nil, state: nil).logits }
        }
        eval(out)
        let observedLayers = Set(probe.counts.keys).union(sites.keys).filter { layers?.contains($0) ?? true }.sorted()
        return observedLayers.compactMap { layer -> MoERouterLayerSummary? in
            let site = sites[layer]
            let counts = probe.counts[layer] ?? [:]
            let numExperts = max(site?.numExperts ?? 0, (counts.keys.max() ?? -1) + 1)
            guard numExperts > 0 else { return nil }
            return buildRouterSummary(
                layer: layer,
                numExperts: numExperts,
                counts: counts,
                tokensObserved: probe.tokenCounts[layer] ?? 0,
                scale: site?.expertScale
            )
        }
    }

    private static func buildRouterSummary(
        layer: Int,
        numExperts: Int,
        counts: [Int: Int],
        tokensObserved: Int,
        scale: MLXArray?
    ) -> MoERouterLayerSummary {
        let totalSelections = counts.values.reduce(0, +)
        let entropy: Double
        if totalSelections > 0, numExperts > 1 {
            let raw = counts.values.reduce(0.0) { acc, c in
                let p = Double(c) / Double(totalSelections)
                return p > 0 ? acc - p * log(p) : acc
            }
            entropy = raw / log(Double(numExperts))
        } else {
            entropy = 0.0
        }
        let ranked = counts.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }.prefix(5)
        let distribution = MoERoutingDistribution(
            activeExpertCount: counts.count,
            normalizedEntropy: entropy,
            topExperts: ranked.map(\.key),
            topExpertCounts: ranked.map(\.value)
        )
        return MoERouterLayerSummary(
            layer: layer,
            numExperts: numExperts,
            tokensObserved: tokensObserved,
            distribution: distribution,
            perExpertScale: scale.map(perExpertScaleStats)
        )
    }

    private static func perExpertScaleStats(_ array: MLXArray) -> ScalarRangeSummary {
        eval(array)
        let vals = array.asType(.float32).flattened().asArray(Float.self)
        guard !vals.isEmpty else { return ScalarRangeSummary(minValue: 0, maxValue: 0, mean: 0) }
        let lo = vals.min() ?? 0
        let hi = vals.max() ?? 0
        let avg = vals.reduce(Float(0), +) / Float(vals.count)
        return ScalarRangeSummary(minValue: lo, maxValue: hi, mean: avg)
    }
}

private final class RouterHistogramProbe: ForwardProbe {
    let layerFilter: Set<Int>?
    var counts: [Int: [Int: Int]] = [:]
    var tokenCounts: [Int: Int] = [:]

    init(layerFilter: Set<Int>?) {
        self.layerFilter = layerFilter
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray { hidden }
    func attentionWeights(layer: Int, _ weights: MLXArray) {}
    func wantsAttention(layer: Int) -> Bool { false }
    func headMask(layer: Int) -> MLXArray? { nil }
    func routerLogits(layer: Int, _ logits: MLXArray) {}
    func expertMask(layer: Int) -> MLXArray? { nil }

    func routerTopK(layer: Int, indices: MLXArray, gates: MLXArray) {
        if let layerFilter, !layerFilter.contains(layer) { return }
        eval(indices)
        let topK = max(1, indices.dim(-1))
        tokenCounts[layer, default: 0] += indices.size / topK
        var layerCounts = counts[layer] ?? [:]
        for idx in indices.asType(.int32).flattened().asArray(Int32.self) {
            layerCounts[Int(idx), default: 0] += 1
        }
        counts[layer] = layerCounts
    }
}
