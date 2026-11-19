import Foundation

public struct InfluentialHeadRef: Sendable, Codable, Equatable {
    public var layer: Int
    public var head: Int
    public var influenceScore: Float
    public var residualShare: Float
    public var logitAlignment: Float

    public init(layer: Int, head: Int, influenceScore: Float, residualShare: Float, logitAlignment: Float) {
        self.layer = layer
        self.head = head
        self.influenceScore = influenceScore
        self.residualShare = residualShare
        self.logitAlignment = logitAlignment
    }

    enum CodingKeys: String, CodingKey {
        case layer, head
        case influenceScore = "influence_score"
        case residualShare = "residual_share"
        case logitAlignment = "logit_alignment"
    }
}

public struct LayerInfluenceSummary: Sendable, Codable, Equatable {
    public var layer: Int
    public var layerInfluence: Float
    public var dominantHead: Int?
    public var headConcentration: Float

    public init(layer: Int, layerInfluence: Float, dominantHead: Int?, headConcentration: Float) {
        self.layer = layer
        self.layerInfluence = layerInfluence
        self.dominantHead = dominantHead
        self.headConcentration = headConcentration
    }

    enum CodingKeys: String, CodingKey {
        case layer
        case layerInfluence = "layer_influence"
        case dominantHead = "dominant_head"
        case headConcentration = "head_concentration"
    }
}

public struct NetworkInfluenceReport: Sendable, Codable, Equatable {
    public static let defaultAnomalyThreshold: Float = 0.65

    public var layers: [LayerInfluenceSummary]
    public var topInfluentialHeads: [InfluentialHeadRef]
    public var injectionAnomalyScore: Float
    public var anomalyFlagged: Bool

    public init(
        layers: [LayerInfluenceSummary],
        topInfluentialHeads: [InfluentialHeadRef],
        injectionAnomalyScore: Float,
        anomalyThreshold: Float = NetworkInfluenceReport.defaultAnomalyThreshold
    ) {
        let clamped = min(1.0, max(0.0, injectionAnomalyScore.isFinite ? injectionAnomalyScore : 0.0))
        self.layers = layers
        self.topInfluentialHeads = topInfluentialHeads
        self.injectionAnomalyScore = clamped
        self.anomalyFlagged = clamped >= anomalyThreshold
    }

    enum CodingKeys: String, CodingKey {
        case layers
        case topInfluentialHeads = "top_influential_heads"
        case injectionAnomalyScore = "injection_anomaly_score"
        case anomalyFlagged = "anomaly_flagged"
    }
}

public enum HeadInfluenceCore {
    public static let residualWeight: Float = 0.55
    public static let logitWeight: Float = 0.45

    public static func enrich(
        pathways: [TokenLayerPathway],
        sampledId: Int,
        finalProbability: Float
    ) -> [TokenLayerPathway] {
        guard !pathways.isEmpty else { return [] }
        let ordered = pathways.sorted { $0.layer < $1.layer }
        var previousProb: Float = 0.0
        var result: [TokenLayerPathway] = []
        result.reserveCapacity(ordered.count)
        for var pathway in ordered {
            let currentProb = resolvedTargetProbability(
                pathway: pathway, sampledId: sampledId, fallback: finalProbability
            )
            let delta = max(0.0, currentProb - previousProb)
            let alignment = min(1.0, max(0.0, 0.5 * currentProb + 0.5 * min(1.0, delta * 2.0)))
            let scored = scoreLayerHeads(pathway.topHeads, logitAlignment: alignment)
            pathway.topHeads = scored.heads
            pathway.layerInfluence = scored.layerInfluence
            result.append(pathway)
            previousProb = currentProb
        }
        return result
    }

    public static func summarizeNetwork(
        tokens: [OutputTokenTelemetry],
        topK: Int = 6,
        anomalyThreshold: Float = NetworkInfluenceReport.defaultAnomalyThreshold
    ) -> NetworkInfluenceReport? {
        let allPathways = tokens.compactMap(\.layers).flatMap { $0 }
        guard !allPathways.isEmpty else { return nil }
        let grouped = Dictionary(grouping: allPathways, by: \.layer)
        var layerSummaries: [LayerInfluenceSummary] = []
        var headRefs: [InfluentialHeadRef] = []
        for layer in grouped.keys.sorted() {
            let entries = grouped[layer] ?? []
            let summary = aggregateLayer(layer: layer, entries: entries, into: &headRefs)
            layerSummaries.append(summary)
        }
        let sortedHeads = headRefs
            .sorted { $0.influenceScore > $1.influenceScore }
            .prefix(max(1, topK))
        let anomaly = computeAnomalyScore(tokens: tokens, layers: layerSummaries, headRefs: headRefs)
        return NetworkInfluenceReport(
            layers: layerSummaries,
            topInfluentialHeads: Array(sortedHeads),
            injectionAnomalyScore: anomaly,
            anomalyThreshold: anomalyThreshold
        )
    }

    private static func resolvedTargetProbability(
        pathway: TokenLayerPathway, sampledId: Int, fallback: Float
    ) -> Float {
        guard let lensId = pathway.lensTopTokenId, let lensProb = pathway.lensTopProbability else {
            return min(1.0, max(0.0, fallback.isFinite ? fallback : 0.0))
        }
        let clean = min(1.0, max(0.0, lensProb.isFinite ? lensProb : 0.0))
        return lensId == sampledId ? clean : (1.0 - clean) * 0.25
    }

    private static func scoreLayerHeads(
        _ heads: [TokenHeadActivation],
        logitAlignment: Float
    ) -> (heads: [TokenHeadActivation], layerInfluence: Float) {
        guard !heads.isEmpty else { return ([], logitAlignment * logitWeight) }
        let totalNorm = heads.reduce(Float(0.0)) { $0 + max(0.0, $1.norm.isFinite ? $1.norm : 0.0) }
        let denom = totalNorm > 1e-6 ? totalNorm : Float(heads.count)
        var updated: [TokenHeadActivation] = []
        updated.reserveCapacity(heads.count)
        var peakInfluence: Float = 0.0
        for var head in heads {
            let safeNorm = max(0.0, head.norm.isFinite ? head.norm : 0.0)
            let share = totalNorm > 1e-6 ? min(1.0, safeNorm / denom) : 1.0 / denom
            let entFactor = entropyAttenuation(head.attentionEntropy)
            let aligned = min(1.0, max(0.0, logitAlignment * entFactor))
            let combined = min(1.0, max(0.0, residualWeight * share + logitWeight * aligned))
            head.residualShare = share
            head.logitAlignment = aligned
            head.influenceScore = combined
            peakInfluence = max(peakInfluence, combined)
            updated.append(head)
        }
        updated.sort { ($0.influenceScore ?? 0) > ($1.influenceScore ?? 0) }
        return (updated, peakInfluence)
    }

    private static func entropyAttenuation(_ attentionEntropy: Float?) -> Float {
        guard let ent = attentionEntropy, ent.isFinite else { return 1.0 }
        let clamped = min(4.0, max(0.0, ent))
        return 1.0 - 0.35 * (clamped / 4.0)
    }

    private static func aggregateLayer(
        layer: Int,
        entries: [TokenLayerPathway],
        into headRefs: inout [InfluentialHeadRef]
    ) -> LayerInfluenceSummary {
        var byHead: [Int: (score: Float, res: Float, logit: Float, count: Int)] = [:]
        var layerSum: Float = 0.0
        for entry in entries {
            layerSum += entry.layerInfluence ?? 0.0
            for h in entry.topHeads {
                let s = h.influenceScore ?? 0.0
                let r = h.residualShare ?? 0.0
                let l = h.logitAlignment ?? 0.0
                var cur = byHead[h.head] ?? (0, 0, 0, 0)
                cur.score += s
                cur.res += r
                cur.logit += l
                cur.count += 1
                byHead[h.head] = cur
            }
        }
        var shares: [Float] = []
        var dominant: (head: Int, score: Float)?
        for (headId, agg) in byHead {
            let n = Float(max(1, agg.count))
            let meanS = agg.score / n
            let meanR = agg.res / n
            let meanL = agg.logit / n
            shares.append(meanR)
            headRefs.append(InfluentialHeadRef(
                layer: layer, head: headId, influenceScore: meanS, residualShare: meanR, logitAlignment: meanL
            ))
            if dominant == nil || meanS > (dominant?.score ?? -1) {
                dominant = (headId, meanS)
            }
        }
        let hhi = shares.reduce(Float(0.0)) { $0 + $1 * $1 }
        return LayerInfluenceSummary(
            layer: layer,
            layerInfluence: layerSum / Float(max(1, entries.count)),
            dominantHead: dominant?.head,
            headConcentration: min(1.0, max(0.0, hhi))
        )
    }

    private static func computeAnomalyScore(
        tokens: [OutputTokenTelemetry],
        layers: [LayerInfluenceSummary],
        headRefs: [InfluentialHeadRef]
    ) -> Float {
        let maxConc = layers.map(\.headConcentration).max() ?? 0.0
        let mismatch = headRefs.map { max(0.0, $0.residualShare - $0.logitAlignment) }.max() ?? 0.0
        let meanEnt = tokens.isEmpty
            ? Float(0.0)
            : tokens.reduce(Float(0.0)) { $0 + min(1.0, max(0.0, $1.entropy / 4.0)) } / Float(tokens.count)
        let raw = 0.40 * maxConc + 0.35 * mismatch + 0.25 * meanEnt
        return min(1.0, max(0.0, raw))
    }
}
