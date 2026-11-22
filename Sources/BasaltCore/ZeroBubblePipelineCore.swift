import Foundation

public enum ClusterTopologyMode: String, Sendable, Codable, Equatable, CaseIterable {
    case auto
    case ddp
    case pipeline
    case diloco
}

public enum ClusterCompressionMode: String, Sendable, Codable, Equatable, CaseIterable {
    case none
    case fp8
    case powersgd
    case qjl1
}

public struct ClusterOrchestrationSpec: Sendable, Codable, Equatable {
    public var peers: [String]
    public var topology: ClusterTopologyMode
    public var compression: ClusterCompressionMode
    public var maxMemoryFraction: Double

    public init(
        peers: [String],
        topology: ClusterTopologyMode = .auto,
        compression: ClusterCompressionMode = .none,
        maxMemoryFraction: Double = 0.82
    ) {
        self.peers = peers
        self.topology = topology
        self.compression = compression
        self.maxMemoryFraction = maxMemoryFraction
    }

    public func validate() -> [String] {
        var issues: [String] = []
        if peers.isEmpty {
            issues.append("cluster.peers must declare at least one peer endpoint")
        }
        if Set(peers).count != peers.count {
            issues.append("cluster.peers contains duplicate endpoints")
        }
        for peer in peers {
            let trimmed = peer.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty || trimmed.contains("0.0.0.0") || trimmed.contains("[::]") {
                issues.append("cluster.peers rejects empty or wildcard bind host '\(peer)'")
            }
        }
        if !(0.20 ... 0.92).contains(maxMemoryFraction) {
            issues.append("cluster.max_memory_fraction must lie within 0.20...0.92")
        }
        return issues
    }

    public static func renderPrometheusMetrics(
        activeRanks: Int,
        linkGbps: Double,
        bubbleRatio: Double
    ) -> String {
        [
            "# TYPE basalt_cluster_ranks_active gauge",
            "basalt_cluster_ranks_active \(max(0, activeRanks))",
            "# TYPE basalt_cluster_link_gbps gauge",
            String(format: "basalt_cluster_link_gbps %.2f", max(0.0, linkGbps)),
            "# TYPE basalt_cluster_bubble_ratio gauge",
            String(format: "basalt_cluster_bubble_ratio %.4f", max(0.0, bubbleRatio)),
        ].joined(separator: "\n") + "\n"
    }
}

public struct StagePartitionResult: Sendable, Equatable {
    public let stageRanges: [Range<Int>]
    public let stageBytes: [Int64]
    public let fitsSafetyCeiling: Bool
    public let maxUtilization: Double
}

public enum LayerStagePartitioner {
    public static let defaultSafetyCeiling = 0.82

    public static func partition(
        layerMemoryBytes: [Int64],
        peerRAMBytes: [Int64],
        safetyFraction: Double = defaultSafetyCeiling
    ) -> StagePartitionResult {
        let numLayers = layerMemoryBytes.count
        let stages = peerRAMBytes.count
        guard numLayers > 0, stages > 0, numLayers >= stages, safetyFraction.isFinite, safetyFraction > 0 else {
            return StagePartitionResult(stageRanges: [], stageBytes: [], fitsSafetyCeiling: false, maxUtilization: .infinity)
        }
        var prefix = [Int64](repeating: 0, count: numLayers + 1)
        for idx in 0 ..< numLayers {
            prefix[idx + 1] = prefix[idx] + max(1, layerMemoryBytes[idx])
        }
        let budgets = peerRAMBytes.map { Double(max(1, $0)) * safetyFraction }
        var dp = Array(repeating: Array(repeating: Double.infinity, count: numLayers + 1), count: stages + 1)
        var splitAt = Array(repeating: Array(repeating: 0, count: numLayers + 1), count: stages + 1)
        dp[0][0] = 0.0

        for s in 1 ... stages {
            for end in s ... (numLayers - (stages - s)) {
                for start in (s - 1) ..< end {
                    let segBytes = Double(prefix[end] - prefix[start])
                    let util = segBytes / budgets[s - 1]
                    let candidate = max(dp[s - 1][start], util)
                    if candidate < dp[s][end] {
                        dp[s][end] = candidate
                        splitAt[s][end] = start
                    }
                }
            }
        }
        var ranges = [Range<Int>](repeating: 0 ..< 0, count: stages)
        var bytes = [Int64](repeating: 0, count: stages)
        var cursor = numLayers
        for s in stride(from: stages, through: 1, by: -1) {
            let prev = splitAt[s][cursor]
            ranges[s - 1] = prev ..< cursor
            bytes[s - 1] = prefix[cursor] - prefix[prev]
            cursor = prev
        }
        let maxUtil = dp[stages][numLayers]
        return StagePartitionResult(
            stageRanges: ranges,
            stageBytes: bytes,
            fitsSafetyCeiling: maxUtil <= 1.000001,
            maxUtilization: maxUtil
        )
    }
}

public enum ZBH1OpKind: String, Sendable, Codable, Equatable {
    case forward = "F"
    case backwardInputB = "B"
    case backwardWeightW = "W"
}

public struct ZBH1SlotOp: Sendable, Codable, Equatable {
    public let stage: Int
    public let microbatch: Int
    public let kind: ZBH1OpKind
    public let clockTick: Int
}

public struct ZBH1Schedule: Sendable, Equatable {
    public let numStages: Int
    public let numMicrobatches: Int
    public let perStageOps: [[ZBH1SlotOp]]
    public let makespanTicks: Int
    public let zbBubbleRatio: Double
    public let standard1F1BBubbleRatio: Double
}

public enum ZBH1SchedulePlanner {
    private struct GridState {
        let p: Int
        let m: Int
        var fDone: [[Bool]]
        var bDone: [[Bool]]
        var wDone: [[Bool]]
    }

    public static func build(numStages: Int, numMicrobatches: Int) -> ZBH1Schedule {
        let p = max(1, numStages)
        let m = max(p, numMicrobatches)
        var state = GridState(
            p: p,
            m: m,
            fDone: Array(repeating: Array(repeating: false, count: m), count: p),
            bDone: Array(repeating: Array(repeating: false, count: m), count: p),
            wDone: Array(repeating: Array(repeating: false, count: m), count: p)
        )
        var perStage: [[ZBH1SlotOp]] = Array(repeating: [], count: p)
        let totalOps = p * m * 3
        var completed = 0
        var tick = 0

        while completed < totalOps {
            let chosen = (0 ..< p).compactMap { selectStageOp(stage: $0, state: state) }
            for op in chosen {
                applyOp(op, to: &state)
                perStage[op.stage].append(ZBH1SlotOp(stage: op.stage, microbatch: op.mb, kind: op.kind, clockTick: tick))
                completed += 1
            }
            tick += 1
        }
        let idealTicks = Double(m * 3)
        let zbBubble = max(0.0, (Double(tick) - idealTicks) / Double(max(1, tick)))
        let std1F1BMakespan = Double(3 * (m + p - 1))
        let stdBubble = max(0.0, (std1F1BMakespan - idealTicks) / max(1.0, std1F1BMakespan))
        return ZBH1Schedule(
            numStages: p,
            numMicrobatches: m,
            perStageOps: perStage,
            makespanTicks: tick,
            zbBubbleRatio: zbBubble,
            standard1F1BBubbleRatio: stdBubble
        )
    }

    private static func selectStageOp(
        stage s: Int,
        state: GridState
    ) -> (stage: Int, mb: Int, kind: ZBH1OpKind)? {
        let readyB = (0 ..< state.m).first { mb in
            !state.bDone[s][mb] && state.fDone[s][mb] && (s == state.p - 1 || state.bDone[s + 1][mb])
        }
        let readyF = (0 ..< state.m).first { mb in
            !state.fDone[s][mb] && (s == 0 || state.fDone[s - 1][mb])
        }
        if let bMB = readyB {
            let warmupLimit = min(state.m, 2 * (state.p - 1 - s) + 1)
            let fCount = state.fDone[s].filter { $0 }.count
            if fCount >= warmupLimit || readyF == nil {
                return (s, bMB, .backwardInputB)
            }
        }
        if let fMB = readyF {
            return (s, fMB, .forward)
        }
        if let wMB = (0 ..< state.m).first(where: { !state.wDone[s][$0] && state.bDone[s][$0] }) {
            return (s, wMB, .backwardWeightW)
        }
        return nil
    }

    private static func applyOp(_ op: (stage: Int, mb: Int, kind: ZBH1OpKind), to state: inout GridState) {
        switch op.kind {
        case .forward: state.fDone[op.stage][op.mb] = true
        case .backwardInputB: state.bDone[op.stage][op.mb] = true
        case .backwardWeightW: state.wDone[op.stage][op.mb] = true
        }
    }
}
