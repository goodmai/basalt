import Foundation

public struct HeterogeneousNodeProfile: Sendable, Codable, Equatable {
    public var nodeId: String
    public var effectiveTFLOPS: Double
    public var powerWatts: Double
    public var availableMemoryBytes: Int64
    public var linkGbps: Double

    public init(
        nodeId: String,
        effectiveTFLOPS: Double,
        powerWatts: Double,
        availableMemoryBytes: Int64,
        linkGbps: Double = 80.0
    ) {
        self.nodeId = nodeId
        self.effectiveTFLOPS = effectiveTFLOPS
        self.powerWatts = powerWatts
        self.availableMemoryBytes = availableMemoryBytes
        self.linkGbps = linkGbps
    }

    public init(capability: PeerCapability, powerWatts: Double) {
        self.nodeId = capability.peerId
        let compute = max(0.05, capability.measuredTFLOPS.isFinite ? capability.measuredTFLOPS : 0.05)
        let thermal = min(1.0, max(0.05, capability.thermalFactor.isFinite ? capability.thermalFactor : 1.0))
        self.effectiveTFLOPS = compute * thermal
        self.powerWatts = powerWatts
        self.availableMemoryBytes = capability.unifiedMemoryBytes
        self.linkGbps = capability.linkGbps
    }

    public var computeRateUnitsPerSecond: Double {
        let compute = max(0.05, effectiveTFLOPS.isFinite ? effectiveTFLOPS : 0.05)
        let bw = max(0.1, linkGbps.isFinite ? linkGbps : 10.0)
        let linkScale = bw / (bw + 10.0)
        return compute * linkScale
    }
}

public struct SplittableWorkload: Sendable, Codable, Equatable {
    public var workloadId: String
    public var totalWorkUnits: Int
    public var minChunkUnits: Int
    public var memoryBytesPerUnit: Int64
    public var energyWeight: Double

    public init(
        workloadId: String,
        totalWorkUnits: Int,
        minChunkUnits: Int = 1,
        memoryBytesPerUnit: Int64 = 1_048_576,
        energyWeight: Double = 0.25
    ) {
        self.workloadId = workloadId
        self.totalWorkUnits = totalWorkUnits
        self.minChunkUnits = minChunkUnits
        self.memoryBytesPerUnit = memoryBytesPerUnit
        self.energyWeight = energyWeight
    }

    public func validate() throws(BasaltError) {
        guard totalWorkUnits >= 1, minChunkUnits >= 1 else {
            throw .invalidRequestStructure(details: "heterogeneous planner: totalWorkUnits and minChunkUnits must be >= 1")
        }
        guard memoryBytesPerUnit >= 0 else {
            throw .invalidRequestStructure(details: "heterogeneous planner: memoryBytesPerUnit must be non-negative")
        }
        guard energyWeight.isFinite, (0.0 ... 1.0).contains(energyWeight) else {
            throw .invalidRequestStructure(details: "heterogeneous planner: energyWeight must lie in 0...1")
        }
    }
}

public struct AdaptiveSplitAssignment: Sendable, Codable, Equatable {
    public var nodeId: String
    public var assignedUnits: Int
    public var predictedSeconds: Double
    public var predictedJoules: Double

    public init(nodeId: String, assignedUnits: Int, predictedSeconds: Double, predictedJoules: Double) {
        self.nodeId = nodeId
        self.assignedUnits = assignedUnits
        self.predictedSeconds = predictedSeconds
        self.predictedJoules = predictedJoules
    }
}

public struct AdaptiveSplitPlan: Sendable, Codable, Equatable {
    public var workloadId: String
    public var assignments: [AdaptiveSplitAssignment]
    public var makespanSeconds: Double
    public var totalEnergyJoules: Double
    public var unassignedUnits: Int

    public init(
        workloadId: String,
        assignments: [AdaptiveSplitAssignment],
        makespanSeconds: Double,
        totalEnergyJoules: Double,
        unassignedUnits: Int
    ) {
        self.workloadId = workloadId
        self.assignments = assignments
        self.makespanSeconds = makespanSeconds
        self.totalEnergyJoules = totalEnergyJoules
        self.unassignedUnits = unassignedUnits
    }
}

public enum AdaptiveHeterogeneousPlanner {
    public static func plan(
        workload: SplittableWorkload,
        nodes: [HeterogeneousNodeProfile]
    ) throws(BasaltError) -> AdaptiveSplitPlan {
        try workload.validate()
        guard !nodes.isEmpty else {
            throw .invalidRequestStructure(details: "heterogeneous planner: at least one node is required")
        }
        let sortedNodes = nodes.sorted { $0.nodeId < $1.nodeId }
        var assigned = Array(repeating: 0, count: sortedNodes.count)
        let capacities = sortedNodes.map { node -> Int in
            guard workload.memoryBytesPerUnit > 0 else { return workload.totalWorkUnits }
            return max(0, Int(node.availableMemoryBytes / workload.memoryBytesPerUnit))
        }
        var remaining = workload.totalWorkUnits
        let avgWatts = max(1.0, sortedNodes.map(\.powerWatts).reduce(0, +) / Double(sortedNodes.count))

        while remaining > 0 {
            let chunk = min(remaining, max(workload.minChunkUnits, remaining / max(1, sortedNodes.count * 2)))
            guard let chosen = pickBestNode(
                nodes: sortedNodes,
                assigned: assigned,
                capacities: capacities,
                chunk: chunk,
                energyWeight: workload.energyWeight,
                avgWatts: avgWatts
            ) else {
                break
            }
            let grant = min(chunk, capacities[chosen] - assigned[chosen])
            assigned[chosen] += grant
            remaining -= grant
        }

        let assignments: [AdaptiveSplitAssignment] = sortedNodes.enumerated().compactMap { idx, node in
            let units = assigned[idx]
            guard units > 0 else { return nil }
            let seconds = Double(units) / node.computeRateUnitsPerSecond
            let joules = seconds * max(0.0, node.powerWatts)
            return AdaptiveSplitAssignment(nodeId: node.nodeId, assignedUnits: units, predictedSeconds: seconds, predictedJoules: joules)
        }
        let makespan = assignments.map(\.predictedSeconds).max() ?? 0.0
        let totalJoules = assignments.reduce(0.0) { $0 + $1.predictedJoules }
        return AdaptiveSplitPlan(
            workloadId: workload.workloadId,
            assignments: assignments,
            makespanSeconds: makespan,
            totalEnergyJoules: totalJoules,
            unassignedUnits: remaining
        )
    }

    private static func pickBestNode(
        nodes: [HeterogeneousNodeProfile],
        assigned: [Int],
        capacities: [Int],
        chunk: Int,
        energyWeight: Double,
        avgWatts: Double
    ) -> Int? {
        var bestIdx: Int?
        var bestCost = Double.infinity
        for idx in nodes.indices {
            let headroom = capacities[idx] - assigned[idx]
            guard headroom > 0 else { continue }
            let grant = min(chunk, headroom)
            let rate = nodes[idx].computeRateUnitsPerSecond
            let finishSeconds = Double(assigned[idx] + grant) / rate
            let deltaEnergyNorm = (Double(grant) / rate) * (max(0.0, nodes[idx].powerWatts) / avgWatts)
            let cost = (1.0 - energyWeight) * finishSeconds + energyWeight * deltaEnergyNorm
            if cost < bestCost {
                bestCost = cost
                bestIdx = idx
            }
        }
        return bestIdx
    }
}
