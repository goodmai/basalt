import Foundation

public struct RCOSensitivityAllocator: Sendable {
    public enum LayerRole: String, Sendable, Equatable, Hashable, CaseIterable {
        case attention
        case ffn
        case moeRouted
        case moeShared
        case embedding
        case lmHead

        public var requiresSensitiveFloor: Bool {
            switch self {
            case .embedding, .lmHead, .moeShared: true
            case .attention, .ffn, .moeRouted: false
            }
        }
    }

    public struct LayerSensitivity: Sendable, Equatable {
        public let name: String
        public let weightCount: Int
        public let sensitivity: Double
        public let role: LayerRole
        public let columns: Int?

        public init(
            name: String,
            weightCount: Int,
            sensitivity: Double,
            role: LayerRole,
            columns: Int? = nil
        ) {
            self.name = name
            self.weightCount = weightCount
            self.sensitivity = sensitivity
            self.role = role
            self.columns = columns
        }
    }

    public struct BudgetConfig: Sendable, Equatable {
        public let targetBPW: Double
        public let allowedTypes: [GGMLType]
        public let minFloorForSensitiveRoles: GGMLType?

        public init(
            targetBPW: Double,
            allowedTypes: [GGMLType] = IQBlockDescriptor.all.map(\.ggmlType),
            minFloorForSensitiveRoles: GGMLType? = .iq4XS
        ) {
            self.targetBPW = targetBPW
            self.allowedTypes = allowedTypes
            self.minFloorForSensitiveRoles = minFloorForSensitiveRoles
        }
    }

    public struct LayerAllocation: Sendable, Equatable {
        public let name: String
        public let role: LayerRole
        public let weightCount: Int
        public let sensitivity: Double
        public let descriptor: IQBlockDescriptor

        public var ggmlType: GGMLType { descriptor.ggmlType }
        public var bitsPerWeight: Double { descriptor.bitsPerWeight }
        public var totalBits: Double { Double(weightCount) * descriptor.bitsPerWeight }
        public var distortion: Double { max(sensitivity, 1e-12) * Double(weightCount) * descriptor.unitDistortion }
    }

    public struct AllocationPlan: Sendable, Equatable {
        public let allocations: [LayerAllocation]
        public let targetBPW: Double
        public let achievedBPW: Double
        public let totalBits: Double
        public let totalWeights: Int
        public let totalDistortion: Double
        public let lagrangeMultiplier: Double
    }

    public init() {}

    public func allocate(layers: [LayerSensitivity], config: BudgetConfig) -> AllocationPlan {
        RCOSolverCore.solve(layers: layers, config: config)
    }
}

private enum RCOSolverCore {
    struct GroupKey: Hashable {
        let role: RCOSensitivityAllocator.LayerRole
        let columns: Int?
    }

    static func solve(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        config: RCOSensitivityAllocator.BudgetConfig
    ) -> RCOSensitivityAllocator.AllocationPlan {
        guard !layers.isEmpty else {
            return RCOSensitivityAllocator.AllocationPlan(
                allocations: [], targetBPW: config.targetBPW, achievedBPW: 0,
                totalBits: 0, totalWeights: 0, totalDistortion: 0, lagrangeMultiplier: 0)
        }
        let rawAllowed = config.allowedTypes.compactMap(IQBlockDescriptor.descriptor(for:))
        let allowed = rawAllowed.isEmpty ? IQBlockDescriptor.all : rawAllowed
        let floorBPW = config.minFloorForSensitiveRoles.flatMap(IQBlockDescriptor.descriptor(for:))?.bitsPerWeight
        let perLayer = layers.map { paretoFrontier(candidates(for: $0, allowed: allowed, floorBPW: floorBPW)) }
        let hulls = perLayer.map(convexHull(_:))
        let totalWeights = layers.reduce(0) { $0 + max(0, $1.weightCount) }
        let budgetBits = config.targetBPW * Double(totalWeights)
        let (hullChoice, lambda) = bisectLambda(layers: layers, hulls: hulls, budgetBits: budgetBits)
        var selected = zip(perLayer, zip(hulls, hullChoice)).map { ladder, pair in
            let chosen = pair.0[pair.1]
            return ladder.firstIndex(of: chosen) ?? 0
        }
        refineGreedy(layers: layers, ladders: perLayer, indices: &selected, budgetBits: budgetBits)
        return buildPlan(layers: layers, ladders: perLayer, indices: selected, config: config, lambda: lambda)
    }

    static func paretoFrontier(_ descriptors: [IQBlockDescriptor]) -> [IQBlockDescriptor] {
        let sorted = descriptors.sorted {
            if $0.bitsPerWeight != $1.bitsPerWeight { return $0.bitsPerWeight < $1.bitsPerWeight }
            return $0.unitDistortion < $1.unitDistortion
        }
        var result: [IQBlockDescriptor] = []
        for item in sorted {
            if let last = result.last {
                if abs(last.bitsPerWeight - item.bitsPerWeight) < 1e-9 { continue }
                if item.unitDistortion >= last.unitDistortion { continue }
            }
            result.append(item)
        }
        return result.isEmpty ? [sorted[0]] : result
    }

    static func candidates(
        for layer: RCOSensitivityAllocator.LayerSensitivity,
        allowed: [IQBlockDescriptor],
        floorBPW: Double?
    ) -> [IQBlockDescriptor] {
        let aligned = allowed.filter { desc in
            guard let cols = layer.columns, cols > 0 else { return true }
            return cols % desc.elementsPerBlock == 0
        }
        let pool = (aligned.isEmpty ? allowed : aligned).sorted { $0.bitsPerWeight < $1.bitsPerWeight }
        guard layer.role.requiresSensitiveFloor, let floorBPW else { return pool }
        let floored = pool.filter { $0.bitsPerWeight + 1e-9 >= floorBPW }
        return floored.isEmpty ? [pool[pool.count - 1]] : floored
    }

    static func convexHull(_ ladder: [IQBlockDescriptor]) -> [IQBlockDescriptor] {
        var hull: [IQBlockDescriptor] = []
        for candidate in ladder {
            if let last = hull.last, candidate.unitDistortion >= last.unitDistortion { continue }
            while hull.count >= 2 {
                let a = hull[hull.count - 2]
                let b = hull[hull.count - 1]
                let slopeAB = (a.unitDistortion - b.unitDistortion) / (b.bitsPerWeight - a.bitsPerWeight)
                let slopeBC = (b.unitDistortion - candidate.unitDistortion) / (candidate.bitsPerWeight - b.bitsPerWeight)
                if slopeAB <= slopeBC { hull.removeLast() } else { break }
            }
            hull.append(candidate)
        }
        return hull.isEmpty ? [ladder[0]] : hull
    }

    static func pickOnHull(sigma: Double, hull: [IQBlockDescriptor], lambda: Double) -> Int {
        var best = 0
        var bestCost = sigma * hull[0].unitDistortion + lambda * hull[0].bitsPerWeight
        for idx in 1 ..< hull.count {
            let cost = sigma * hull[idx].unitDistortion + lambda * hull[idx].bitsPerWeight
            if cost <= bestCost {
                bestCost = cost
                best = idx
            }
        }
        return best
    }

    static func bisectLambda(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        hulls: [[IQBlockDescriptor]],
        budgetBits: Double
    ) -> ([Int], Double) {
        let minBits = zip(layers, hulls).reduce(0.0) { $0 + Double($1.0.weightCount) * $1.1[0].bitsPerWeight }
        if minBits >= budgetBits { return (Array(repeating: 0, count: layers.count), 1e9) }
        let maxChoice = hulls.map { $0.count - 1 }
        let maxBits = zip(layers, zip(hulls, maxChoice)).reduce(0.0) {
            $0 + Double($1.0.weightCount) * $1.1.0[$1.1.1].bitsPerWeight
        }
        if maxBits <= budgetBits { return (maxChoice, 0.0) }
        var lo = 0.0
        var hi = 1.0
        while totalHullBits(layers: layers, hulls: hulls, lambda: hi) > budgetBits { hi *= 2.0 }
        for _ in 0 ..< 64 {
            let mid = 0.5 * (lo + hi)
            if totalHullBits(layers: layers, hulls: hulls, lambda: mid) <= budgetBits { hi = mid } else { lo = mid }
        }
        let picks = zip(layers, hulls).map { pickOnHull(sigma: max($0.sensitivity, 1e-12), hull: $1, lambda: hi) }
        return (picks, hi)
    }

    static func totalHullBits(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        hulls: [[IQBlockDescriptor]],
        lambda: Double
    ) -> Double {
        var sum = 0.0
        for idx in layers.indices {
            let pick = pickOnHull(sigma: max(layers[idx].sensitivity, 1e-12), hull: hulls[idx], lambda: lambda)
            sum += Double(layers[idx].weightCount) * hulls[idx][pick].bitsPerWeight
        }
        return sum
    }
}

extension RCOSolverCore {
    static func refineGreedy(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        ladders: [[IQBlockDescriptor]],
        indices: inout [Int],
        budgetBits: Double
    ) {
        var groups: [GroupKey: [Int]] = [:]
        for idx in layers.indices {
            let key = GroupKey(role: layers[idx].role, columns: layers[idx].columns)
            groups[key, default: []].append(idx)
        }
        let sortedGroups = groups.values.map { members in
            members.sorted {
                if layers[$0].sensitivity != layers[$1].sensitivity {
                    return layers[$0].sensitivity > layers[$1].sensitivity
                }
                return $0 < $1
            }
        }
        var usedBits = zip(layers, zip(ladders, indices)).reduce(0.0) {
            $0 + Double($1.0.weightCount) * $1.1.0[$1.1.1].bitsPerWeight
        }
        while let upgrade = bestUpgrade(
            layers: layers, ladders: ladders, indices: indices,
            groups: sortedGroups, remainingBits: budgetBits - usedBits
        ) {
            indices[upgrade.layer] += 1
            usedBits += upgrade.deltaBits
        }
    }

    static func bestUpgrade(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        ladders: [[IQBlockDescriptor]],
        indices: [Int],
        groups: [[Int]],
        remainingBits: Double
    ) -> (layer: Int, deltaBits: Double)? {
        var bestLayer = -1
        var bestGainPerBit = -1.0
        var bestDeltaBits = 0.0
        for group in groups {
            var strictPredMinIdx = Int.max
            var pos = 0
            while pos < group.count {
                let sVal = layers[group[pos]].sensitivity
                var end = pos + 1
                while end < group.count && layers[group[end]].sensitivity == sVal { end += 1 }
                for k in pos ..< end {
                    let i = group[k]
                    let next = indices[i] + 1
                    guard next < ladders[i].count, next <= strictPredMinIdx else { continue }
                    let cur = ladders[i][indices[i]]
                    let nxt = ladders[i][next]
                    let dBits = Double(layers[i].weightCount) * (nxt.bitsPerWeight - cur.bitsPerWeight)
                    guard dBits <= remainingBits + 1e-9 else { continue }
                    let gain = max(sVal, 1e-12) * max(0.0, cur.unitDistortion - nxt.unitDistortion)
                        / (nxt.bitsPerWeight - cur.bitsPerWeight)
                    if gain > bestGainPerBit {
                        bestGainPerBit = gain
                        bestLayer = i
                        bestDeltaBits = dBits
                    }
                }
                for k in pos ..< end { strictPredMinIdx = min(strictPredMinIdx, indices[group[k]]) }
                pos = end
            }
        }
        return bestLayer >= 0 ? (bestLayer, bestDeltaBits) : nil
    }

    static func buildPlan(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        ladders: [[IQBlockDescriptor]],
        indices: [Int],
        config: RCOSensitivityAllocator.BudgetConfig,
        lambda: Double
    ) -> RCOSensitivityAllocator.AllocationPlan {
        let allocations = layers.indices.map { i in
            RCOSensitivityAllocator.LayerAllocation(
                name: layers[i].name, role: layers[i].role, weightCount: layers[i].weightCount,
                sensitivity: layers[i].sensitivity, descriptor: ladders[i][indices[i]])
        }
        let totalWeights = allocations.reduce(0) { $0 + $1.weightCount }
        let totalBits = allocations.reduce(0.0) { $0 + $1.totalBits }
        let totalDistortion = allocations.reduce(0.0) { $0 + $1.distortion }
        let achievedBPW = totalWeights > 0 ? totalBits / Double(totalWeights) : 0.0
        return RCOSensitivityAllocator.AllocationPlan(
            allocations: allocations, targetBPW: config.targetBPW, achievedBPW: achievedBPW,
            totalBits: totalBits, totalWeights: totalWeights,
            totalDistortion: totalDistortion, lagrangeMultiplier: lambda)
    }
}
