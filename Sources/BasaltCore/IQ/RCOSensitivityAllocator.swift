import Foundation

public struct RCOSensitivityAllocator: Sendable {
    public enum LayerRole: String, Sendable, Equatable, Hashable, CaseIterable {
        case attention
        case attentionValue
        case ssm
        case router
        case ffn
        case moeRouted
        case moeShared
        case embedding
        case lmHead

        public var requiresSensitiveFloor: Bool {
            switch self {
            case .embedding, .lmHead, .moeShared, .attentionValue, .ssm, .router: true
            case .attention, .ffn, .moeRouted: false
            }
        }

        private static let leafRoles: [String: LayerRole] = [
            "embed": .embedding,
            "lm_head": .lmHead,
            "gate": .router,
            "moe_gate": .router,
            "v_proj": .attentionValue,
        ]

        private static let substringRoles: [(String, LayerRole)] = [
            ("embed_tokens", .embedding),
            ("tok_embeddings", .embedding),
            ("router", .router),
            ("linear_attn", .ssm),
            ("ssm", .ssm),
            ("mamba", .ssm),
            ("in_proj_", .ssm),
            ("attn_v", .attentionValue),
            ("shared_expert", .moeShared),
            ("routed_experts", .moeRouted),
            ("experts", .moeRouted),
            ("switch_mlp", .moeRouted),
            ("self_attn", .attention),
            ("attn", .attention),
        ]

        public static func infer(from path: String) -> LayerRole {
            let clean = path.hasSuffix(".weight") ? String(path.dropLast(".weight".count)) : path
            if clean == "output" { return .lmHead }
            let leaf = clean.split(separator: ".").last.map(String.init) ?? ""
            if let exact = leafRoles[leaf] { return exact }
            for (needle, role) in substringRoles where clean.contains(needle) {
                return role
            }
            return .ffn
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

    public struct RateCandidate: Sendable, Equatable, Hashable {
        public let id: String
        public let bitsPerWeight: Double
        public let distortionFactor: Double
        public let blockAlignment: Int
        public let ggmlType: GGMLType?

        public init(
            id: String,
            bitsPerWeight: Double,
            distortionFactor: Double = 1.0,
            blockAlignment: Int = 1,
            ggmlType: GGMLType? = nil
        ) {
            self.id = id
            self.bitsPerWeight = bitsPerWeight
            self.distortionFactor = distortionFactor
            self.blockAlignment = max(1, blockAlignment)
            self.ggmlType = ggmlType
        }

        public init(_ descriptor: IQBlockDescriptor) {
            self.id = descriptor.ggmlType.name
            self.bitsPerWeight = descriptor.bitsPerWeight
            self.distortionFactor = descriptor.distortionFactor
            self.blockAlignment = descriptor.elementsPerBlock
            self.ggmlType = descriptor.ggmlType
        }

        public var unitDistortion: Double {
            distortionFactor * pow(2.0, -2.0 * bitsPerWeight)
        }
    }

    public struct RateAllocation: Sendable, Equatable {
        public let name: String
        public let role: LayerRole
        public let weightCount: Int
        public let sensitivity: Double
        public let candidate: RateCandidate

        public var bitsPerWeight: Double { candidate.bitsPerWeight }
        public var totalBits: Double { Double(weightCount) * candidate.bitsPerWeight }
        public var distortion: Double { max(sensitivity, 1e-12) * Double(weightCount) * candidate.unitDistortion }
    }

    public struct RateAllocationPlan: Sendable, Equatable {
        public let allocations: [RateAllocation]
        public let targetBPW: Double
        public let achievedBPW: Double
        public let totalBits: Double
        public let totalWeights: Int
        public let totalDistortion: Double
        public let lagrangeMultiplier: Double
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
}

extension RCOSensitivityAllocator {
    public static let gsqRCOCandidates: [RateCandidate] = IQBlockDescriptor.all.map(RateCandidate.init)

    public static let qtipBudgetCandidates: [RateCandidate] = [
        RateCandidate(id: "qtip:2.0", bitsPerWeight: 2.0, distortionFactor: 0.88, blockAlignment: 16),
        RateCandidate(id: "qtip:2.5", bitsPerWeight: 2.5, distortionFactor: 0.85, blockAlignment: 16),
        RateCandidate(id: "qtip:3.0", bitsPerWeight: 3.0, distortionFactor: 0.82, blockAlignment: 16),
        RateCandidate(id: "affine:4.0", bitsPerWeight: 4.0, distortionFactor: 0.94, blockAlignment: 32),
    ]

    public func allocate(layers: [LayerSensitivity], config: BudgetConfig) -> AllocationPlan {
        let rawAllowed = config.allowedTypes.compactMap(IQBlockDescriptor.descriptor(for:))
        let descriptors = rawAllowed.isEmpty ? IQBlockDescriptor.all : rawAllowed
        let floorBPW = config.minFloorForSensitiveRoles.flatMap(IQBlockDescriptor.descriptor(for:))?.bitsPerWeight
        let solved = RCOSolverCore.solve(
            layers: layers,
            targetBPW: config.targetBPW,
            candidates: descriptors.map(RateCandidate.init),
            floorBPW: floorBPW
        )
        let mapped = solved.allocations.map { item in
            let desc = item.candidate.ggmlType.flatMap(IQBlockDescriptor.descriptor(for:)) ?? descriptors[0]
            return LayerAllocation(
                name: item.name, role: item.role, weightCount: item.weightCount,
                sensitivity: item.sensitivity, descriptor: desc)
        }
        return AllocationPlan(
            allocations: mapped, targetBPW: solved.targetBPW, achievedBPW: solved.achievedBPW,
            totalBits: solved.totalBits, totalWeights: solved.totalWeights,
            totalDistortion: solved.totalDistortion, lagrangeMultiplier: solved.lagrangeMultiplier)
    }

    public func allocate(
        layers: [LayerSensitivity],
        targetBitsPerWeight: Double,
        candidates: [IQBlockDescriptor],
        minFloorForSensitiveRoles: GGMLType? = .iq4XS
    ) -> AllocationPlan {
        let types = candidates.isEmpty ? IQBlockDescriptor.all.map(\.ggmlType) : candidates.map(\.ggmlType)
        return allocate(
            layers: layers,
            config: BudgetConfig(
                targetBPW: targetBitsPerWeight,
                allowedTypes: types,
                minFloorForSensitiveRoles: minFloorForSensitiveRoles))
    }

    public func allocate(
        layers: [LayerSensitivity],
        targetBitsPerWeight: Double,
        candidates: [RateCandidate] = Self.gsqRCOCandidates,
        sensitiveFloorBPW: Double? = nil
    ) -> RateAllocationPlan {
        let pool = candidates.isEmpty ? Self.gsqRCOCandidates : candidates
        let resolvedFloor = sensitiveFloorBPW ?? Self.defaultSensitiveFloor(
            layers: layers, targetBPW: targetBitsPerWeight, candidates: pool)
        return RCOSolverCore.solve(
            layers: layers,
            targetBPW: targetBitsPerWeight,
            candidates: pool,
            floorBPW: resolvedFloor
        )
    }

    private static func defaultSensitiveFloor(
        layers: [LayerSensitivity],
        targetBPW: Double,
        candidates: [RateCandidate]
    ) -> Double? {
        let sorted = candidates.map(\.bitsPerWeight).sorted()
        guard sorted.count >= 2 else { return nil }
        let totalWeights = layers.reduce(0) { $0 + max(0, $1.weightCount) }
        guard totalWeights > 0 else { return nil }
        let sensitiveWeights = layers.reduce(0) { $0 + ($1.role.requiresSensitiveFloor ? max(0, $1.weightCount) : 0) }
        guard sensitiveWeights > 0 else { return nil }
        let nonSensitiveWeights = totalWeights - sensitiveWeights
        let minRate = sorted[0]
        var bestFloor: Double?
        for rate in sorted.dropFirst() where rate <= 4.5 {
            let minNeededBits = Double(sensitiveWeights) * rate + Double(nonSensitiveWeights) * minRate
            if minNeededBits <= targetBPW * Double(totalWeights) + 1e-9 {
                bestFloor = rate
                if rate >= 3.0 { break }
            }
        }
        return bestFloor
    }
}

private enum RCOSolverCore {
    struct GroupKey: Hashable {
        let role: RCOSensitivityAllocator.LayerRole
        let columns: Int?
    }

    static func solve(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        targetBPW: Double,
        candidates: [RCOSensitivityAllocator.RateCandidate],
        floorBPW: Double?
    ) -> RCOSensitivityAllocator.RateAllocationPlan {
        guard !layers.isEmpty, !candidates.isEmpty else {
            return RCOSensitivityAllocator.RateAllocationPlan(
                allocations: [], targetBPW: targetBPW, achievedBPW: 0,
                totalBits: 0, totalWeights: 0, totalDistortion: 0, lagrangeMultiplier: 0)
        }
        let perLayer = layers.map { paretoFrontier(layerCandidates(for: $0, allowed: candidates, floorBPW: floorBPW)) }
        let hulls = perLayer.map(convexHull(_:))
        let totalWeights = layers.reduce(0) { $0 + max(0, $1.weightCount) }
        let budgetBits = targetBPW * Double(totalWeights)
        let (hullChoice, lambda) = bisectLambda(layers: layers, hulls: hulls, budgetBits: budgetBits)
        var selected = zip(perLayer, zip(hulls, hullChoice)).map { ladder, pair in
            let chosen = pair.0[pair.1]
            return ladder.firstIndex(of: chosen) ?? 0
        }
        refineGreedy(layers: layers, ladders: perLayer, indices: &selected, budgetBits: budgetBits)
        return buildPlan(layers: layers, ladders: perLayer, indices: selected, targetBPW: targetBPW, lambda: lambda)
    }

    static func paretoFrontier(
        _ candidates: [RCOSensitivityAllocator.RateCandidate]
    ) -> [RCOSensitivityAllocator.RateCandidate] {
        let sorted = candidates.sorted {
            if $0.bitsPerWeight != $1.bitsPerWeight { return $0.bitsPerWeight < $1.bitsPerWeight }
            return $0.unitDistortion < $1.unitDistortion
        }
        var result: [RCOSensitivityAllocator.RateCandidate] = []
        for item in sorted {
            if let last = result.last {
                if abs(last.bitsPerWeight - item.bitsPerWeight) < 1e-9 { continue }
                if item.unitDistortion >= last.unitDistortion { continue }
            }
            result.append(item)
        }
        return result.isEmpty ? [sorted[0]] : result
    }

    static func layerCandidates(
        for layer: RCOSensitivityAllocator.LayerSensitivity,
        allowed: [RCOSensitivityAllocator.RateCandidate],
        floorBPW: Double?
    ) -> [RCOSensitivityAllocator.RateCandidate] {
        let aligned = allowed.filter { cand in
            guard let cols = layer.columns, cols > 0 else { return true }
            return cols % cand.blockAlignment == 0
        }
        let pool = (aligned.isEmpty ? allowed : aligned).sorted { $0.bitsPerWeight < $1.bitsPerWeight }
        guard layer.role.requiresSensitiveFloor, let floorBPW else { return pool }
        let floored = pool.filter { $0.bitsPerWeight + 1e-9 >= floorBPW }
        return floored.isEmpty ? [pool[pool.count - 1]] : floored
    }

    static func convexHull(
        _ ladder: [RCOSensitivityAllocator.RateCandidate]
    ) -> [RCOSensitivityAllocator.RateCandidate] {
        var hull: [RCOSensitivityAllocator.RateCandidate] = []
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

    static func pickOnHull(
        sigma: Double,
        hull: [RCOSensitivityAllocator.RateCandidate],
        lambda: Double
    ) -> Int {
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
}

extension RCOSolverCore {
    static func bisectLambda(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        hulls: [[RCOSensitivityAllocator.RateCandidate]],
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
        hulls: [[RCOSensitivityAllocator.RateCandidate]],
        lambda: Double
    ) -> Double {
        var sum = 0.0
        for idx in layers.indices {
            let pick = pickOnHull(sigma: max(layers[idx].sensitivity, 1e-12), hull: hulls[idx], lambda: lambda)
            sum += Double(layers[idx].weightCount) * hulls[idx][pick].bitsPerWeight
        }
        return sum
    }

    static func refineGreedy(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        ladders: [[RCOSensitivityAllocator.RateCandidate]],
        indices: inout [Int],
        budgetBits: Double
    ) {
        var groups: [GroupKey: [Int]] = [:]
        for idx in layers.indices {
            let key = GroupKey(role: layers[idx].role, columns: layers[idx].columns)
            groups[key, default: []].append(idx)
        }
        let ordered = groups.sorted { ($0.key.role.rawValue, $0.key.columns ?? -1) < ($1.key.role.rawValue, $1.key.columns ?? -1) }
        let sortedGroups = ordered.map(\.value).map { members in
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
        ladders: [[RCOSensitivityAllocator.RateCandidate]],
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
        ladders: [[RCOSensitivityAllocator.RateCandidate]],
        indices: [Int],
        targetBPW: Double,
        lambda: Double
    ) -> RCOSensitivityAllocator.RateAllocationPlan {
        let allocations = layers.indices.map { i in
            RCOSensitivityAllocator.RateAllocation(
                name: layers[i].name, role: layers[i].role, weightCount: layers[i].weightCount,
                sensitivity: layers[i].sensitivity, candidate: ladders[i][indices[i]])
        }
        let totalWeights = allocations.reduce(0) { $0 + $1.weightCount }
        let totalBits = allocations.reduce(0.0) { $0 + $1.totalBits }
        let totalDistortion = allocations.reduce(0.0) { $0 + $1.distortion }
        let achievedBPW = totalWeights > 0 ? totalBits / Double(totalWeights) : 0.0
        return RCOSensitivityAllocator.RateAllocationPlan(
            allocations: allocations, targetBPW: targetBPW, achievedBPW: achievedBPW,
            totalBits: totalBits, totalWeights: totalWeights,
            totalDistortion: totalDistortion, lagrangeMultiplier: lambda)
    }
}
