import Foundation

public enum ThermalThrottleLevel: String, Sendable, Codable, Equatable, CaseIterable {
    case nominal
    case fair
    case serious
    case critical

    public var throughputMultiplier: Double {
        switch self {
        case .nominal: 1.0
        case .fair: 0.85
        case .serious: 0.55
        case .critical: 0.25
        }
    }
}

public struct PeerStepTelemetry: Sendable, Codable, Equatable {
    public var rank: Int
    public var stepLatencyMs: Double
    public var thermal: ThermalThrottleLevel
    public var baseWeight: Double

    public init(
        rank: Int,
        stepLatencyMs: Double,
        thermal: ThermalThrottleLevel = .nominal,
        baseWeight: Double = 1.0
    ) {
        self.rank = rank
        self.stepLatencyMs = max(0.1, stepLatencyMs)
        self.thermal = thermal
        self.baseWeight = max(0.05, baseWeight)
    }

    public var effectiveScore: Double {
        let speed = 1_000.0 / max(1.0, stepLatencyMs)
        return baseWeight * speed * thermal.throughputMultiplier
    }
}

public enum StragglerThermalPacer {
    public static func rebalanceMicrobatches(
        globalBatchSize: Int,
        telemetry: [PeerStepTelemetry]
    ) -> [Int] {
        let world = telemetry.count
        guard world > 0 else { return [] }
        let totalBatch = max(world, globalBatchSize)
        let ordered = telemetry.sorted { $0.rank < $1.rank }
        let scores = ordered.map { max(1e-6, $0.effectiveScore) }
        let scoreSum = scores.reduce(0, +)
        var counts = [Int](repeating: 1, count: world)
        let distributable = totalBatch - world
        guard distributable > 0, scoreSum > 0 else { return counts }
        var remainders: [(idx: Int, frac: Double)] = []
        var assigned = 0
        for idx in 0 ..< world {
            let exact = Double(distributable) * (scores[idx] / scoreSum)
            let whole = Int(floor(exact))
            counts[idx] += whole
            assigned += whole
            remainders.append((idx, exact - Double(whole)))
        }
        remainders.sort {
            if abs($0.frac - $1.frac) > 1e-12 { return $0.frac > $1.frac }
            return $0.idx < $1.idx
        }
        let leftover = distributable - assigned
        for item in remainders.prefix(max(0, leftover)) {
            counts[item.idx] += 1
        }
        return counts
    }
}

public struct ZeRO2ParameterSlice: Sendable, Codable, Equatable {
    public let paramKey: String
    public let ownerRank: Int
    public let elementStart: Int
    public let elementCount: Int
    public let totalElements: Int
    public let shape: [Int]

    public init(
        paramKey: String,
        ownerRank: Int,
        elementStart: Int,
        elementCount: Int,
        totalElements: Int,
        shape: [Int]
    ) {
        self.paramKey = paramKey
        self.ownerRank = ownerRank
        self.elementStart = elementStart
        self.elementCount = elementCount
        self.totalElements = totalElements
        self.shape = shape
    }
}

public struct ZeRO2PartitionPlan: Sendable, Codable, Equatable {
    public let worldSize: Int
    public let totalElements: Int
    public let slices: [ZeRO2ParameterSlice]

    public static func build(
        parameterShapes: [(key: String, shape: [Int])],
        worldSize: Int
    ) -> ZeRO2PartitionPlan {
        let ranks = max(1, worldSize)
        let sorted = parameterShapes.sorted { $0.key < $1.key }
        let grandTotal = sorted.reduce(0) { $0 + max(1, $1.shape.reduce(1, *)) }
        var builtSlices: [ZeRO2ParameterSlice] = []
        var globalCursor = 0
        for entry in sorted {
            let count = max(1, entry.shape.reduce(1, *))
            var localOffset = 0
            while localOffset < count {
                let absPos = globalCursor + localOffset
                let owner = min(ranks - 1, (absPos * ranks) / max(1, grandTotal))
                let ownerEndGlobal = ((owner + 1) * grandTotal + ranks - 1) / ranks
                let maxInRank = max(1, ownerEndGlobal - absPos)
                let chunk = min(count - localOffset, maxInRank)
                builtSlices.append(ZeRO2ParameterSlice(
                    paramKey: entry.key,
                    ownerRank: owner,
                    elementStart: localOffset,
                    elementCount: chunk,
                    totalElements: count,
                    shape: entry.shape
                ))
                localOffset += chunk
            }
            globalCursor += count
        }
        return ZeRO2PartitionPlan(worldSize: ranks, totalElements: grandTotal, slices: builtSlices)
    }

    public func slices(ownedBy rank: Int) -> [ZeRO2ParameterSlice] {
        slices.filter { $0.ownerRank == rank }
    }

    public func elementCount(ownedBy rank: Int) -> Int {
        slices(ownedBy: rank).reduce(0) { $0 + $1.elementCount }
    }
}

public struct QJL1CompressedPacket: Sendable, Codable, Equatable {
    public let originalCount: Int
    public let scale: Float
    public let seed: UInt64
    public let packedSigns: [UInt64]
}

public enum QJL1GradientCodec {
    public static func compressWithFeedback(
        gradient: [Float],
        errorBuffer: inout [Float],
        seed: UInt64
    ) -> QJL1CompressedPacket {
        let n = gradient.count
        if errorBuffer.count != n {
            errorBuffer = [Float](repeating: 0, count: n)
        }
        var u = [Float](repeating: 0, count: n)
        var normSq: Double = 0
        for i in 0 ..< n {
            let val = gradient[i] + errorBuffer[i]
            u[i] = val
            normSq += Double(val) * Double(val)
        }
        let l2 = Float(normSq.squareRoot())
        guard n > 0, l2 > 1e-12 else {
            return QJL1CompressedPacket(originalCount: n, scale: 0, seed: seed, packedSigns: [])
        }
        let meanAbsScale = l2 / Float(n).squareRoot()
        let wordCount = (n + 63) / 64
        var words = [UInt64](repeating: 0, count: wordCount)
        for i in 0 ..< n {
            let rad = rademacher(index: i, seed: seed)
            let projected = u[i] * rad
            if projected >= 0 {
                words[i >> 6] |= (UInt64(1) << UInt64(i & 63))
            }
            let recon = (projected >= 0 ? meanAbsScale : -meanAbsScale) * rad
            errorBuffer[i] = u[i] - recon
        }
        return QJL1CompressedPacket(originalCount: n, scale: meanAbsScale, seed: seed, packedSigns: words)
    }

    public static func decompress(_ packet: QJL1CompressedPacket) -> [Float] {
        let n = packet.originalCount
        guard n > 0, packet.scale > 0 else { return [Float](repeating: 0, count: max(0, n)) }
        var out = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            let wordIdx = i >> 6
            let bit = wordIdx < packet.packedSigns.count
                ? ((packet.packedSigns[wordIdx] >> UInt64(i & 63)) & 1)
                : 0
            let sign: Float = bit == 1 ? 1.0 : -1.0
            out[i] = sign * packet.scale * rademacher(index: i, seed: packet.seed)
        }
        return out
    }

    private static func rademacher(index: Int, seed: UInt64) -> Float {
        var z = seed &+ (UInt64(index) &* 0x9E37_79B9_7F4A_7C15)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return ((z ^ (z >> 31)) & 1) == 0 ? 1.0 : -1.0
    }
}
