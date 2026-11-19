import Foundation

public struct SpeculativeAcceptanceEMA: Sendable, Codable, Equatable {
    public static let defaultSmoothing = 0.2
    public static let defaultInitial = 0.7

    public private(set) var value: Double
    public let smoothing: Double

    public init(initial: Double = Self.defaultInitial, smoothing: Double = Self.defaultSmoothing) {
        self.value = Self.clampUnit(initial, fallback: 0)
        self.smoothing = smoothing.isFinite ? min(1, max(0.01, smoothing)) : Self.defaultSmoothing
    }

    public mutating func observe(accepted: Int, drafted: Int) {
        guard drafted > 0 else { return }
        let rate = min(1, max(0, Double(max(0, accepted)) / Double(drafted)))
        value = (1 - smoothing) * value + smoothing * rate
    }

    public func observing(accepted: Int, drafted: Int) -> SpeculativeAcceptanceEMA {
        var next = self
        next.observe(accepted: accepted, drafted: drafted)
        return next
    }

    static func clampUnit(_ raw: Double, fallback: Double) -> Double {
        raw.isFinite ? min(1, max(0, raw)) : fallback
    }
}

public struct DynamicTreeBudget: Sendable, Codable, Equatable {
    public static let depthRange = 1 ... 6
    public static let widthRange = 1 ... 4

    public let depth: Int
    public let width: Int

    public init(depth: Int, width: Int) {
        self.depth = min(Self.depthRange.upperBound, max(Self.depthRange.lowerBound, depth))
        self.width = min(Self.widthRange.upperBound, max(Self.widthRange.lowerBound, width))
    }

    public var verifyBudget: Int { depth * width }

    public var maxDraftNodes: Int {
        guard width > 1 else { return depth }
        var sum = 0
        var levelNodes = 1
        for _ in 0 ..< depth {
            levelNodes *= width
            sum += levelNodes
        }
        return sum
    }

    public static func compute(
        thermalState: ProcessInfo.ThermalState,
        isLowPowerModeEnabled: Bool = false,
        acceptanceEMA: Double,
        activeBatchSize: Int = 1
    ) -> DynamicTreeBudget {
        compute(
            thermalLevel: ThermalLevel(thermalState),
            isLowPowerModeEnabled: isLowPowerModeEnabled,
            acceptanceEMA: acceptanceEMA,
            activeBatchSize: activeBatchSize)
    }

    public static func compute(
        thermalLevel: ThermalLevel,
        isLowPowerModeEnabled: Bool = false,
        acceptanceEMA: Double,
        activeBatchSize: Int = 1
    ) -> DynamicTreeBudget {
        let alpha = SpeculativeAcceptanceEMA.clampUnit(acceptanceEMA, fallback: 0)
        let base = baseBudget(for: alpha)
        let thermal = thermalPenalty(for: thermalLevel)
        let power = isLowPowerModeEnabled ? (depth: 2, width: 1) : (depth: 0, width: 0)
        let batch = batchPenalty(for: activeBatchSize)
        return DynamicTreeBudget(
            depth: base.depth - thermal.depth - power.depth - batch.depth,
            width: base.width - thermal.width - power.width - batch.width)
    }

    public static func current(
        acceptanceEMA: Double,
        activeBatchSize: Int = 1,
        processInfo: ProcessInfo = .processInfo
    ) -> DynamicTreeBudget {
        compute(
            thermalState: processInfo.thermalState,
            isLowPowerModeEnabled: processInfo.isLowPowerModeEnabled,
            acceptanceEMA: acceptanceEMA,
            activeBatchSize: activeBatchSize)
    }

    private static func baseBudget(for alpha: Double) -> (depth: Int, width: Int) {
        let depth: Int
        if alpha < 0.15 {
            depth = 1
        } else if alpha < 0.30 {
            depth = 2
        } else if alpha < 0.50 {
            depth = 3
        } else if alpha < 0.65 {
            depth = 4
        } else if alpha < 0.80 {
            depth = 5
        } else {
            depth = 6
        }
        let width: Int
        if alpha < 0.20 {
            width = 1
        } else if alpha < 0.45 {
            width = 2
        } else if alpha < 0.75 {
            width = 3
        } else {
            width = 4
        }
        return (depth, width)
    }

    private static func thermalPenalty(for level: ThermalLevel) -> (depth: Int, width: Int) {
        switch level {
        case .nominal: (0, 0)
        case .fair: (1, 1)
        case .serious: (3, 2)
        case .critical: (5, 3)
        }
    }

    private static func batchPenalty(for activeBatchSize: Int) -> (depth: Int, width: Int) {
        let batch = max(1, activeBatchSize)
        if batch == 1 { return (0, 0) }
        if batch == 2 { return (1, 1) }
        if batch <= 4 { return (2, 2) }
        if batch <= 7 { return (4, 3) }
        return (5, 3)
    }
}
