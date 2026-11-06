import Foundation

public struct WeightSpectrum: Sendable, Codable, Equatable {
    public var rows: Int
    public var columns: Int
    public var noiseVariance: Double
    public var bulkMin: Double
    public var bulkMax: Double
    public var massAboveBulk: Double
    public var rank: Int
    public var alpha: Double?
    public var tailSize: Int
    public var seconds: Double = 0

    public init?(eigenvalues: [Double], rows: Int, columns: Int) {
        let long = Double(max(rows, columns))
        guard eigenvalues.allSatisfy(\.isFinite) else { return nil }
        let values = eigenvalues.map { max($0, 0) / long }.sorted()
        guard let top = values.last, top > 0 else { return nil }
        let ratio = Double(values.count) / long
        let median = Self.median(values)
        self.rows = rows
        self.columns = columns
        let variance = median / Self.marchenkoPasturMedian(ratio)
        let edge = variance * pow(1 + ratio.squareRoot(), 2)
        noiseVariance = variance
        bulkMin = variance * pow(1 - ratio.squareRoot(), 2)
        bulkMax = edge
        massAboveBulk = values.filter { $0 > edge }.reduce(0, +) / values.reduce(0, +)
        let threshold = Self.gavishDonohoOmega(ratio) * median.squareRoot()
        rank = values.count(where: { $0.squareRoot() > threshold })
        let tail = Self.powerLawTail(values.reversed().filter { $0 > 0 })
        alpha = tail.alpha
        tailSize = tail.size
    }

    static func median(_ sorted: [Double]) -> Double {
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    static func gavishDonohoOmega(_ ratio: Double) -> Double {
        0.56 * ratio * ratio * ratio - 0.95 * ratio * ratio + 1.82 * ratio + 1.43
    }

    static func marchenkoPasturMedian(_ ratio: Double, steps: Int = 4096) -> Double {
        let lower = pow(1 - ratio.squareRoot(), 2), upper = pow(1 + ratio.squareRoot(), 2)
        let width = Double.pi / 2 / Double(steps)
        func point(_ angle: Double) -> Double { lower + (upper - lower) * pow(sin(angle), 2) }
        let masses = (0 ..< steps).map { step in
            let angle = (Double(step) + 0.5) * width
            return pow((upper - lower) * sin(angle) * cos(angle), 2) / point(angle)
        }
        let half = masses.reduce(0, +) / 2
        var cumulative = 0.0
        for (step, mass) in masses.enumerated() {
            if cumulative + mass >= half { return point((Double(step) + (half - cumulative) / mass) * width) }
            cumulative += mass
        }
        return upper
    }

    static func powerLawTail(_ descending: [Double]) -> (alpha: Double?, size: Int) {
        let smallest = min(10, descending.count), largest = descending.count / 2
        guard smallest >= 2, largest >= smallest else { return (nil, 0) }
        var best: (distance: Double, alpha: Double, size: Int)?
        var logSum = descending.prefix(smallest - 1).reduce(0) { $0 + log($1) }
        for size in smallest ... largest {
            let floor = descending[size - 1]
            logSum += log(floor)
            let spread = logSum - Double(size) * log(floor)
            guard spread > 0 else { continue }
            let alpha = 1 + Double(size - 1) / spread
            let distance = ksDistance(descending.prefix(size), floor: floor, alpha: alpha)
            if distance < best?.distance ?? .infinity { best = (distance, alpha, size) }
        }
        return (best?.alpha, best?.size ?? 0)
    }

    static func ksDistance(_ tail: ArraySlice<Double>, floor: Double, alpha: Double) -> Double {
        let count = Double(tail.count)
        var worst = 0.0
        for (index, value) in tail.enumerated() {
            let survival = pow(value / floor, 1 - alpha)
            worst = max(worst, abs(survival - Double(index + 1) / count), abs(survival - Double(index) / count))
        }
        return worst
    }
}
