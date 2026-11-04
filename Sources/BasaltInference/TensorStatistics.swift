import MLX

struct TensorStatistics: Sendable, Equatable {
    static let percentileRanks: [Double] = [0.01, 0.05, 0.5, 0.95, 0.99]

    var count = 0
    var mean = 0.0
    var squaredDeviations = 0.0
    var min = Float.infinity
    var max = -Float.infinity

    var std: Float { count == 0 ? 0 : Float((squaredDeviations / Double(count)).squareRoot()) }
    var l2: Float { Float((squaredDeviations + Double(count) * mean * mean).squareRoot()) }

    static func + (lhs: Self, rhs: Self) -> Self {
        guard lhs.count > 0, rhs.count > 0 else { return lhs.count > 0 ? lhs : rhs }
        let count = lhs.count + rhs.count
        let delta = rhs.mean - lhs.mean
        let share = Double(rhs.count) / Double(count)
        return Self(
            count: count, mean: lhs.mean + delta * share,
            squaredDeviations: lhs.squaredDeviations + rhs.squaredDeviations + delta * delta * Double(lhs.count) * share,
            min: Swift.min(lhs.min, rhs.min), max: Swift.max(lhs.max, rhs.max))
    }

    static func percentileIndices(count: Int) -> [Int] {
        guard count > 0 else { return [] }
        return percentileRanks.map { Swift.min(count - 1, Int((Double(count - 1) * $0).rounded())) }
    }

    static func measure(
        rows: (Range<Int>) -> MLXArray, count rowCount: Int, rowElements: Int,
        chunkElements: Int = 1 << 24, sampleLimit: Int = 1 << 20
    ) -> (statistics: TensorStatistics, percentiles: [Float]) {
        let chunk = Swift.max(1, chunkElements / Swift.max(1, rowElements))
        let step = Swift.max(1, rowCount * rowElements / Swift.max(1, sampleLimit))
        var statistics = TensorStatistics()
        var samples: [MLXArray] = []
        for start in Swift.stride(from: 0, to: rowCount, by: chunk) {
            let values = rows(start ..< Swift.min(rowCount, start + chunk)).asType(.float32).flattened()
            guard values.size > 0 else { continue }
            let mean = values.mean()
            let deviations = (values - mean).square().sum()
            let low = values.min()
            let high = values.max()
            let sample = values[.stride(by: step)].abs()
            eval(mean, deviations, low, high, sample)
            statistics = statistics + TensorStatistics(
                count: values.size, mean: Double(mean.item(Float.self)), squaredDeviations: Double(deviations.item(Float.self)),
                min: low.item(Float.self), max: high.item(Float.self))
            samples.append(sample)
        }
        return (statistics, percentiles(samples))
    }

    static func percentiles(_ samples: [MLXArray]) -> [Float] {
        guard !samples.isEmpty else { return [] }
        let sorted = MLX.sorted(concatenated(samples))
        let indices = percentileIndices(count: sorted.size).map(Int32.init)
        return sorted[MLXArray(indices)].asArray(Float.self)
    }
}
