public enum TNorm: String, Sendable, Codable, CaseIterable {
    case godel
    case product
    case lukasiewicz

    public static func clamp(_ value: Double) -> Double {
        guard !value.isNaN else { return 0.0 }
        return min(max(value, 0.0), 1.0)
    }

    public func evaluate(_ values: Span<Double>) -> Double {
        switch self {
        case .godel: Self.evaluateGodel(values)
        case .product: Self.evaluateProduct(values)
        case .lukasiewicz: Self.evaluateLukasiewicz(values)
        }
    }

    public func evaluate(_ first: Double, _ second: Double) -> Double {
        let pair: InlineArray<2, Double> = [first, second]
        return evaluate(pair.span)
    }

    public func tConorm(_ values: Span<Double>) -> Double {
        switch self {
        case .godel: Self.conormGodel(values)
        case .product: Self.conormProduct(values)
        case .lukasiewicz: Self.conormLukasiewicz(values)
        }
    }

    public func tConorm(_ first: Double, _ second: Double) -> Double {
        let pair: InlineArray<2, Double> = [first, second]
        return tConorm(pair.span)
    }

    public func residuum(from antecedent: Double, to consequent: Double) -> Double {
        let a = Self.clamp(antecedent)
        let b = Self.clamp(consequent)
        guard a > b else { return 1.0 }
        return switch self {
        case .godel: b
        case .product: Self.clamp(b / a)
        case .lukasiewicz: Self.clamp(1.0 - a + b)
        }
    }

    private static func evaluateGodel(_ values: Span<Double>) -> Double {
        var minimum = 1.0
        for index in values.indices {
            let x = clamp(values[index])
            if x == 0.0 { return 0.0 }
            if x < minimum { minimum = x }
        }
        return minimum
    }

    private static func evaluateProduct(_ values: Span<Double>) -> Double {
        var nonUnitCount = 0
        var singleValue = 1.0
        var product = 1.0
        for index in values.indices {
            let x = clamp(values[index])
            if x == 0.0 { return 0.0 }
            if x == 1.0 { continue }
            nonUnitCount += 1
            singleValue = x
            product *= x
        }
        if nonUnitCount == 0 { return 1.0 }
        if nonUnitCount == 1 { return singleValue }
        return clamp(product)
    }

    private static func evaluateLukasiewicz(_ values: Span<Double>) -> Double {
        var nonUnitCount = 0
        var singleValue = 1.0
        var shortfallSum = 0.0
        for index in values.indices {
            let x = clamp(values[index])
            if x == 0.0 { return 0.0 }
            if x == 1.0 { continue }
            nonUnitCount += 1
            singleValue = x
            shortfallSum += (1.0 - x)
        }
        if nonUnitCount == 0 { return 1.0 }
        if nonUnitCount == 1 { return singleValue }
        return shortfallSum >= 1.0 ? 0.0 : clamp(1.0 - shortfallSum)
    }

    private static func conormGodel(_ values: Span<Double>) -> Double {
        var maximum = 0.0
        for index in values.indices {
            let x = clamp(values[index])
            if x == 1.0 { return 1.0 }
            if x > maximum { maximum = x }
        }
        return maximum
    }

    private static func conormProduct(_ values: Span<Double>) -> Double {
        var nonZeroCount = 0
        var singleValue = 0.0
        var complement = 1.0
        for index in values.indices {
            let x = clamp(values[index])
            if x == 1.0 { return 1.0 }
            if x == 0.0 { continue }
            nonZeroCount += 1
            singleValue = x
            complement *= (1.0 - x)
        }
        if nonZeroCount == 0 { return 0.0 }
        if nonZeroCount == 1 { return singleValue }
        return clamp(1.0 - complement)
    }

    private static func conormLukasiewicz(_ values: Span<Double>) -> Double {
        var nonZeroCount = 0
        var singleValue = 0.0
        var total = 0.0
        for index in values.indices {
            let x = clamp(values[index])
            if x == 1.0 { return 1.0 }
            if x == 0.0 { continue }
            nonZeroCount += 1
            singleValue = x
            total += x
        }
        if nonZeroCount == 0 { return 0.0 }
        if nonZeroCount == 1 { return singleValue }
        return total >= 1.0 ? 1.0 : clamp(total)
    }
}

public struct TNormSummary: Sendable, Codable, Equatable {
    public let godelSatisfaction: Double
    public let productSatisfaction: Double
    public let lukasiewiczSatisfaction: Double
    public let checkCount: Int

    public init(_ values: Span<Double>) {
        self.godelSatisfaction = TNorm.godel.evaluate(values)
        self.productSatisfaction = TNorm.product.evaluate(values)
        self.lukasiewiczSatisfaction = TNorm.lukasiewicz.evaluate(values)
        self.checkCount = values.count
    }

    public init(_ values: [Double]) {
        self.init(values.span)
    }

    public init(
        godelSatisfaction: Double,
        productSatisfaction: Double,
        lukasiewiczSatisfaction: Double,
        checkCount: Int
    ) {
        self.godelSatisfaction = TNorm.clamp(godelSatisfaction)
        self.productSatisfaction = TNorm.clamp(productSatisfaction)
        self.lukasiewiczSatisfaction = TNorm.clamp(lukasiewiczSatisfaction)
        self.checkCount = max(0, checkCount)
    }
}

public enum ClauseMargin {
    public static func upperCeiling(actual: Double, limit: Double, ideal: Double = 0.0) -> Double {
        guard actual.isFinite, limit.isFinite, ideal.isFinite, limit >= ideal else { return 0.0 }
        if limit == ideal { return actual <= limit ? 1.0 : 0.0 }
        if actual <= ideal { return 1.0 }
        if actual >= limit { return 0.0 }
        return TNorm.clamp((limit - actual) / (limit - ideal))
    }

    public static func lowerFloor(actual: Double, floor: Double, ideal: Double? = nil) -> Double {
        guard actual.isFinite, floor.isFinite else { return 0.0 }
        if let ideal {
            guard ideal.isFinite, ideal >= floor else { return 0.0 }
            if ideal == floor {
                guard floor > 0 else { return actual >= floor ? 1.0 : 0.0 }
                return actual >= floor ? 1.0 : TNorm.clamp(max(0.0, actual) / floor)
            }
            if actual >= ideal { return 1.0 }
            if actual <= floor { return 0.0 }
            return TNorm.clamp((actual - floor) / (ideal - floor))
        }
        guard floor > 0 else { return actual >= floor ? 1.0 : 0.0 }
        return TNorm.clamp(max(0.0, actual) / floor)
    }
}
