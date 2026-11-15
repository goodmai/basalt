import Foundation
public import ComplexModule
public import RealModule

public struct PhasorVector: Sendable, Codable, Equatable {
    public var elements: [Complex<Float>]
    public var count: Int { elements.count }

    public init(elements: [Complex<Float>]) { self.elements = elements.map(FHRRBinding.unitPhasor) }
    public init(real hidden: Span<Float>) { self.elements = PhasorInputEncoding.unitPhasors.encode(hidden) }
    public init(real hidden: [Float]) { self.init(real: hidden.span) }
    fileprivate init(validated elements: [Complex<Float>]) { self.elements = elements }

    public static func random(dimension: Int, seed: UInt64) -> PhasorVector {
        var rng = SplitMix64(seed: seed)
        let count = max(0, dimension)
        var out: [Complex<Float>] = []
        out.reserveCapacity(count)
        for _ in 0 ..< count {
            let unit = Double(rng.next() >> 11) * (1.0 / Double(1 << 53))
            let angle = 2.0 * Double.pi * unit
            out.append(Complex(Float(Foundation.cos(angle)), Float(Foundation.sin(angle))))
        }
        return PhasorVector(validated: out)
    }

    public func bound(with other: PhasorVector) -> PhasorVector {
        let lhs = elements, rhs = other.elements
        return PhasorVector(validated: FHRRBinding.bind(lhs.span, rhs.span))
    }

    public func unbound(key: PhasorVector) -> PhasorVector {
        let lhs = elements, rhs = key.elements
        return PhasorVector(validated: FHRRBinding.unbind(bound: lhs.span, key: rhs.span))
    }

    public func similarity(to other: PhasorVector) -> Float {
        let lhs = elements, rhs = other.elements
        return FHRRBinding.similarity(lhs.span, rhs.span)
    }
}

public enum FHRRBinding {
    public static func unitPhasor(_ z: Complex<Float>) -> Complex<Float> {
        guard z.isFinite else { return Complex(1, 0) }
        let length = z.length
        return length >= 1e-12 ? z.divided(by: length) : Complex(1, 0)
    }

    public static func bind(_ a: Span<Complex<Float>>, _ b: Span<Complex<Float>>) -> [Complex<Float>] {
        let count = min(a.count, b.count)
        guard count > 0 else { return [] }
        var out: [Complex<Float>] = []
        out.reserveCapacity(count)
        for index in 0 ..< count { out.append(unitPhasor(a[index] * b[index])) }
        return out
    }

    public static func unbind(bound: Span<Complex<Float>>, key: Span<Complex<Float>>) -> [Complex<Float>] {
        let count = min(bound.count, key.count)
        guard count > 0 else { return [] }
        var out: [Complex<Float>] = []
        out.reserveCapacity(count)
        for index in 0 ..< count { out.append(unitPhasor(bound[index] * key[index].conjugate)) }
        return out
    }

    public static func bundle(_ items: [[Complex<Float>]]) -> [Complex<Float>] {
        guard let count = items.map(\.count).min(), count > 0 else { return [] }
        var out: [Complex<Float>] = []
        out.reserveCapacity(count)
        for index in 0 ..< count {
            var sumRe = 0.0, sumIm = 0.0
            for item in items {
                sumRe += Double(item[index].real)
                sumIm += Double(item[index].imaginary)
            }
            out.append(unitPhasor(Complex(Float(sumRe), Float(sumIm))))
        }
        return out
    }

    public static func bundle(_ items: [PhasorVector]) -> PhasorVector {
        PhasorVector(validated: bundle(items.map(\.elements)))
    }

    public static func similarity(_ a: Span<Complex<Float>>, _ b: Span<Complex<Float>>) -> Float {
        let count = min(a.count, b.count)
        guard count > 0 else { return 0 }
        var sum = 0.0
        for index in 0 ..< count {
            sum += Double(a[index].real) * Double(b[index].real) + Double(a[index].imaginary) * Double(b[index].imaginary)
        }
        return Float(min(max(sum / Double(count), -1.0), 1.0))
    }

    public static func evaluate(
        layer: Int = 0,
        triples: [(bound: [Float], role: [Float], filler: [Float])],
        nullTrials: Int = 16,
        seed: UInt64 = 42
    ) -> FHRRBindingReport {
        guard !triples.isEmpty else {
            return FHRRBindingReport(layer: layer, unbindingSimilarity: 0, randomBaselineSimilarity: 0, additiveBaselineSimilarity: 0, zScore: 0)
        }
        let trials = max(1, nullTrials)
        var unbindSum = 0.0, additiveSum = 0.0, randomSum = 0.0, totalDimension = 0
        for (offset, triple) in triples.enumerated() {
            let bound = PhasorVector(real: triple.bound)
            let role = PhasorVector(real: triple.role)
            let filler = PhasorVector(real: triple.filler)
            let dimension = min(bound.count, min(role.count, filler.count))
            guard dimension > 0 else { continue }
            totalDimension += dimension
            let boundElems = bound.elements, roleElems = role.elements, fillerElems = filler.elements
            let recovered = unbind(bound: boundElems.span, key: roleElems.span)
            unbindSum += Double(similarity(recovered.span, fillerElems.span))
            let additive = PhasorVector(real: zip(triple.bound, triple.role).map(-))
            let additiveElems = additive.elements
            additiveSum += Double(similarity(additiveElems.span, fillerElems.span))
            for trial in 0 ..< trials {
                let randomElems = PhasorVector.random(dimension: dimension, seed: seed &+ UInt64(offset * trials + trial + 1)).elements
                let nullRecovered = unbind(bound: boundElems.span, key: randomElems.span)
                randomSum += Double(similarity(nullRecovered.span, fillerElems.span))
            }
        }
        let count = Double(triples.count)
        let meanDim = max(1.0, Double(totalDimension) / count)
        let unbindingMean = unbindSum / count
        let randomMean = randomSum / (count * Double(trials))
        let nullStd = 1.0 / (2.0 * meanDim * count).squareRoot()
        return FHRRBindingReport(
            layer: layer, unbindingSimilarity: unbindingMean, randomBaselineSimilarity: randomMean,
            additiveBaselineSimilarity: additiveSum / count, zScore: (unbindingMean - randomMean) / max(nullStd, 1e-9))
    }
}

public struct FHRRBindingReport: Sendable, Codable, Equatable {
    public var layer: Int
    public var unbindingSimilarity: Double
    public var randomBaselineSimilarity: Double
    public var additiveBaselineSimilarity: Double
    public var zScore: Double

    public init(
        layer: Int = 0, unbindingSimilarity: Double, randomBaselineSimilarity: Double,
        additiveBaselineSimilarity: Double, zScore: Double
    ) {
        self.layer = layer
        self.unbindingSimilarity = unbindingSimilarity
        self.randomBaselineSimilarity = randomBaselineSimilarity
        self.additiveBaselineSimilarity = additiveBaselineSimilarity
        self.zScore = zScore
    }
}
