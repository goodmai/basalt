import Foundation
public import ComplexModule
public import RealModule

public enum MVNActivation: Sendable, Codable, Equatable {
    case continuousUnitCircle
    case discreteRoots(sectors: Int)
    case periodicRoots(sectors: Int, folds: Int)

    public static func positiveAngle(_ z: Complex<Float>) -> Float {
        guard z.isFinite, z.length >= 1e-12 else { return 0 }
        let raw = Foundation.atan2(Double(z.imaginary), Double(z.real))
        var angle = raw < 0 ? raw + 2.0 * .pi : raw
        if angle >= 2.0 * .pi { angle = 0 }
        return Float(angle)
    }

    public static func wedgeIndex(_ z: Complex<Float>, wedges: Int) -> Int {
        let count = max(1, wedges)
        guard z.isFinite, z.length >= 1e-12 else { return 0 }
        let raw = Foundation.atan2(Double(z.imaginary), Double(z.real))
        var angle = raw < 0 ? raw + 2.0 * .pi : raw
        if angle >= 2.0 * .pi { angle = 0 }
        let scaled = angle * Double(count) / (2.0 * .pi) + 1e-6
        let index = Int(scaled.rounded(.down))
        return ((index % count) + count) % count
    }

    public static func rootOfUnity(sector: Int, sectors: Int) -> Complex<Float> {
        let count = max(1, sectors)
        let wrapped = ((sector % count) + count) % count
        let angle = 2.0 * Double.pi * Double(wrapped) / Double(count)
        return Complex(Float(Foundation.cos(angle)), Float(Foundation.sin(angle)))
    }

    public static func wedgeBisector(wedge: Int, wedges: Int) -> Complex<Float> {
        let count = max(1, wedges)
        let wrapped = ((wedge % count) + count) % count
        let angle = 2.0 * Double.pi * (Double(wrapped) + 0.5) / Double(count)
        return Complex(Float(Foundation.cos(angle)), Float(Foundation.sin(angle)))
    }

    public func apply(_ z: Complex<Float>) -> Complex<Float> {
        switch self {
        case .continuousUnitCircle:
            guard z.isFinite else { return Complex(1, 0) }
            let length = z.length
            return length >= 1e-12 ? z.divided(by: length) : Complex(1, 0)
        case .discreteRoots(let sectors):
            let count = max(1, sectors)
            return Self.rootOfUnity(sector: Self.wedgeIndex(z, wedges: count), sectors: count)
        case .periodicRoots(let sectors, let folds):
            let count = max(1, sectors)
            let wedges = count * max(1, folds)
            return Self.rootOfUnity(sector: Self.wedgeIndex(z, wedges: wedges) % count, sectors: count)
        }
    }

    public func sector(of z: Complex<Float>, fallbackSectors: Int = 2) -> Int {
        switch self {
        case .continuousUnitCircle:
            return Self.wedgeIndex(z, wedges: max(1, fallbackSectors))
        case .discreteRoots(let sectors):
            return Self.wedgeIndex(z, wedges: max(1, sectors))
        case .periodicRoots(let sectors, let folds):
            let count = max(1, sectors)
            return Self.wedgeIndex(z, wedges: count * max(1, folds)) % count
        }
    }

    public func wedgeHalfAngle(fallbackSectors: Int = 2) -> Double {
        switch self {
        case .continuousUnitCircle:
            return .pi / Double(max(1, fallbackSectors))
        case .discreteRoots(let sectors):
            return .pi / Double(max(1, sectors))
        case .periodicRoots(let sectors, let folds):
            return .pi / Double(max(1, sectors) * max(1, folds))
        }
    }

    public func targetBisector(for z: Complex<Float>, targetSector: Int, fallbackSectors: Int = 2) -> Complex<Float> {
        switch self {
        case .continuousUnitCircle:
            return Self.wedgeBisector(wedge: targetSector, wedges: max(1, fallbackSectors))
        case .discreteRoots(let sectors):
            return Self.wedgeBisector(wedge: targetSector, wedges: max(1, sectors))
        case .periodicRoots(let sectors, let folds):
            let count = max(1, sectors)
            let foldCount = max(1, folds)
            let wedges = count * foldCount
            let wrapped = ((targetSector % count) + count) % count
            let direction = MVNActivation.continuousUnitCircle.apply(z)
            var best = Self.wedgeBisector(wedge: wrapped, wedges: wedges)
            var bestDot: Float = -2
            for fold in 0 ..< foldCount {
                let candidate = Self.wedgeBisector(wedge: wrapped + fold * count, wedges: wedges)
                let dot = (direction * candidate.conjugate).real
                if dot > bestDot {
                    bestDot = dot
                    best = candidate
                }
            }
            return best
        }
    }
}

public enum PhasorInputEncoding: String, Sendable, Codable, Equatable, CaseIterable {
    case rawPairs
    case unitPhasors

    public func encode(_ hidden: Span<Float>) -> [Complex<Float>] {
        let pairs = hidden.count / 2
        guard pairs > 0 else { return [] }
        var encoded: [Complex<Float>] = []
        encoded.reserveCapacity(pairs)
        for pair in 0 ..< pairs {
            let z = Complex(hidden[2 * pair], hidden[2 * pair + 1])
            switch self {
            case .rawPairs:
                encoded.append(z.isFinite ? z : .zero)
            case .unitPhasors:
                let length = z.length
                encoded.append(z.isFinite && length >= 1e-12 ? z.divided(by: length) : Complex(1, 0))
            }
        }
        return encoded
    }

    public func encode(_ hidden: [Float]) -> [Complex<Float>] {
        encode(hidden.span)
    }
}

public struct MVNProbe: Sendable, Codable, Equatable {
    public var bias: Complex<Float>
    public var weights: [Complex<Float>]
    public var activation: MVNActivation
    public var encoding: PhasorInputEncoding

    public init(
        bias: Complex<Float> = .zero,
        weights: [Complex<Float>],
        activation: MVNActivation = .discreteRoots(sectors: 2),
        encoding: PhasorInputEncoding = .unitPhasors
    ) {
        self.bias = bias
        self.weights = weights
        self.activation = activation
        self.encoding = encoding
    }

    public init(
        complexDimension: Int,
        activation: MVNActivation = .discreteRoots(sectors: 2),
        encoding: PhasorInputEncoding = .unitPhasors,
        seed: UInt64 = 42
    ) {
        var rng = SplitMix64(seed: seed)
        func uniform(_ bound: Double) -> Float {
            let unit = Double(rng.next() >> 11) * (1.0 / Double(1 << 53))
            return Float((2.0 * unit - 1.0) * bound)
        }
        let count = max(0, complexDimension)
        self.bias = Complex(uniform(0.1), uniform(0.1))
        self.weights = (0 ..< count).map { _ in Complex(uniform(0.25), uniform(0.25)) }
        self.activation = activation
        self.encoding = encoding
    }

    public func weightedSum(_ inputs: Span<Complex<Float>>) -> Complex<Float> {
        var sumRe = Double(bias.real)
        var sumIm = Double(bias.imaginary)
        let limit = min(weights.count, inputs.count)
        for index in 0 ..< limit {
            let w = weights[index]
            let x = inputs[index]
            sumRe += Double(w.real) * Double(x.real) - Double(w.imaginary) * Double(x.imaginary)
            sumIm += Double(w.real) * Double(x.imaginary) + Double(w.imaginary) * Double(x.real)
        }
        return Complex(Float(sumRe), Float(sumIm))
    }

    public func weightedSum(_ inputs: [Complex<Float>]) -> Complex<Float> {
        weightedSum(inputs.span)
    }

    public func forward(_ inputs: Span<Complex<Float>>) -> Complex<Float> {
        activation.apply(weightedSum(inputs))
    }

    public func forward(_ inputs: [Complex<Float>]) -> Complex<Float> {
        forward(inputs.span)
    }

    public func predictSector(_ inputs: Span<Complex<Float>>, sectors: Int = 2) -> Int {
        activation.sector(of: weightedSum(inputs), fallbackSectors: sectors)
    }

    public func predictSector(real hidden: Span<Float>, sectors: Int = 2) -> Int {
        let encoded = encoding.encode(hidden)
        return predictSector(encoded.span, sectors: sectors)
    }

    @discardableResult
    public mutating func step(
        _ inputs: Span<Complex<Float>>,
        targetSector: Int,
        sectors: Int = 2,
        learningRate: Float = 1.0
    ) -> Bool {
        let count = max(1, sectors)
        let wrapped = ((targetSector % count) + count) % count
        let z = weightedSum(inputs)
        if activation.sector(of: z, fallbackSectors: count) == wrapped {
            let target = activation.targetBisector(for: z, targetSector: wrapped, fallbackSectors: count)
            let unit = MVNActivation.continuousUnitCircle.apply(z)
            if case .continuousUnitCircle = activation {
                if (unit - target).length <= 1e-2 { return true }
            } else {
                let minCosine = Foundation.cos(activation.wedgeHalfAngle(fallbackSectors: count) * 0.2)
                if Double((unit * target.conjugate).real) >= minCosine { return true }
            }
        }
        let target = activation.targetBisector(for: z, targetSector: wrapped, fallbackSectors: count)
        let radius = max(z.isFinite ? z.length : 1.0, 1.0)
        applyDelta(target.multiplied(by: radius) - z, inputs: inputs, learningRate: learningRate)
        return false
    }

    @discardableResult
    public mutating func stepContinuous(
        _ inputs: Span<Complex<Float>>,
        targetPhasor: Complex<Float>,
        learningRate: Float = 1.0,
        tolerance: Float = 1e-3
    ) -> Bool {
        let z = weightedSum(inputs)
        let unit = MVNActivation.continuousUnitCircle.apply(z)
        let target = MVNActivation.continuousUnitCircle.apply(targetPhasor)
        if (unit - target).length <= tolerance { return true }
        let radius = max(z.isFinite ? z.length : 1.0, 1.0)
        applyDelta(target.multiplied(by: radius) - z, inputs: inputs, learningRate: learningRate)
        return false
    }

    @discardableResult
    public mutating func fit(
        samples: [[Complex<Float>]],
        labels: [Int],
        sectors: Int = 2,
        maxEpochs: Int = 200,
        learningRate: Float = 1.0
    ) -> Int {
        let pairCount = min(samples.count, labels.count)
        guard pairCount > 0, maxEpochs > 0 else { return 0 }
        for epoch in 1 ... maxEpochs {
            var allCorrect = true
            for index in 0 ..< pairCount {
                let sample = samples[index]
                if !step(sample.span, targetSector: labels[index], sectors: sectors, learningRate: learningRate) {
                    allCorrect = false
                }
            }
            if allCorrect { return epoch }
        }
        return maxEpochs
    }

    public func angularError(_ inputs: Span<Complex<Float>>, targetSector: Int, sectors: Int = 2) -> Double {
        let z = weightedSum(inputs)
        let unit = MVNActivation.continuousUnitCircle.apply(z)
        let target = activation.targetBisector(for: z, targetSector: targetSector, fallbackSectors: sectors)
        let cosine = min(max(Double((unit * target.conjugate).real), -1.0), 1.0)
        return Foundation.acos(cosine)
    }

    private mutating func applyDelta(_ delta: Complex<Float>, inputs: Span<Complex<Float>>, learningRate: Float) {
        let limit = min(weights.count, inputs.count)
        var normSq = 1.0
        for index in 0 ..< limit {
            normSq += Double(inputs[index].lengthSquared)
        }
        let scale = Float(Double(learningRate) / max(normSq, 1e-6))
        let stepValue = delta.multiplied(by: scale)
        bias += stepValue
        for index in 0 ..< limit {
            weights[index] += stepValue * inputs[index].conjugate
        }
    }
}

public struct LinearProbeBaseline: Sendable, Codable, Equatable {
    public var classes: Int
    public var featureDimension: Int
    public var weights: [Float]
    public var biases: [Float]

    public init(classes: Int, featureDimension: Int, seed: UInt64 = 42) {
        let k = max(2, classes)
        let d = max(0, featureDimension)
        var rng = SplitMix64(seed: seed)
        let scale = d > 0 ? 0.05 / Double(d).squareRoot() : 0.05
        self.classes = k
        self.featureDimension = d
        self.weights = (0 ..< k * d).map { _ in
            let unit = Double(rng.next() >> 11) * (1.0 / Double(1 << 53))
            return Float((2.0 * unit - 1.0) * scale)
        }
        self.biases = Array(repeating: 0, count: k)
    }

    public func predict(_ features: Span<Float>) -> Int {
        let limit = min(featureDimension, features.count)
        var bestClass = 0
        var bestLogit = -Double.infinity
        for c in 0 ..< classes {
            let offset = c * featureDimension
            var logit = Double(biases[c])
            for j in 0 ..< limit {
                logit += Double(weights[offset + j]) * Double(features[j])
            }
            if logit > bestLogit {
                bestLogit = logit
                bestClass = c
            }
        }
        return bestClass
    }

    public func predict(_ features: [Float]) -> Int {
        predict(features.span)
    }

    public mutating func fit(
        samples: [[Float]],
        labels: [Int],
        epochs: Int = 200,
        learningRate: Float = 1.0,
        weightDecay: Float = 1e-4
    ) {
        let count = min(samples.count, labels.count)
        guard count > 0, epochs > 0, classes >= 2 else { return }
        var normSum = 0.0
        for index in 0 ..< count {
            let row = samples[index]
            for j in 0 ..< min(featureDimension, row.count) {
                normSum += Double(row[j]) * Double(row[j])
            }
        }
        let stepSize = Float(Double(learningRate) / max(normSum / Double(count) + 1.0, 1.0))
        let invCount = 1.0 / Float(count)
        var gradW = Array(repeating: Float(0), count: weights.count)
        var gradB = Array(repeating: Float(0), count: biases.count)
        for _ in 0 ..< epochs {
            for idx in gradW.indices { gradW[idx] = 0 }
            for idx in gradB.indices { gradB[idx] = 0 }
            for index in 0 ..< count {
                let sample = samples[index]
                accumulateGradient(sample: sample.span, label: labels[index], gradW: &gradW, gradB: &gradB)
            }
            for idx in weights.indices {
                weights[idx] -= stepSize * (gradW[idx] * invCount + weightDecay * weights[idx])
            }
            for idx in biases.indices {
                biases[idx] -= stepSize * gradB[idx] * invCount
            }
        }
    }

    public func accuracy(samples: [[Float]], labels: [Int]) -> Double {
        let count = min(samples.count, labels.count)
        guard count > 0 else { return 0 }
        var hits = 0
        for index in 0 ..< count {
            let expected = ((labels[index] % classes) + classes) % classes
            let sample = samples[index]
            if predict(sample.span) == expected { hits += 1 }
        }
        return Double(hits) / Double(count)
    }

    private func accumulateGradient(
        sample: Span<Float>,
        label: Int,
        gradW: inout [Float],
        gradB: inout [Float]
    ) {
        let limit = min(featureDimension, sample.count)
        let target = ((label % classes) + classes) % classes
        var logits = Array(repeating: 0.0, count: classes)
        var maxLogit = -Double.infinity
        for c in 0 ..< classes {
            let offset = c * featureDimension
            var sum = Double(biases[c])
            for j in 0 ..< limit {
                sum += Double(weights[offset + j]) * Double(sample[j])
            }
            logits[c] = sum
            if sum > maxLogit { maxLogit = sum }
        }
        var denom = 0.0
        for c in 0 ..< classes {
            logits[c] = Foundation.exp(logits[c] - maxLogit)
            denom += logits[c]
        }
        for c in 0 ..< classes {
            let diff = Float(logits[c] / max(denom, 1e-12) - (c == target ? 1.0 : 0.0))
            gradB[c] += diff
            let offset = c * featureDimension
            for j in 0 ..< limit {
                gradW[offset + j] += diff * sample[j]
            }
        }
    }
}

public struct PhasorProbeReport: Sendable, Codable, Equatable {
    public var layer: Int
    public var sectors: Int
    public var trainAccuracy: Double
    public var heldOutAccuracy: Double
    public var linearHeldOutAccuracy: Double
    public var vsLinearDelta: Double
    public var angularMAE: Double
    public var epochsRun: Int

    public init(
        layer: Int,
        sectors: Int,
        trainAccuracy: Double = 0,
        heldOutAccuracy: Double = 0,
        linearHeldOutAccuracy: Double = 0,
        angularMAE: Double = 0
    ) {
        self.layer = layer
        self.sectors = sectors
        self.trainAccuracy = trainAccuracy
        self.heldOutAccuracy = heldOutAccuracy
        self.linearHeldOutAccuracy = linearHeldOutAccuracy
        self.vsLinearDelta = heldOutAccuracy - linearHeldOutAccuracy
        self.angularMAE = angularMAE
        self.epochsRun = 0
    }
}
