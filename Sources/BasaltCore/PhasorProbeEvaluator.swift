import Foundation
import ComplexModule
import RealModule

public struct LabeledPrompt: Sendable, Codable, Equatable {
    public var prompt: String
    public var sector: Int

    public init(prompt: String, sector: Int) {
        self.prompt = prompt
        self.sector = sector
    }
}

public struct RoleFillerTriple: Sendable, Codable, Equatable {
    public var bound: String
    public var role: String
    public var filler: String

    public init(bound: String, role: String, filler: String) {
        self.bound = bound
        self.role = role
        self.filler = filler
    }
}

public enum PhasorPooling: String, Sendable, Codable, Equatable, CaseIterable {
    case lastToken
    case meanPool
}

public struct PhasorProbeRequest: Sendable, Codable, Equatable {
    public var train: [LabeledPrompt]
    public var heldOut: [LabeledPrompt]
    public var triples: [RoleFillerTriple]
    public var sectors: Int = 2
    public var folds: Int = 1
    public var continuous = false
    public var encoding: PhasorInputEncoding = .unitPhasors
    public var pooling: PhasorPooling = .lastToken
    public var layers: [Int]?
    public var maxEpochs: Int = 200
    public var learningRate: Float = 1.0
    public var seed: UInt64 = 42

    public static let trainCount = 2...64
    public static let heldOutCount = 0...64
    public static let tripleCount = 1...32
    public static let sectorRange = 2...64
    public static let foldRange = 1...16
    public static let epochRange = 1...2000

    public init(
        train: [LabeledPrompt] = [],
        heldOut: [LabeledPrompt] = [],
        triples: [RoleFillerTriple] = [],
        sectors: Int = 2,
        layers: [Int]? = nil
    ) {
        self.train = train
        self.heldOut = heldOut
        self.triples = triples
        self.sectors = sectors
        self.layers = layers
    }

    public var activation: MVNActivation {
        if continuous { return .continuousUnitCircle }
        return folds > 1 ? .periodicRoots(sectors: sectors, folds: folds) : .discreteRoots(sectors: sectors)
    }

    public var uniquePrompts: [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for item in train + heldOut where seen.insert(item.prompt).inserted {
            ordered.append(item.prompt)
        }
        for triple in triples {
            for prompt in [triple.bound, triple.role, triple.filler] where seen.insert(prompt).inserted {
                ordered.append(prompt)
            }
        }
        return ordered
    }

    public func validate(layers count: Int) throws(BasaltError) {
        guard !train.isEmpty || !triples.isEmpty else {
            throw .invalidRequestStructure(details: "phasor_probe needs train prompts or role-filler triples")
        }
        try validateTrain()
        try validateTriples()
        guard Self.epochRange.contains(maxEpochs), learningRate.isFinite, learningRate > 0 else {
            throw .invalidRequestStructure(details: "maxEpochs must be in \(Self.epochRange) and learningRate positive")
        }
        if let layers, layers.isEmpty || layers.contains(where: { !(0 ..< count).contains($0) }) {
            throw .invalidRequestStructure(details: "layers must list at least one layer in 0..<\(count)")
        }
    }

    private func validateTrain() throws(BasaltError) {
        guard !train.isEmpty else { return }
        guard Self.trainCount.contains(train.count), Self.heldOutCount.contains(heldOut.count) else {
            throw .invalidRequestStructure(details: "phasor_probe needs \(Self.trainCount) train and \(Self.heldOutCount) heldOut prompts")
        }
        guard Self.sectorRange.contains(sectors), Self.foldRange.contains(folds) else {
            throw .invalidRequestStructure(details: "sectors must be in \(Self.sectorRange) and folds in \(Self.foldRange)")
        }
        for item in train + heldOut {
            guard !item.prompt.isEmpty, (0 ..< sectors).contains(item.sector) else {
                throw .invalidRequestStructure(details: "each labeled prompt must be non-empty with sector in 0..<\(sectors)")
            }
        }
    }

    private func validateTriples() throws(BasaltError) {
        guard !triples.isEmpty else { return }
        guard Self.tripleCount.contains(triples.count) else {
            throw .invalidRequestStructure(details: "triples must hold \(Self.tripleCount) {bound, role, filler} entries")
        }
        for triple in triples {
            guard !triple.bound.isEmpty, !triple.role.isEmpty, !triple.filler.isEmpty else {
                throw .invalidRequestStructure(details: "each triple must provide non-empty bound, role, and filler prompts")
            }
        }
    }
}

public struct PhasorProbeSummary: Sendable, Codable, Equatable {
    public var layers: [PhasorProbeReport]
    public var bestLayer: Int?
    public var binding: [FHRRBindingReport]
    public var bestBindingLayer: Int?

    public init(
        layers: [PhasorProbeReport] = [],
        bestLayer: Int? = nil,
        binding: [FHRRBindingReport] = [],
        bestBindingLayer: Int? = nil
    ) {
        self.layers = layers
        self.bestLayer = bestLayer
        self.binding = binding
        self.bestBindingLayer = bestBindingLayer
    }
}

public struct ExtractedPhasorStates: Sendable {
    public var layers: [Int]
    public var vectorsByPrompt: [String: [Int: [Float]]]

    public init(layers: [Int], vectorsByPrompt: [String: [Int: [Float]]]) {
        self.layers = layers
        self.vectorsByPrompt = vectorsByPrompt
    }
}

public struct PhasorLayerDataset: Sendable {
    public var trainStates: [[Float]]
    public var trainLabels: [Int]
    public var heldOutStates: [[Float]]
    public var heldOutLabels: [Int]

    public init(
        trainStates: [[Float]],
        trainLabels: [Int],
        heldOutStates: [[Float]] = [],
        heldOutLabels: [Int] = []
    ) {
        self.trainStates = trainStates
        self.trainLabels = trainLabels
        self.heldOutStates = heldOutStates
        self.heldOutLabels = heldOutLabels
    }
}

public enum PhasorProbeEvaluator {
    public static func evaluateLayer(
        layer: Int,
        dataset: PhasorLayerDataset,
        request: PhasorProbeRequest
    ) -> PhasorProbeReport {
        let evalStates = dataset.heldOutStates.isEmpty ? dataset.trainStates : dataset.heldOutStates
        let evalLabels = dataset.heldOutLabels.isEmpty ? dataset.trainLabels : dataset.heldOutLabels
        let featureDim = dataset.trainStates.first?.count ?? 0
        let trainComplex = dataset.trainStates.map { state in request.encoding.encode(state.span) }
        let evalComplex = evalStates.map { state in request.encoding.encode(state.span) }
        var mvn = MVNProbe(
            complexDimension: featureDim / 2,
            activation: request.activation,
            encoding: request.encoding,
            seed: request.seed
        )
        let epochs = mvn.fit(
            samples: trainComplex,
            labels: dataset.trainLabels,
            sectors: request.sectors,
            maxEpochs: request.maxEpochs,
            learningRate: request.learningRate
        )
        let trainAcc = sectorAccuracy(probe: mvn, samples: trainComplex, labels: dataset.trainLabels, sectors: request.sectors)
        let heldOutAcc = sectorAccuracy(probe: mvn, samples: evalComplex, labels: evalLabels, sectors: request.sectors)
        let mae = meanAngularError(probe: mvn, samples: evalComplex, labels: evalLabels, sectors: request.sectors)
        var linear = LinearProbeBaseline(classes: request.sectors, featureDimension: featureDim, seed: request.seed)
        linear.fit(samples: dataset.trainStates, labels: dataset.trainLabels, epochs: request.maxEpochs)
        let linearAcc = linear.accuracy(samples: evalStates, labels: evalLabels)
        var report = PhasorProbeReport(
            layer: layer,
            sectors: request.sectors,
            trainAccuracy: trainAcc,
            heldOutAccuracy: heldOutAcc,
            linearHeldOutAccuracy: linearAcc,
            angularMAE: mae
        )
        report.epochsRun = epochs
        return report
    }

    public static func evaluate(request: PhasorProbeRequest, extracted: ExtractedPhasorStates) -> PhasorProbeSummary {
        var reports: [PhasorProbeReport] = []
        if !request.train.isEmpty {
            for layer in extracted.layers {
                guard let dataset = makeLayerDataset(layer: layer, request: request, extracted: extracted) else { continue }
                reports.append(evaluateLayer(layer: layer, dataset: dataset, request: request))
            }
        }
        var bindingReports: [FHRRBindingReport] = []
        if !request.triples.isEmpty {
            for layer in extracted.layers {
                let stateTriples: [(bound: [Float], role: [Float], filler: [Float])] = request.triples.compactMap { triple in
                    guard let bound = extracted.vectorsByPrompt[triple.bound]?[layer],
                          let role = extracted.vectorsByPrompt[triple.role]?[layer],
                          let filler = extracted.vectorsByPrompt[triple.filler]?[layer] else { return nil }
                    return (bound: bound, role: role, filler: filler)
                }
                bindingReports.append(FHRRBinding.evaluate(layer: layer, triples: stateTriples, seed: request.seed))
            }
        }
        let best = reports.max { ($0.heldOutAccuracy, $0.vsLinearDelta) < ($1.heldOutAccuracy, $1.vsLinearDelta) }?.layer
        let bestBinding = bindingReports.max { $0.zScore < $1.zScore }?.layer
        return PhasorProbeSummary(layers: reports, bestLayer: best, binding: bindingReports, bestBindingLayer: bestBinding)
    }

    private static func makeLayerDataset(
        layer: Int,
        request: PhasorProbeRequest,
        extracted: ExtractedPhasorStates
    ) -> PhasorLayerDataset? {
        let trainPairs = request.train.compactMap { item -> ([Float], Int)? in
            guard let vec = extracted.vectorsByPrompt[item.prompt]?[layer] else { return nil }
            return (vec, item.sector)
        }
        guard !trainPairs.isEmpty else { return nil }
        let heldOutPairs = request.heldOut.compactMap { item -> ([Float], Int)? in
            guard let vec = extracted.vectorsByPrompt[item.prompt]?[layer] else { return nil }
            return (vec, item.sector)
        }
        return PhasorLayerDataset(
            trainStates: trainPairs.map(\.0),
            trainLabels: trainPairs.map(\.1),
            heldOutStates: heldOutPairs.map(\.0),
            heldOutLabels: heldOutPairs.map(\.1)
        )
    }

    private static func sectorAccuracy(
        probe: MVNProbe,
        samples: [[Complex<Float>]],
        labels: [Int],
        sectors: Int
    ) -> Double {
        let count = min(samples.count, labels.count)
        guard count > 0 else { return 0 }
        let k = max(1, sectors)
        var hits = 0
        for index in 0 ..< count {
            let expected = ((labels[index] % k) + k) % k
            let sample = samples[index]
            if probe.predictSector(sample.span, sectors: k) == expected { hits += 1 }
        }
        return Double(hits) / Double(count)
    }

    private static func meanAngularError(
        probe: MVNProbe,
        samples: [[Complex<Float>]],
        labels: [Int],
        sectors: Int
    ) -> Double {
        let count = min(samples.count, labels.count)
        guard count > 0 else { return 0 }
        var sum = 0.0
        for index in 0 ..< count {
            let sample = samples[index]
            sum += probe.angularError(sample.span, targetSector: labels[index], sectors: sectors)
        }
        return sum / Double(count)
    }
}
