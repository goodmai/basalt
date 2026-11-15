import Foundation

public enum BankSlotKind: String, Sendable, Codable, Equatable, CaseIterable {
    case noul
    case choice
    case score64
}

public struct BankSlot: Sendable, Codable, Equatable {
    public static let scoreBins = 64
    public static let scoreMax = 10.0

    public var kind: BankSlotKind
    public var slice: Range<Int>
    public var options: [String]
    public var temperature: Double
    public var bias: Double

    public init(
        kind: BankSlotKind,
        slice: Range<Int>,
        options: [String] = [],
        temperature: Double = 1.0,
        bias: Double = 0.0
    ) {
        self.kind = kind
        self.slice = slice
        self.options = options
        self.temperature = temperature
        self.bias = bias
    }
}

public struct SystemOneBankManifest: Sendable, Codable, Equatable {
    public static let version = 1
    public static let fileName = "bank_manifest.json"

    public var schemaVersion: Int
    public var baseModelID: String
    public var hiddenSize: Int
    public var totalOutputs: Int
    public var modelID: String?
    public var slots: [String: BankSlot]

    public init(
        baseModelID: String,
        hiddenSize: Int,
        totalOutputs: Int,
        slots: [String: BankSlot],
        modelID: String? = nil,
        schemaVersion: Int = SystemOneBankManifest.version
    ) {
        self.schemaVersion = schemaVersion
        self.baseModelID = baseModelID
        self.hiddenSize = hiddenSize
        self.totalOutputs = totalOutputs
        self.modelID = modelID
        self.slots = slots
    }

    public func validate(path: String) throws(BasaltError) {
        guard schemaVersion == Self.version else {
            throw .modelConfigurationInvalid(path: path, reason: "bank_manifest schemaVersion \(schemaVersion) != \(Self.version)")
        }
        guard !baseModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw .modelConfigurationInvalid(path: path, reason: "bank_manifest baseModelID cannot be empty")
        }
        guard hiddenSize > 0, totalOutputs > 0 else {
            throw .modelConfigurationInvalid(path: path, reason: "bank_manifest hiddenSize and totalOutputs must be positive")
        }
        guard !slots.isEmpty else {
            throw .modelConfigurationInvalid(path: path, reason: "bank_manifest must declare at least one slot")
        }
        for (name, slot) in slots {
            try Self.validateSlot(slot, name: name, totalOutputs: totalOutputs, path: path)
        }
    }

    static func validateSlot(_ slot: BankSlot, name: String, totalOutputs: Int, path: String) throws(BasaltError) {
        guard !name.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw .modelConfigurationInvalid(path: path, reason: "bank_manifest contains a blank slot ID")
        }
        guard slot.slice.lowerBound >= 0, slot.slice.upperBound <= totalOutputs, !slot.slice.isEmpty else {
            throw .modelConfigurationInvalid(path: path, reason: "slot \(name) slice \(slot.slice) lies outside 0..<\(totalOutputs)")
        }
        guard slot.temperature.isFinite, slot.temperature > 0, slot.bias.isFinite else {
            throw .modelConfigurationInvalid(path: path, reason: "slot \(name) needs finite bias and positive temperature")
        }
        switch slot.kind {
        case .noul:
            guard slot.slice.count == 1 || slot.slice.count == 2 else {
                throw .modelConfigurationInvalid(path: path, reason: "noul slot \(name) needs 1 or 2 outputs, got \(slot.slice.count)")
            }
        case .choice:
            guard slot.slice.count >= 2, slot.options.count == slot.slice.count, Set(slot.options).count == slot.options.count else {
                throw .modelConfigurationInvalid(path: path, reason: "choice slot \(name) needs >= 2 distinct options matching slice length")
            }
        case .score64:
            guard slot.slice.count == BankSlot.scoreBins else {
                throw .modelConfigurationInvalid(path: path, reason: "score64 slot \(name) needs \(BankSlot.scoreBins) outputs, got \(slot.slice.count)")
            }
        }
    }
}

public enum SystemOneBankCore {
    public typealias Covered = (id: String, question: SystemOneQuestion, slot: BankSlot)

    public static func partition(
        _ questions: [String: SystemOneQuestion],
        manifest: SystemOneBankManifest
    ) -> (covered: [Covered], uncovered: [String: SystemOneQuestion]) {
        var covered: [Covered] = []
        var uncovered: [String: SystemOneQuestion] = [:]
        for key in questions.keys.sorted() {
            guard let question = questions[key] else { continue }
            if let slot = manifest.slots[key], matches(question, slot: slot) {
                covered.append((id: key, question: question, slot: slot))
            } else {
                uncovered[key] = question
            }
        }
        return (covered, uncovered)
    }

    public static func matches(_ question: SystemOneQuestion, slot: BankSlot) -> Bool {
        switch (question.type, slot.kind) {
        case (.noul, .noul):
            return slot.slice.count == 1 || slot.slice.count == 2
        case (.choice, .choice):
            let requested = Set(SystemOneCore.options(question).keys)
            return slot.options.count == slot.slice.count && requested == Set(slot.options)
        case (.score, .score64):
            return slot.slice.count == BankSlot.scoreBins
        default:
            return false
        }
    }

    public static func decode(
        _ covered: [Covered],
        logits: Span<Float>,
        request: SystemOneRequest
    ) -> [String: SystemOneAnswer] {
        let calibrated = request.calibrated != false
        var answers: [String: SystemOneAnswer] = [:]
        answers.reserveCapacity(covered.count)
        for entry in covered {
            guard entry.slot.slice.lowerBound >= 0, entry.slot.slice.upperBound <= logits.count else { continue }
            let slice = logits.extracting(entry.slot.slice)
            let raw = decodeSlot(entry.slot, question: entry.question, slice: slice, calibrated: calibrated)
            answers[entry.id] = SystemOneCore.gated(raw, abstainBelow: request.abstainBelow, minMargin: request.minMargin)
        }
        return answers
    }

    public static func decodeSlot(
        _ slot: BankSlot,
        question: SystemOneQuestion,
        slice: Span<Float>,
        calibrated: Bool = true
    ) -> SystemOneAnswer {
        switch slot.kind {
        case .noul:
            return decodeNoul(slot, slice: slice, calibrated: calibrated)
        case .choice:
            return decodeChoice(slot, question: question, slice: slice, calibrated: calibrated)
        case .score64:
            return decodeScore64(slot, question: question, slice: slice, calibrated: calibrated)
        }
    }

    static func decodeNoul(_ slot: BankSlot, slice: Span<Float>, calibrated: Bool) -> SystemOneAnswer {
        var answer = SystemOneAnswer()
        guard slice.count >= 1 else { return answer }
        let raw = slice.count >= 2 ? Double(slice[1]) - Double(slice[0]) : Double(slice[0])
        let scale = calibrated ? slot.temperature : 1.0
        let shift = calibrated ? slot.bias : 0.0
        let probability = 1.0 / (1.0 + exp(-(raw / scale + shift)))
        answer.noul = probability
        answer.confidence = abs(2.0 * probability - 1.0)
        return answer
    }

    static func decodeChoice(
        _ slot: BankSlot,
        question: SystemOneQuestion,
        slice: Span<Float>,
        calibrated: Bool
    ) -> SystemOneAnswer {
        let scale = calibrated ? slot.temperature : 1.0
        let p = softmax(slice, temperature: scale)
        var answer = SystemOneAnswer()
        guard p.count == slot.options.count, !p.isEmpty else { return answer }
        answer.probabilities = Dictionary(uniqueKeysWithValues: zip(slot.options, p))
        answer.confidence = SystemOneCore.choiceConfidence(p)
        let best = p.max() ?? 0
        let named: [String: String] = if case .named(let map)? = question.criteria { map } else { [:] }
        let tied = p.indices.filter { p[$0] >= best - 1e-7 }
        let selected = tied.min { lhs, rhs in
            let leftDesc = named[slot.options[lhs]] ?? slot.options[lhs]
            let rightDesc = named[slot.options[rhs]] ?? slot.options[rhs]
            return (leftDesc.lowercased(), leftDesc, slot.options[lhs]) < (rightDesc.lowercased(), rightDesc, slot.options[rhs])
        }
        answer.choice = selected.map { slot.options[$0] }
        return answer
    }

    static func decodeScore64(
        _ slot: BankSlot,
        question: SystemOneQuestion,
        slice: Span<Float>,
        calibrated: Bool
    ) -> SystemOneAnswer {
        let scale = calibrated ? slot.temperature : 1.0
        let shift = calibrated ? slot.bias : 0.0
        let p = softmax(slice, temperature: scale)
        var answer = SystemOneAnswer()
        guard p.count == BankSlot.scoreBins else { return answer }
        var expectation = 0.0
        var entropy = 0.0
        let denom = Double(BankSlot.scoreBins - 1)
        for index in 0 ..< BankSlot.scoreBins {
            let prob = p[index]
            expectation += prob * (BankSlot.scoreMax * Double(index) / denom)
            if prob > 0 { entropy -= prob * log(prob) }
        }
        answer.score = min(max(expectation + shift, 0.0), BankSlot.scoreMax)
        let maxEntropy = log(Double(BankSlot.scoreBins))
        answer.confidence = min(max(1.0 - entropy / maxEntropy, 0.0), 1.0)
        if case .levels(let levels)? = question.criteria, levels.count >= 2 {
            answer.legend = levels
            var buckets = [Double](repeating: 0.0, count: levels.count)
            for index in 0 ..< BankSlot.scoreBins {
                let bucket = min(levels.count - 1, (index * levels.count) / BankSlot.scoreBins)
                buckets[bucket] += p[index]
            }
            answer.probabilities = Dictionary(uniqueKeysWithValues: buckets.indices.map { (String($0), buckets[$0]) })
        }
        return answer
    }

    public static func softmax(_ slice: Span<Float>, temperature: Double) -> [Double] {
        guard slice.count > 0 else { return [] }
        var top = -Double.infinity
        for index in 0 ..< slice.count {
            let scaled = Double(slice[index]) / temperature
            if scaled > top { top = scaled }
        }
        guard top.isFinite else { return [Double](repeating: 1.0 / Double(slice.count), count: slice.count) }
        var weights = [Double](repeating: 0.0, count: slice.count)
        var total = 0.0
        for index in 0 ..< slice.count {
            let w = exp(Double(slice[index]) / temperature - top)
            weights[index] = w
            total += w
        }
        guard total > 0, total.isFinite else { return [Double](repeating: 1.0 / Double(slice.count), count: slice.count) }
        for index in 0 ..< weights.count { weights[index] /= total }
        return weights
    }
}
