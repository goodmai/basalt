import Foundation

public struct PreferencePair: Sendable, Equatable, Codable {
    public var prompt: String
    public var chosen: String
    public var rejected: String

    public init(prompt: String, chosen: String, rejected: String) {
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected
    }
}

public enum PreferencePairsLoader {
    public static func load(from path: String) throws(BasaltError) -> [PreferencePair] {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let text: String
        do {
            text = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw .invalidRequestStructure(details: "dpo: cannot read preference pairs file \(url.path)")
        }
        return try parse(jsonl: text, sourceName: url.lastPathComponent)
    }

    public static func parse(jsonl: String, sourceName: String = "pairs.jsonl") throws(BasaltError) -> [PreferencePair] {
        let lines = jsonl.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        var pairs: [PreferencePair] = []
        pairs.reserveCapacity(lines.count)
        for (index, rawLine) in lines.enumerated() {
            let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            pairs.append(try parseLine(trimmed, lineNumber: index + 1, sourceName: sourceName))
        }
        guard !pairs.isEmpty else {
            throw .invalidRequestStructure(details: "dpo: preference pairs file \(sourceName) is empty")
        }
        return pairs
    }

    private static func parseLine(_ line: String, lineNumber: Int, sourceName: String) throws(BasaltError) -> PreferencePair {
        guard let object = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else {
            throw .invalidRequestStructure(details: "dpo: line \(lineNumber) of \(sourceName) is not a JSON object")
        }
        guard let prompt = extractPrompt(from: object) else {
            throw .invalidRequestStructure(details: "dpo: line \(lineNumber) of \(sourceName) is missing a non-empty prompt or messages field")
        }
        guard let chosen = extractText(from: object["chosen"]) else {
            throw .invalidRequestStructure(details: "dpo: line \(lineNumber) of \(sourceName) is missing a non-empty chosen field")
        }
        guard let rejected = extractText(from: object["rejected"]) else {
            throw .invalidRequestStructure(details: "dpo: line \(lineNumber) of \(sourceName) is missing a non-empty rejected field")
        }
        let chosenNorm = chosen.trimmingCharacters(in: .whitespacesAndNewlines)
        let rejectedNorm = rejected.trimmingCharacters(in: .whitespacesAndNewlines)
        guard chosen != rejected, chosenNorm != rejectedNorm else {
            throw .invalidRequestStructure(details: "dpo: line \(lineNumber) of \(sourceName) has identical chosen and rejected responses")
        }
        return PreferencePair(prompt: prompt, chosen: chosen, rejected: rejected)
    }

    private static func extractPrompt(from object: [String: Any]) -> String? {
        if let direct = object["prompt"] as? String,
           !direct.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return direct
        }
        guard let messages = object["messages"] as? [[String: Any]], !messages.isEmpty else {
            return nil
        }
        return formatMessages(messages)
    }

    private static func formatMessages(_ messages: [[String: Any]]) -> String? {
        var parts: [(role: String, content: String)] = []
        parts.reserveCapacity(messages.count)
        for message in messages {
            guard let content = message["content"] as? String,
                  !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }
            let role = (message["role"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            parts.append((role: role, content: content))
        }
        guard !parts.isEmpty else { return nil }
        if parts.count == 1 { return parts[0].content }
        return parts.map { $0.role.isEmpty ? $0.content : "\($0.role): \($0.content)" }.joined(separator: "\n")
    }

    private static func extractText(from value: Any?) -> String? {
        if let text = value as? String {
            return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : text
        }
        if let messages = value as? [[String: Any]] {
            for message in messages.reversed() {
                if let content = message["content"] as? String,
                   !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    return content
                }
            }
        }
        return nil
    }
}

public struct OfflineDPOConfig: Sendable, Equatable, Codable {
    public var pairsPath: String
    public var beta: Float = 0.1
    public var iterations: Int = 50
    public var learningRate: Float = 5e-5
    public var batchSize: Int = 2
    public var maxSeqLen: Int = 1024
    public var labelSmoothing: Float = 0.0
    public var referenceFree: Bool = false
    public var rank: Int = 16
    public var layers: [Int]?
    public var method: AdapterMethod?
    public var optimizer: TrainingOptimizerKind = .adamw
    public var seed: UInt64 = 42
    public var maxGradNorm: Float = 1.0
    public var weightDecay: Float = 0.01
    public var initialAdapterPath: String?
    public var output: String?
    public var applyOverlay: Bool = false
    public var gate: GateRequest?

    public var effectiveMethod: AdapterMethod { method ?? .lora(rank: rank) }

    public init(
        pairsPath: String,
        beta: Float = 0.1,
        iterations: Int = 50,
        learningRate: Float = 5e-5,
        batchSize: Int = 2,
        maxSeqLen: Int = 1024
    ) {
        self.pairsPath = pairsPath
        self.beta = beta
        self.iterations = iterations
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.maxSeqLen = maxSeqLen
    }

    public func validate(prefix: String = "dpo") throws(BasaltError) {
        guard !pairsPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw .invalidRequestStructure(details: "\(prefix): pairsPath cannot be empty")
        }
        try ConfigBounds.requirePositive(beta, max: 100, name: "beta", prefix: prefix)
        try ConfigBounds.requireInt(iterations, in: 1 ... 10_000, name: "iterations", prefix: prefix)
        try ConfigBounds.requirePositive(learningRate, max: 1, name: "learningRate", prefix: prefix)
        try ConfigBounds.requireInt(batchSize, in: 1 ... 64, name: "batchSize", prefix: prefix)
        try ConfigBounds.requireInt(maxSeqLen, in: 16 ... 32_768, name: "maxSeqLen", prefix: prefix)
        try ConfigBounds.requireHalfOpen(labelSmoothing, in: 0.0 ..< 0.5, name: "labelSmoothing", prefix: prefix)
        try ConfigBounds.requirePositive(maxGradNorm, max: Float.greatestFiniteMagnitude, name: "maxGradNorm", prefix: prefix)
        try ConfigBounds.requireNonNegative(weightDecay, max: Float.greatestFiniteMagnitude, name: "weightDecay", prefix: prefix)
        try effectiveMethod.validate(prefix: prefix)
    }
}

public struct OfflineDPOReport: Sendable, Equatable, Codable {
    public var losses: [Float] = []
    public var meanMarginBefore: Float = 0
    public var meanMarginAfter: Float = 0
    public var preferenceAccuracyBefore: Float = 0
    public var preferenceAccuracyAfter: Float = 0
    public var klDivergence: Float = 0
    public var outputPath: String = ""
    public var gatePassed: Bool?

    public init(outputPath: String = "") {
        self.outputPath = outputPath
    }
}

public struct GRPOConfig: Sendable, Equatable, Codable {
    public var groupSize: Int = 4
    public var betaKL: Float = 0.04
    public var clipEpsilon: Float = 0.2
    public var rounds: Int = 8
    public var ppoEpochs: Int = 2
    public var maxTokens: Int = 64
    public var temperature: Float = 0.7
    public var learningRate: Float = 5e-5
    public var seed: UInt64 = 42
    public var tasks: TrainingTasks = .arithmetic(count: 8, seed: 42)
    public var rank: Int = 16
    public var layers: [Int]?
    public var method: AdapterMethod?
    public var optimizer: TrainingOptimizerKind = .adamw
    public var maxGradNorm: Float = 1.0
    public var weightDecay: Float = 0.01
    public var output: String?
    public var gate: GateRequest = GateRequest(heldOut: CorpusTrainRequest.defaultHeldOutPrompts)
    public var applyOverlay: Bool = false

    public var effectiveMethod: AdapterMethod { method ?? .lora(rank: rank) }

    public init(
        groupSize: Int = 4,
        betaKL: Float = 0.04,
        clipEpsilon: Float = 0.2,
        rounds: Int = 8,
        learningRate: Float = 5e-5,
        seed: UInt64 = 42
    ) {
        self.groupSize = groupSize
        self.betaKL = betaKL
        self.clipEpsilon = clipEpsilon
        self.rounds = rounds
        self.learningRate = learningRate
        self.seed = seed
    }

    public func validate(prefix: String = "grpo") throws(BasaltError) {
        try ConfigBounds.requireInt(groupSize, in: 2 ... 64, name: "groupSize", prefix: prefix)
        try ConfigBounds.requireNonNegative(betaKL, max: 100, name: "betaKL", prefix: prefix)
        try ConfigBounds.requireOpenUnit(clipEpsilon, name: "clipEpsilon", prefix: prefix)
        try ConfigBounds.requireInt(rounds, in: 1 ... 10_000, name: "rounds", prefix: prefix)
        try ConfigBounds.requireInt(ppoEpochs, in: 1 ... 16, name: "ppoEpochs", prefix: prefix)
        try ConfigBounds.requireInt(maxTokens, in: 1 ... 4096, name: "maxTokens", prefix: prefix)
        try ConfigBounds.requirePositive(temperature, max: 2, name: "temperature", prefix: prefix)
        try ConfigBounds.requirePositive(learningRate, max: 1, name: "learningRate", prefix: prefix)
        try ConfigBounds.requirePositive(maxGradNorm, max: Float.greatestFiniteMagnitude, name: "maxGradNorm", prefix: prefix)
        try ConfigBounds.requireNonNegative(weightDecay, max: Float.greatestFiniteMagnitude, name: "weightDecay", prefix: prefix)
        try effectiveMethod.validate(prefix: prefix)
    }
}

private enum ConfigBounds {
    static func requireInt(_ value: Int, in range: ClosedRange<Int>, name: String, prefix: String) throws(BasaltError) {
        guard range.contains(value) else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must be in \(range)")
        }
    }

    static func requirePositive(_ value: Float, max upper: Float, name: String, prefix: String) throws(BasaltError) {
        guard value.isFinite, value > 0, value <= upper else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must be finite and in (0, \(upper)]")
        }
    }

    static func requireNonNegative(_ value: Float, max upper: Float, name: String, prefix: String) throws(BasaltError) {
        guard value.isFinite, value >= 0, value <= upper else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must be finite and in [0, \(upper)]")
        }
    }

    static func requireHalfOpen(_ value: Float, in range: Range<Float>, name: String, prefix: String) throws(BasaltError) {
        guard value.isFinite, range.contains(value) else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must be in [\(range.lowerBound), \(range.upperBound))")
        }
    }

    static func requireOpenUnit(_ value: Float, name: String, prefix: String) throws(BasaltError) {
        guard value.isFinite, value > 0, value < 1 else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must be in (0, 1)")
        }
    }
}

public struct GRPOReport: Sendable, Equatable, Codable {
    public var losses: [Float] = []
    public var meanRewards: [Float] = []
    public var accuracyBefore: Float = 0
    public var accuracyAfter: Float = 0
    public var meanKL: Float = 0
    public var updatesApplied: Int = 0
    public var skippedGroups: Int = 0
    public var gatePassed: Bool = false
    public var outputPath: String = ""

    public init(outputPath: String = "") {
        self.outputPath = outputPath
    }
}

public enum GRPOAdvantage {
    public static func normalize(rewards: [Float], eps: Float = 1e-6) -> [Float] {
        guard !rewards.isEmpty else { return [] }
        guard rewards.allSatisfy(\.isFinite) else {
            return Array(repeating: Float(0), count: rewards.count)
        }
        let count = Float(rewards.count)
        let mean = rewards.reduce(Float(0), +) / count
        let variance = rewards.reduce(Float(0)) { acc, value in
            let diff = value - mean
            return acc + diff * diff
        } / count
        let floor = max(eps, 1e-12)
        guard variance.isFinite, variance >= floor else {
            return Array(repeating: Float(0), count: rewards.count)
        }
        let denom = max(variance.squareRoot(), floor)
        return rewards.map { ($0 - mean) / denom }
    }
}

public enum DPOMath {
    public static func loss(
        policyChosenLogp: Float,
        policyRejectedLogp: Float,
        refChosenLogp: Float,
        refRejectedLogp: Float,
        beta: Float,
        labelSmoothing: Float = 0.0
    ) -> (loss: Float, margin: Float, chosenReward: Float, rejectedReward: Float) {
        let chosenReward = beta * (policyChosenLogp - refChosenLogp)
        let rejectedReward = beta * (policyRejectedLogp - refRejectedLogp)
        let margin = chosenReward - rejectedReward
        let primary = negLogSigmoid(margin)
        let smoothLoss: Float
        if labelSmoothing > 0 {
            smoothLoss = (1.0 - labelSmoothing) * primary + labelSmoothing * negLogSigmoid(-margin)
        } else {
            smoothLoss = primary
        }
        return (loss: smoothLoss, margin: margin, chosenReward: chosenReward, rejectedReward: rejectedReward)
    }

    public static func negLogSigmoid(_ value: Float) -> Float {
        let x = Double(value)
        if x >= 0 {
            return Float(Foundation.log(1.0 + Foundation.exp(-x)))
        }
        return Float(-x + Foundation.log(1.0 + Foundation.exp(x)))
    }

    public static func schulmanKL(logPolicy: Float, logReference: Float) -> Float {
        let d = Double(logReference - logPolicy)
        guard d.isFinite else { return 0 }
        if abs(d) < 1e-5 {
            return Float(max(0.0, 0.5 * d * d + (d * d * d) / 6.0))
        }
        let clamped = min(d, 80.0)
        return Float(max(0.0, Foundation.exp(clamped) - d - 1.0))
    }
}
