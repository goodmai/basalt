import Foundation

public enum PreferenceObjective: String, Sendable, Codable, CaseIterable, Equatable {
    case dpo
    case adpo
    case cadpo
}

public struct ADPOPair: Sendable, Codable, Equatable {
    public var prompt: String
    public var chosen: String
    public var rejected: String

    public init(prompt: String, chosen: String, rejected: String) {
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected
    }
}

public struct ADPOConfig: Sendable, Codable, Equatable {
    public static let allowedSegments = 1 ... 512
    public static let allowedBeta: ClosedRange<Float> = 1e-4 ... 100.0
    public static let allowedDiversitySamples = 0 ... 16

    public var objective: PreferenceObjective
    public var segments: Int
    public var beta: Float
    public var conservatism: Float
    public var minImportanceWeight: Float = 0.1
    public var emaDecay: Float?
    public var swapEMAOnFinish: Bool = true
    public var maxSharpeningTax: Float = 0.8
    public var minEntropy: Float = 0.05
    public var diversitySamples: Int = 4
    public var useGroundTruthChosen: Bool = true
    public var pairs: [ADPOPair] = []

    public init(
        objective: PreferenceObjective = .adpo,
        segments: Int? = nil,
        beta: Float = 0.1,
        conservatism: Float = 0.8,
        emaDecay: Float? = 0.995
    ) {
        self.objective = objective
        self.segments = segments ?? (objective == .dpo ? 1 : 128)
        self.beta = beta
        self.conservatism = conservatism
        self.emaDecay = emaDecay
    }

    public func validate(prefix: String = "train") throws(BasaltError) {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "\(prefix): \(reason)") }
        guard Self.allowedSegments.contains(segments) else {
            throw refused("segments must be in \(Self.allowedSegments)")
        }
        if objective == .dpo, segments != 1 {
            throw refused("dpo requires segments == 1")
        }
        guard beta.isFinite, Self.allowedBeta.contains(beta) else {
            throw refused("beta must be in \(Self.allowedBeta)")
        }
        guard conservatism.isFinite, conservatism > 0, conservatism <= 1 else {
            throw refused("conservatism must lie in (0, 1]")
        }
        guard minImportanceWeight.isFinite, minImportanceWeight > 0, minImportanceWeight <= 1 else {
            throw refused("minImportanceWeight must lie in (0, 1]")
        }
        if let emaDecay {
            guard emaDecay.isFinite, emaDecay > 0, emaDecay < 1 else {
                throw refused("emaDecay must lie in (0, 1)")
            }
        }
        guard maxSharpeningTax.isFinite, maxSharpeningTax > 0, maxSharpeningTax <= 1 else {
            throw refused("maxSharpeningTax must lie in (0, 1]")
        }
        guard minEntropy.isFinite, minEntropy >= 0 else {
            throw refused("minEntropy must be finite and non-negative")
        }
        guard Self.allowedDiversitySamples.contains(diversitySamples) else {
            throw refused("diversitySamples must be in \(Self.allowedDiversitySamples)")
        }
        guard pairs.allSatisfy({ !$0.prompt.isEmpty && !$0.chosen.isEmpty && !$0.rejected.isEmpty }) else {
            throw refused("every preference pair requires non-empty prompt, chosen and rejected text")
        }
    }
}

extension ADPOConfig {
    public static func parseYAML(_ text: String) throws(BasaltError) -> ADPOConfig {
        let entries = try extractYAMLEntries(text)
        guard !entries.isEmpty else {
            throw .invalidRequestStructure(details: "train: empty YAML train configuration")
        }
        var config = ADPOConfig()
        var explicitSegments = false
        for (key, value) in entries {
            if key == "segments" { explicitSegments = true }
            try config.applyYAMLField(key: key, value: value)
        }
        if config.objective == .dpo, !explicitSegments {
            config.segments = 1
        }
        try config.validate(prefix: "train")
        return config
    }

    public func toYAMLInline() -> String {
        var parts = ["method: \(objective.rawValue)", "segments: \(segments)", "beta: \(beta)"]
        if objective == .cadpo {
            parts.append("conservatism: \(conservatism)")
        }
        if let emaDecay {
            parts.append("ema: \(emaDecay)")
        } else {
            parts.append("ema: none")
        }
        return "train: { \(parts.joined(separator: ", ")) }"
    }

    private static func extractYAMLEntries(_ text: String) throws(BasaltError) -> [(String, String)] {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("train:") {
            trimmed = String(trimmed.dropFirst("train:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if trimmed.hasPrefix("{") || trimmed.hasSuffix("}") {
            guard trimmed.hasPrefix("{"), trimmed.hasSuffix("}") else {
                throw .invalidRequestStructure(details: "train: unbalanced braces in inline YAML map")
            }
            let inner = trimmed.dropFirst().dropLast()
            var inlinePairs: [(String, String)] = []
            for item in inner.split(separator: ",") {
                let piece = item.trimmingCharacters(in: .whitespacesAndNewlines)
                if piece.isEmpty { continue }
                inlinePairs.append(try splitKeyValue(piece))
            }
            return inlinePairs
        }
        var pairs: [(String, String)] = []
        for rawLine in trimmed.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            pairs.append(try splitKeyValue(line))
        }
        return pairs
    }

    private static func splitKeyValue(_ raw: String) throws(BasaltError) -> (String, String) {
        guard let colon = raw.firstIndex(of: ":") else {
            throw .invalidRequestStructure(details: "train: malformed YAML entry '\(raw)'")
        }
        let key = raw[..<colon].trimmingCharacters(in: .whitespaces)
        var value = raw[raw.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        if value.count >= 2, (value.hasPrefix("\"") && value.hasSuffix("\"")) || (value.hasPrefix("'") && value.hasSuffix("'")) {
            value = String(value.dropFirst().dropLast())
        }
        guard !key.isEmpty, !value.isEmpty else {
            throw .invalidRequestStructure(details: "train: empty key or value in '\(raw)'")
        }
        return (key, value)
    }

    private mutating func applyYAMLField(key: String, value: String) throws(BasaltError) {
        switch key {
        case "method", "objective":
            guard let parsed = PreferenceObjective(rawValue: value.lowercased()) else {
                throw .invalidRequestStructure(details: "train: method must be dpo, adpo or cadpo")
            }
            objective = parsed
        case "segments":
            segments = try Self.parseInt(value, name: key)
        case "diversity_samples", "diversitySamples":
            diversitySamples = try Self.parseInt(value, name: key)
        case "beta":
            beta = try Self.parseFloat(value, name: key)
        case "conservatism":
            conservatism = try Self.parseFloat(value, name: key)
        case "min_importance_weight", "minImportanceWeight":
            minImportanceWeight = try Self.parseFloat(value, name: key)
        case "max_sharpening_tax", "maxSharpeningTax":
            maxSharpeningTax = try Self.parseFloat(value, name: key)
        case "min_entropy", "minEntropy":
            minEntropy = try Self.parseFloat(value, name: key)
        case "ema", "ema_decay", "emaDecay":
            let lower = value.lowercased()
            emaDecay = (lower == "none" || lower == "null" || lower == "off") ? nil : try Self.parseFloat(value, name: key)
        case "swap_ema", "swapEMAOnFinish":
            swapEMAOnFinish = try Self.parseBool(value, name: key)
        default:
            throw .invalidRequestStructure(details: "train: unknown YAML key '\(key)'")
        }
    }

    private static func parseInt(_ raw: String, name: String) throws(BasaltError) -> Int {
        guard let parsed = Int(raw) else {
            throw .invalidRequestStructure(details: "train: \(name) must be an integer")
        }
        return parsed
    }

    private static func parseFloat(_ raw: String, name: String) throws(BasaltError) -> Float {
        guard let parsed = Float(raw), parsed.isFinite else {
            throw .invalidRequestStructure(details: "train: \(name) must be a finite number")
        }
        return parsed
    }

    private static func parseBool(_ raw: String, name: String) throws(BasaltError) -> Bool {
        switch raw.lowercased() {
        case "true", "yes": return true
        case "false", "no": return false
        default: throw .invalidRequestStructure(details: "train: \(name) must be true or false")
        }
    }
}

public enum ADPOSegments {
    public static func effectiveCount(requested: Int, chosenLength: Int, rejectedLength: Int) -> Int {
        max(1, min(requested, min(chosenLength, rejectedLength)))
    }

    public static func partition(length: Int, count: Int) -> [Range<Int>] {
        guard length > 0 else { return [] }
        let m = max(1, min(count, length))
        return (0 ..< m).map { index in
            let start = (index * length) / m
            let end = ((index + 1) * length) / m
            return start ..< end
        }
    }

    public static func aligned(
        chosenLength: Int,
        rejectedLength: Int,
        segments: Int
    ) -> (chosen: [Range<Int>], rejected: [Range<Int>]) {
        guard chosenLength > 0, rejectedLength > 0 else { return ([], []) }
        let m = effectiveCount(requested: segments, chosenLength: chosenLength, rejectedLength: rejectedLength)
        return (partition(length: chosenLength, count: m), partition(length: rejectedLength, count: m))
    }

    public static func indicatorMatrices(ranges: [Range<Int>], length: Int) -> (sumMask: [Float], meanMask: [Float]) {
        guard !ranges.isEmpty, length > 0 else { return ([], []) }
        var sumMask = [Float](repeating: 0, count: ranges.count * length)
        var meanMask = [Float](repeating: 0, count: ranges.count * length)
        for (row, range) in ranges.enumerated() where !range.isEmpty {
            let inv = 1.0 / Float(range.count)
            let base = row * length
            for col in range where col >= 0 && col < length {
                sumMask[base + col] = 1.0
                meanMask[base + col] = inv
            }
        }
        return (sumMask, meanMask)
    }
}

public enum ADPOMetrics {
    public static func sharpeningTax(before: Float, after: Float) -> Float {
        guard before.isFinite, after.isFinite, before > 1e-8 else { return 0 }
        return 1.0 - (after / before)
    }

    public static func isEntropyCollapsed(before: Float, current: Float, config: ADPOConfig) -> Bool {
        guard current.isFinite else { return true }
        if current < config.minEntropy { return true }
        return sharpeningTax(before: before, after: current) > config.maxSharpeningTax
    }

    public static func passAtKDiversity(_ samplesPerPrompt: [[[Int]]]) -> Float {
        var promptScores: [Float] = []
        for samples in samplesPerPrompt where samples.count >= 2 {
            var total: Float = 0
            var pairs = 0
            for lhs in 0 ..< (samples.count - 1) {
                for rhs in (lhs + 1) ..< samples.count {
                    total += sequenceDistance(samples[lhs], samples[rhs])
                    pairs += 1
                }
            }
            if pairs > 0 {
                promptScores.append(total / Float(pairs))
            }
        }
        guard !promptScores.isEmpty else { return 0 }
        return promptScores.reduce(0, +) / Float(promptScores.count)
    }

    public static func sequenceDistance(_ lhs: [Int], _ rhs: [Int]) -> Float {
        if lhs == rhs { return 0 }
        let a = ngramFeatures(lhs)
        let b = ngramFeatures(rhs)
        let unionCount = a.union(b).count
        guard unionCount > 0 else { return 0 }
        let sharedCount = a.intersection(b).count
        return 1.0 - (Float(sharedCount) / Float(unionCount))
    }

    private static func ngramFeatures(_ tokens: [Int]) -> Set<UInt64> {
        var features: Set<UInt64> = []
        features.reserveCapacity(tokens.count * 2)
        for index in tokens.indices {
            let low = UInt64(bitPattern: Int64(tokens[index])) & 0xFFFF_FFFF
            features.insert(low)
            if index + 1 < tokens.count {
                let next = UInt64(bitPattern: Int64(tokens[index + 1])) & 0xFFFF_FFFF
                features.insert((UInt64(1) << 63) | ((low & 0x7FFF_FFFF) << 32) | next)
            }
        }
        return features
    }
}
