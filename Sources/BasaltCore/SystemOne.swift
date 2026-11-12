import Foundation

public enum SystemOneKind: String, Codable, Sendable, CaseIterable {
    case noul, choice, score
}

public enum SystemOneCriteria: Codable, Sendable, Equatable {
    case named([String: String])
    case levels([String])

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let levels = try? container.decode([String].self) { self = .levels(levels) } else { self = .named(try container.decode([String: String].self)) }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .named(let named): try container.encode(named)
        case .levels(let levels): try container.encode(levels)
        }
    }
}

public struct SystemOneQuestion: Codable, Sendable, Equatable {
    public var type: SystemOneKind
    public var instructions: String
    public var criteria: SystemOneCriteria?

    public init(type: SystemOneKind, instructions: String, criteria: SystemOneCriteria? = nil) {
        self.type = type
        self.instructions = instructions
        self.criteria = criteria
    }
}

public struct DecisionMedia: Codable, Sendable, Equatable {
    public static let kinds: [String: String] = ["image": "image", "audio": "audio", "video": "video", "pdf": "document", "document": "document"]
    public static let byteLimit = 50 << 20

    public var kind: String
    public var mimeType: String
    public var data: Data

    public init?(kind: String, mimeType: String, data: Data) {
        guard let kind = Self.kinds[kind], !mimeType.isEmpty, !data.isEmpty, data.count <= Self.byteLimit else { return nil }
        (self.kind, self.mimeType, self.data) = (kind, mimeType, data)
    }

    enum CodingKeys: String, CodingKey {
        case kind, data
        case mimeType = "mime_type"
    }
}

public struct SystemOneRequest: Codable, Sendable, Equatable {
    public var state: String
    public var questions: [String: SystemOneQuestion]
    public var calibrated: Bool?
    public var abstainBelow: Double?
    public var minMargin: Double?
    public var media: [DecisionMedia]?

    public init(state: String, questions: [String: SystemOneQuestion], calibrated: Bool? = nil, abstainBelow: Double? = nil, minMargin: Double? = nil) {
        (self.state, self.questions, self.calibrated) = (state, questions, calibrated)
        (self.abstainBelow, self.minMargin) = (abstainBelow, minMargin)
    }
}

public struct SystemOneAnswer: Codable, Sendable, Equatable {
    public var noul: Double?
    public var probabilities: [String: Double]?
    public var confidence: Double?
    public var choice: String?
    public var score: Double?
    public var legend: [String]?
    public var margin: Double?
    public var abstain: Bool?

    public init() {}
}

public struct SystemOneUsage: Codable, Sendable, Equatable {
    public var prefixTokens: Int
    public var candidateBranches: Int
    public var prefixEvaluations: Int
    public var branchForwardCalls: Int
    public var suffixTokens: Int
    public var outputDecodingTokens = 0
    public var cost: Double?
    public var stages: [SystemOneStage]?

    public init(prefixTokens: Int, candidateBranches: Int, prefixEvaluations: Int, branchForwardCalls: Int, suffixTokens: Int) {
        (self.prefixTokens, self.candidateBranches, self.prefixEvaluations) = (prefixTokens, candidateBranches, prefixEvaluations)
        (self.branchForwardCalls, self.suffixTokens) = (branchForwardCalls, suffixTokens)
    }

    enum CodingKeys: String, CodingKey {
        case prefixTokens = "prefix_tokens", candidateBranches = "candidate_branches", prefixEvaluations = "prefix_evaluations"
        case branchForwardCalls = "branch_forward_calls", suffixTokens = "suffix_tokens", outputDecodingTokens = "output_decoding_tokens"
        case cost, stages
    }
}

public struct SystemOneStage: Codable, Sendable, Equatable {
    public var model: String
    public var questions: [String]
    public var latencyMs: Int
    public var cost: Double?
    public var fallback: String?

    public init(model: String, questions: [String], latencyMs: Int, cost: Double? = nil, fallback: String? = nil) {
        (self.model, self.questions, self.latencyMs, self.cost, self.fallback) = (model, questions, latencyMs, cost, fallback)
    }

    enum CodingKeys: String, CodingKey {
        case model, questions, cost, fallback, latencyMs = "latency_ms"
    }
}

public struct SystemOneResponse: Codable, Sendable, Equatable {
    public var model: String
    public var answers: [String: SystemOneAnswer]
    public var usage: SystemOneUsage
    public var calibrated: Bool
    public var escalate: Bool

    public init(model: String, answers: [String: SystemOneAnswer], usage: SystemOneUsage, calibrated: Bool) {
        (self.model, self.answers, self.usage, self.calibrated) = (model, answers, usage, calibrated)
        escalate = answers.values.contains { $0.abstain == true }
    }
}

public protocol DecisionEngine: Sendable {
    var modelID: String { get }
    var runsBesideDecode: Bool { get }
    var acceptsMedia: Bool { get }
    var onDevice: (any DecisionEngine)? { get }
    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse
}

extension DecisionEngine {
    public var runsBesideDecode: Bool { false }
    public var acceptsMedia: Bool { false }
    public var onDevice: (any DecisionEngine)? { self }
}

public enum SystemOneCore {
    public static let maxQuestions = 128
    public static let maxStateCharacters = 100_000
    public static let maxText = 4096

    public static func options(_ question: SystemOneQuestion) -> (keys: [String], descriptions: [String]) {
        switch (question.type, question.criteria) {
        case (.choice, .named(let named)?):
            let keys = named.keys.sorted()
            return (keys, keys.map { named[$0] ?? "" })
        case (.score, .levels(let levels)?):
            return (levels.indices.map(String.init), levels)
        case (.noul, .named(let named)?):
            return (["false", "true"], [named["false"] ?? "", named["true"] ?? ""])
        default:
            return (["false", "true"], ["The following statement is false: \(question.instructions)", "The following statement is true: \(question.instructions)"])
        }
    }

    public static func validate(_ request: SystemOneRequest) throws(BasaltError) {
        func refuse(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "systemone: \(reason)") }
        for limit in [request.abstainBelow, request.minMargin].compactMap({ $0 }) where !(0 ... 1).contains(limit) {
            throw refuse("abstain_below and min_margin must lie in 0...1, got \(limit)")
        }
        guard !request.state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, request.state.count <= maxStateCharacters else {
            throw refuse("state must contain 1–\(maxStateCharacters) characters")
        }
        guard (1 ... maxQuestions).contains(request.questions.count) else { throw refuse("1–\(maxQuestions) questions") }
        for (name, question) in request.questions {
            guard !name.trimmingCharacters(in: .whitespaces).isEmpty else { throw refuse("question IDs cannot be blank") }
            try validate(question, name: name)
        }
    }

    static func validate(_ question: SystemOneQuestion, name: String) throws(BasaltError) {
        func refuse(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "systemone: question \(name): \(reason)") }
        let instructions = question.instructions.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !instructions.isEmpty, question.instructions.count <= maxText else { throw refuse("instructions must contain 1–\(maxText) characters") }
        switch (question.type, question.criteria) {
        case (.choice, .named(let named)?) where (2 ... 255).contains(named.count): break
        case (.score, .levels(let levels)?) where (2 ... 10).contains(levels.count): break
        case (.noul, nil): break
        case (.noul, .named(let named)?) where Set(named.keys) == ["false", "true"]: break
        case (.choice, _): throw refuse("choice needs 2–255 named candidates")
        case (.score, _): throw refuse("score needs 2–10 ordered level descriptions")
        case (.noul, _): throw refuse("noul criteria must describe exactly false and true")
        }
        let descriptions = options(question).descriptions
        guard descriptions.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.count <= maxText }) else {
            throw refuse("candidate descriptions must contain 1–\(maxText) characters")
        }
        guard Set(descriptions).count == descriptions.count else { throw refuse("candidate descriptions must be distinct") }
    }

    public static func branch(_ question: SystemOneQuestion, candidate: String, tail: String) -> String {
        "\n\nQuestion:\n" + question.instructions + "\n\nCandidate answer:\n" + candidate + "\n\nIs this candidate correct? Answer Yes or No." + tail
    }

    public static func answer(_ question: SystemOneQuestion, scores: [Double], temperature: Double) -> SystemOneAnswer {
        let (keys, descriptions) = options(question)
        let logits = scores.map { $0 / temperature }
        let top = logits.max() ?? 0
        let weights = logits.map { exp($0 - top) }
        let total = weights.reduce(0, +)
        let p = weights.map { $0 / total }
        var answer = SystemOneAnswer()
        guard question.type != .noul else {
            answer.noul = p.count > 1 ? p[1] : nil
            answer.confidence = answer.noul.map { abs(2 * $0 - 1) }
            return answer
        }
        let best = p.max() ?? 0
        answer.probabilities = Dictionary(uniqueKeysWithValues: zip(keys, p))
        answer.confidence = question.type == .score ? scoreConfidence(p) : choiceConfidence(p)
        if question.type == .choice {
            let tied = p.indices.filter { p[$0] >= best - 1e-7 }
            let selected = tied.min { (descriptions[$0].lowercased(), descriptions[$0]) < (descriptions[$1].lowercased(), descriptions[$1]) }
            answer.choice = selected.map { keys[$0] }
        } else {
            answer.score = zip(p, p.indices).reduce(0) { $0 + $1.0 * Double($1.1) }
            answer.legend = descriptions
        }
        return answer
    }

    public static func choiceConfidence(_ p: [Double]) -> Double {
        guard p.count > 1, let top = p.max() else { return 1 }
        let uniform = 1 / Double(p.count)
        return max(0, (top - uniform) / (1 - uniform))
    }

    public static func scoreConfidence(_ p: [Double]) -> Double {
        guard p.count > 1, let mode = p.indices.max(by: { p[$0] < p[$1] }) else { return 1 }
        let spread = zip(p, p.indices).reduce(0) { $0 + $1.0 * Double(abs($1.1 - mode)) }
        let center = Double(p.count - 1) / 2
        let uniform = p.indices.reduce(0) { $0 + abs(Double($1) - center) } / Double(p.count)
        return max(0, 1 - spread / uniform)
    }

    public static func gated(_ answer: SystemOneAnswer, abstainBelow: Double?, minMargin: Double?) -> SystemOneAnswer {
        let ranked = (answer.noul.map { [max($0, 1 - $0), min($0, 1 - $0)] } ?? answer.probabilities.map { $0.values.sorted(by: >) } ?? [])
        guard let top = ranked.first else { return answer }
        var gated = answer
        gated.margin = top - (ranked.dropFirst().first ?? 0)
        gated.abstain = (answer.confidence ?? top) < (abstainBelow ?? 0) || (gated.margin ?? 1) < (minMargin ?? 0)
        return gated
    }
}
