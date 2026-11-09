import Foundation

public enum TeacherProvider: String, Sendable, Codable, Equatable, CaseIterable {
    case openai
    case anthropic
}

public enum DistillFilter: String, Sendable, Codable, Equatable, CaseIterable {
    case none
    case verify
}

public struct TeacherSpec: Sendable, Codable, Equatable {
    public var url: String
    public var model: String
    public var provider: TeacherProvider = .openai
    public var maxTokens = 2048
    public var temperature: Float = 0.7
    public var concurrency = 4
    public var reasoningEffort: String?
    public var inputPricePerMillion: Double?
    public var outputPricePerMillion: Double?

    public init(url: String, model: String) {
        self.url = url
        self.model = model
    }
}

public struct DistillRequest: Sendable, Codable, Equatable {
    public var prompts: String
    public var teacher: TeacherSpec
    public var samplesPerPrompt = 1
    public var filter: DistillFilter = .none
    public var heldOutFraction: Float = 0.1
    public var includeReasoning = false
    public var budgetUSD: Double?
    public var layers: [Int]?
    public var rank = 16
    public var iterations = 200
    public var batchSize = 2
    public var learningRate: Float = 1e-4
    public var maxSeqLen = 2048
    public var gate: GateRequest
    public var output: String?
    public var onPolicy = false

    public init(prompts: String, teacher: TeacherSpec, gate: GateRequest) {
        self.prompts = prompts
        self.teacher = teacher
        self.gate = gate
    }
}

public enum DistillStop: String, Sendable, Codable, Equatable {
    case completed
    case budget
}

public struct DistillReport: Sendable, Codable {
    public var prompts = 0
    public var cached = 0
    public var teacherCalls = 0
    public var kept = 0
    public var rejected: [String: Int] = [:]
    public var teacherInputTokens = 0
    public var teacherOutputTokens = 0
    public var spentUSD = 0.0
    public var stoppedBy: DistillStop = .completed
    public var losses: [Float] = []
    public var heldOut = 0
    public var teacherNLLBefore: Float?
    public var teacherNLLAfter: Float?
    public var reverseKLBefore: Float?
    public var reverseKLAfter: Float?
    public var cacheDirectory: String?
    public var adapterDirectory: String?
    public var verdict: AdapterVerdict?

    public init() {}
}

public struct ScoreRequest: Sendable, Codable, Equatable {
    public var messages: [[String: String]]
    public var completion: [Int]
    public var topLogprobs: Int

    enum CodingKeys: String, CodingKey {
        case messages, completion
        case topLogprobs = "top_logprobs"
    }

    public init(messages: [[String: String]], completion: [Int], topLogprobs: Int) {
        self.messages = messages
        self.completion = completion
        self.topLogprobs = topLogprobs
    }
}

public struct TopLogprobs: Sendable, Codable, Equatable {
    public var ids: [Int]
    public var logprobs: [Float]

    public init(ids: [Int], logprobs: [Float]) {
        self.ids = ids
        self.logprobs = logprobs
    }
}

public struct ScoreReport: Sendable, Codable, Equatable {
    public var tokenizer: String
    public var promptTokens: Int
    public var positions: [TopLogprobs]

    enum CodingKeys: String, CodingKey {
        case tokenizer, positions
        case promptTokens = "prompt_tokens"
    }

    public init(tokenizer: String, promptTokens: Int, positions: [TopLogprobs]) {
        self.tokenizer = tokenizer
        self.promptTokens = promptTokens
        self.positions = positions
    }
}
