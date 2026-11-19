import Foundation

public enum TokenRiskState: String, Sendable, Codable, Equatable, CaseIterable {
    case grounded
    case uncertain
    case highRisk = "high_risk"
}

public struct TokenCandidateProbability: Sendable, Codable, Equatable {
    public var token: String
    public var tokenId: Int
    public var bytes: [UInt8]
    public var logprob: Float
    public var probability: Float

    public init(token: String, tokenId: Int, logprob: Float) {
        self.token = token
        self.tokenId = tokenId
        self.bytes = Array(token.utf8)
        self.logprob = logprob
        let p = Foundation::exp(logprob)
        self.probability = p.isFinite ? min(1.0, max(0.0, p)) : 0.0
    }

    enum CodingKeys: String, CodingKey {
        case token, bytes, logprob, probability
        case tokenId = "token_id"
    }
}

public struct TokenHeadActivation: Sendable, Codable, Equatable {
    public var head: Int
    public var norm: Float
    public var attentionEntropy: Float?

    public init(head: Int, norm: Float, attentionEntropy: Float? = nil) {
        self.head = head
        self.norm = norm
        self.attentionEntropy = attentionEntropy
    }

    enum CodingKeys: String, CodingKey {
        case head, norm
        case attentionEntropy = "attention_entropy"
    }
}

public struct TokenExpertActivation: Sendable, Codable, Equatable {
    public var expert: Int
    public var weight: Float

    public init(expert: Int, weight: Float) {
        self.expert = expert
        self.weight = weight
    }
}

public struct TokenLayerPathway: Sendable, Codable, Equatable {
    public var layer: Int
    public var topHeads: [TokenHeadActivation]
    public var experts: [TokenExpertActivation]
    public var routerEntropy: Float?
    public var routerMargin: Float?
    public var lensTopTokenId: Int?
    public var lensTopToken: String?
    public var lensTopProbability: Float?

    public init(
        layer: Int,
        topHeads: [TokenHeadActivation] = [],
        experts: [TokenExpertActivation] = []
    ) {
        self.layer = layer
        self.topHeads = topHeads
        self.experts = experts
    }

    enum CodingKeys: String, CodingKey {
        case layer, experts
        case topHeads = "top_heads"
        case routerEntropy = "router_entropy"
        case routerMargin = "router_margin"
        case lensTopTokenId = "lens_top_token_id"
        case lensTopToken = "lens_top_token"
        case lensTopProbability = "lens_top_probability"
    }
}

public struct OutputTokenTelemetry: Sendable, Codable, Equatable {
    public var index: Int
    public var token: String
    public var tokenId: Int
    public var bytes: [UInt8]
    public var logprob: Float
    public var probability: Float
    public var surprisalBits: Float
    public var entropy: Float
    public var margin: Float
    public var topLogprobs: [TokenCandidateProbability]
    public var hallucinationRisk: Float
    public var riskState: TokenRiskState
    public var onsetLayer: Int?
    public var layers: [TokenLayerPathway]?

    public init(
        index: Int,
        token: String,
        tokenId: Int,
        logprob: Float,
        entropy: Float,
        margin: Float
    ) {
        self.index = index
        self.token = token
        self.tokenId = tokenId
        self.bytes = Array(token.utf8)
        self.logprob = logprob
        let prob = Foundation::exp(logprob)
        let cleanProb = prob.isFinite ? min(1.0, max(0.0, prob)) : 0.0
        self.probability = cleanProb
        let bits = -logprob / Float(Foundation::log(2.0))
        self.surprisalBits = bits.isFinite ? max(0.0, bits) : 0.0
        self.entropy = entropy.isFinite ? max(0.0, entropy) : 0.0
        self.margin = margin.isFinite ? min(1.0, max(0.0, margin)) : 0.0
        self.topLogprobs = []
        let scored = TokenHallucinationScorer.evaluate(
            probability: cleanProb,
            entropy: self.entropy,
            margin: self.margin,
            onsetLayer: nil,
            totalLayers: nil
        )
        self.hallucinationRisk = scored.risk
        self.riskState = scored.state
    }

    enum CodingKeys: String, CodingKey {
        case index, token, bytes, logprob, probability, entropy, margin, layers
        case tokenId = "token_id"
        case surprisalBits = "surprisal_bits"
        case topLogprobs = "top_logprobs"
        case hallucinationRisk = "hallucination_risk"
        case riskState = "risk_state"
        case onsetLayer = "onset_layer"
    }
}

public enum TokenHallucinationScorer {
    public static func evaluate(
        probability: Float,
        entropy: Float,
        margin: Float,
        onsetLayer: Int?,
        totalLayers: Int?
    ) -> (risk: Float, state: TokenRiskState) {
        let p = min(1.0, max(0.0, probability.isFinite ? probability : 0.0))
        let hNorm = min(1.0, max(0.0, (entropy.isFinite ? entropy : 0.0) / 4.0))
        let m = min(1.0, max(0.0, margin.isFinite ? margin : 0.0))
        var lateOnsetPenalty: Float = 0.0
        if let onsetLayer, let totalLayers, totalLayers >= 4, onsetLayer >= totalLayers - 1, p < 0.85 {
            lateOnsetPenalty = 0.15
        }
        let raw = 0.45 * (1.0 - p) + 0.30 * hNorm + 0.25 * (1.0 - m) + lateOnsetPenalty
        let risk = min(1.0, max(0.0, raw))
        let state: TokenRiskState = if risk >= 0.62 || (p < 0.20 && hNorm > 0.55) {
            .highRisk
        } else if risk >= 0.32 || p < 0.45 {
            .uncertain
        } else {
            .grounded
        }
        return (risk, state)
    }
}

public struct AttributedTokenTraceRequest: Sendable, Codable, Equatable {
    public static let maxGeneratedLimit = 256
    public static let topLogprobsRange = 0 ... 20
    public static let topHeadsRange = 1 ... 32

    public var prompt: String
    public var messages: [ChatMessageInput]?
    public var maxTokens: Int
    public var temperature: Float
    public var topLogprobs: Int
    public var topHeads: Int
    public var layers: [Int]?
    public var includeLogitLens: Bool
    public var includeAttentionEntropy: Bool

    public init(
        prompt: String = "",
        messages: [ChatMessageInput]? = nil,
        maxTokens: Int = 32,
        temperature: Float = 0.0,
        topLogprobs: Int = 5,
        topHeads: Int = 4
    ) {
        self.prompt = prompt
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topLogprobs = topLogprobs
        self.topHeads = topHeads
        self.layers = nil
        self.includeLogitLens = true
        self.includeAttentionEntropy = false
    }

    public func validate(against anatomy: ModelAnatomy) throws(BasaltError) {
        let hasText = !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (messages?.isEmpty == false)
        guard hasText else {
            throw .invalidRequestStructure(details: "xray tokens: provide non-empty prompt or messages")
        }
        guard (1 ... Self.maxGeneratedLimit).contains(maxTokens) else {
            throw .invalidRequestStructure(details: "xray tokens: max_tokens must lie in 1...\(Self.maxGeneratedLimit)")
        }
        guard temperature.isFinite, (0.0 ... 2.0).contains(temperature) else {
            throw .invalidRequestStructure(details: "xray tokens: temperature must lie in 0.0...2.0")
        }
        guard Self.topLogprobsRange.contains(topLogprobs) else {
            throw .invalidRequestStructure(details: "xray tokens: top_logprobs must lie in 0...20")
        }
        guard Self.topHeadsRange.contains(topHeads) else {
            throw .invalidRequestStructure(details: "xray tokens: top_heads must lie in 1...32")
        }
        let count = anatomy.layers.count
        if let layers, layers.isEmpty || layers.contains(where: { !(0 ..< count).contains($0) }) {
            throw .invalidRequestStructure(details: "xray tokens: layers must list valid indices in 0..<\(count)")
        }
    }

    enum CodingKeys: String, CodingKey {
        case prompt, messages, temperature, layers
        case maxTokens = "max_tokens"
        case topLogprobs = "top_logprobs"
        case topHeads = "top_heads"
        case includeLogitLens = "include_logit_lens"
        case includeAttentionEntropy = "include_attention_entropy"
    }
}

public struct AttributedTokenTraceSummary: Sendable, Codable, Equatable {
    public var meanProbability: Double
    public var minProbability: Double
    public var meanEntropy: Double
    public var minMargin: Double
    public var belnapState: BelnapState
    public var highRiskTokenIndices: [Int]

    public init(tokens: [OutputTokenTelemetry]) {
        if tokens.isEmpty {
            self.meanProbability = 1.0
            self.minProbability = 1.0
            self.meanEntropy = 0.0
            self.minMargin = 1.0
            self.belnapState = .true
            self.highRiskTokenIndices = []
            return
        }
        let probs = tokens.map { Double($0.probability) }
        let ents = tokens.map { Double($0.entropy) }
        let margins = tokens.map { Double($0.margin) }
        let meanP = probs.reduce(0.0, +) / Double(probs.count)
        let minP = probs.min() ?? 0.0
        let meanH = ents.reduce(0.0, +) / Double(ents.count)
        let minM = margins.min() ?? 0.0
        var rc = ResponseConfidence(min: minP, mean: meanP)
        rc.meanEntropy = meanH
        rc.minMargin = minM
        self.meanProbability = meanP
        self.minProbability = minP
        self.meanEntropy = meanH
        self.minMargin = minM
        self.belnapState = rc.belnapState()
        self.highRiskTokenIndices = tokens.filter { $0.riskState == .highRisk }.map(\.index)
    }

    enum CodingKeys: String, CodingKey {
        case meanProbability = "mean_probability"
        case minProbability = "min_probability"
        case meanEntropy = "mean_entropy"
        case minMargin = "min_margin"
        case belnapState = "belnap_state"
        case highRiskTokenIndices = "high_risk_token_indices"
    }
}

public struct AttributedTokenTraceReport: Sendable, Codable, Equatable {
    public var model: String
    public var promptTokens: Int
    public var completionTokens: Int
    public var generatedText: String
    public var summary: AttributedTokenTraceSummary
    public var tokens: [OutputTokenTelemetry]

    public init(
        model: String,
        promptTokens: Int,
        generatedText: String,
        tokens: [OutputTokenTelemetry]
    ) {
        self.model = model
        self.promptTokens = promptTokens
        self.completionTokens = tokens.count
        self.generatedText = generatedText
        self.summary = AttributedTokenTraceSummary(tokens: tokens)
        self.tokens = tokens
    }

    enum CodingKeys: String, CodingKey {
        case model, summary, tokens
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case generatedText = "generated_text"
    }
}
