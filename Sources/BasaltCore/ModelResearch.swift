import Foundation

public protocol ModelResearch: Sendable {
    func anatomy() async throws(BasaltError) -> ModelAnatomy
    func tensors(matching filter: String?) async throws(BasaltError) -> [TensorInfo]
    func tensor(_ request: TensorRequest) async throws(BasaltError) -> TensorReport
    func promptCacheReport(namespace: String?) async -> PromptCacheReport
    func trace(_ request: TraceRequest) async throws(BasaltError) -> TraceReport
    func intervene(_ intervention: Intervention) async throws(BasaltError) -> OverlayInfo
    func applyDelta(_ delta: WeightDelta) async throws(BasaltError) -> OverlayInfo
    func overlays() async -> [OverlayInfo]
    func undo(_ overlay: OverlayID) async throws(BasaltError)
    func compare(_ request: CompareRequest) async throws(BasaltError) -> CompareReport
}

public enum OverlayTag {}
public typealias OverlayID = Tagged<OverlayTag, Int>

public struct ModelAnatomy: Sendable, Codable {
    public var modelType: String
    public var architecture: String
    public var layers: [LayerAnatomy]
    public var vocabulary = 0
    public var hiddenSize = 0
    public var quantization: String?
    public var parameterCount: Int
    public var bytes: ByteCount

    public init(modelType: String, architecture: String, layers: [LayerAnatomy], parameterCount: Int, bytes: ByteCount) {
        self.modelType = modelType
        self.architecture = architecture
        self.layers = layers
        self.parameterCount = parameterCount
        self.bytes = bytes
    }
}

public struct LayerAnatomy: Sendable, Codable, Equatable {
    public var index: Int
    public var kind: String
    public var heads: Int?
    public var kvHeads: Int?
    public var headDim: Int?
    public var window: Int?
    public var experts: Int?
    public var expertsPerToken: Int?
    public var cache: String

    public init(index: Int, kind: String, cache: String) {
        self.index = index
        self.kind = kind
        self.cache = cache
    }
}

public struct TensorInfo: Sendable, Codable, Equatable {
    public var name: String
    public var shape: [Int]
    public var dtype: String
    public var quantized: Bool

    public init(name: String, shape: [Int], dtype: String, quantized: Bool) {
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.quantized = quantized
    }
}

public struct TensorRequest: Sendable, Codable, Equatable {
    public var name: String
    public var slice: [Range<Int>]?
    public var maxValues: Int = 256

    public init(name: String, slice: [Range<Int>]? = nil, maxValues: Int = 256) {
        self.name = name
        self.slice = slice
        self.maxValues = maxValues
    }
}

public struct TensorReport: Sendable, Codable {
    public var info: TensorInfo
    public var mean: Float
    public var std: Float
    public var min: Float
    public var max: Float
    public var l2: Float
    public var percentiles: [Float] = []
    public var values: [Float]?

    public init(info: TensorInfo, mean: Float, std: Float, min: Float, max: Float, l2: Float) {
        self.info = info
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.l2 = l2
    }
}

public struct PromptCacheReport: Sendable, Codable {
    public var budget: ByteCount
    public var bytes: ByteCount
    public var evictions: Int
    public var namespaces: [NamespaceReport]

    public init(budget: ByteCount, bytes: ByteCount, evictions: Int, namespaces: [NamespaceReport]) {
        self.budget = budget
        self.bytes = bytes
        self.evictions = evictions
        self.namespaces = namespaces
    }
}

public struct NamespaceReport: Sendable, Codable {
    public var name: String
    public var nodes: Int
    public var leaves: Int
    public var tokens: Int
    public var bytes: ByteCount
    public var snapshots = 0
    public var pinned = 0
    public var lastUse = 0

    public init(name: String, nodes: Int, leaves: Int, tokens: Int, bytes: ByteCount) {
        self.name = name
        self.nodes = nodes
        self.leaves = leaves
        self.tokens = tokens
        self.bytes = bytes
    }
}

public struct TraceRequest: Sendable, Codable, Equatable {
    public var prompt: String
    public var position: Int?
    public var layers: [Int]?
    public var topK: Int = 10
    public var attention: Bool = true
    public var router: Bool = true
    public var logitLens: Bool = true

    public init(prompt: String, position: Int? = nil, layers: [Int]? = nil, topK: Int = 10) {
        self.prompt = prompt
        self.position = position
        self.layers = layers
        self.topK = topK
    }
}

public struct TraceReport: Sendable, Codable {
    public var tokens: [String]
    public var position: Int
    public var layers: [LayerTrace]
    public var final: [TokenScore]

    public init(tokens: [String], position: Int, layers: [LayerTrace], final: [TokenScore]) {
        self.tokens = tokens
        self.position = position
        self.layers = layers
        self.final = final
    }
}

public struct LayerTrace: Sendable, Codable {
    public var index: Int
    public var hiddenNorm: Float
    public var lens: [TokenScore]?
    public var attention: [HeadAttention]?
    public var experts: [ExpertChoice]?

    public init(
        index: Int, hiddenNorm: Float, lens: [TokenScore]? = nil, attention: [HeadAttention]? = nil,
        experts: [ExpertChoice]? = nil
    ) {
        self.index = index
        self.hiddenNorm = hiddenNorm
        self.lens = lens
        self.attention = attention
        self.experts = experts
    }
}

public struct TokenScore: Sendable, Codable {
    public var token: String
    public var id: Int
    public var probability: Float

    public init(token: String, id: Int, probability: Float) {
        self.token = token
        self.id = id
        self.probability = probability
    }
}

public struct HeadAttention: Sendable, Codable {
    public var head: Int
    public var top: [PositionWeight]
    public var entropy: Float

    public init(head: Int, top: [PositionWeight], entropy: Float) {
        self.head = head
        self.top = top
        self.entropy = entropy
    }
}

public struct PositionWeight: Sendable, Codable {
    public var position: Int
    public var weight: Float

    public init(position: Int, weight: Float) {
        self.position = position
        self.weight = weight
    }
}

public struct ExpertChoice: Sendable, Codable {
    public var expert: Int
    public var weight: Float

    public init(expert: Int, weight: Float) {
        self.expert = expert
        self.weight = weight
    }
}

public enum Intervention: Sendable, Codable, Equatable {
    case steer(layer: Int, vector: [Float], scale: Float)
    case ablateHead(layer: Int, head: Int)
    case ablateExpert(layer: Int, expert: Int)
}

public struct WeightDelta: Sendable, Codable, Equatable {
    public var tensor: String
    public var slice: [Range<Int>]?
    public var values: [Float]?
    public var scale: Float = 1

    public init(tensor: String, slice: [Range<Int>]? = nil, values: [Float]? = nil, scale: Float = 1) {
        self.tensor = tensor
        self.slice = slice
        self.values = values
        self.scale = scale
    }
}

public struct OverlayInfo: Sendable, Codable {
    public var id: OverlayID
    public var description: String
    public var createdAt: Date
    public var expiresAt: Date?

    public init(id: OverlayID, description: String, createdAt: Date, expiresAt: Date? = nil) {
        self.id = id
        self.description = description
        self.createdAt = createdAt
        self.expiresAt = expiresAt
    }
}

public struct CompareRequest: Sendable, Codable, Equatable {
    public var prompts: [String]
    public var overlays: [OverlayID]
    public var maxTokens: Int = 32

    public init(prompts: [String], overlays: [OverlayID], maxTokens: Int = 32) {
        self.prompts = prompts
        self.overlays = overlays
        self.maxTokens = maxTokens
    }
}

public struct CompareReport: Sendable, Codable {
    public var rows: [CompareRow]

    public init(rows: [CompareRow]) {
        self.rows = rows
    }
}

public struct CompareRow: Sendable, Codable {
    public var prompt: String
    public var baselineText: String
    public var overlayText: String
    public var klDivergence: Float
    public var baselinePerplexity: Float
    public var overlayPerplexity: Float

    public init(
        prompt: String, baselineText: String, overlayText: String, klDivergence: Float,
        baselinePerplexity: Float, overlayPerplexity: Float
    ) {
        self.prompt = prompt
        self.baselineText = baselineText
        self.overlayText = overlayText
        self.klDivergence = klDivergence
        self.baselinePerplexity = baselinePerplexity
        self.overlayPerplexity = overlayPerplexity
    }
}
