import Foundation

public struct TraceDirection: Sendable, Codable, Equatable {
    public var layer: Int
    public var vector: [Float]
    public var threshold: Float

    public init(layer: Int, vector: [Float], threshold: Float) {
        self.layer = layer
        self.vector = vector
        self.threshold = threshold
    }
}

public struct TokenTraceRequest: Sendable, Codable, Equatable {
    public var prompt: String
    public var maxTokens = 32
    public var layers: [Int]?
    public var heads = true
    public var experts = true
    public var attentionEntropy = false
    public var projections: [TraceDirection] = []
    public var lensTopK = 0
    public var topK = 5
    public var maxFileBytes = TokenTraceRequest.defaultFileBytes
    public var directory = TokenTraceRequest.defaultDirectory

    public static let tokenLimit = 2048
    public static let generatedTokens = 0...1024
    public static let lensLimit = 0...20
    public static let summaryTopK = 1...50
    public static let projectionLimit = 0...16
    public static let fileBytes = 1...(64 << 20)
    public static let defaultFileBytes = 16 << 20
    public static let defaultDirectory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".basalt/traces")

    public init(prompt: String, maxTokens: Int = 32, layers: [Int]? = nil) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.layers = layers
    }

    public func tracedLayers(of count: Int) -> [Int] {
        layers.map { Array(Set($0)).sorted() } ?? Array(0 ..< count)
    }

    public func validate(against anatomy: ModelAnatomy) throws(BasaltError) {
        try validateLimits()
        let count = anatomy.layers.count
        if let layers, layers.isEmpty || layers.contains(where: { !(0 ..< count).contains($0) }) {
            throw .invalidRequestStructure(details: "layers must list at least one layer in 0..<\(count)")
        }
        let traced = Set(tracedLayers(of: count))
        for direction in projections {
            guard traced.contains(direction.layer) else {
                throw .invalidRequestStructure(details: "projection layer \(direction.layer) is not among the traced layers")
            }
            guard direction.vector.count == anatomy.hiddenSize else {
                throw .invalidRequestStructure(details: "a projection direction has \(direction.vector.count) values; the hidden size is \(anatomy.hiddenSize)")
            }
            guard direction.threshold.isFinite, direction.vector.allSatisfy(\.isFinite) else {
                throw .invalidRequestStructure(details: "projection directions and thresholds must be finite")
            }
        }
    }

    func validateLimits() throws(BasaltError) {
        guard Self.generatedTokens.contains(maxTokens), Self.lensLimit.contains(lensTopK), Self.summaryTopK.contains(topK) else {
            throw .invalidRequestStructure(details: "maxTokens must be in \(Self.generatedTokens), lensTopK in \(Self.lensLimit), topK in \(Self.summaryTopK)")
        }
        guard Self.projectionLimit.contains(projections.count), Self.fileBytes.contains(maxFileBytes) else {
            throw .invalidRequestStructure(details: "at most \(Self.projectionLimit.upperBound) projections; maxFileBytes in \(Self.fileBytes)")
        }
        guard heads || experts || attentionEntropy || lensTopK > 0 || !projections.isEmpty else {
            throw .invalidRequestStructure(details: "nothing to record: turn on heads, experts, attentionEntropy, lensTopK or projections")
        }
    }

    public func estimatedBytes(anatomy: ModelAnatomy, tokens: Int) -> Int {
        var floats = projections.count, ints = 0
        for index in tracedLayers(of: anatomy.layers.count) where anatomy.layers.indices.contains(index) {
            let layer = anatomy.layers[index]
            let heads = layer.heads ?? 0
            floats += (self.heads ? heads : 0) + (attentionEntropy ? heads : 0) + lensTopK
            ints += lensTopK
            if experts, let count = layer.experts {
                let k = layer.expertsPerToken ?? count
                floats += k + 2
                ints += k
            }
        }
        return tokens * (floats * 4 + ints * 11 + Self.recordOverhead) + Self.headerOverhead
    }

    static let recordOverhead = 160
    static let headerOverhead = 64 << 10
}

public enum TraceField: String, Sendable, Codable {
    case heads, attentionEntropy, gates, routerEntropy, routerMargin, projection, lensProbabilities, experts, lens
}

public struct TraceSegment: Sendable, Codable, Equatable {
    public var layer: Int
    public var field: TraceField
    public var width: Int

    public init(layer: Int, field: TraceField, width: Int) {
        self.layer = layer
        self.field = field
        self.width = width
    }
}

public struct TokenTraceLayout: Sendable, Codable, Equatable {
    public var floats: [TraceSegment]
    public var ints: [TraceSegment]

    public init(floats: [TraceSegment], ints: [TraceSegment]) {
        self.floats = floats
        self.ints = ints
    }

    public var floatWidth: Int { floats.reduce(0) { $0 + $1.width } }
    public var intWidth: Int { ints.reduce(0) { $0 + $1.width } }

    public static func columns(_ field: TraceField, layer: Int, in segments: [TraceSegment]) -> Range<Int>? {
        var start = 0
        for segment in segments {
            if segment.field == field, segment.layer == layer { return start ..< start + segment.width }
            start += segment.width
        }
        return nil
    }
}

public struct TokenTraceReport: Sendable, Codable {
    public var file: String
    public var bytes: Int
    public var promptTokens: Int
    public var generated: [Int]
    public var text: String
    public var layers: [LayerActivity] = []
    public var crossings: [ProjectionCrossing] = []

    public init(file: String, bytes: Int, promptTokens: Int, generated: [Int], text: String) {
        self.file = file
        self.bytes = bytes
        self.promptTokens = promptTokens
        self.generated = generated
        self.text = text
    }
}

public struct LayerActivity: Sendable, Codable, Equatable {
    public var index: Int
    public var heads: [HeadActivity] = []
    public var experts: [ExpertUsage] = []
    public var routerEntropy: Float?
    public var routerMargin: Float?

    public init(index: Int) {
        self.index = index
    }
}

public struct HeadActivity: Sendable, Codable, Equatable {
    public var head: Int
    public var meanNorm: Float
    public var maxNorm: Float

    public init(head: Int, meanNorm: Float, maxNorm: Float) {
        self.head = head
        self.meanNorm = meanNorm
        self.maxNorm = maxNorm
    }
}

public struct ExpertUsage: Sendable, Codable, Equatable {
    public var expert: Int
    public var tokens: Int
    public var meanGate: Float

    public init(expert: Int, tokens: Int, meanGate: Float) {
        self.expert = expert
        self.tokens = tokens
        self.meanGate = meanGate
    }
}

public struct ProjectionCrossing: Sendable, Codable, Equatable {
    public var layer: Int
    public var threshold: Float
    public var count: Int
    public var positions: [Int]
    public var max: Float

    public init(layer: Int, threshold: Float, count: Int, positions: [Int], max: Float) {
        self.layer = layer
        self.threshold = threshold
        self.count = count
        self.positions = positions
        self.max = max
    }
}
