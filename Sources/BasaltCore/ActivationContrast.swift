import Foundation

public struct ActivationContrastRequest: Sendable, Codable, Equatable {
    public var a: [String]
    public var b: [String]
    public var maxTokens = 16
    public var layers: [Int]?
    public var onsetWindow = 4
    public var topK = 20

    public static let generatedTokens = 0...256
    public static let onsetTokens = 1...64
    public static let ranked = 1...100

    public init(a: [String], b: [String], maxTokens: Int = 16, layers: [Int]? = nil) {
        self.a = a
        self.b = b
        self.maxTokens = maxTokens
        self.layers = layers
    }

    public func validate(layers count: Int) throws(BasaltError) {
        guard ContrastRequest.classPrompts.contains(a.count), ContrastRequest.classPrompts.contains(b.count) else {
            throw .invalidRequestStructure(details: "activation_contrast needs \(ContrastRequest.classPrompts) prompts in a and in b")
        }
        guard Self.generatedTokens.contains(maxTokens), Self.onsetTokens.contains(onsetWindow), Self.ranked.contains(topK) else {
            throw .invalidRequestStructure(details: "maxTokens must be in \(Self.generatedTokens), onsetWindow in \(Self.onsetTokens), topK in \(Self.ranked)")
        }
        if let layers, layers.isEmpty || layers.contains(where: { !(0 ..< count).contains($0) }) {
            throw .invalidRequestStructure(details: "layers must list at least one layer in 0..<\(count)")
        }
    }

    public var trace: TokenTraceRequest {
        var request = TokenTraceRequest(prompt: "", maxTokens: maxTokens, layers: layers)
        request.maxFileBytes = TokenTraceRequest.fileBytes.upperBound
        return request
    }
}

public enum ComponentKind: String, Sendable, Codable {
    case head, expert
}

public struct ComponentContrast: Sendable, Codable, Equatable {
    public var layer: Int
    public var kind: ComponentKind
    public var index: Int
    public var rateA: Float = 0
    public var rateB: Float = 0
    public var meanA: Float = 0
    public var meanB: Float = 0
    public var onsetA: Float = 0
    public var onsetB: Float = 0
    public var cohensD: Float = 0
    public var auroc: Float = 0.5

    public init(layer: Int, kind: ComponentKind, index: Int) {
        self.layer = layer
        self.kind = kind
        self.index = index
    }
}

public struct ActivationContrastReport: Sendable, Codable {
    public var components: [ComponentContrast]
    public var layers: [DirectionLayer]
    public var bestLayer: Int
    public var direction: [Float]

    public init(components: [ComponentContrast], layers: [DirectionLayer], bestLayer: Int, direction: [Float]) {
        self.components = components
        self.layers = layers
        self.bestLayer = bestLayer
        self.direction = direction
    }
}

public struct TracedSample: Sendable, Equatable {
    public var table: TokenTraceTable
    public var promptTokens: Int

    public init(table: TokenTraceTable, promptTokens: Int) {
        self.table = table
        self.promptTokens = promptTokens
    }
}

public enum ActivationStatistics {
    struct Series {
        var mean: Float
        var rate: Float
        var onset: Float
    }

    public static func cohensD(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count > 1, b.count > 1 else { return 0 }
        func moments(_ x: [Float]) -> (mean: Float, squares: Float) {
            let mean = x.reduce(0, +) / Float(x.count)
            return (mean, x.reduce(0) { $0 + ($1 - mean) * ($1 - mean) })
        }
        let (ma, sa) = moments(a), (mb, sb) = moments(b)
        let pooled = ((sa + sb) / Float(a.count + b.count - 2)).squareRoot()
        return pooled > 1e-12 ? (ma - mb) / pooled : 0
    }

    public static func auroc(_ a: [Float], _ b: [Float]) -> Float {
        guard !a.isEmpty, !b.isEmpty else { return 0.5 }
        var wins: Float = 0
        for x in a {
            for y in b { wins += x > y ? 1 : x == y ? 0.5 : 0 }
        }
        return wins / Float(a.count * b.count)
    }

    public static func rank(a: [TracedSample], b: [TracedSample], onsetWindow: Int, topK: Int) -> [ComponentContrast] {
        guard let layout = (a + b).first?.table.layout else { return [] }
        let components = layout.floats.flatMap { segment -> [(Int, ComponentKind, Int)] in
            switch segment.field {
            case .heads: (0 ..< segment.width).map { (segment.layer, .head, $0) }
            case .gates: experts(in: a + b, layer: segment.layer).map { (segment.layer, .expert, $0) }
            case .attentionEntropy, .routerEntropy, .routerMargin, .projection, .lensProbabilities, .experts, .lens: []
            }
        }
        let ranked = components.map { layer, kind, index in
            contrast(layer: layer, kind: kind, index: index, a: a, b: b, onsetWindow: onsetWindow)
        }
        return Array(ranked.sorted { (abs($0.cohensD), abs($0.auroc - 0.5)) > (abs($1.cohensD), abs($1.auroc - 0.5)) }.prefix(topK))
    }

    static func experts(in samples: [TracedSample], layer: Int) -> [Int] {
        var seen = Set<Int>()
        for sample in samples {
            for (ids, _) in sample.table.routed(layer: layer) ?? [] { ids.forEach { seen.insert(Int($0)) } }
        }
        return seen.sorted()
    }

    static func contrast(layer: Int, kind: ComponentKind, index: Int, a: [TracedSample], b: [TracedSample], onsetWindow: Int) -> ComponentContrast {
        let values = (a + b).map { activation($0, layer: layer, kind: kind, index: index) }
        let pooled = values.flatMap { $0 }
        let threshold = kind == .head ? pooled.reduce(0, +) / Float(max(pooled.count, 1)) : 0
        let series = zip(a + b, values).map { sample, x in summary(x, threshold: threshold, onset: sample.promptTokens, window: onsetWindow) }
        let (sa, sb) = (Array(series.prefix(a.count)), Array(series.suffix(b.count)))
        func average(_ s: [Series], _ key: KeyPath<Series, Float>) -> Float { s.isEmpty ? 0 : s.reduce(0) { $0 + $1[keyPath: key] } / Float(s.count) }
        var result = ComponentContrast(layer: layer, kind: kind, index: index)
        (result.rateA, result.rateB) = (average(sa, \.rate), average(sb, \.rate))
        (result.meanA, result.meanB) = (average(sa, \.mean), average(sb, \.mean))
        (result.onsetA, result.onsetB) = (average(sa, \.onset), average(sb, \.onset))
        result.cohensD = cohensD(sa.map(\.mean), sb.map(\.mean))
        result.auroc = auroc(sa.map(\.mean), sb.map(\.mean))
        return result
    }

    static func activation(_ sample: TracedSample, layer: Int, kind: ComponentKind, index: Int) -> [Float] {
        switch kind {
        case .head: return sample.table.column(.heads, layer: layer, index) ?? []
        case .expert:
            return (sample.table.routed(layer: layer) ?? []).map { ids, gates in
                ids.firstIndex(of: Int32(index)).map { gates[gates.startIndex + ($0 - ids.startIndex)] } ?? 0
            }
        }
    }

    static func summary(_ x: [Float], threshold: Float, onset: Int, window: Int) -> Series {
        guard !x.isEmpty else { return Series(mean: 0, rate: 0, onset: 0) }
        let before = x[max(0, min(onset, x.count) - window) ..< min(onset, x.count)]
        return Series(
            mean: x.reduce(0, +) / Float(x.count),
            rate: Float(x.filter { $0 > threshold }.count) / Float(x.count),
            onset: before.isEmpty ? 0 : before.reduce(0, +) / Float(before.count))
    }
}
