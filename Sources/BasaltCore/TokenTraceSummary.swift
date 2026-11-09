import Foundation
import System

public struct TokenTraceTable: Sendable, Equatable {
    public var layout: TokenTraceLayout
    public var tokens: Int
    public var floats: [Float]
    public var ints: [Int32]

    public init(layout: TokenTraceLayout, tokens: Int, floats: [Float], ints: [Int32]) {
        self.layout = layout
        self.tokens = tokens
        self.floats = floats
        self.ints = ints
    }

    public func column(_ field: TraceField, layer: Int, _ offset: Int = 0) -> [Float]? {
        guard let columns = TokenTraceLayout.columns(field, layer: layer, in: layout.floats), offset < columns.count else { return nil }
        let width = layout.floatWidth
        return (0 ..< tokens).map { floats[$0 * width + columns.lowerBound + offset] }
    }

    public func row(_ field: TraceField, layer: Int, token: Int) -> [Int32]? {
        guard let columns = TokenTraceLayout.columns(field, layer: layer, in: layout.ints) else { return nil }
        let start = token * layout.intWidth + columns.lowerBound
        return Array(ints[start ..< start + columns.count])
    }

    public func floatRow(_ field: TraceField, layer: Int, token: Int) -> [Float]? {
        guard let columns = TokenTraceLayout.columns(field, layer: layer, in: layout.floats) else { return nil }
        let start = token * layout.floatWidth + columns.lowerBound
        return Array(floats[start ..< start + columns.count])
    }

    public func routed(layer: Int) -> [(ids: ArraySlice<Int32>, gates: ArraySlice<Float>)]? {
        guard let ids = TokenTraceLayout.columns(.experts, layer: layer, in: layout.ints),
              let gates = TokenTraceLayout.columns(.gates, layer: layer, in: layout.floats) else { return nil }
        let (intWidth, floatWidth) = (layout.intWidth, layout.floatWidth)
        return (0 ..< tokens).map { token in
            (ints[token * intWidth + ids.lowerBound ..< token * intWidth + ids.upperBound],
             floats[token * floatWidth + gates.lowerBound ..< token * floatWidth + gates.upperBound])
        }
    }

    public var layers: [Int] { Array(Set(layout.floats.map(\.layer) + layout.ints.map(\.layer))).sorted() }
}

public enum TokenTraceSummary {
    public static let crossingPositions = 32

    public static func layers(_ table: TokenTraceTable, topK: Int) -> [LayerActivity] {
        table.layers.compactMap { index in
            var activity = LayerActivity(index: index)
            activity.heads = heads(table, layer: index, topK: topK)
            activity.experts = experts(table, layer: index, topK: topK)
            activity.routerEntropy = table.column(.routerEntropy, layer: index).map(mean)
            activity.routerMargin = table.column(.routerMargin, layer: index).map(mean)
            let empty = activity.heads.isEmpty && activity.experts.isEmpty && activity.routerEntropy == nil
            return empty ? nil : activity
        }
    }

    static func heads(_ table: TokenTraceTable, layer: Int, topK: Int) -> [HeadActivity] {
        guard table.tokens > 0, let width = TokenTraceLayout.columns(.heads, layer: layer, in: table.layout.floats)?.count else { return [] }
        let activity = (0 ..< width).compactMap { head in
            table.column(.heads, layer: layer, head).map { HeadActivity(head: head, meanNorm: mean($0), maxNorm: $0.max() ?? 0) }
        }
        return Array(activity.sorted { ($0.meanNorm, -$0.head) > ($1.meanNorm, -$1.head) }.prefix(topK))
    }

    static func experts(_ table: TokenTraceTable, layer: Int, topK: Int) -> [ExpertUsage] {
        var counts: [Int: (tokens: Int, gate: Float)] = [:]
        for (ids, gates) in table.routed(layer: layer) ?? [] {
            for (id, gate) in zip(ids, gates) {
                let previous = counts[Int(id), default: (0, 0)]
                counts[Int(id)] = (previous.tokens + 1, previous.gate + gate)
            }
        }
        let usage = counts.map { ExpertUsage(expert: $0.key, tokens: $0.value.tokens, meanGate: $0.value.gate / Float($0.value.tokens)) }
        return Array(usage.sorted { ($0.tokens, $0.meanGate, -$0.expert) > ($1.tokens, $1.meanGate, -$1.expert) }.prefix(topK))
    }

    public static func crossings(_ table: TokenTraceTable, directions: [TraceDirection]) -> [ProjectionCrossing] {
        directions.compactMap { direction in
            guard let values = table.column(.projection, layer: direction.layer) else { return nil }
            let above = values.indices.filter { values[$0] > direction.threshold }
            return ProjectionCrossing(
                layer: direction.layer, threshold: direction.threshold, count: above.count,
                positions: Array(above.prefix(crossingPositions)), max: values.max() ?? -.infinity)
        }
    }

    static func mean(_ values: [Float]) -> Float {
        values.isEmpty ? 0 : values.reduce(0, +) / Float(values.count)
    }
}

public enum TokenTraceFile {
    public struct Header: Codable, Equatable {
        public var format = "basalt-token-trace"
        public var version = 1
        public var model: String
        public var tokens: Int
        public var promptTokens: Int
        public var layout: TokenTraceLayout
        public var floatsPerToken: Int
        public var intsPerToken: Int
        public var binary = "float32 little-endian, tokens x floatsPerToken, after the token lines"
    }

    public struct Record: Codable, Equatable {
        public var position: Int
        public var id: Int
        public var token: String
        public var generated: Bool
        public var ints: [Int32]
    }

    public static func encode(model: String, table: TokenTraceTable, ids: [Int], texts: [String], promptTokens: Int, floats: Data) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let header = Header(
            model: model, tokens: table.tokens, promptTokens: promptTokens, layout: table.layout,
            floatsPerToken: table.layout.floatWidth, intsPerToken: table.layout.intWidth)
        var data = try encoder.encode(header) + Data("\n".utf8)
        let width = table.layout.intWidth
        for position in 0 ..< table.tokens {
            let record = Record(
                position: position, id: ids[position], token: texts[position], generated: position >= promptTokens,
                ints: Array(table.ints[position * width ..< (position + 1) * width]))
            data += try encoder.encode(record) + Data("\n".utf8)
        }
        return data + floats
    }

    public static func write(_ data: Data, into directory: URL, at now: Date = Date()) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        let url = directory.appendingPathComponent("\(Int(now.timeIntervalSince1970))-\(UUID().uuidString.prefix(8).lowercased()).trace")
        let fd = try FileDescriptor.open(
            FilePath(url.path), .writeOnly, options: [.create, .exclusiveCreate, .closeOnExec], permissions: .ownerReadWrite)
        _ = try fd.closeAfter { try fd.writeAll(data) }
        return url
    }
}
