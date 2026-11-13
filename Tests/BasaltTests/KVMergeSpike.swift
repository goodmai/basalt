import Foundation
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

enum KVMergeSpike {
    enum Arm: String, CaseIterable { case drop, merge, mergeNoBias }

    struct Middle {
        var keys: [[[Float]]]
        var values: [[[Float]]]
        var counts: [[Int]]
        var mergedRows = 0
        var droppedRows = 0
    }

    static func batches(appends: [Int], zones: KVZones) -> [Range<Int>] {
        var (window, migrated, total, moved) = (0, 0, 0, [Range<Int>]())
        for count in appends {
            let sinkBefore = min(zones.sinkTokens, total)
            total += count
            window += count - (min(zones.sinkTokens, total) - sinkBefore)
            let overflow = window - zones.windowTokens
            if overflow >= zones.migrationStep {
                let start = min(zones.sinkTokens, total) + migrated
                moved.append(start ..< start + overflow)
                (migrated, window) = (migrated + overflow, window - overflow)
            }
        }
        return moved
    }

    static func distance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in a.indices { sum += (a[i] - b[i]) * (a[i] - b[i]) }
        return sum.squareRoot()
    }

    static func norm(_ a: [Float]) -> Float { a.reduce(0) { $0 + $1 * $1 }.squareRoot() }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        for i in a.indices { dot += a[i] * b[i] }
        return dot / max(norm(a) * norm(b), 1e-12)
    }

    static func mean(_ rows: [[Float]]) -> [Float] {
        var out = [Float](repeating: 0, count: rows[0].count)
        for row in rows { for i in row.indices { out[i] += row[i] } }
        return out.map { $0 / Float(rows.count) }
    }

    static func compress(keys: [[Float]], values: [[Float]], keep: Int, radius: Float?) -> (keys: [[Float]], values: [[Float]], counts: [Int], merged: Int) {
        let order = keys.indices.sorted { norm(keys[$0]) < norm(keys[$1]) }
        let (kept, dropped) = (Array(order.prefix(keep)).sorted(), Array(order.dropFirst(keep)))
        var groups = Dictionary(uniqueKeysWithValues: kept.map { ($0, [$0]) })
        var merged = 0
        if let radius, radius > 0, !kept.isEmpty {
            for row in dropped {
                let target = kept.max { cosine(keys[row], keys[$0]) < cosine(keys[row], keys[$1]) }!
                let members = groups[target]! + [row]
                let center = mean(members.map { keys[$0] })
                if members.allSatisfy({ distance(keys[$0], center) <= radius * norm(center) }) {
                    groups[target] = members
                    merged += 1
                }
            }
        }
        let rows = kept.map { groups[$0]! }
        return (rows.map { mean($0.map { keys[$0] }) }, rows.map { mean($0.map { values[$0] }) }, rows.map(\.count), merged)
    }

    static func attention(query: [Float], keys: [[Float]], values: [[Float]], bias: [Float]) -> [Float] {
        let scale = 1 / Float(query.count).squareRoot()
        let scores = keys.indices.map { i in zip(query, keys[i]).reduce(0) { $0 + $1.0 * $1.1 } * scale + bias[i] }
        let top = scores.max() ?? 0
        let weights = scores.map { exp($0 - top) }
        let total = weights.reduce(0, +)
        var out = [Float](repeating: 0, count: values.first?.count ?? 0)
        for i in values.indices { for d in out.indices { out[d] += weights[i] / total * values[i][d] } }
        return out
    }

    static func relativeError(_ a: [Float], _ b: [Float]) -> Float {
        distance(a, b) / max(norm(b), 1e-12)
    }
}

extension KVMergeSpike {
    struct LayerOutcome {
        var errors: [Arm: [Float]] = [:]
        var merged = 0
        var dropped = 0
    }

    static func heads(_ array: MLXArray) -> [[[Float]]] {
        let (count, rows, dim) = (array.dim(1), array.dim(2), array.dim(3))
        let flat = array.asType(.float32).reshaped(-1).asArray(Float.self)
        return (0 ..< count).map { h in (0 ..< rows).map { r in Array(flat[(h * rows + r) * dim ..< (h * rows + r + 1) * dim]) } }
    }

    static func middle(keys: [[[Float]]], values: [[[Float]]], batches: [Range<Int>], zones: KVZones, radius: Float?) -> Middle {
        var result = Middle(keys: keys.map { _ in [] }, values: values.map { _ in [] }, counts: keys.map { _ in [] })
        var (quantized, evicted) = (0, 0)
        for batch in batches {
            let keep = min(batch.count, max(0, zones.keptMiddleTokens(quantized + evicted + batch.count) - quantized))
            (quantized, evicted) = (quantized + keep, evicted + batch.count - keep)
            for h in keys.indices {
                let c = compress(keys: Array(keys[h][batch]), values: Array(values[h][batch]), keep: keep, radius: radius)
                result.keys[h] += c.keys
                result.values[h] += c.values
                result.counts[h] += c.counts
                result.mergedRows += c.merged
                result.droppedRows += batch.count - keep - c.merged
            }
        }
        return result
    }

    static func evaluate(keys: MLXArray, values: MLXArray, appends: [Int], queries: [(position: Int, query: MLXArray)], zones: KVZones, radius: Float) -> LayerOutcome {
        let (k, v) = (heads(keys), heads(values))
        let batches = batches(appends: appends, zones: zones)
        guard let last = batches.last else { return LayerOutcome() }
        let sinks = min(zones.sinkTokens, keys.dim(2))
        let (drop, merge) = (middle(keys: k, values: v, batches: batches, zones: zones, radius: nil), middle(keys: k, values: v, batches: batches, zones: zones, radius: radius))
        var outcome = LayerOutcome(merged: merge.mergedRows, dropped: drop.droppedRows)
        let window = last.upperBound ..< keys.dim(2)
        let groups = queries.first.map { $0.query.dim(1) / k.count } ?? 1
        let arms = Arm.allCases.map { arm in
            let source = arm == .drop ? drop : merge
            let rows = sinks + source.keys[0].count + window.count
            let flatKeys = k.indices.flatMap { h in (Array(k[h][..<sinks]) + source.keys[h] + Array(k[h][window])).flatMap { $0 } }
            let flatValues = v.indices.flatMap { h in (Array(v[h][..<sinks]) + source.values[h] + Array(v[h][window])).flatMap { $0 } }
            let bias = k.indices.flatMap { h in
                let row = [Float](repeating: 0, count: sinks) + source.counts[h].map { arm == .merge ? log(Float($0)) : 0 } + [Float](repeating: 0, count: window.count)
                return Array(repeating: row, count: groups).flatMap { $0 }
            }
            return (arm, MLXArray(flatKeys, [1, k.count, rows, keys.dim(3)]), MLXArray(flatValues, [1, v.count, rows, values.dim(3)]),
                    MLXArray(bias, [1, k.count * groups, 1, rows]), sinks + source.keys[0].count)
        }
        let (fullKeys, fullValues) = (keys.asType(.float32), values.asType(.float32))
        for (position, query) in queries {
            let visible = min(position, keys.dim(2))
            let scale = 1 / Float(query.dim(3)).squareRoot()
            let full = MLXFast.scaledDotProductAttention(
                queries: query, keys: fullKeys[.ellipsis, ..<visible, 0...], values: fullValues[.ellipsis, ..<visible, 0...], scale: scale, mask: .none)
            for (arm, armKeys, armValues, bias, head) in arms {
                let rows = head + max(0, visible - window.lowerBound)
                let output = MLXFast.scaledDotProductAttention(
                    queries: query, keys: armKeys[.ellipsis, ..<rows, 0...], values: armValues[.ellipsis, ..<rows, 0...], scale: scale,
                    mask: .array(bias[.ellipsis, ..<rows]))
                let error = (sqrt((output - full).square().sum(axis: -1)) / sqrt(full.square().sum(axis: -1))).reshaped(-1).asArray(Float.self)
                outcome.errors[arm, default: []] += error
            }
        }
        return outcome
    }
}

final class KVRecordingCache: KVCacheAttentionProtocol {
    private let exact = KVCacheSimple()
    private(set) var appends: [Int] = []
    private(set) var queries: [(position: Int, query: MLXArray)] = []

    var offset: Int { exact.offset }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }
    func innerState() -> [MLXArray] { exact.innerState() }

    var state: [MLXArray] {
        get { exact.state }
        set { preconditionFailure("KVRecordingCache is never restored") }
    }

    var metaState: [String] {
        get { [] }
        set { preconditionFailure("KVRecordingCache is never restored") }
    }

    var rows: (keys: MLXArray, values: MLXArray) { (exact.state[0], exact.state[1]) }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        appends.append(keys.dim(2))
        return exact.update(keys: keys, values: values)
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        appends.append(keys.dim(2))
        let (allKeys, allValues) = exact.update(keys: keys, values: values)
        if queries.dim(2) == 1 { self.queries.append((exact.offset, queries.asType(.float32))) }
        return MLXFast.scaledDotProductAttention(queries: queries, keys: allKeys, values: allValues, scale: scale, mask: mask)
    }

    func trim(_ n: Int) -> Int { 0 }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n == 1 ? .none : .causal
    }

    func copy() -> any KVCache { preconditionFailure("KVRecordingCache is never copied") }

    static func replacingFullAttention(_ caches: [any KVCache]) -> [any KVCache] {
        caches.map { type(of: $0) == KVCacheSimple.self ? KVRecordingCache() : $0 }
    }
}
