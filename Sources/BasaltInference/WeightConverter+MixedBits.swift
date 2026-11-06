import BasaltCore
import Foundation
import MLX

package struct MixedBits: Sendable, Equatable {
    package var low: Int
    package var high: Int
    package var groupSize: Int
    package var budgetBytes: Int
    package var highLayers: Set<String> = []

    package init(low: Int, high: Int, groupSize: Int, budgetBytes: Int) {
        self.low = low
        self.high = high
        self.groupSize = groupSize
        self.budgetBytes = budgetBytes
    }

    func bits(layer: String) -> Int {
        highLayers.contains(layer) ? high : low
    }

    var configBlock: [String: Any] {
        var block: [String: Any] = ["bits": low, "group_size": groupSize, "mode": "affine"]
        for layer in highLayers {
            block[layer] = ["bits": high, "group_size": groupSize, "mode": "affine"]
        }
        return block
    }
}

package struct BitPlan: Sendable, Equatable {
    struct Candidate: Equatable {
        var layer: String
        var elements: Int
        var scaleBytes: Int
        var score: Double?
    }

    package var highLayers: Set<String> = []
    package var layers = 0
    package var unscored = 0
    package var bytes = 0

    init(_ candidates: [Candidate], fixedBytes: Int, mixed: MixedBits) throws(WeightConverter.Failure) {
        layers = candidates.count
        unscored = candidates.count { $0.score == nil }
        bytes = candidates.reduce(fixedBytes) { $0 + Self.bytes($1, bits: mixed.low, groupSize: mixed.groupSize) }
        guard bytes <= mixed.budgetBytes else {
            throw .refused("the budget of \(mixed.budgetBytes) bytes is below \(bytes) bytes, the size with every quantized layer at \(mixed.low) bits")
        }
        let ranked = candidates.filter { $0.score != nil }.sorted(by: Self.ranksAbove)
        for candidate in ranked {
            let extra = Self.bytes(candidate, bits: mixed.high, groupSize: mixed.groupSize)
                - Self.bytes(candidate, bits: mixed.low, groupSize: mixed.groupSize)
            guard bytes + extra <= mixed.budgetBytes else { continue }
            highLayers.insert(candidate.layer)
            bytes += extra
        }
    }

    static func ranksAbove(_ a: Candidate, _ b: Candidate) -> Bool {
        let left = a.score ?? -.infinity, right = b.score ?? -.infinity
        return left == right ? a.layer < b.layer : left > right
    }

    static func bytes(_ candidate: Candidate, bits: Int, groupSize: Int) -> Int {
        candidate.elements * bits / 8 + 2 * (candidate.elements / groupSize) * candidate.scaleBytes
    }
}

extension WeightConverter {
    static let sampledExperts = 4

    func plan(_ layout: SourceLayout, _ mixed: MixedBits) throws -> BitPlan {
        var candidates: [BitPlan.Candidate] = []
        var fixedBytes = 0
        for shard in layout.shards {
            var arrays = try loadArrays(url: shard)
            for key in arrays.keys.sorted() where !layout.isCompanion(key) {
                guard let tensor = arrays.removeValue(forKey: key) else { continue }
                let packed = try layout.packed(key, taking: &arrays)
                guard Self.quantizable(key, tensor, packed: packed, sourceQuantized: layout.quantization != nil, groupSize: mixed.groupSize) else {
                    fixedBytes += packed.map { $0.scales.size * $0.quantization.groupSize * $0.scales.dtype.size } ?? tensor.nbytes
                    continue
                }
                let weight = Self.dense(tensor, packed)
                candidates.append(BitPlan.Candidate(
                    layer: String(key.dropLast(".weight".count)), elements: weight.size, scaleBytes: weight.dtype.size,
                    score: Self.isPerExpert(key) ? nil : Self.noiseVariance(weight)))
            }
            Memory.clearCache()
        }
        return try BitPlan(candidates, fixedBytes: fixedBytes, mixed: mixed)
    }

    static func isPerExpert(_ key: String) -> Bool {
        let parts = key.split(separator: ".")
        return zip(parts, parts.dropFirst()).contains { $0 == "experts" && Int($1) != nil }
    }

    static func noiseVariance(_ weight: MLXArray) -> Double? {
        let count = weight.ndim == 3 ? weight.dim(0) : 1
        let picks = min(count, sampledExperts)
        let matrices = weight.ndim == 3 ? (0 ..< picks).map { weight[$0 * count / picks] } : [weight]
        eval(matrices)
        let values = matrices.compactMap { matrix in
            (try? WeightSpectrum.measure(rows: matrix.dim(0), columns: matrix.dim(1)) { matrix[$0] })?.noiseVariance
        }.filter(\.isFinite)
        return values.isEmpty ? nil : values.reduce(0, +) / Double(values.count)
    }
}
