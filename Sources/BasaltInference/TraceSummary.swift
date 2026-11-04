import Foundation
import BasaltCore

enum TraceSummary {
    struct Ranked: Equatable {
        let index: Int
        let value: Float
    }

    static func top(_ row: [Float], k: Int) -> [Ranked] {
        guard k > 0 else { return [] }
        var best: [Ranked] = []
        best.reserveCapacity(k + 1)
        for (index, value) in row.enumerated() where !value.isNaN && (best.count < k || value > best[best.count - 1].value) {
            best.insert(Ranked(index: index, value: value), at: best.firstIndex { value > $0.value } ?? best.endIndex)
            if best.count > k { best.removeLast() }
        }
        return best
    }

    static func entropy(_ row: some Sequence<Float>) -> Float {
        -row.reduce(0) { $1 > 0 ? $0 + $1 * log($1) : $0 }
    }

    static func heads(_ weights: [Float], heads: Int, k: Int, firstPosition: Int) -> [HeadAttention] {
        guard heads > 0, weights.count % heads == 0 else { return [] }
        let keys = weights.count / heads
        return (0 ..< heads).map { head in
            let row = Array(weights[head * keys ..< (head + 1) * keys])
            return HeadAttention(
                head: head,
                top: top(row, k: k).map { PositionWeight(position: firstPosition + $0.index, weight: $0.value) },
                entropy: entropy(row))
        }
    }
}
