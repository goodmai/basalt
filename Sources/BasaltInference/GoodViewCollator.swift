import Foundation
import MLX
import MLXLMCommon
import BasaltCore

struct GoodViewScene: Sendable, Equatable {
    let views: [String]
}

struct GoodViewDataset: Sendable, Equatable {
    let scenes: [GoodViewScene]

    init(scenes: [GoodViewScene]) throws(BasaltError) {
        guard !scenes.isEmpty else {
            throw .invalidRequestStructure(details: "good-view dataset: dataset must contain at least one scene")
        }
        for (index, scene) in scenes.enumerated() {
            let trimmed = scene.views.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            guard trimmed.count >= 2 else {
                throw .invalidRequestStructure(
                    details: "good-view dataset: scene \(index) has \(trimmed.count) valid view(s); at least 2 are required")
            }
        }
        self.scenes = scenes.map { scene in
            GoodViewScene(views: scene.views.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty })
        }
    }

    static func load(from path: String) throws(BasaltError) -> GoodViewDataset {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let text: String
        do {
            text = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw .invalidRequestStructure(details: "good-view dataset: could not read \(path): \(error.localizedDescription)")
        }
        return try parse(jsonl: text)
    }

    static func parse(jsonl: String) throws(BasaltError) -> GoodViewDataset {
        let lines = jsonl.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else {
            throw .invalidRequestStructure(details: "good-view dataset: JSONL is empty")
        }
        var scenes: [GoodViewScene] = []
        scenes.reserveCapacity(lines.count)
        let decoder = JSONDecoder()
        for (lineIndex, line) in lines.enumerated() {
            let data = Data(line.utf8)
            if let array = try? decoder.decode([String].self, from: data) {
                scenes.append(GoodViewScene(views: array))
            } else if let wrapper = try? decoder.decode(ViewsRecord.self, from: data) {
                scenes.append(GoodViewScene(views: wrapper.views))
            } else {
                throw .invalidRequestStructure(
                    details: "good-view dataset: line \(lineIndex + 1) is neither a JSON string array nor {\"views\": [...]}")
            }
        }
        return try GoodViewDataset(scenes: scenes)
    }

    private struct ViewsRecord: Decodable {
        let views: [String]
    }
}

struct GoodViewBatch {
    let inputIds: MLXArray
    let attentionMask: MLXArray
    let sceneCount: Int
    let viewsPerScene: Int
    let maxSeqLen: Int
    let spanLengths: [Int]
    let sourceLengths: [Int]
}

struct GoodViewCollator: Sendable {
    static let globalViewsPerScene = 2
    static let localViewsPerScene = 6
    static let totalViewsPerScene = 8

    var maxSeqLen: Int
    var globalRatio: ClosedRange<Double>
    var localRatio: ClosedRange<Double>
    var maskRatio: Double
    var maskTokenId: Int
    var padTokenId: Int

    init(
        maxSeqLen: Int = 64,
        globalRatio: ClosedRange<Double> = 0.50...1.00,
        localRatio: ClosedRange<Double> = 0.15...0.50,
        maskRatio: Double = 0.15,
        maskTokenId: Int = 0,
        padTokenId: Int = 0
    ) {
        self.maxSeqLen = max(4, maxSeqLen)
        self.globalRatio = globalRatio
        self.localRatio = localRatio
        self.maskRatio = min(max(0.0, maskRatio), 1.0)
        self.maskTokenId = maskTokenId
        self.padTokenId = padTokenId
    }

    func collate(
        scenes: [GoodViewScene],
        encode: (String) -> [Int],
        rng: inout SplitMix64
    ) throws(BasaltError) -> GoodViewBatch {
        guard !scenes.isEmpty else {
            throw .invalidRequestStructure(details: "good-view collator: batch must contain at least one scene")
        }
        let rowCount = scenes.count * Self.totalViewsPerScene
        var acc = SpanAccumulator(rowCount: rowCount, maxSeqLen: maxSeqLen)

        for scene in scenes {
            guard scene.views.count >= 2 else {
                throw .invalidRequestStructure(details: "good-view collator: every scene must have at least 2 views")
            }
            let tokenized = try scene.views.map { view throws(BasaltError) -> [Int] in
                let raw = Array(encode(view).prefix(maxSeqLen))
                guard !raw.isEmpty else {
                    throw .invalidRequestStructure(details: "good-view collator: tokenized view is empty")
                }
                return raw
            }
            let globalIndices = Self.sampleTwoDistinct(count: tokenized.count, rng: &rng)
            for idx in globalIndices {
                appendSpan(from: tokenized[idx], ratio: globalRatio, rng: &rng, into: &acc)
            }
            for _ in 0 ..< Self.localViewsPerScene {
                let idx = Int.random(in: 0 ..< tokenized.count, using: &rng)
                appendSpan(from: tokenized[idx], ratio: localRatio, rng: &rng, into: &acc)
            }
        }

        return GoodViewBatch(
            inputIds: MLXArray(acc.flatIds).reshaped(rowCount, maxSeqLen),
            attentionMask: MLXArray(acc.flatMask).reshaped(rowCount, maxSeqLen),
            sceneCount: scenes.count,
            viewsPerScene: Self.totalViewsPerScene,
            maxSeqLen: maxSeqLen,
            spanLengths: acc.spanLengths,
            sourceLengths: acc.sourceLengths
        )
    }

    static func spanBounds(tokenCount: Int, ratio: ClosedRange<Double>, maxSeqLen: Int) -> ClosedRange<Int> {
        let capped = min(max(1, tokenCount), maxSeqLen)
        let low = max(1, min(capped, Int(ceil(Double(capped) * ratio.lowerBound))))
        let high = max(low, min(capped, Int(floor(Double(capped) * ratio.upperBound))))
        return low...high
    }

    private static func sampleTwoDistinct(count: Int, rng: inout SplitMix64) -> [Int] {
        let first = Int.random(in: 0 ..< count, using: &rng)
        let offset = Int.random(in: 1 ..< count, using: &rng)
        let second = (first + offset) % count
        return [first, second]
    }
}

extension GoodViewCollator {
    fileprivate struct SpanAccumulator {
        var flatIds: [Int32] = []
        var flatMask: [Int32] = []
        var spanLengths: [Int] = []
        var sourceLengths: [Int] = []

        init(rowCount: Int, maxSeqLen: Int) {
            flatIds.reserveCapacity(rowCount * maxSeqLen)
            flatMask.reserveCapacity(rowCount * maxSeqLen)
            spanLengths.reserveCapacity(rowCount)
            sourceLengths.reserveCapacity(rowCount)
        }
    }

    fileprivate func appendSpan(
        from tokens: [Int],
        ratio: ClosedRange<Double>,
        rng: inout SplitMix64,
        into acc: inout SpanAccumulator
    ) {
        let bounds = Self.spanBounds(tokenCount: tokens.count, ratio: ratio, maxSeqLen: maxSeqLen)
        let spanLen = Int.random(in: bounds, using: &rng)
        let start = Int.random(in: 0 ... max(0, tokens.count - spanLen), using: &rng)
        var span = Array(tokens[start ..< (start + spanLen)])
        applyTokenMask(&span, rng: &rng)
        acc.spanLengths.append(spanLen)
        acc.sourceLengths.append(tokens.count)
        let padCount = max(0, maxSeqLen - spanLen)
        for token in span {
            acc.flatIds.append(Int32(token))
            acc.flatMask.append(1)
        }
        if padCount > 0 {
            acc.flatIds.append(contentsOf: repeatElement(Int32(padTokenId), count: padCount))
            acc.flatMask.append(contentsOf: repeatElement(Int32(0), count: padCount))
        }
    }

    fileprivate func applyTokenMask(_ span: inout [Int], rng: inout SplitMix64) {
        guard maskRatio > 0, !span.isEmpty else { return }
        let maskCount = min(span.count, Int((Double(span.count) * maskRatio).rounded()))
        guard maskCount > 0 else { return }
        var indices = Array(span.indices)
        for i in 0 ..< maskCount {
            let j = Int.random(in: i ..< indices.count, using: &rng)
            indices.swapAt(i, j)
            span[indices[i]] = maskTokenId
        }
    }
}
