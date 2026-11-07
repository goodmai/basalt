import Foundation
import Testing
import MLX
@testable import BasaltCore
@testable import BasaltInference

@Suite("GoodViewDataset & GoodViewCollator: multi-view JSONL parsing, 2 global + 6 local spans, 15% masking (#385)")
struct GoodViewCollatorTests {
    static let sampleEncode: @Sendable (String) -> [Int] = { text in
        text.split(separator: " ").enumerated().map { idx, word in
            10 + ((word.utf8.reduce(0) { $0 + Int($1) } + idx) % 200)
        }
    }

    @Test("GoodViewDataset parses array and object JSONL records and refuses K < 2 or empty inputs", arguments: [
        (#"["a red car parked by a tree", "crimson sedan next to an oak"]"#, true),
        (#"{"views": ["view one of the scene", "view two of the scene", "view three of the scene"]}"#, true),
        (#"["single view only"]"#, false),
        (#"{"views": ["only one", "   "]}"#, false),
        (#"not valid json"#, false),
        ("", false),
    ])
    func datasetParsing(jsonl: String, valid: Bool) {
        if valid {
            let dataset = try? GoodViewDataset.parse(jsonl: jsonl)
            #expect(dataset != nil)
            #expect((dataset?.scenes.first?.views.count ?? 0) >= 2)
        } else {
            #expect(throws: BasaltError.self) {
                try GoodViewDataset.parse(jsonl: jsonl)
            }
        }
    }

    @Test("GoodViewCollator outputs deterministic [B * 8, SeqLen] tensors with global (50..100%) and local (15..50%) span bounds and 15% token masking",
          arguments: [(1, 32, UInt64(7)), (3, 64, UInt64(42)), (4, 48, UInt64(99))])
    func collatorBoundsAndDeterminism(batchSize: Int, maxSeqLen: Int, seed: UInt64) throws {
        let baseViews = (0 ..< 4).map { idx in
            (0 ..< 24).map { tok in "scene\(idx)_tok\(tok)" }.joined(separator: " ")
        }
        let scenes = (0 ..< batchSize).map { _ in GoodViewScene(views: baseViews) }
        let collator = GoodViewCollator(maxSeqLen: maxSeqLen, maskRatio: 0.15, maskTokenId: 999, padTokenId: 0)

        var rng1 = SplitMix64(seed: seed)
        var rng2 = SplitMix64(seed: seed)
        var rng3 = SplitMix64(seed: seed ^ 0xFF)

        let batch1 = try collator.collate(scenes: scenes, encode: Self.sampleEncode, rng: &rng1)
        let batch2 = try collator.collate(scenes: scenes, encode: Self.sampleEncode, rng: &rng2)
        let batch3 = try collator.collate(scenes: scenes, encode: Self.sampleEncode, rng: &rng3)
        eval(batch1.inputIds, batch1.attentionMask, batch2.inputIds, batch3.inputIds)

        #expect(batch1.inputIds.shape == [batchSize * 8, maxSeqLen])
        #expect(batch1.attentionMask.shape == [batchSize * 8, maxSeqLen])
        #expect(batch1.inputIds.asArray(Int32.self) == batch2.inputIds.asArray(Int32.self))
        #expect(batch1.attentionMask.asArray(Int32.self) == batch2.attentionMask.asArray(Int32.self))
        #expect(batch1.inputIds.asArray(Int32.self) != batch3.inputIds.asArray(Int32.self))

        let idsFlat = batch1.inputIds.asArray(Int32.self)
        for row in 0 ..< (batchSize * 8) {
            let viewInScene = row % 8
            let srcLen = batch1.sourceLengths[row]
            let spanLen = batch1.spanLengths[row]
            let ratio = viewInScene < 2 ? collator.globalRatio : collator.localRatio
            let expectedBounds = GoodViewCollator.spanBounds(tokenCount: srcLen, ratio: ratio, maxSeqLen: maxSeqLen)
            #expect(expectedBounds.contains(spanLen), "row \(row) spanLen \(spanLen) outside \(expectedBounds)")

            let rowSlice = Array(idsFlat[(row * maxSeqLen) ..< ((row + 1) * maxSeqLen)])
            let activeTokens = Array(rowSlice.prefix(spanLen))
            let maskedCount = activeTokens.filter { $0 == 999 }.count
            let expectedMasked = Int((Double(spanLen) * 0.15).rounded())
            #expect(maskedCount == expectedMasked)
        }
    }
}
