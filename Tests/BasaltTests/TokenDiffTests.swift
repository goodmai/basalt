import Foundation
import Testing
@testable import BasaltCore

@Suite("Token divergence diagnostics with CustomTestReflectable")
struct TokenDiffTests {
    @Test("Identical sequences compare equal")
    func identicalSequences() {
        let seq1 = [1, 2, 3, 4, 5]
        let seq2 = [1, 2, 3, 4, 5]
        let diff1 = TokenDiff(seq1, expected: seq2)
        let diff2 = TokenDiff(seq2, expected: seq1)
        #expect(diff1 == diff2)
        expectTokenSequence(seq1, seq2)
    }

    @Test("Divergent sequence mirror exposes first divergence index and window")
    func divergentSequenceMirror() {
        let actual = [10, 11, 12, 13, 14, 999, 16, 17, 18, 19]
        let expected = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        let diff = TokenDiff(actual, expected: expected)
        let mirror = diff.customTestMirror

        let children = Dictionary(uniqueKeysWithValues: mirror.children.compactMap { child in
            child.label.map { ($0, child.value) }
        })

        #expect(children["divergenceIndex"] as? Int == 5)
        #expect(diff.testDescription.contains("divergenceIndex: 5"))
    }
}
