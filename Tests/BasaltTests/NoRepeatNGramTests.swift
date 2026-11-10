import Testing
import MLX
import MLXLMCommon
@testable import BasaltInference

@Suite("No-repeat n-gram for DeepSeek-OCR: the token that would repeat an n-gram inside the window is blocked, as in the official NoRepeatNGramLogitsProcessor (#400)")
struct NoRepeatNGramTests {
    @Test("The continuation of a repeated prefix is banned only when the earlier n-gram lies inside the window")
    func banned() {
        let loop = [7, 1, 2, 3, 1, 2]
        #expect(NoRepeatNGram.banned(loop, size: 3, window: 50) == [3])
        #expect(NoRepeatNGram.banned(loop, size: 3, window: 3) == [])
        #expect(NoRepeatNGram.banned([1, 2, 1, 2, 1, 2], size: 2, window: 50) == [1])
        #expect(NoRepeatNGram.banned([1, 2], size: 3, window: 50) == [])
        #expect(NoRepeatNGram.banned([5, 6, 7, 8], size: 2, window: 50) == [])
    }

    @Test("process sets banned logits to -inf after the sampled tokens are read, keeps whitelisted ones, and prompt resets the history")
    func masks() {
        var processor = NoRepeatNGram(size: 2, window: 50, whitelist: [4])
        processor.prompt(MLXArray([Int32(0)]))
        for token in [3, 1, 4, 1] {
            _ = processor.process(logits: MLXArray.zeros([1, 6]))
            processor.didSample(token: MLXArray(Int32(token)))
        }
        let masked = processor.process(logits: MLXArray.zeros([1, 6]))[0].asArray(Float.self)
        #expect(masked == [0, 0, 0, 0, 0, 0])
        processor.didSample(token: MLXArray(Int32(3)))
        let afterThree = processor.process(logits: MLXArray.zeros([1, 6]))[0].asArray(Float.self)
        #expect(afterThree[1] == -Float.infinity && afterThree.enumerated().allSatisfy { $0.offset == 1 || $0.element == 0 })
        processor.prompt(MLXArray([Int32(0)]))
        #expect(processor.process(logits: MLXArray.zeros([1, 6]))[0].asArray(Float.self).allSatisfy { $0 == 0 })
    }
}
