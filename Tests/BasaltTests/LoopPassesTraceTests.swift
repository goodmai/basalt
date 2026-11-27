import Foundation
import MLXLMCommon
import Testing
@testable import BasaltInference

@Suite("X-ray loop passes per token for looped models under --experimental-loop-exit (#820)", .serialized)
struct LoopPassesTraceTests {
    static let ids = [5, 19, 42, 7, 88, 13]

    @Test("Looped families expose their loop-exit counter and a plain decoder does not")
    func counterPerFamily() throws {
        #expect(LoopExit.counter(of: try TinyModel.make(.huginn)) != nil)
        #expect(LoopExit.counter(of: try TinyModel.make(.ouro)) != nil)
        #expect(LoopExit.counter(of: try TinyModel.make(.qwen3Small)) == nil)
    }

    @Test("No threshold reports nothing; a threshold that never fires reports num_steps per token; any threshold sums to the model's own pass counter")
    func passesPerToken() throws {
        let model = try #require(try TinyModel.make(.huginn) as? HuginnModel)
        let core = model.transformer
        core.loopExitThreshold = nil
        #expect(try LoopExit.passesPerToken(Self.ids, model: model) == nil)
        core.loopExitThreshold = 1e-9
        #expect(try LoopExit.passesPerToken(Self.ids, model: model) == Array(repeating: core.config.steps, count: Self.ids.count))
        core.loopExitThreshold = 0.5
        let before = core.exitPasses
        let passes = try #require(try LoopExit.passesPerToken(Self.ids, model: model))
        #expect(passes.count == Self.ids.count && passes.allSatisfy { (1 ... core.config.steps).contains($0) })
        #expect(passes.reduce(0, +) == core.exitPasses - before)
    }
}
