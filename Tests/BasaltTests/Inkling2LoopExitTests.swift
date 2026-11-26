import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCLI
@testable import BasaltInference

@Suite("Inkling 2 --experimental-loop-exit: training-free early exit from the looped core on the inkling2Loop TinyModel (#820)", .serialized)
struct Inkling2LoopExitTests {
    static let prompt: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 200, 3]
    static let steps: [Int32] = [150, 77, 12, 44, 101, 6]

    static func model(threshold: Float?) throws -> Inkling2Model {
        let model = try #require(try TinyModel.make(.inkling2Loop) as? Inkling2Model)
        model.model.languageModel.loopExitThreshold = threshold
        return model
    }

    static func decode(_ model: Inkling2Model) throws -> (logits: MLXArray, cache: [KVCache]) {
        let cache = try model.newCache(parameters: nil)
        _ = model(MLXArray(prompt)[.newAxis], cache: cache)
        let logits = steps.map { model(MLXArray([$0])[.newAxis], cache: cache).asType(.float32) }
        return (concatenated(logits, axis: 1), cache)
    }

    @Test("A threshold no pass meets decodes exactly like the stock fixed-pass model, and every one-token step counts 3 passes")
    func unmetThresholdIsStock() throws {
        let stock = try Self.decode(Self.model(threshold: nil)).logits
        let model = try Self.model(threshold: 1e-12)
        let gated = try Self.decode(model).logits
        #expect(abs(gated - stock).max().item(Float.self) == 0)
        let text = model.model.languageModel
        #expect(text.exitSteps == Self.steps.count && text.exitPasses == 3 * Self.steps.count)
    }

    @Test("A threshold every pass meets ends each one-token step after pass 2: caches of every visit stay aligned, pass 3 holds pass 2's newest entries, and decoding continues finite")
    func metThresholdSkipsPassThree() throws {
        let model = try Self.model(threshold: .greatestFiniteMagnitude)
        let (logits, cache) = try Self.decode(model)
        let text = model.model.languageModel
        #expect(text.exitSteps == Self.steps.count && text.exitPasses == 2 * Self.steps.count)
        #expect(!isNaN(logits).any().item(Bool.self) && abs(logits).max().item(Float.self).isFinite)
        let total = Self.prompt.count + Self.steps.count
        #expect(stride(from: 0, to: cache.count, by: 2).allSatisfy { cache[$0].offset == total })
        let loop = try #require(model.config.loop)
        let core = loop.coreEnd - loop.coreStart
        for layer in 0 ..< core {
            let pass2 = cache[2 * (loop.coreStart + core + layer)].state[0][0..., 0..., (-1)..., 0...]
            let pass3 = cache[2 * (loop.coreStart + 2 * core + layer)].state[0][0..., 0..., (-1)..., 0...]
            #expect(abs(pass3 - pass2).max().item(Float.self) == 0, "core layer \(layer)")
        }
    }

    @Test("Prefill and multi-token verify chunks never exit, so the prompt always runs every pass")
    func multiTokenNeverExits() throws {
        let model = try Self.model(threshold: .greatestFiniteMagnitude)
        _ = model(MLXArray(Self.prompt)[.newAxis], cache: try model.newCache(parameters: nil))
        #expect(model.model.languageModel.exitSteps == 0)
    }

    @Test("--experimental-loop-exit reaches the model through BASALT_LOOP_EXIT, takes only a fraction in (0, 1), and the default sets nothing")
    func flagReachesModel() throws {
        var set: [String: String] = [:]
        try ServeCommand.parse(["--experimental-loop-exit", "0.05"]).experimental.applyKernelEnvironment { set[$0] = $1 }
        #expect(set == [LoopExit.environmentKey: "0.05"])
        set = [:]
        try ServeCommand.parse([]).experimental.applyKernelEnvironment { set[$0] = $1 }
        #expect(set.isEmpty)
        for value in ["0", "1", "1.5", "-0.1"] {
            #expect(throws: (any Error).self, "\(value)") { _ = try ServeCommand.parse(["--experimental-loop-exit", value]) }
        }
    }
}
