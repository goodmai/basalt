import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Hybrid prefix reuse on a real model: restored snapshot versus one- and two-phase prefill (#79)", .serialized)
struct HybridPrefixDriftTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_HYBRID_MODEL_PATH"]
    }

    static let system = "You are a careful coding agent.\n" + (0..<40).flatMap { i in
        (0..<10).map { j in "File src/module_\(i).swift line \(j): let value\(i)_\(j) = compute(\(i), \(j)) // note \(i * j)" }
    }.joined(separator: "\n")

    static func lastLogits(
        _ tokens: MLXArray, from start: Int, cache: PromptCache, state: LMOutput.State?, model: any LanguageModel,
        prefill: PrefillParameters = GenerateParameters().prefill
    ) throws -> MLXArray {
        let tail = LMInput(tokens: tokens[.ellipsis, start...])
        switch try model.prepare(tail, cache: cache.layers, state: state, prefill: prefill) {
        case .logits(let output):
            return output.logits[0, -1].asType(.float32)
        case .tokens(let rest):
            return model(rest[text: .newAxis], cache: cache.layers, state: state).logits[0, -1].asType(.float32)
        }
    }

    static func topFive(_ logits: MLXArray) -> [Int] {
        argSort(-logits)[..<5].asArray(Int.self)
    }

    @Test("Restored-snapshot logits equal a from-scratch two-phase prefill with the same cuts; one-phase differs only by drift",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_HYBRID_MODEL_PATH"))
    func restoredSnapshotIsBitwiseTwoPhase() async throws {
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: try #require(Self.modelPath))
        let container = try #require(await engine.container)
        let makeInput: @Sendable (String) -> UserInput = { UserInput(chat: [.user($0)]) }
        let first = Self.system + "\n\n[User]: Summarise module 0."
        let second = first + "\n\n[Assistant]: Calling the tool.\n\n[User]: Tool result: build succeeded in 3.2 s, 0 warnings. Next step?"
        let tokens1 = try await container.prepare(input: makeInput(first)).text.tokens
        let tokens2 = try await container.prepare(input: makeInput(second)).text.tokens
        let suffix = await HybridPrefixSnapshot.templateSuffixLength(container, makeInput)
        let b1 = tokens1.dim(-1) - suffix
        let b2 = tokens2.dim(-1) - suffix

        let report = try await container.perform(nonSendable: (tokens1, tokens2)) { context, pair in
            let (tokens1, tokens2) = pair
            let model = context.model
            let prefill = GenerateParameters().prefill
            let ids1 = tokens1.asArray(Int.self)
            let ids2 = tokens2.asArray(Int.self)

            let turn1 = try HybridPrefixSnapshot.prefillHead(
                tokens1, into: PromptCache(try model.newCache(parameters: nil)), reused: 0, boundary: b1,
                model: model, prefill: prefill)
            var tree = PromptCacheTree(byteBudget: 1 << 34)
            tree.store("shared", tokens: Array(ids1.prefix(b1)), cache: try #require(turn1.kept))
            let leased = tree.lease("shared", for: ids2)
            let lease = try #require(leased)
            let reused = lease.reusedTokens
            let turn2 = try HybridPrefixSnapshot.prefillHead(
                tokens2, into: lease.cache, reused: reused, boundary: b2, model: model, prefill: prefill)
            let a = try Self.lastLogits(tokens2, from: b2, cache: lease.cache, state: turn2.state, model: model)

            let one = PromptCache(try model.newCache(parameters: nil))
            let b = try Self.lastLogits(tokens2, from: 0, cache: one, state: nil, model: model)

            let two = PromptCache(try model.newCache(parameters: nil))
            let cut1 = try HybridPrefixSnapshot.prefillHead(
                tokens2, into: two, reused: 0, boundary: b1, model: model, prefill: prefill)
            two.state = cut1.state
            let cut2 = try HybridPrefixSnapshot.prefillHead(
                tokens2, into: two, reused: b1, boundary: b2, model: model, prefill: prefill)
            let c = try Self.lastLogits(tokens2, from: b2, cache: two, state: cut2.state, model: model)
            let d = try Self.lastLogits(
                tokens2, from: 0, cache: PromptCache(try model.newCache(parameters: nil)), state: nil, model: model,
                prefill: PrefillParameters(stepSize: 256))
            eval(a, b, c, d)
            tree.release(lease)
            return (
                reused: reused,
                ac: abs(a - c).max().item(Float.self),
                ab: abs(a - b).max().item(Float.self),
                abMean: abs(a - b).mean().item(Float.self),
                bd: abs(b - d).max().item(Float.self),
                scale: abs(b).max().item(Float.self),
                top: (Self.topFive(a), Self.topFive(b), Self.topFive(c)))
        }
        print("HybridPrefixDrift b1=\(b1) b2=\(b2) n2=\(tokens2.dim(-1)) reused=\(report.reused) |a-c|max=\(report.ac) |a-b|max=\(report.ab) |a-b|mean=\(report.abMean) |b-b256|max=\(report.bd) |b|max=\(report.scale) top5 a=\(report.top.0) b=\(report.top.1) c=\(report.top.2)")
        #expect(report.reused == b1)
        #expect(report.ac == 0)
        #expect(report.top.0 == report.top.2)
        #expect(report.top.0.first == report.top.1.first)
    }
}
