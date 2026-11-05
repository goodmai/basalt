import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

struct VerifyRoundSample: Sendable {
    var build: Double
    var encode: Double
    var wait: Double
}

struct VerifyRoundRow: Sendable, CustomStringConvertible {
    let block: Int
    let compiled: Bool
    let build: Double
    let encode: Double
    let wait: Double
    let greedy: [Int]

    init(block: Int, compiled: Bool, samples: [VerifyRoundSample], greedy: [Int]) {
        func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
        self.block = block
        self.compiled = compiled
        self.build = median(samples.map(\.build))
        self.encode = median(samples.map(\.encode))
        self.wait = median(samples.map(\.wait))
        self.greedy = greedy
    }

    var round: Double { build + encode + wait }
    var hostShare: Double { (build + encode) / round }

    var description: String {
        String(
            format: "K=%d compile=%@ build %.2f ms dispatch %.2f ms gpu-wait %.2f ms round %.2f ms host %.1f%%",
            block, compiled ? "on" : "off", build * 1000, encode * 1000, wait * 1000, round * 1000, hostShare * 100)
    }
}

enum VerifyRound {
    static let warmup = 4
    static let rounds = 32

    static func sample(_ model: any LanguageModel, cache: [KVCache], fed: [Int], mtp: Bool, staged: Bool) -> (VerifyRoundSample, [Int]) {
        let recurrent = cache.compactMap { $0 as? MambaCache }
        let checkpoints = recurrent.map { (conv: $0[0], state: $0[1], offset: $0.offset) }
        let storage = staged ? KVCacheStorage(cache) : nil
        let round = storage?.beginRound(maximumPositions: fed.count)
        let layers = round?.caches ?? cache
        var state: LMOutput.State?
        if mtp {
            var emit = LMOutput.State()
            emit[mtpEmitFlagKey] = true
            emit[mtpCacheCheckpointIndexKey] = recurrent.isEmpty ? nil : fed.count - 1
            state = emit
        }
        let input = LMInput.Text(tokens: MLXArray(fed.map(Int32.init)))
        let clock = ContinuousClock()
        let start = clock.now
        let output = model(input[text: .newAxis], cache: layers, state: state)
        let greedy = argMax(output.logits[0], axis: -1)
        let hidden = output.state?[mtpLastHiddenStatesKey].map { [$0] } ?? []
        let targets = [greedy] + hidden + layers.flatMap { $0.innerState() }
        let built = clock.now
        asyncEval(targets)
        let encoded = clock.now
        let ids = greedy.asArray(Int.self)
        let done = clock.now
        if let storage, let round {
            _ = storage.rollback(round)
        } else if recurrent.isEmpty {
            _ = trimPromptCache(cache, numTokens: fed.count)
        } else {
            for layer in cache where layer.isTrimmable { layer.trim(fed.count) }
            for (layer, checkpoint) in zip(recurrent, checkpoints) {
                layer[0] = checkpoint.conv
                layer[1] = checkpoint.state
                layer.offset = checkpoint.offset
            }
        }
        func seconds(_ duration: Duration) -> Double { Double(duration.components.attoseconds) / 1e18 + Double(duration.components.seconds) }
        return (VerifyRoundSample(build: seconds(built - start), encode: seconds(encoded - built), wait: seconds(done - encoded)), ids)
    }

    static func measure(_ model: any LanguageModel, prompt: MLXArray, blocks: [Int], mtp: Bool) throws -> [VerifyRoundRow] {
        let cache = try model.newCache(parameters: nil)
        let promptCache = PromptCache(cache)
        eval(try HybridPrefixDriftTests.lastLogits(prompt, from: 0, cache: promptCache, state: nil, model: model))
        eval(cache.flatMap { $0.innerState() })
        let staged = !promptCache.isSharable && !PromptLookupTokenIterator.isReplayableHybrid(cache)
            && PromptLookupTokenIterator.stagesRounds(cache)
        let ids = prompt.asArray(Int.self)
        var rows: [VerifyRoundRow] = []
        defer { compile(enable: true) }
        for block in blocks {
            let fed = Array(ids.suffix(block))
            for compiled in [true, false] {
                compile(enable: compiled)
                for _ in 0 ..< warmup { _ = sample(model, cache: cache, fed: fed, mtp: mtp, staged: staged) }
                let runs = (0 ..< rounds).map { _ in sample(model, cache: cache, fed: fed, mtp: mtp, staged: staged) }
                rows.append(VerifyRoundRow(block: block, compiled: compiled, samples: runs.map(\.0), greedy: runs[0].1))
            }
        }
        return rows
    }
}

@Suite("Verify-round host share on a real model: graph build and dispatch against GPU wait, MLX compile on and off (#210)", .serialized)
struct VerifyRoundOverheadTests {
    static let env = ProcessInfo.processInfo.environment
    static let gate = 0.15

    static var modelPath: String? {
        env["BASALT_E2E"] == "1" ? env["BASALT_E2E_VERIFY_MODEL_PATH"] : nil
    }

    static var mtp: Bool { env["BASALT_E2E_VERIFY_MTP"] == "1" }

    static var blocks: [Int] {
        (env["BASALT_E2E_VERIFY_BLOCKS"] ?? (mtp ? "1,2,3,4" : "1,3,5,9,17")).split(separator: ",").compactMap { Int($0) }
    }

    static let promptURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appending(path: "benchmarks/speculative-lookup/code.txt")

    @Test("Each verify block splits into graph build, dispatch and GPU wait with compile on and off, the greedy rows match across the two, and the log states whether host time reaches the 15% gate for compiled verify graphs",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_VERIFY_MODEL_PATH (BASALT_E2E_VERIFY_MTP=1 for an MTP target)"))
    func splitsVerifyRounds() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("verify-overhead", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine()
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let text = try String(contentsOf: Self.promptURL, encoding: .utf8)
        let prompt = try await container.prepare(input: UserInput(chat: [.user(text)])).text.tokens
        let (blocks, mtp, mode) = (Self.blocks, Self.mtp, Self.mtp ? "mtp" : "lookup")
        let rows = try await log.step("verify-rounds \(mode) prompt \(prompt.dim(-1)) blocks \(blocks)") {
            try await container.perform(nonSendable: prompt) { context, prompt in
                try VerifyRound.measure(context.model, prompt: prompt, blocks: blocks, mtp: mtp)
            }
        }
        for row in rows { log.record(mode, row.description) }
        let current = rows.filter(\.compiled)
        for row in current {
            let plain = try #require(rows.first { !$0.compiled && $0.block == row.block })
            log.record("\(mode) K=\(row.block)", String(format: "compile saves %.1f%% of the round, greedy rows %@", (1 - row.round / plain.round) * 100, row.greedy == plain.greedy ? "identical" : "differ"))
            #expect(row.greedy == plain.greedy, "K=\(row.block)")
        }
        let peak = try #require(current.max { $0.hostShare < $1.hostShare })
        log.record("gate", String(format: "host share peaks at %.1f%% (K=%d); %@", peak.hostShare * 100, peak.block,
                                  peak.hostShare >= Self.gate ? "at or above 15%: prototype compiled verify graphs" : "below 15%: compiled verify graphs not needed"))
    }
}
