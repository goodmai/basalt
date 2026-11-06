import Foundation
import Synchronization
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Head split with zoned k8v4 retrieval heads on real models: KV bytes, passkey, multikey and multihop, perplexity, decode speed (#294)", .serialized)
struct HeadSplitZonedRealModelTests {
    typealias Mode = ZonedKVCacheBenchmarkTests.Mode
    typealias Fact = (depth: Double, text: String)

    struct Generated: Sendable {
        let tokens: [Int]
        let stateBytes: Int
        let heldBytes: Int
        let decodeTPS: Double
    }

    struct StepFailure: Error, CustomStringConvertible {
        let description: String
    }

    static let env = ProcessInfo.processInfo.environment
    static let defaultModes: [Mode] = [.fp16, .zonedK8V4, .headSplit25, .headSplit50, .headSplit25K8V4, .headSplit50K8V4]

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return ["QWEN38", "LFM25", "GEMMA4"].compactMap { env["BASALT_E2E_\($0)_MODEL_PATH"] }
    }

    static func list(_ key: String, _ fallback: String) -> [String] { ZonedKVCacheBenchmarkTests.list(key, default: fallback) }

    static var modes: [Mode] {
        let picked = list("BASALT_HSQ_MODES", defaultModes.map(\.description).joined(separator: ","))
        return Mode.allCases.filter { picked.contains($0.description) }
    }

    static func estimate(_ geometry: KVCacheGeometry, tokens: Int, mode: Mode) -> Int64 {
        let count = TokenCount(tokens)
        let dense = geometry.bytes(tokens: count, kvPrecision: mode.precision, zones: mode.zones).rawValue
        guard let ratio = mode.headRatio else { return dense }
        let other = geometry.bytes(tokens: count).rawValue - Int64(Double(tokens) * geometry.bytesPerToken(nil))
        let streamedTokens = min(tokens, HeadSplitKVCache.sinkTokens + HeadSplitKVCache.windowTokens + HeadSplitKVCache.trimStep)
        let streamed = Double(streamedTokens) * geometry.bytesPerToken(nil)
        return other + Int64(ratio * Double(dense - other) + (1 - ratio) * streamed)
    }

    static func generate(_ mode: Mode, _ tokens: [Int], maxTokens: Int, temperature: Float, _ context: ModelContext) throws -> Generated {
        Memory.clearCache()
        let before = Memory.activeMemory
        let cache = try ZonedKVCacheBenchmarkTests.cache(mode, context)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(tokens)), model: context.model, cache: cache,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: temperature))
        guard let first = iterator.next() else { throw StepFailure(description: "no first token") }
        var out = [first]
        Stream.gpu.synchronize()
        let started = Date()
        while let token = iterator.next() { out.append(token) }
        Stream.gpu.synchronize()
        let seconds = Date().timeIntervalSince(started)
        Memory.clearCache()
        let held = Memory.activeMemory - before
        let state = cache.reduce(0) { total, layer in total + layer.state.reduce(0) { $0 + $1.nbytes } }
        return Generated(tokens: out, stateBytes: state, heldBytes: held, decodeTPS: Double(out.count - 1) / seconds)
    }

    static func prompt(_ facts: [Fact], question: String, length: Int, filler: [Int], tokenizer: any MLXLMCommon.Tokenizer) throws -> [Int] {
        let inserts = facts.map { (at: Int(Double(length) * $0.depth), tokens: tokenizer.encode(text: "\n\($0.text)\n")) }
        let asked = tokenizer.encode(text: "\n\(question)")
        let end = length - inserts.map(\.tokens.count).reduce(0, +) - asked.count
        guard filler.count >= end else { throw StepFailure(description: "corpus has \(filler.count) tokens, \(end) needed") }
        var body = [Int](), start = 0
        for insert in inserts.sorted(by: { $0.at < $1.at }) {
            body += filler[start ..< insert.at] + insert.tokens
            start = insert.at
        }
        return body + filler[start ..< end] + asked
    }

    @Test("KV bytes against the geometry, decode speed at ~30k, perplexity, passkey at 16k and 32k and multikey/multihop for dense, zones, head split and head split with zoned k8v4",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_QWEN38_MODEL_PATH, BASALT_E2E_LFM25_MODEL_PATH or BASALT_E2E_GEMMA4_MODEL_PATH"),
          arguments: models)
    func measure(path: String) async throws {
        let log = try await ModelRunLog("head-split-zoned", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(gpuCacheLimit: 0, kvPrecision: .k8v4, kvZones: .standard, kvHeadSplit: 0.5)
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let geometry = try #require(KVCacheGeometry.load(modelDirectory: URL(fileURLWithPath: path)))
        let (zones, mask) = await (engine.supportsKVZones, engine.kvHeadMask)
        log.record("serve", "--kv-zones applies \(zones), --kv-head-split 0.5 applies \(mask != nil)")
        guard zones else {
            log.record("refused", "attention does not go through the cache, so serve keeps head split and zones off; only fp16 is measured")
            await run(Runner(log: log, container: container, geometry: geometry, modes: [.fp16]))
            return
        }
        let scores = try await KVHeadClassifier.loadOrScore(directory: URL(fileURLWithPath: path), container: container)
        let key = await container.perform { ObjectIdentifier($0.model as AnyObject) }
        ZonedKVCacheBenchmarkTests.headScores.withLock { $0[key] = scores }
        log.record("heads", "concentrated \(KVHeadMask.isConcentrated(scores)) (serve refuses the split when false; the split modes are measured anyway)")
        await run(Runner(log: log, container: container, geometry: geometry, modes: Self.modes))
    }

    func run(_ runner: Runner) async {
        let steps = Set(Self.list("BASALT_HSQ_STEPS", "mem,ppl,passkey,tasks"))
        if steps.contains("mem") { await runner.memory() }
        let corpusPath = Self.env["BASALT_ZONED_CORPUS"] ?? ""
        if let corpus = try? String(contentsOfFile: corpusPath, encoding: .utf8) {
            if steps.contains("ppl") { await runner.perplexity(corpus) }
            if steps.contains("passkey") { await runner.passkey(corpus) }
            if steps.contains("tasks") { await runner.tasks(corpus) }
        } else if !steps.isDisjoint(with: ["ppl", "passkey", "tasks"]) {
            runner.fail("corpus", "BASALT_ZONED_CORPUS is not a readable text file: \(corpusPath)")
        }
        #expect(runner.failures.withLock { $0 } == 0, "see \(runner.log.url.path)")
    }
}

extension HeadSplitZonedRealModelTests {
    struct Probe: Sendable {
        let facts: [Fact]
        let question: String
        let expected: String
        let maxTokens: Int
    }

    final class Runner: Sendable {
        let log: ModelRunLog
        let container: ModelContainer
        let geometry: KVCacheGeometry
        let modes: [Mode]
        let failures = Mutex(0)
        let room = PromptCacheBudget.gpuWorkingSet().rawValue - Int64(Memory.activeMemory)

        init(log: ModelRunLog, container: ModelContainer, geometry: KVCacheGeometry, modes: [Mode]) {
            (self.log, self.container, self.geometry, self.modes) = (log, container, geometry, modes)
        }

        var lengths: [Int] { HeadSplitZonedRealModelTests.list("BASALT_HSQ_LENGTHS", "16000,32000").compactMap { Int($0) } }

        func fail(_ step: String, _ detail: String) {
            failures.withLock { $0 += 1 }
            log.record("FAIL \(step)", detail)
        }

        func step(_ name: String, _ body: () async throws -> String) async {
            do {
                log.record(name, try await log.step(name, body))
            } catch {
                failures.withLock { $0 += 1 }
            }
        }

        func fits(_ length: Int, _ step: String) -> Bool {
            let dense = geometry.bytes(tokens: TokenCount(length)).rawValue
            guard dense * 2 > room else { return true }
            log.record("SKIP \(step) length \(length)", "fp16 KV \(dense) B twice exceeds the \(room) B left after the weights")
            return false
        }

        func memory() async {
            let length = Int(HeadSplitZonedRealModelTests.env["BASALT_HSQ_PROMPT"] ?? "") ?? 30_000
            guard fits(length, "mem") else { return }
            let prompt = (0 ..< length).map { 1_000 + $0 * 7_919 % 20_000 }
            for mode in modes {
                await step("mem \(mode) prompt \(length) generated 128 temperature 0.7") {
                    let made = try await container.perform { context in
                        try HeadSplitZonedRealModelTests.generate(mode, prompt, maxTokens: 128, temperature: 0.7, context)
                    }
                    let estimate = HeadSplitZonedRealModelTests.estimate(geometry, tokens: length + made.tokens.count, mode: mode)
                    let ratio = String(format: "%.4f", Double(made.stateBytes) / Double(estimate))
                    return "kvStateBytes \(made.stateBytes) geometry \(estimate) ratio \(ratio) held \(made.heldBytes) decodeTPS \(String(format: "%.1f", made.decodeTPS)) \(log.host())"
                }
            }
        }

        func perplexity(_ corpus: String) async {
            let total = Int(HeadSplitZonedRealModelTests.env["BASALT_HSQ_PPL_TOKENS"] ?? "") ?? 16_384
            for mode in modes {
                await step("ppl \(mode) context \(total / 2) scored \(total - total / 2)") {
                    let ppl = try await container.perform { context in
                        let tokens = Array(context.tokenizer.encode(text: corpus).prefix(total + 1))
                        guard tokens.count == total + 1 else { throw StepFailure(description: "corpus has \(tokens.count) tokens") }
                        return try ZonedKVCacheBenchmarkTests.perplexity(mode, tokens: tokens, context: context)
                    }
                    guard ppl.isFinite else { throw StepFailure(description: "perplexity \(ppl)") }
                    return "ppl \(String(format: "%.4f", ppl))"
                }
            }
        }

        func passkey(_ corpus: String) async {
            let depths = HeadSplitZonedRealModelTests.list("BASALT_HSQ_DEPTHS", "0.1,0.3,0.5,0.7,0.9").compactMap { Double($0) }
            for length in lengths where fits(length, "passkey") {
                for depth in depths {
                    let key = 10_000 + Int(depth * 1_000) * 37 + length % 997
                    let probe = Probe(
                        facts: [(depth, "The pass key is \(key). Remember it. \(key) is the pass key.")],
                        question: "What is the pass key? The pass key is", expected: String(key), maxTokens: 8)
                    for mode in modes { await answer("passkey \(mode) length \(length) depth \(depth)", probe, length: length, mode: mode, corpus: corpus) }
                }
            }
        }

        func tasks(_ corpus: String) async {
            let trials = Int(HeadSplitZonedRealModelTests.env["BASALT_HSQ_TRIALS"] ?? "") ?? 3
            for length in lengths where fits(length, "tasks") {
                for name in HeadSplitZonedRealModelTests.list("BASALT_HSQ_TASKS", "multikey,multihop") {
                    for trial in 0 ..< trials {
                        let task = ZonedKVCacheBenchmarkTests.task(name, trial: trial)
                        let probe = Probe(facts: task.facts, question: task.question, expected: task.expected, maxTokens: name == "multikey" ? 10 : 40)
                        for mode in modes { await answer("task \(name) \(mode) length \(length) trial \(trial)", probe, length: length, mode: mode, corpus: corpus) }
                    }
                }
            }
        }

        func answer(_ name: String, _ probe: Probe, length: Int, mode: Mode, corpus: String) async {
            await step("\(name) expected \(probe.expected)") {
                let text = try await container.perform { context in
                    let body = try HeadSplitZonedRealModelTests.prompt(
                        probe.facts, question: probe.question, length: length, filler: context.tokenizer.encode(text: corpus), tokenizer: context.tokenizer)
                    let made = try HeadSplitZonedRealModelTests.generate(mode, body, maxTokens: probe.maxTokens, temperature: 0, context)
                    return context.tokenizer.decode(tokenIds: made.tokens)
                }
                return "ok \(text.contains(probe.expected)) answer \(text.debugDescription)"
            }
        }
    }
}
