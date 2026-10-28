import Foundation
import CryptoKit
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Zoned KV cache benchmark on real models", .serialized)
struct ZonedKVCacheBenchmarkTests {
    enum Mode: CustomStringConvertible, CaseIterable {
        case fp16, uniform8, uniform4, zoned8, zoned4

        var description: String {
            switch self {
            case .fp16: "fp16"
            case .uniform8: "uniform-q8"
            case .uniform4: "uniform-q4"
            case .zoned8: "zoned-q8"
            case .zoned4: "zoned-q4"
            }
        }

        var bits: Int? {
            switch self {
            case .fp16: nil
            case .uniform8, .zoned8: 8
            case .uniform4, .zoned4: 4
            }
        }

        var zones: KVZones? {
            switch self {
            case .zoned8, .zoned4: .standard
            default: nil
            }
        }
    }

    struct StepFailure: Error, CustomStringConvertible {
        let description: String
    }

    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var models: [String] { list("BASALT_E2E_ZONED_BENCH", default: "") }

    static func list(_ key: String, default fallback: String) -> [String] {
        (env[key] ?? fallback).split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    static func modes(_ key: String) -> [Mode] {
        let picked = list(key, default: Mode.allCases.map(\.description).joined(separator: ","))
        return Mode.allCases.filter { picked.contains($0.description) }
    }

    final class Log: @unchecked Sendable {
        let url: URL
        private(set) var failures = 0

        init(url: URL) {
            self.url = url
        }

        func line(_ text: String) {
            let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(text)\n"
            print(stamped, terminator: "")
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(Data(stamped.utf8))
                try? handle.close()
            } else {
                try? stamped.write(to: url, atomically: true, encoding: .utf8)
            }
        }

        func error(_ text: String) {
            failures += 1
            line("ERROR \(text)")
        }

        func step(_ name: String, _ body: () async throws -> String) async {
            line("START \(name)")
            let started = Date()
            do {
                let result = try await body()
                line("DONE \(name) \(result) seconds=\(String(format: "%.1f", Date().timeIntervalSince(started)))")
            } catch {
                self.error("\(name): \(error)")
            }
        }
    }

    static func shell(_ arguments: [String]) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        guard (try? process.run()) != nil else { return "unknown" }
        process.waitUntilExit()
        return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func hostLoad() -> (summary: String, quiet: Bool) {
        let average = shell(["sysctl", "-n", "vm.loadavg"])
        let oneMinute = average.split(separator: " ").compactMap { Double($0) }.first ?? .infinity
        let ciJob = !shell(["pgrep", "-f", "Runner.Worker"]).isEmpty
        let quiet = oneMinute <= 8 && !ciJob
        return ("loadavg=\(average) ciJob=\(ciJob) speedValid=\(quiet)", quiet)
    }

    static func incompleteSnapshot(_ directory: URL) async -> String? {
        let report = await ModelInspector().inspect(modelIdOrPath: directory.path)
        if !report.intact { return report.errors.joined(separator: "; ") }
        let blobs = directory.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("blobs")
        let partial = ((try? FileManager.default.contentsOfDirectory(atPath: blobs.path)) ?? []).filter { $0.hasSuffix(".incomplete") }
        return partial.isEmpty ? nil : "\(partial.count) .incomplete blobs, a download is still running"
    }

    static func acquire(_ lock: String, log: Log) async throws -> String {
        let owner = "\(ProcessInfo.processInfo.processIdentifier) \(UUID().uuidString)"
        var announced = false
        while (try? FileManager.default.createDirectory(atPath: lock, withIntermediateDirectories: false)) == nil {
            if !announced {
                log.line("WAIT gpu lock \(lock) held by \((try? String(contentsOfFile: lock + "/owner", encoding: .utf8)) ?? "unknown")")
                announced = true
            }
            try await Task.sleep(for: .seconds(15))
        }
        try owner.write(toFile: lock + "/owner", atomically: true, encoding: .utf8)
        log.line("LOCKED \(lock) owner=\(owner)")
        return owner
    }

    static func release(_ lock: String, owner: String, log: Log) {
        let current = try? String(contentsOfFile: lock + "/owner", encoding: .utf8)
        guard current == owner else {
            log.error("gpu lock \(lock) is no longer ours (owner \(current ?? "none")), left in place")
            return
        }
        try? FileManager.default.removeItem(atPath: lock)
        log.line("UNLOCKED \(lock) owner=\(owner)")
    }

    static func holdQuiet(_ marker: String, log: Log) async throws -> String {
        let owner = "\(ProcessInfo.processInfo.processIdentifier) #139 \(UUID().uuidString)"
        while true {
            let ciRuns = shell(["gh", "run", "list", "-R", "goodmai/basalt", "--status", "in_progress", "--json", "databaseId", "--jq", "length"])
            if ciRuns == "0", createExclusively(marker, contents: owner) {
                log.line("QUIET \(marker) owner=\(owner)")
                return owner
            }
            log.line("WAIT quiet: ciRunsInProgress=\(ciRuns) marker=\(FileManager.default.fileExists(atPath: marker))")
            try await Task.sleep(for: .seconds(30))
        }
    }

    static func createExclusively(_ path: String, contents: String) -> Bool {
        let descriptor = open(path, O_CREAT | O_EXCL | O_WRONLY, 0o644)
        guard descriptor >= 0 else { return false }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        handle.write(Data(contents.utf8))
        return true
    }

    static func releaseQuiet(_ marker: String, owner: String, log: Log) {
        guard (try? String(contentsOfFile: marker, encoding: .utf8)) == owner else {
            log.error("quiet marker \(marker) is no longer ours, left in place")
            return
        }
        try? FileManager.default.removeItem(atPath: marker)
        log.line("UNQUIET \(marker)")
    }

    static func cache(_ mode: Mode, _ model: any LanguageModel) throws -> [any KVCache] {
        let fresh = try model.newCache(parameters: nil)
        guard let zones = mode.zones, let bits = mode.bits else { return fresh }
        return ZonedKVCache.replacingFullAttention(fresh, zones: zones, bits: bits)
    }

    static func iterator(
        _ mode: Mode, tokens: [Int], maxTokens: Int, context: ModelContext
    ) throws -> (iterator: TokenIterator, cache: [any KVCache]?) {
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        let input = LMInput(tokens: MLXArray(tokens))
        guard mode.zones == nil else {
            let cache = try Self.cache(mode, context.model)
            return (try TokenIterator(input: input, model: context.model, cache: cache, parameters: parameters), cache)
        }
        parameters.kvBits = mode.bits
        parameters.quantizedKVStart = KVCacheGeometry.defaultQuantizedKVStart
        return (try TokenIterator(input: input, model: context.model, cache: nil, parameters: parameters), nil)
    }

    @Test("KV bytes against the geometry, perplexity, passkey and speed for fp16, uniform and zoned KV on each model in BASALT_E2E_ZONED_BENCH",
          .enabled(if: !ZonedKVCacheBenchmarkTests.models.isEmpty, "set BASALT_E2E_ZONED_BENCH to a comma-separated list of cached repo ids"))
    func benchmark() async throws {
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
        try FileManager.default.createDirectory(atPath: "logs", withIntermediateDirectories: true)
        let log = Log(url: URL(fileURLWithPath: Self.env["BASALT_ZONED_LOG"] ?? "logs/zoned-kv-bench-\(stamp).log"))
        let lock = Self.env["BASALT_GPU_LOCK"] ?? "/private/tmp/claude-501/basalt-gpu.lock"
        let steps = Set(Self.list("BASALT_ZONED_STEPS", default: "mem,ppl,passkey"))
        let corpusPath = Self.env["BASALT_ZONED_CORPUS"]
        let corpus = corpusPath.flatMap { try? String(contentsOfFile: $0, encoding: .utf8) }
        log.line("RUN commit=\(Self.shell(["git", "rev-parse", "HEAD"])) dirty=\(!Self.shell(["git", "status", "--porcelain", "--untracked-files=no"]).isEmpty)")
        log.line("MACHINE \(Self.shell(["sysctl", "-n", "machdep.cpu.brand_string"])) memory=\(Self.shell(["sysctl", "-n", "hw.memsize"])) os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
        if let corpus {
            let digest = SHA256.hash(data: Data(corpus.utf8)).map { String(format: "%02x", $0) }.joined()
            log.line("CORPUS \(corpusPath ?? "") bytes=\(corpus.utf8.count) sha256=\(digest)")
        } else if steps.contains("ppl") || steps.contains("passkey") {
            log.error("BASALT_ZONED_CORPUS is not a readable text file: \(corpusPath ?? "unset")")
        }

        for repo in Self.models {
            guard let directory = ModelCache.resolve(repoId: repo) else {
                log.error("\(repo): not in the Hugging Face cache")
                continue
            }
            if let problem = await Self.incompleteSnapshot(directory) {
                log.error("\(repo): incomplete snapshot \(directory.path): \(problem)")
                continue
            }
            log.line("MODEL \(repo) revision=\(directory.lastPathComponent) path=\(directory.path)")
            let owner = try await Self.acquire(lock, log: log)
            await run(repo: repo, directory: directory, steps: steps, corpus: corpus, log: log)
            Self.release(lock, owner: owner, log: log)
        }
        log.line("END failures=\(log.failures) log=\(log.url.path)")
        #expect(log.failures == 0, "see \(log.url.path)")
    }

    func run(repo: String, directory: URL, steps: Set<String>, corpus: String?, log: Log) async {
        let engine = MLXInferenceEngine(gpuCacheLimit: 0)
        do {
            try await engine.load(modelPath: directory.path)
        } catch {
            log.error("\(repo) load: \(error)")
            return
        }
        guard let container = await engine.container,
              let geometry = KVCacheGeometry.load(modelDirectory: directory) else {
            log.error("\(repo): no container or KV geometry after load")
            return
        }
        let room = Int64(PromptCacheBudget.gpuWorkingSet()) - Int64(Memory.activeMemory)
        log.line("PROBE \(repo) attentionThroughZonedCache=\(await engine.supportsKVZones)")
        log.line("GEOMETRY \(repo) fp16BytesPerToken=\(geometry.bytes(tokens: 1_000) / 1_000) window=\(geometry.window) roomAfterWeights=\(room)")

        if steps.contains("mem") {
            let marker = Self.env["BASALT_QUIET_MARKER"] ?? "/private/tmp/claude-501/basalt-quiet"
            let quiet: String
            do {
                quiet = try await Self.holdQuiet(marker, log: log)
            } catch {
                log.error("\(repo) quiet marker: \(error)")
                return
            }
            defer { Self.releaseQuiet(marker, owner: quiet, log: log) }
            let prompt = Int(Self.env["BASALT_ZONED_PROMPT"] ?? "16380") ?? 16_380
            for mode in Self.modes("BASALT_ZONED_MODES") {
                let before = Self.hostLoad()
                await log.step("mem \(repo) \(mode) prompt=\(prompt) generated=128 before: \(before.summary)") {
                    let measured = try await container.perform { (context: ModelContext) -> (Int, Int, Double, Double) in
                        Memory.clearCache()
                        let before = Memory.activeMemory
                        Memory.peakMemory = 0
                        let tokens = (0..<prompt).map { 1_000 + $0 * 7_919 % 20_000 }
                        let started = Date()
                        let made = try Self.iterator(mode, tokens: tokens, maxTokens: 128, context: context)
                        var iterator = made.iterator
                        guard iterator.next() != nil else { throw StepFailure(description: "no first token") }
                        Stream.gpu.synchronize()
                        let prefill = Date().timeIntervalSince(started)
                        let decodeStarted = Date()
                        var generated = 1
                        while iterator.next() != nil { generated += 1 }
                        Stream.gpu.synchronize()
                        let decode = Double(generated - 1) / Date().timeIntervalSince(decodeStarted)
                        Memory.clearCache()
                        withExtendedLifetime(made.cache) {}
                        return (Memory.activeMemory - before, Memory.peakMemory - before, Double(prompt) / prefill, decode)
                    }
                    let estimate = geometry.bytes(tokens: prompt + 128, kvBits: mode.bits, zones: mode.zones)
                    let after = Self.hostLoad()
                    return "after: \(after.summary) speedValid=\(before.quiet && after.quiet) held=\(measured.0) geometry=\(estimate) ratio=\(String(format: "%.4f", Double(measured.0) / Double(estimate))) peak=\(measured.1) prefillTPS=\(String(format: "%.0f", measured.2)) decodeTPS=\(String(format: "%.1f", measured.3))"
                }
            }
        }

        guard let corpus else { return }
        if steps.contains("greedy") {
            for length in Self.list("BASALT_ZONED_GREEDY_LENGTHS", default: "1000,16000").compactMap({ Int($0) }) {
                var reference: [Int]?
                for mode in Self.modes("BASALT_ZONED_MODES") {
                    await log.step("greedy \(repo) \(mode) prompt=\(length) generated=64") {
                        let tokens = try await container.perform { (context: ModelContext) -> [Int] in
                            let prompt = Array(context.tokenizer.encode(text: corpus).prefix(length))
                            let made = try Self.iterator(mode, tokens: prompt, maxTokens: 64, context: context)
                            var iterator = made.iterator
                            var out = [Int]()
                            while let token = iterator.next() { out.append(token) }
                            withExtendedLifetime(made.cache) {}
                            return out
                        }
                        let expected = reference ?? tokens
                        if reference == nil { reference = tokens }
                        let same = zip(tokens, expected).prefix { $0 == $1 }.count
                        return "identicalToFirstMode=\(tokens == expected) sharedPrefix=\(same)/\(expected.count)"
                    }
                }
            }
        }
        if steps.contains("ppl") {
            let total = Int(Self.env["BASALT_ZONED_PPL_TOKENS"] ?? "16384") ?? 16_384
            for mode in Self.modes("BASALT_ZONED_MODES") {
                await log.step("ppl \(repo) \(mode) context=\(total / 2) scored=\(total - total / 2)") {
                    let ppl = try await container.perform { (context: ModelContext) -> Double in
                        let tokens = Array(context.tokenizer.encode(text: corpus).prefix(total + 1))
                        guard tokens.count == total + 1 else { throw StepFailure(description: "corpus has \(tokens.count) tokens") }
                        return try Self.perplexity(mode, tokens: tokens, context: context)
                    }
                    guard ppl.isFinite else { throw StepFailure(description: "perplexity \(ppl)") }
                    return "ppl=\(String(format: "%.4f", ppl))"
                }
            }
        }

        if steps.contains("passkey") {
            let lengths = Self.list("BASALT_ZONED_LENGTHS", default: "16000,32000").compactMap { Int($0) }
            let depths = Self.list("BASALT_ZONED_DEPTHS", default: "0.2,0.8").compactMap { Double($0) }
            for length in lengths {
                if geometry.bytes(tokens: length) * 2 > room {
                    log.line("SKIP passkey \(repo) length=\(length): fp16 KV \(geometry.bytes(tokens: length)) B twice exceeds the \(room) B left after the weights")
                    continue
                }
                for depth in depths {
                    for mode in Self.modes("BASALT_ZONED_PASSKEY_MODES") {
                        let key = 10_000 + Int(depth * 1_000) * 37 + length % 997
                        await log.step("passkey \(repo) \(mode) length=\(length) depth=\(depth) key=\(key)") {
                            let answer = try await container.perform { (context: ModelContext) -> String in
                                let filler = context.tokenizer.encode(text: corpus)
                                let needle = context.tokenizer.encode(text: "\nThe pass key is \(key). Remember it. \(key) is the pass key.\n")
                                let question = context.tokenizer.encode(text: "\nWhat is the pass key? The pass key is")
                                let at = Int(Double(length) * depth)
                                let end = length - needle.count - question.count
                                guard filler.count >= end else { throw StepFailure(description: "corpus has \(filler.count) tokens") }
                                let body = Array(filler[0..<at]) + needle + Array(filler[at..<end]) + question
                                let made = try Self.iterator(mode, tokens: body, maxTokens: 8, context: context)
                                var iterator = made.iterator
                                var out = [Int]()
                                while let token = iterator.next() { out.append(token) }
                                withExtendedLifetime(made.cache) {}
                                return context.tokenizer.decode(tokenIds: out)
                            }
                            return "ok=\(answer.contains(String(key))) answer=\(answer.debugDescription)"
                        }
                    }
                }
            }
        }
    }

    static func perplexity(_ mode: Mode, tokens: [Int], context: ModelContext) throws -> Double {
        let total = tokens.count - 1
        let prefix = total / 2
        let chunk = 512
        var cache = try Self.cache(mode, context.model)
        var start = 0
        while start < prefix {
            let end = min(prefix, start + chunk)
            _ = context.model(MLXArray(Array(tokens[start..<end]))[.newAxis], cache: cache)
            eval(cache)
            start = end
        }
        if mode.zones == nil, let bits = mode.bits {
            maybeQuantizeKVCache(cache: &cache, kvBits: bits, quantizedKVStart: 0)
        }
        var nll = 0.0
        while start < total {
            let end = min(total, start + chunk)
            let logits = context.model(MLXArray(Array(tokens[start..<end]))[.newAxis], cache: cache)[0].asType(.float32)
            let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
            let targets = MLXArray(Array(tokens[(start + 1)...end]))
            nll -= Double(takeAlong(logp, targets[0..., .newAxis], axis: -1).sum().item(Float.self))
            eval(cache)
            start = end
        }
        return exp(nll / Double(total - prefix))
    }
}
