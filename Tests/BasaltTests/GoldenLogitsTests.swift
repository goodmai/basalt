import Foundation
import Darwin
import Testing
@testable import BasaltCore
@testable import BasaltInference

struct GoldenCandidate: Codable, Equatable {
    let id: Int
    let logprob: Double

    init(id: Int, logprob: Double) {
        self.id = id
        self.logprob = logprob
    }

    init(from decoder: Decoder) throws {
        var pair = try decoder.unkeyedContainer()
        id = try pair.decode(Int.self)
        logprob = try pair.decode(Double.self)
    }

    func encode(to encoder: Encoder) throws {
        var pair = encoder.unkeyedContainer()
        try pair.encode(id)
        try pair.encode(logprob)
    }
}

struct GoldenTrace: Codable, Equatable {
    var tokens: [Int]
    var steps: [[GoldenCandidate]]

    init(probed: some Collection<LogitProbe.Step>) {
        tokens = probed.map(\.token)
        steps = probed.map { step in
            zip(step.ids, step.logprobs).map { GoldenCandidate(id: $0, logprob: (Double($1) * 10_000).rounded() / 10_000) }
        }
    }
}

struct GoldenVerdict: CustomStringConvertible {
    var maxKL = 0.0
    var failure: String?

    mutating func fail(_ reason: String) {
        if failure == nil { failure = reason }
    }

    var description: String { "max KL \(maxKL)" + (failure.map { ", first failure: \($0)" } ?? "") }
}

struct GoldenFixture: Codable, Equatable {
    static let defaultTolerance = 1e-3
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/golden")

    var revision: String
    var tolerance: Double
    var prompts: [String: GoldenTrace]

    static func url(repo: String) -> URL {
        directory.appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--") + ".json")
    }

    static func load(_ url: URL) throws -> GoldenFixture {
        try JSONDecoder().decode(GoldenFixture.self, from: Data(contentsOf: url))
    }

    func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    func write(to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try encoded().write(to: url, options: .atomic)
    }

    static func divergence(reference: [GoldenCandidate], new: [GoldenCandidate]) -> Double {
        let found = Dictionary(new.map { ($0.id, $0.logprob) }, uniquingKeysWith: max)
        let floor = (new.map(\.logprob).min() ?? 0) - log(1000)
        let p = normalized(reference.map(\.logprob))
        let q = normalized(reference.map { found[$0.id] ?? floor })
        return zip(p, q).reduce(0) { $0 + exp($1.0) * ($1.0 - $1.1) }
    }

    static func normalized(_ logprobs: [Double]) -> [Double] {
        let peak = logprobs.max() ?? 0
        let total = peak + log(logprobs.reduce(0) { $0 + exp($1 - peak) })
        return logprobs.map { $0 - total }
    }

    func compare(_ fresh: [String: GoldenTrace]) -> GoldenVerdict {
        var verdict = GoldenVerdict()
        for (name, expected) in prompts.sorted(by: { $0.key < $1.key }) {
            guard let actual = fresh[name] else {
                verdict.fail("\(name): not run")
                continue
            }
            if actual.tokens != expected.tokens {
                let step = zip(expected.tokens, actual.tokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                    ?? min(expected.tokens.count, actual.tokens.count)
                verdict.fail("\(name) step \(step): greedy token \(actual.tokens.dropFirst(step).prefix(1)) instead of \(expected.tokens.dropFirst(step).prefix(1))")
            }
            for (step, pair) in zip(expected.steps, actual.steps).enumerated() {
                let kl = Self.divergence(reference: pair.0, new: pair.1)
                verdict.maxKL = max(verdict.maxKL, kl)
                if kl > tolerance { verdict.fail("\(name) step \(step): KL \(kl) > \(tolerance)") }
            }
        }
        return verdict
    }
}

struct GoldenPrompt: Sendable {
    let name: String
    var tools: [ToolDefinition]? = nil
    let messages: [ChatMessageInput]
    var opening: [ChatMessageInput]? = nil

    func request(_ messages: [ChatMessageInput], maxTokens: Int, run: Int, probe: LogitProbe?) -> GenerationRequest {
        var request = GenerationRequest(prompt: "", maxTokens: maxTokens, temperature: 0, tools: tools, messages: messages)
            .withCacheKey(opening == nil ? nil : "golden-\(name)-\(run)")
        request.logitProbe = probe
        return request
    }

    static let readFile = ToolDefinition(
        name: "read_file", description: "Read a file of the repository",
        parametersJSON: #"{"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}"#)

    static let weather = ToolDefinition(
        name: "get_weather", description: "Get the weather for a city",
        parametersJSON: #"{"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}"#)

    static func user(_ text: String) -> ChatMessageInput { ChatMessageInput(role: .user, text: text) }

    static func agentTrace(turns: Int) -> [ChatMessageInput] {
        let opening = [
            ChatMessageInput(role: .system, text: "You are a careful coding agent. Read files with read_file before you answer."),
            user("Find where the cache budget is computed and summarise it."),
        ]
        let calls = (0..<turns).flatMap { turn in
            let body = (0..<8).map { line in "let value\(turn)_\(line) = budget(\(turn), \(line)) * \(turn + line)" }
            return [
                ChatMessageInput(role: .assistant, text: "", toolCalls: [
                    ToolCallOutput(id: "call_\(turn)", name: "read_file", arguments: #"{"path": "Sources/Module\#(turn)/File\#(turn).swift"}"#),
                ]),
                ChatMessageInput(role: .tool, text: body.joined(separator: "\n"), toolCallId: "call_\(turn)", toolName: "read_file"),
            ]
        }
        return opening + calls + [user("Which file computes the budget? Answer in one sentence.")]
    }

    static let trace = GoldenPrompt(name: "agent-trace-8k", tools: [readFile], messages: agentTrace(turns: 52))

    static let resumeOpening = [
        ChatMessageInput(role: .system, text: "You are a concise assistant for an inference server team."),
        user("Summarise what a prompt cache does."),
    ]

    static let all = [
        GoldenPrompt(name: "chat-capital", messages: [user("What is the capital of Australia? Answer in one sentence.")]),
        GoldenPrompt(name: "chat-sky", messages: [user("Explain in two sentences why the sky is blue.")]),
        GoldenPrompt(name: "chat-commits", messages: [
            ChatMessageInput(role: .system, text: "You are a concise assistant."), user("Give me three tips for writing clear commit messages."),
        ]),
        GoldenPrompt(name: "code-fibonacci", messages: [user("Write a Swift function that returns the n-th Fibonacci number iteratively.")]),
        trace,
        GoldenPrompt(name: "tool-call-json", tools: [weather], messages: [user("What is the weather in Paris in celsius? Use the tool.")]),
        GoldenPrompt(name: "chat-russian", messages: [user("Объясни коротко, что такое фотосинтез и зачем он растениям.")]),
        GoldenPrompt(
            name: "hybrid-anchor-resume",
            messages: resumeOpening + [
                ChatMessageInput(role: .assistant, text: "It keeps the KV state of a prompt prefix so a later request skips recomputing it."),
                user("Now explain what happens when the prefix changes in the middle."),
            ],
            opening: resumeOpening),
    ]
}

enum GoldenRun {
    static let maxTokens = 32
    static let topK = 64

    static func traces(_ engine: MLXInferenceEngine, run: Int, log: ModelRunLog) async throws -> [String: GoldenTrace] {
        var traces: [String: GoldenTrace] = [:]
        for prompt in GoldenPrompt.all {
            traces[prompt.name] = try await log.step("\(prompt.name) run \(run)") {
                try await trace(engine, prompt, run: run, log: log)
            }
        }
        return traces
    }

    static func trace(_ engine: MLXInferenceEngine, _ prompt: GoldenPrompt, run: Int, log: ModelRunLog) async throws -> GoldenTrace {
        if let opening = prompt.opening {
            _ = try await engine.generate(request: prompt.request(opening, maxTokens: maxTokens, run: run, probe: nil))
        }
        let probe = LogitProbe(topK: topK)
        let response = try await engine.generate(request: prompt.request(prompt.messages, maxTokens: maxTokens, run: run, probe: probe))
        log.record("generate", "\(prompt.name) prompt \(response.promptTokens) cached \(response.cachedPromptTokens) completion \(response.completionTokens) probed \(probe.steps.count)")
        return GoldenTrace(probed: probe.steps.prefix(response.completionTokens))
    }
}

enum MachMemory {
    static func swapOuts() -> UInt64 {
        var info = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        return result == KERN_SUCCESS ? info.swapouts : 0
    }

    static func footprint() -> UInt64 { taskInfo()?.phys_footprint ?? 0 }

    static func compressed() -> UInt64 { taskInfo()?.compressed ?? 0 }

    static func taskInfo() -> task_vm_info_data_t? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? info : nil
    }
}

@Suite("Golden logit gate helpers: KL over the reference top-k, fixture round trip and probe recording (#291)")
struct GoldenLogitsTests {
    static func candidates(_ pairs: [(Int, Double)]) -> [GoldenCandidate] { pairs.map { GoldenCandidate(id: $0.0, logprob: $0.1) } }

    static let steps = (0..<4).map { _ in LogitProbe.Step(token: 1, ids: [1, 2], logprobs: [-0.1, -2.4]) }

    @Test("KL is zero for the same distribution and for one shifted by a constant")
    func klIgnoresShift() {
        let reference = Self.candidates([(1, log(0.5)), (2, log(0.3)), (3, log(0.2))])
        #expect(GoldenFixture.divergence(reference: reference, new: reference) == 0)
        let shifted = reference.map { GoldenCandidate(id: $0.id, logprob: $0.logprob - 2) }
        #expect(abs(GoldenFixture.divergence(reference: reference, new: shifted)) < 1e-12)
    }

    @Test("A reference id missing from the new top-k gets the new smallest logprob minus ln 1000, then both renormalize over the reference ids")
    func missingIdGetsFloor() {
        let reference = Self.candidates([(1, log(0.5)), (2, log(0.3)), (3, log(0.2))])
        let new = Self.candidates([(2, log(0.4)), (1, log(0.6)), (9, log(0.001))])
        let q = [0.6, 0.4, 1e-6].map { $0 / 1.000001 }
        let expected = zip([0.5, 0.3, 0.2], q).reduce(0.0) { $0 + $1.0 * log($1.0 / $1.1) }
        #expect(abs(GoldenFixture.divergence(reference: reference, new: new) - expected) < 1e-9)
    }

    @Test("A fixture survives a JSON round trip, rounds logprobs to 1e-4 and stores each candidate as an [id, logprob] pair")
    func fixtureRoundTrip() throws {
        let trace = GoldenTrace(probed: [LogitProbe.Step(token: 7, ids: [7, 3], logprobs: [-0.123456, -2.5])])
        let fixture = GoldenFixture(revision: "abc", tolerance: 1e-3, prompts: ["chat": trace])
        let data = try fixture.encoded()
        #expect(String(decoding: data, as: UTF8.self).contains("[[7,-0.1235],[3,-2.5]]"))
        #expect(try JSONDecoder().decode(GoldenFixture.self, from: data) == fixture)
    }

    @Test("The gate passes an identical run and names the first prompt and step whose KL or greedy token breaks the fixture")
    func compareNamesFirstFailure() throws {
        let reference = GoldenFixture(
            revision: "r", tolerance: 1e-3, prompts: ["a": GoldenTrace(probed: Self.steps), "b": GoldenTrace(probed: Self.steps)])
        #expect(reference.compare(reference.prompts).failure == nil)
        var drifted = Self.steps
        drifted[2].logprobs = [-0.9, -0.6]
        let shifted = reference.compare(["a": try #require(reference.prompts["a"]), "b": GoldenTrace(probed: drifted)])
        #expect(shifted.failure?.hasPrefix("b step 2: KL") == true, "\(shifted)")
        #expect(shifted.maxKL > 1e-3)
        drifted[1].token = 2
        let diverged = reference.compare(["a": try #require(reference.prompts["a"]), "b": GoldenTrace(probed: drifted)])
        #expect(diverged.failure?.hasPrefix("b step 1: greedy token") == true, "\(diverged)")
        #expect(reference.compare(["a": try #require(reference.prompts["a"])]).failure == "b: not run")
    }

    @Test("Through the engine on a tiny Qwen3 the probe's top-1 is the sampled greedy token at every step, it records one row per step plus the iterator's one-token lookahead, and the output is unchanged")
    func probeRecordsGreedySteps() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let request = GenerationRequest(prompt: "golden probe", maxTokens: 12, temperature: 0)
        let plain = try await engine.generate(request: request)
        let probe = LogitProbe(topK: 64)
        var probed = request
        probed.logitProbe = probe
        let response = try await engine.generate(request: probed)
        let steps = probe.steps
        #expect(response.generatedText == plain.generatedText)
        #expect(response.completionTokens == plain.completionTokens)
        #expect(steps.count == response.completionTokens + 1)
        #expect(steps.allSatisfy { $0.ids.count == 64 && $0.logprobs.count == 64 && $0.ids.first == $0.token })
        #expect(steps.allSatisfy { zip($0.logprobs, $0.logprobs.dropFirst()).allSatisfy { $0 >= $1 } && ($0.logprobs.first ?? 1) <= 0 })
    }
}

@Suite("Real-model regression gate: golden top-k logit signatures and the zero swap-out invariant (#291)", .serialized)
struct GoldenLogitsRealModelTests {
    static let env = ProcessInfo.processInfo.environment

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_GOLDEN_MODELS"] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    static func path(_ entry: String) -> String {
        entry.hasPrefix("/") ? entry : ModelCache.cacheDir(for: entry).path
    }

    @Test("Greedy tokens equal the fixture and every step's KL over the reference top-64 stays within the fixture's tolerance",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_GOLDEN_MODELS"), arguments: models)
    func goldenLogits(entry: String) async throws {
        let path = Self.path(entry)
        let log = try await ModelRunLog("golden-logits", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let revision = URL(fileURLWithPath: path).resolvingSymlinksInPath().lastPathComponent
        let url = GoldenFixture.url(repo: log.model)
        let traces = try await GoldenRun.traces(engine, run: 0, log: log)
        if Self.env["BASALT_GOLDEN_NOISE"] == "1" {
            let again = try await GoldenRun.traces(engine, run: 1, log: log)
            log.record("result", "GOLDEN-NOISE \(GoldenFixture(revision: revision, tolerance: .infinity, prompts: traces).compare(again))")
            return
        }
        if Self.env["BASALT_GOLDEN_UPDATE"] == "1" {
            let tolerance = Self.env["BASALT_GOLDEN_TOLERANCE"].flatMap(Double.init)
                ?? (try? GoldenFixture.load(url).tolerance) ?? GoldenFixture.defaultTolerance
            try GoldenFixture(revision: revision, tolerance: tolerance, prompts: traces).write(to: url)
            log.record("result", "GOLDEN-UPDATE \(url.path) tolerance \(tolerance)")
            return
        }
        let reference = try GoldenFixture.load(url)
        let verdict = reference.compare(traces)
        log.record("result", "GOLDEN \(verdict) tolerance \(reference.tolerance)")
        #expect(reference.revision == revision, "\(log.model): fixture recorded on snapshot \(reference.revision), running \(revision)")
        #expect(verdict.failure == nil, "\(log.model): \(verdict)")
    }

    @Test("An 8k-token generation of 256 tokens peaks within the basalt fit estimate; system swap-outs and basalt pages compressed are reported, not asserted",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_GOLDEN_MODELS"), arguments: models)
    func memoryFootprint(entry: String) async throws {
        let path = Self.path(entry)
        let log = try await ModelRunLog("golden-swap", modelPath: path)
        defer { log.releaseGPU() }
        let available = SystemProfiler.availableSystemMemory()
        let engine = MLXInferenceEngine()
        try await log.step("load") { try await engine.load(modelPath: path) }
        let prompt = GoldenPrompt.trace
        let swapOuts = MachMemory.swapOuts()
        let compressed = MachMemory.compressed()
        var peak = MachMemory.footprint()
        var metadata: GenerationResponse?
        var chunks = 0
        for await chunk in try await engine.generateStream(request: prompt.request(prompt.messages, maxTokens: 256, run: 0, probe: nil)) {
            chunks += 1
            if chunks % 32 == 0 { peak = max(peak, MachMemory.footprint()) }
            if case .metadata(let response) = chunk { metadata = response }
        }
        peak = max(peak, MachMemory.footprint())
        let swapped = Int64(MachMemory.swapOuts()) - Int64(swapOuts)
        let pushedOut = Int64(MachMemory.compressed()) - Int64(compressed)
        let response = try #require(metadata)
        let fit = await MemoryEvaluator().evaluate(modelId: log.model, contextLength: response.promptTokens + 256)
        log.record("result", "SWAP prompt \(response.promptTokens) completion \(response.completionTokens) system swap-outs \(swapped) basalt pages compressed or swapped \(formatRAMBytes(max(0, pushedOut))) peak \(formatRAMBytes(Int64(peak))) fit \(fit.requiredRAMFormatted) available before load \(formatRAMBytes(available)) fits \(fit.fitsInMemory)")
        try #require(fit.fitsInMemory, "\(log.model): basalt models check says it does not fit, so the invariant does not apply")
        #expect(Int64(peak) <= fit.requiredRAMBytes, "\(log.model): peak footprint \(formatRAMBytes(Int64(peak))) above the fit estimate \(fit.requiredRAMFormatted)")
    }
}
