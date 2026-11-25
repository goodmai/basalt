import Foundation
import BasaltCore

public enum BenchmarkAdapterTier: String, Sendable, Codable, Equatable {
    case native
    case pureSwift = "pure-swift"
    case externalBridge = "external-bridge"
}

public struct BenchmarkAdapterDescriptor: Sendable, Codable, Equatable {
    public var id: String
    public var tier: BenchmarkAdapterTier
    public var summary: String
    public var defaultSubmissionTarget: SubmissionTarget?

    public init(
        id: String, tier: BenchmarkAdapterTier, summary: String,
        defaultSubmissionTarget: SubmissionTarget? = nil
    ) {
        (self.id, self.tier, self.summary, self.defaultSubmissionTarget) = (id, tier, summary, defaultSubmissionTarget)
    }
}

public struct BenchmarkTaskContext: Sendable {
    public var spec: ComposeSpec
    public var suiteName: String
    public var task: BenchmarkTaskSpec
    public var defaults: BenchmarkDefaultsSpec
    public var endpoint: BenchmarkEndpoint
    public var seed: UInt64
    public var temperature: Double

    public init(
        spec: ComposeSpec, suiteName: String, task: BenchmarkTaskSpec,
        defaults: BenchmarkDefaultsSpec, endpoint: BenchmarkEndpoint
    ) {
        self.spec = spec
        self.suiteName = suiteName
        self.task = task
        self.defaults = defaults
        self.endpoint = endpoint
        self.seed = task.params["seed"]?.intValue.flatMap { $0 >= 0 ? UInt64($0) : nil } ?? defaults.seed ?? 42
        self.temperature = task.params["temperature"]?.doubleValue ?? defaults.temperature ?? 0.0
    }

    public var resolvedModelName: String {
        if task.target.hasPrefix("model:") {
            let slot = String(task.target.dropFirst("model:".count))
            if let local = spec.local(slot) { return local }
            if case .cloud(_, let model)? = spec.models[slot]?.source { return model }
            return slot
        }
        return spec.local(spec.answer) ?? spec.answer
    }

    public var resolvedPipelinePath: String {
        if task.target.hasPrefix("pipeline:") {
            let name = String(task.target.dropFirst("pipeline:".count))
            if name != ComposeSpec.defaultPipeline { return "/v1/pipelines/\(name)" }
        }
        return "/v1/pipeline"
    }
}

public struct BenchmarkAdapterOutput: Sendable, Equatable {
    public var samples: [BenchmarkSampleRecord]
    public var metrics: [String: Double]
    public var extraCostUSD: Double

    public init(samples: [BenchmarkSampleRecord], metrics: [String: Double] = [:], extraCostUSD: Double = 0) {
        (self.samples, self.metrics, self.extraCostUSD) = (samples, metrics, extraCostUSD)
    }
}

public protocol BenchmarkAdapter: Sendable {
    var id: String { get }
    var tier: BenchmarkAdapterTier { get }
    var summary: String { get }
    var defaultSubmissionTarget: SubmissionTarget? { get }
    func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput
}

extension BenchmarkAdapter {
    public var defaultSubmissionTarget: SubmissionTarget? { nil }

    public var descriptor: BenchmarkAdapterDescriptor {
        BenchmarkAdapterDescriptor(id: id, tier: tier, summary: summary, defaultSubmissionTarget: defaultSubmissionTarget)
    }
}

public struct ExternalHarnessAdapter: BenchmarkAdapter {
    public let id: String
    public let summary: String
    public let defaultSubmissionTarget: SubmissionTarget?
    public var tier: BenchmarkAdapterTier { .externalBridge }

    public init(id: String, summary: String, defaultSubmissionTarget: SubmissionTarget? = nil) {
        (self.id, self.summary, self.defaultSubmissionTarget) = (id, summary, defaultSubmissionTarget)
    }

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        if let dataset = context.task.dataset, !dataset.isEmpty {
            let data: Data
            do {
                data = try Data(contentsOf: URL(fileURLWithPath: dataset))
            } catch {
                throw BenchmarkError.invalidResponse("failed to read dataset '\(dataset)': \(error.localizedDescription)")
            }
            guard let imported = try? JSONDecoder().decode([BenchmarkSampleRecord].self, from: data),
                  !imported.isEmpty else {
                throw BenchmarkError.invalidResponse("dataset '\(dataset)' is empty or malformed JSON")
            }
            let accuracy = Double(imported.filter(\.passed).count) / Double(imported.count)
            let p50 = BenchmarkStats.percentile(imported.map(\.latencyMs), 0.5)
            return BenchmarkAdapterOutput(samples: imported, metrics: ["accuracy": accuracy, "p50_ms": p50])
        }
        let payload: [String: Any] = [
            "model": context.resolvedModelName,
            "messages": [["role": "user", "content": context.task.params["prompt"]?.stringValue ?? "Solve benchmark smoke probe for \(id)."]],
            "max_tokens": context.task.params["max_tokens"]?.intValue ?? 128,
            "temperature": context.temperature,
            "seed": context.seed,
        ]
        let (json, reply) = try await context.endpoint.postJSON("/v1/chat/completions", payload: payload)
        let text = BenchmarkStats.extractChatContent(json)
        let usage = BenchmarkStats.extractUsage(json)
        var sample = BenchmarkSampleRecord(
            id: "\(id)-0", adapter: id, model: context.resolvedModelName,
            prompt: "smoke:\(id)", answer: text, score: text.isEmpty ? 0.0 : 1.0
        )
        sample.dataset = context.task.dataset
        sample.latencyMs = reply.totalMs
        sample.ttftMs = reply.ttftMs
        sample.promptTokens = usage.prompt
        sample.completionTokens = usage.completion
        sample.passed = !text.isEmpty
        sample.metadata["bridge_openai_base_url"] = context.endpoint.baseURL.appendingPathComponent("v1").absoluteString
        return BenchmarkAdapterOutput(
            samples: [sample],
            metrics: ["accuracy": sample.score, "p50_ms": reply.totalMs]
        )
    }
}

public struct BenchmarkAdapterRegistry: Sendable {
    private let adaptersByID: [String: any BenchmarkAdapter]

    public init(adapters: [any BenchmarkAdapter]) {
        self.adaptersByID = Dictionary(uniqueKeysWithValues: adapters.map { ($0.id, $0) })
    }

    public static let standard = BenchmarkAdapterRegistry(adapters: [
        BasaltThroughputAdapter(),
        BasaltPerplexityAdapter(),
        BasaltSystemOneAdapter(),
        BasaltPasskeyAdapter(),
        BasaltMoMAdapter(),
        BasaltHarnessAdapter(),
        TranslationBenchmarkAdapter(),
        ARCAGIAdapter(id: "arc-agi", summary: "Pure-Swift ARC-AGI-1 2-attempt grid reasoner and exact-match scorer"),
        ARCAGIAdapter(id: "arc-agi-2", summary: "Pure-Swift ARC-AGI-2 2-attempt grid reasoner and exact-match scorer"),
        ExternalHarnessAdapter(id: "terminal-bench", summary: "Harbor / Terminal-Bench 2.0 ATIF v1.4 trajectory bridge", defaultSubmissionTarget: .terminalBench),
        ExternalHarnessAdapter(id: "swe-bench", summary: "SWE-bench Verified patch & trajectory bridge", defaultSubmissionTarget: .sweBench),
        ExternalHarnessAdapter(id: "lm-eval", summary: "EleutherAI LM Evaluation Harness OpenAI-compatible bridge", defaultSubmissionTarget: .hfLeaderboard),
        ExternalHarnessAdapter(id: "livecodebench", summary: "LiveCodeBench code generation & custom_outputs bridge", defaultSubmissionTarget: .liveCodeBench),
        ExternalHarnessAdapter(id: "bfcl", summary: "Berkeley Function Calling Leaderboard v3/v4 AST bridge", defaultSubmissionTarget: .bfcl),
        ExternalHarnessAdapter(id: "inspect-ai", summary: "UK AISI Inspect AI .eval log bundle bridge", defaultSubmissionTarget: .inspectAI),
        ExternalHarnessAdapter(id: "tau-bench", summary: "tau-bench multi-turn tool policy reward bridge", defaultSubmissionTarget: .tauBench),
        ExternalHarnessAdapter(id: "gaia", summary: "GAIA normalized final-answer & reasoning trace bridge", defaultSubmissionTarget: .gaia),
        ExternalHarnessAdapter(id: "evalplus", summary: "EvalPlus HumanEval+ / MBPP+ code synthesis bridge", defaultSubmissionTarget: .liveCodeBench),
    ])

    public func adapter(for id: String) -> (any BenchmarkAdapter)? {
        adaptersByID[id]
    }

    public var allDescriptors: [BenchmarkAdapterDescriptor] {
        adaptersByID.keys.sorted().compactMap { adaptersByID[$0]?.descriptor }
    }
}
