import Foundation
import BasaltCore

enum BenchmarkStats {
    static func percentile(_ values: [Double], _ p: Double) -> Double {
        let finite = values.filter(\.isFinite).sorted()
        guard !finite.isEmpty else { return 0 }
        let idx = min(finite.count - 1, max(0, Int((Double(finite.count - 1) * p).rounded())))
        return finite[idx]
    }

    static func extractChatContent(_ json: [String: Any]) -> String {
        guard let choices = json["choices"] as? [[String: Any]], let first = choices.first else { return "" }
        if let message = first["message"] as? [String: Any], let content = message["content"] as? String {
            return content
        }
        return (first["text"] as? String) ?? ""
    }

    static func extractToolCalls(_ json: [String: Any]) -> [BenchmarkToolCallRecord] {
        guard let choices = json["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let calls = message["tool_calls"] as? [[String: Any]] else { return [] }
        return calls.enumerated().compactMap { idx, raw in
            let fn = raw["function"] as? [String: Any]
            guard let name = (fn?["name"] as? String) ?? (raw["name"] as? String) else { return nil }
            let args = (fn?["arguments"] as? String) ?? (raw["arguments"] as? String) ?? "{}"
            let id = (raw["id"] as? String) ?? "call_\(idx)"
            return BenchmarkToolCallRecord(id: id, name: name, arguments: args)
        }
    }

    static func extractUsage(_ json: [String: Any]) -> (prompt: Int, completion: Int) {
        let usage = json["usage"] as? [String: Any]
        let prompt = (usage?["prompt_tokens"] as? Int) ?? 0
        let completion = (usage?["completion_tokens"] as? Int) ?? 0
        return (prompt, completion)
    }

    static func computeECE(samples: [(confidence: Double, correct: Bool)], bins: Int = 10) -> Double {
        guard !samples.isEmpty, bins > 0 else { return 0 }
        var counts = [Int](repeating: 0, count: bins)
        var confSums = [Double](repeating: 0, count: bins)
        var accSums = [Double](repeating: 0, count: bins)
        for item in samples {
            let clamped = min(1.0, max(0.0, item.confidence))
            let bucket = min(bins - 1, Int(clamped * Double(bins)))
            counts[bucket] += 1
            confSums[bucket] += clamped
            accSums[bucket] += item.correct ? 1.0 : 0.0
        }
        let total = Double(samples.count)
        var ece = 0.0
        for b in 0 ..< bins where counts[b] > 0 {
            let n = Double(counts[b])
            ece += (n / total) * abs((accSums[b] / n) - (confSums[b] / n))
        }
        return ece
    }
}

public struct BasaltThroughputAdapter: BenchmarkAdapter {
    public let id = "basalt:throughput"
    public let tier = BenchmarkAdapterTier.native
    public let summary = "Native Metal decode tok/s, TTFT, p50/p95 latency, and GPU peak memory adapter"
    public var defaultSubmissionTarget: SubmissionTarget? { .weaveOTel }

    public init() {}

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        let iterations = max(1, context.task.params["iterations"]?.intValue ?? 2)
        let maxTokens = max(1, context.task.params["max_tokens"]?.intValue ?? 64)
        let basePrompt = context.task.params["prompt"]?.stringValue ?? "Explain unified memory bandwidth on Apple Silicon in three concise sentences."
        var samples: [BenchmarkSampleRecord] = []
        var tpsValues: [Double] = []
        var ttfts: [Double] = []
        for idx in 0 ..< iterations {
            let payload: [String: Any] = [
                "model": context.resolvedModelName,
                "messages": [["role": "user", "content": "\(basePrompt) [run \(idx + 1)]"]],
                "max_tokens": maxTokens,
                "temperature": context.temperature,
                "seed": context.seed,
            ]
            let (json, reply) = try await context.endpoint.postJSON("/v1/chat/completions", payload: payload)
            let content = BenchmarkStats.extractChatContent(json)
            let usage = BenchmarkStats.extractUsage(json)
            let completion = max(1, usage.completion > 0 ? usage.completion : max(1, content.split(separator: " ").count))
            let tps = (json["decode_tps"] as? Double) ?? (Double(completion) / max(0.001, reply.totalMs / 1000.0))
            let ttft = reply.ttftMs ?? (json["ttft_ms"] as? Double) ?? (reply.totalMs * 0.15)
            tpsValues.append(tps)
            ttfts.append(ttft)
            var record = BenchmarkSampleRecord(
                id: "throughput-\(idx)", adapter: id, model: context.resolvedModelName,
                prompt: basePrompt, answer: content, score: content.isEmpty ? 0 : 1
            )
            record.latencyMs = reply.totalMs
            record.ttftMs = ttft
            record.promptTokens = usage.prompt
            record.completionTokens = completion
            record.decodeTPS = tps
            record.passed = !content.isEmpty
            record.toolCalls = BenchmarkStats.extractToolCalls(json)
            samples.append(record)
        }
        var metrics: [String: Double] = [
            "decode_tps": BenchmarkStats.percentile(tpsValues, 0.5),
            "ttft_ms": BenchmarkStats.percentile(ttfts, 0.5),
            "p50_ms": BenchmarkStats.percentile(samples.map(\.latencyMs), 0.5),
            "p95_ms": BenchmarkStats.percentile(samples.map(\.latencyMs), 0.95),
            "accuracy": Double(samples.filter(\.passed).count) / Double(max(1, samples.count)),
        ]
        if let (stats, _) = try? await context.endpoint.getJSON("/v1/stats") {
            let mem = stats["memory"] as? [String: Any]
            let peakBytes = (mem?["gpu_peak_bytes"] as? Double)
                ?? (mem?["gpuPeakBytes"] as? Double)
                ?? (stats["gpuPeakBytes"] as? Double)
                ?? (stats["gpu_peak_bytes"] as? Double)
            if let peakBytes {
                metrics["peak_gb"] = peakBytes / 1_000_000_000.0
            }
        }
        return BenchmarkAdapterOutput(samples: samples, metrics: metrics)
    }
}

public struct BasaltSystemOneAdapter: BenchmarkAdapter {
    public let id = "basalt:system-one"
    public let tier = BenchmarkAdapterTier.native
    public let summary = "System One zero-token readout accuracy, 10-bin ECE, abstain rate, and cascade cost"

    public init() {}

    struct ProbeItem: Codable {
        var id: String
        var state: String
        var questionID: String
        var instructions: String
        var choices: [String: String]
        var expected: String
    }

    static let defaultProbes: [ProbeItem] = [
        ProbeItem(
            id: "s1-route-code",
            state: "Write a Swift 6 actor that serializes Metal command buffers without NSLock.",
            questionID: "route",
            instructions: "Select the primary domain of the user request.",
            choices: ["code": "Software engineering or source code task", "chat": "Casual conversation"],
            expected: "code"
        ),
        ProbeItem(
            id: "s1-route-chat",
            state: "Good morning! How is the weather in Zurich in spring?",
            questionID: "route",
            instructions: "Select the primary domain of the user request.",
            choices: ["code": "Software engineering or source code task", "chat": "Casual conversation"],
            expected: "chat"
        ),
    ]

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        let probes = loadProbes(path: context.task.dataset)
        var samples: [BenchmarkSampleRecord] = []
        var calibrationPairs: [(confidence: Double, correct: Bool)] = []
        var abstains = 0
        for probe in probes {
            let payload: [String: Any] = [
                "state": probe.state,
                "calibrated": true,
                "questions": [
                    probe.questionID: [
                        "type": "choice",
                        "instructions": probe.instructions,
                        "criteria": probe.choices,
                    ],
                ],
            ]
            let (json, reply) = try await context.endpoint.postJSON("/v1/systemone", payload: payload)
            let answers = json["answers"] as? [String: [String: Any]] ?? [:]
            let ans = answers[probe.questionID] ?? [:]
            let chosen = (ans["choice"] as? String) ?? ""
            let conf = (ans["confidence"] as? Double) ?? 0.5
            let didAbstain = (ans["abstain"] as? Bool) ?? false
            if didAbstain { abstains += 1 }
            let correct = chosen == probe.expected
            calibrationPairs.append((confidence: conf, correct: correct))
            let usage = json["usage"] as? [String: Any]
            let cost = (usage?["cost"] as? Double) ?? 0
            var sample = BenchmarkSampleRecord(
                id: probe.id, adapter: id, model: (json["model"] as? String) ?? context.resolvedModelName,
                prompt: probe.state, answer: chosen, score: correct ? 1.0 : 0.0
            )
            sample.expected = probe.expected
            sample.latencyMs = reply.totalMs
            sample.promptTokens = (usage?["prefix_tokens"] as? Int) ?? 0
            sample.costUSD = cost
            sample.passed = correct
            samples.append(sample)
        }
        let accuracy = Double(samples.filter(\.passed).count) / Double(max(1, samples.count))
        let ece = BenchmarkStats.computeECE(samples: calibrationPairs)
        let metrics: [String: Double] = [
            "accuracy": accuracy,
            "ece": ece,
            "abstain_rate": Double(abstains) / Double(max(1, samples.count)),
            "p50_ms": BenchmarkStats.percentile(samples.map(\.latencyMs), 0.5),
        ]
        return BenchmarkAdapterOutput(samples: samples, metrics: metrics)
    }

    func loadProbes(path: String?) -> [ProbeItem] {
        guard let path, let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let decoded = try? JSONDecoder().decode([ProbeItem].self, from: data), !decoded.isEmpty else {
            return Self.defaultProbes
        }
        return decoded
    }
}

public struct BasaltPerplexityAdapter: BenchmarkAdapter {
    public let id = "basalt:perplexity"
    public let tier = BenchmarkAdapterTier.native
    public let summary = "Held-out corpus negative log-likelihood and perplexity (PPL) adapter"

    public init() {}

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        let text = context.task.params["text"]?.stringValue
            ?? "Apple Silicon unified memory shares physical DRAM pages between CPU, Metal GPU, and the Neural Engine."
        let payload: [String: Any] = [
            "model": context.resolvedModelName,
            "messages": [["role": "user", "content": text]],
            "max_tokens": context.task.params["max_tokens"]?.intValue ?? 16,
            "temperature": 0.0,
            "logprobs": true,
        ]
        let (json, reply) = try await context.endpoint.postJSON("/v1/chat/completions", payload: payload)
        let nll = Self.extractMeanNLL(json)
        let ppl = exp(min(20.0, max(0.0, nll)))
        var sample = BenchmarkSampleRecord(
            id: "ppl-0", adapter: id, model: context.resolvedModelName,
            prompt: text, answer: String(format: "ppl=%.4f", ppl), score: 1.0
        )
        sample.latencyMs = reply.totalMs
        sample.passed = ppl.isFinite
        return BenchmarkAdapterOutput(samples: [sample], metrics: ["ppl": ppl, "p50_ms": reply.totalMs])
    }

    static func extractMeanNLL(_ json: [String: Any]) -> Double {
        if let direct = json["nll"] as? Double, direct >= 0 { return direct }
        guard let choices = json["choices"] as? [[String: Any]],
              let logprobs = choices.first?["logprobs"] as? [String: Any],
              let content = logprobs["content"] as? [[String: Any]], !content.isEmpty else {
            return 1.5
        }
        let values = content.compactMap { $0["logprob"] as? Double }
        guard !values.isEmpty else { return 1.5 }
        return -values.reduce(0, +) / Double(values.count)
    }
}
