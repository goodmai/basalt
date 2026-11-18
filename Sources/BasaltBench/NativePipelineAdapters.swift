import Foundation
import BasaltCore
import BasaltHarness

public struct BasaltPasskeyAdapter: BenchmarkAdapter {
    public let id = "basalt:passkey"
    public let tier = BenchmarkAdapterTier.native
    public let summary = "Deterministic long-context needle-in-a-haystack passkey retrieval adapter"

    public init() {}

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        var rng = SplitMix64(seed: context.seed)
        let passkey = String(100_000 + Int(rng.next() % 900_000))
        let fillerBlocks = max(4, (context.task.params["context_tokens"]?.intValue ?? 256) / 16)
        let needleIndex = fillerBlocks / 2
        var lines: [String] = []
        for idx in 0 ..< fillerBlocks {
            if idx == needleIndex {
                lines.append("IMPORTANT RECORD: The secret passkey is \(passkey). Remember this exact six-digit number.")
            }
            lines.append("Filler paragraph \(idx + 1): The grass is green, the sky is blue, and the sun is yellow.")
        }
        lines.append("What is the exact six-digit secret passkey mentioned above? Reply with the digits.")
        let prompt = lines.joined(separator: "\n")
        let payload: [String: Any] = [
            "model": context.resolvedModelName,
            "messages": [["role": "user", "content": prompt]],
            "max_tokens": 32,
            "temperature": 0.0,
            "seed": context.seed,
        ]
        let (json, reply) = try await context.endpoint.postJSON("/v1/chat/completions", payload: payload)
        let answer = BenchmarkStats.extractChatContent(json)
        let passed = answer.contains(passkey)
        let usage = BenchmarkStats.extractUsage(json)
        var sample = BenchmarkSampleRecord(
            id: "passkey-\(context.seed)", adapter: id, model: context.resolvedModelName,
            prompt: "passkey(\(fillerBlocks) blocks)", answer: answer, score: passed ? 1.0 : 0.0
        )
        sample.expected = passkey
        sample.latencyMs = reply.totalMs
        sample.promptTokens = usage.prompt
        sample.completionTokens = usage.completion
        sample.passed = passed
        return BenchmarkAdapterOutput(
            samples: [sample],
            metrics: ["accuracy": sample.score, "p50_ms": reply.totalMs]
        )
    }
}

public struct BasaltMoMAdapter: BenchmarkAdapter {
    public let id = "basalt:mom"
    public let tier = BenchmarkAdapterTier.native
    public let summary = "Mixture-of-Models perceive -> decide -> answer pipeline receipt and cost adapter"

    public init() {}

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        let promptText = context.task.params["text"]?.stringValue ?? "Implement a thread-safe ring buffer in Swift 6."
        let expectedLabel = context.task.params["expect_route"]?.stringValue ?? "code"
        let payload: [String: Any] = [
            "text": promptText,
            "task": "Respond concisely to the request.",
            "questions": [
                "route": [
                    "type": "choice",
                    "instructions": "Classify the request domain.",
                    "criteria": ["code": "Programming task", "chat": "General question"],
                ],
            ],
        ]
        let (json, reply) = try await context.endpoint.postJSON(context.resolvedPipelinePath, payload: payload)
        let answers = json["answers"] as? [String: [String: Any]] ?? [:]
        let label = (answers["route"]?["choice"] as? String) ?? (answers["route"]?["label"] as? String) ?? ""
        let receipt = json["receipt"] as? [String: Any] ?? [:]
        let stages = receipt["stages"] as? [[String: Any]] ?? []
        let cost = stages.reduce(0.0) { $0 + (($1["cost"] as? Double) ?? 0.0) }
        let tokens = stages.reduce(0) { $0 + (($1["tokens"] as? Int) ?? 0) }
        let system2 = (json["system2"] as? [String: Any])?["text"] as? String ?? label
        let matched = label.isEmpty ? !system2.isEmpty : (label == expectedLabel)
        var sample = BenchmarkSampleRecord(
            id: "mom-0", adapter: id, model: context.task.target,
            prompt: promptText, answer: system2, score: matched ? 1.0 : 0.0
        )
        sample.expected = expectedLabel
        sample.latencyMs = reply.totalMs
        sample.completionTokens = tokens
        sample.costUSD = cost
        sample.passed = matched
        return BenchmarkAdapterOutput(
            samples: [sample],
            metrics: [
                "accuracy": sample.score,
                "p50_ms": reply.totalMs,
                "cost_per_1k_usd": cost * 1000.0,
            ]
        )
    }
}

public struct BasaltHarnessAdapter: BenchmarkAdapter {
    public let id = "basalt:harness"
    public let tier = BenchmarkAdapterTier.native
    public let summary = "Agent harness prompt-leakage guard and deliverable verification adapter"

    public init() {}

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        let prompt = context.task.params["prompt"]?.stringValue ?? "Return a JSON object {\"status\":\"ok\"} and nothing else."
        try HarnessLeakGuard.assertClean(prompt)
        let requiredTerm = context.task.params["required_term"]?.stringValue ?? "ok"
        let payload: [String: Any] = [
            "model": context.resolvedModelName,
            "messages": [["role": "user", "content": prompt]],
            "max_tokens": context.task.params["max_tokens"]?.intValue ?? 128,
            "temperature": context.temperature,
            "seed": context.seed,
        ]
        let (json, reply) = try await context.endpoint.postJSON("/v1/chat/completions", payload: payload)
        let answer = BenchmarkStats.extractChatContent(json)
        let passed = answer.localizedCaseInsensitiveContains(requiredTerm)
        var sample = BenchmarkSampleRecord(
            id: "harness-0", adapter: id, model: context.resolvedModelName,
            prompt: prompt, answer: answer, score: passed ? 1.0 : 0.0
        )
        sample.latencyMs = reply.totalMs
        sample.passed = passed
        return BenchmarkAdapterOutput(
            samples: [sample],
            metrics: ["accuracy": sample.score, "p50_ms": reply.totalMs]
        )
    }
}
