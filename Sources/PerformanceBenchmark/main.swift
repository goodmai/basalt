import Foundation
import ArgumentParser
import GemCore

@available(macOS 10.15, *)
@main
struct Benchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmark Gem inference (TPS, TTFT, latency, memory)"
    )

    @Option(name: .shortAndLong, help: "Model path or HF repo ID (must be in local cache)")
    var model: String

    @Option(name: .shortAndLong, help: "Number of timed iterations (warmup is always 1)")
    var iterations: Int = 5

    @Option(name: .shortAndLong, help: "Prompt to use for every iteration")
    var prompt: String = "Explain the importance of open source models."

    @Option(name: .shortAndLong, help: "Max tokens to generate per iteration")
    var tokens: Int = 100

    @Option(name: .customLong("quant"), help: "Quantization subfolder inside the repo (e.g. 4bit) — for repos shipping several variants")
    var quant: String?

    @Option(name: .customLong("reasoning-effort"),
            help: "Reasoning budget (xhigh | medium | low), or `none`. Qwen defaults to xhigh, which spends the whole budget thinking and distorts throughput numbers.")
    var reasoningEffort: String?

    @Option(name: .customLong("gpu-cache-mb"), help: "MLX buffer cache ceiling in MB; 0 means leave MLX's default uncapped")
    var gpuCacheMB: Int?

    @Option(name: .customLong("top-k"), help: "Keep only the k most likely tokens (0/unset disables)")
    var topK: Int?

    @Option(name: .customLong("min-p"), help: "Drop tokens below this fraction of the top token's probability")
    var minP: Float?

    @Option(name: .customLong("seed"), help: "Seed the sampler so a non-greedy run is reproducible")
    var seed: UInt64?

    @Option(name: .customLong("kv-bits"), help: "Quantize the KV cache to 4 or 8 bits (default: full precision)")
    var kvBits: Int?

    @Option(name: .customLong("repetition-penalty"), help: "Penalty on recently emitted tokens, e.g. 1.1 (default: off)")
    var repetitionPenalty: Float?

    @Flag(help: "Skip the warmup iteration")
    var noWarmup: Bool = false

    @Flag(name: .customLong("degradation-test"), help: "Run context degradation benchmark (1k -> 128k) and output to JSON")
    var degradationTest: Bool = false

    func run() async throws {
        let resolvedPath = resolveModelPath()
        print("Benchmark — model: \(model)")
        if resolvedPath != model {
            print("  path: \(resolvedPath)")
        }
        print("  iterations: \(iterations)  tokens: \(tokens)  warmup: \(!noWarmup)\n")

        let engine = MLXInferenceEngine(
            reasoningEffort: reasoningEffort,
            gpuCacheLimit: gpuCacheMB.map { $0 == 0 ? nil : $0 << 20 } ?? (512 << 20),
            kvBits: kvBits,
            repetitionPenalty: repetitionPenalty,
            topK: topK, minP: minP, seed: seed)
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: tokens)

        print("Loading model…", terminator: ""); fflush(stdout)
        try await orchestrator.loadModel(path: resolvedPath)
        print(" done\n")

        if degradationTest {
            try await runDegradationTest(orchestrator: orchestrator)
            return
        }

        // Warmup (excluded from stats)
        if !noWarmup {
            print("Warmup… ", terminator: ""); fflush(stdout)
            let req = GenerationRequest(prompt: prompt, maxTokens: tokens)
            _ = try await orchestrator.generate(request: req)
            print("done\n")
        }

        var tpsSamples:      [Double] = []
        var ttftSamples:     [Double] = []
        var genTimeSamples:  [Double] = []

        print("  #     TPS         TTFT (s)    Time (s)")
        print("  " + String(repeating: "─", count: 42))

        for i in 1...iterations {
            let req = GenerationRequest(prompt: prompt, maxTokens: tokens)
            let r   = try await orchestrator.generate(request: req)

            tpsSamples.append(r.tokensPerSecond)
            ttftSamples.append(r.timeToFirstToken)
            genTimeSamples.append(r.generationTime)

            print(String(format: "  %-4d  %-10.2f  %-10.3f  %-10.2f",
                         i, r.tokensPerSecond, r.timeToFirstToken, r.generationTime))
        }

        print()
        printStats(label: "TPS",          samples: tpsSamples,     unit: "tok/s")
        printStats(label: "TTFT",         samples: ttftSamples,     unit: "s")
        printStats(label: "GenTime",      samples: genTimeSamples,  unit: "s")

        // Memory from last run
        let req = GenerationRequest(prompt: prompt, maxTokens: tokens)
        let last = try await orchestrator.generate(request: req)
        let memMB = last.memory.activeBytes / 1_048_576
        print(String(format: "\n  Memory (active): %d MB", memMB))
    }

    private func runDegradationTest(orchestrator: ModelOrchestratorActor) async throws {
        let profiler = ContextDegradationProfiler(orchestrator: orchestrator)
        let sizes = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
        
        print("Running degradation benchmark with context sizes: \(sizes)")
        let report = try await profiler.runBenchmark(contextSizes: sizes, tokensToGenerate: tokens, iterations: iterations)
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(report)
        let url = URL(fileURLWithPath: "context_latency.json")
        try data.write(to: url)
        
        print("\nDegradation report saved to context_latency.json")
        
        print("  Context   TPS       TTFT      Memory (MB)")
        print("  " + String(repeating: "─", count: 42))
        for p in report.dataPoints {
            print(String(format: "  %-8d  %-8.2f  %-8.3f  %-8d", p.contextSize, p.avgTPS, p.avgTTFT, p.memoryActiveMB))
        }
    }

    private func printStats(label: String, samples: [Double], unit: String) {
        guard !samples.isEmpty else { return }
        let n    = Double(samples.count)
        let mean = samples.reduce(0, +) / n
        let minV = samples.min()!
        let maxV = samples.max()!
        let variance = samples.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n
        let stdDev = sqrt(variance)

        let paddedLabel = label.padding(toLength: 10, withPad: " ", startingAt: 0)
        let stats = String(format: "avg=%-8.3f  min=%-8.3f  max=%-8.3f  σ=%-8.3f", mean, minV, maxV, stdDev)
        print("  \(paddedLabel)  \(stats)  \(unit)")
    }

    private func resolveModelPath() -> String {
        // Shared with `serve` so both agree on which weights are loaded.
        ModelCache.resolve(repoId: model, quant: quant)?.path ?? model
    }
}
