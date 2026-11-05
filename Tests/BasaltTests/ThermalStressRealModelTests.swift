import Foundation
import Testing
import Synchronization
@testable import BasaltCore
@testable import BasaltInference

@Suite("Thermal policy under a sustained prefill/decode load on a real model (#45)", .serialized)
struct ThermalStressRealModelTests {
    static var environment: [String: String] { ProcessInfo.processInfo.environment }

    static var modelPath: String? {
        guard environment["BASALT_E2E"] == "1" else { return nil }
        return environment["BASALT_E2E_THERMAL_MODEL_PATH"]
    }

    static func prompt(_ seed: Int, words: Int) -> String {
        let vocabulary = "kernel buffer cache tensor stream decode anchor memory budget window thermal clock".split(separator: " ")
        return "Report \(seed). List the three most frequent words, then summarise in two sentences.\n"
            + (0..<words).map { "\(vocabulary[($0 * 7 + seed) % vocabulary.count])\(($0 * 13 + seed) % 997)" }.joined(separator: " ")
    }

    static func percentile(_ values: [Double], _ share: Double) -> Double {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        return sorted[min(sorted.count - 1, Int((Double(sorted.count) * share).rounded(.up)) - 1)]
    }

    @Test("15 minutes of 6K- and 10K-token prompts: seconds at each thermal level and p99 TTFT, with the policy on or off",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1, BASALT_E2E_THERMAL_MODEL_PATH and BASALT_E2E_THERMAL_POLICY=on|off"))
    func sustainedLoad() async throws {
        let path = try #require(Self.modelPath)
        let policy = Self.environment["BASALT_E2E_THERMAL_POLICY"] != "off"
        let minutes = Int(Self.environment["BASALT_E2E_THERMAL_MINUTES"] ?? "") ?? 15
        let log = try await ModelRunLog("thermal-stress", modelPath: path)
        defer { log.releaseGPU() }
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(), telemetry: TelemetryTracker())
        try await log.step("load") { try await orchestrator.loadModel(path: path) }
        if policy { await orchestrator.coordinateThermal() }
        log.record("config", "policy \(policy ? "on" : "off") minutes \(minutes) start level \(ThermalLevel.current().rawValue)")
        let seconds = Mutex<[ThermalLevel: Int]>([:])
        let sampler = Task {
            while !Task.isCancelled {
                let level = ThermalLevel.current()
                seconds.withLock { $0[level, default: 0] += 1 }
                try? await Task.sleep(for: .seconds(1))
            }
        }
        var ttft: [String: [Double]] = ["6k": [], "10k": []]
        var refused = 0
        let deadline = ContinuousClock.now + .seconds(minutes * 60)
        var seed = 0
        while ContinuousClock.now < deadline {
            seed += 1
            let kind = seed.isMultiple(of: 4) ? "10k" : "6k"
            let request = GenerationRequest(prompt: Self.prompt(seed, words: kind == "10k" ? 4_200 : 2_500), maxTokens: 128, temperature: 0)
            do {
                let response = try await orchestrator.generate(request: request)
                ttft[kind, default: []].append(response.timeToFirstToken)
                log.record("request", "\(seed) \(kind) prompt \(response.promptTokens) ttft \(String(format: "%.2f", response.timeToFirstToken)) s level \(ThermalLevel.current().rawValue)")
            } catch BasaltError.serverBusy(let retryAfter) {
                refused += 1
                log.record("refused", "\(seed) \(kind) retry after \(retryAfter ?? 0) s level \(ThermalLevel.current().rawValue)")
            }
        }
        sampler.cancel()
        await sampler.value
        let levels = seconds.withLock { $0 }
        let all = ttft.values.flatMap { $0 }
        let summary = ThermalLevel.allCases.map { "\($0.rawValue) \(levels[$0] ?? 0) s" }.joined(separator: ", ")
        log.record("result", "policy \(policy ? "on" : "off"): \(summary); served \(all.count), refused \(refused); p99 TTFT all \(String(format: "%.2f", Self.percentile(all, 0.99))) s, 6k \(String(format: "%.2f", Self.percentile(ttft["6k"] ?? [], 0.99))) s, p50 6k \(String(format: "%.2f", Self.percentile(ttft["6k"] ?? [], 0.5))) s")
        #expect(!all.isEmpty)
        await orchestrator.shutdown()
    }
}
