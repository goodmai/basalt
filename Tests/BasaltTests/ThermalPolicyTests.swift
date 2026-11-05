import Testing
import Foundation
import Synchronization
import Hummingbird
import HummingbirdTesting
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

final class FakeThermal: Sendable {
    private let level: Mutex<ThermalLevel>
    private let reads = Mutex(0)

    init(_ level: ThermalLevel = .nominal) { self.level = Mutex(level) }

    func set(_ level: ThermalLevel) { self.level.withLock { $0 = level } }
    var readCount: Int { reads.withLock { $0 } }

    func read() -> ThermalLevel {
        reads.withLock { $0 += 1 }
        return level.withLock { $0 }
    }
}

final class FakePacingClock: Sendable {
    let base = ContinuousClock.now
    private let ticks = Mutex(0)
    private let pauses = Mutex<[Duration]>([])
    var paused: [Duration] { pauses.withLock { $0 } }

    func pacer(_ thermal: FakeThermal) -> PrefillPacer {
        PrefillPacer(
            directive: { ThermalDirective(thermal.read()) },
            now: { self.base + .milliseconds(400) * self.ticks.withLock { $0 += 1; return $0 } },
            pause: { duration in self.pauses.withLock { $0.append(duration) } })
    }
}

@Suite("Thermal policy: prefill pacing and admission under macOS thermal pressure (#45)")
struct ThermalPolicyTests {
    @Test("Each macOS thermal state maps to its level and directive", arguments: [
        (ProcessInfo.ThermalState.nominal, ThermalLevel.nominal, ThermalDirective.normal),
        (.fair, .fair, .normal),
        (.serious, .serious, .paced),
        (.critical, .critical, .refusing),
    ])
    func stateMapsToDirective(state: ProcessInfo.ThermalState, level: ThermalLevel, directive: ThermalDirective) {
        #expect(ThermalLevel(state) == level)
        #expect(ThermalDirective(level) == directive)
    }

    @Test("Only the refusing directive refuses, and only prompts of 8192+ tokens, with 503 and Retry-After 60")
    func refusesLongPromptsOnlyWhenCritical() throws {
        let refusal = try #require(ThermalDirective.refusing.refusal(promptTokens: MemoryDirective.longContextTokens))
        guard case .serverBusy(let retryAfter) = refusal else { Issue.record("expected serverBusy, got \(refusal)"); return }
        #expect(retryAfter == ThermalDirective.retryAfterSeconds)
        #expect(refusal.httpStatus == 503)
        #expect(ThermalDirective.refusing.refusal(promptTokens: MemoryDirective.longContextTokens - 1) == nil)
        #expect(ThermalDirective.paced.refusal(promptTokens: 1_000_000) == nil)
        #expect(ThermalDirective.normal.refusal(promptTokens: 1_000_000) == nil)
    }

    @Test("Serious and critical cap the prefill chunk at 256 tokens; nominal keeps the requested one", arguments: [
        (ThermalLevel.nominal, Int?.none, Int?.none), (.nominal, 1024, 1024), (.fair, 1024, 1024),
        (.serious, nil, 256), (.serious, 1024, 256), (.serious, 128, 128), (.critical, nil, 256),
    ])
    func chunkCap(level: ThermalLevel, requested: Int?, expected: Int?) {
        #expect(FakePacingClock().pacer(FakeThermal(level)).stepSize(under: requested) == expected)
    }

    @Test("Paced prefill pauses a quarter of each chunk's time after every chunk but the last; normal never pauses")
    func pausesBetweenChunks() {
        let clock = FakePacingClock()
        let chained = Mutex<[Int]>([])
        let progress = clock.pacer(FakeThermal(.serious)).progress { processed, _ in chained.withLock { $0.append(processed) } }
        for processed in [256, 512, 1000] { progress(processed, 1000) }
        #expect(clock.paused == [.milliseconds(100), .milliseconds(100)])
        #expect(chained.withLock { $0 } == [256, 512, 1000])

        let calm = FakePacingClock()
        let unpaced = calm.pacer(FakeThermal(.fair)).progress(chaining: nil)
        for processed in [256, 512, 1000] { unpaced(processed, 1000) }
        #expect(calm.paused.isEmpty)
    }

    @Test("The directive is read again at every chunk, so a state change mid-prefill paces the next chunk")
    func readsDirectiveEveryChunk() {
        let clock = FakePacingClock()
        let thermal = FakeThermal(.nominal)
        let progress = clock.pacer(thermal).progress(chaining: nil)
        progress(256, 1000)
        #expect(clock.paused.isEmpty)
        thermal.set(.serious)
        progress(512, 1000)
        #expect(clock.paused.count == 1)
    }

    @Test("On a tiny model the paced directive is applied inside the launch's prefill: smaller chunks, a pause after each")
    func appliedInsideLaunchPrefill() async throws {
        let context = try GenerationHandOffTests.context()
        func chunks(_ level: ThermalLevel) async throws -> (chunks: Int, pauses: Int, reads: Int) {
            let thermal = FakeThermal(level)
            let clock = FakePacingClock()
            let pacer = clock.pacer(thermal)
            #expect(thermal.readCount == 0)
            let seen = Mutex(0)
            let launch = try MLXInferenceEngine.$prefillProgress.withValue({ _, _ in seen.withLock { $0 += 1 } }) {
                try MLXInferenceEngine.launch(
                    input: LMInput(tokens: MLXArray((0 ..< 3000).map { Int32(1 + $0 % 200) })), prompt: CachedPrompt(boundary: 2999),
                    options: LaunchOptions(params: GenerateParameters(maxTokens: 1, temperature: 0), pacer: pacer), context: context)
            }
            let result = (seen.withLock { $0 }, clock.paused.count, thermal.readCount)
            await MLXInferenceEngine.settle(launch.task, cancelled: true)
            return result
        }
        let calm = try await chunks(.nominal)
        let hot = try await chunks(.serious)
        #expect(calm.pauses == 0 && calm.reads > 0)
        #expect(hot.chunks > calm.chunks)
        #expect(hot.pauses == hot.chunks - 1)
    }

    static func refusedForHeat(_ orchestrator: ModelOrchestratorActor, _ request: GenerationRequest) async -> Bool {
        do {
            _ = try await orchestrator.generate(request: request)
            return false
        } catch {
            guard case .serverBusy(let retryAfter) = error else { return false }
            return retryAfter == ThermalDirective.retryAfterSeconds
        }
    }

    static func orchestrator(_ engine: MockInferenceEngine, _ thermal: FakeThermal) async throws -> ModelOrchestratorActor {
        let orchestrator = try await MemoryCoordinatorTests.orchestrator(engine)
        await orchestrator.coordinateThermal(with: { thermal.read() })
        return orchestrator
    }

    @Test("At critical a long prompt is refused with 503 and Retry-After 60; a short one and any prompt at serious run")
    func criticalRefusesLongPrompts() async throws {
        let engine = MockInferenceEngine()
        let thermal = FakeThermal(.critical)
        let orchestrator = try await Self.orchestrator(engine, thermal)
        await engine.setMemory(kvBudget: 1 << 40, promptTokens: MemoryDirective.longContextTokens)
        let long = GenerationRequest(prompt: "long", maxTokens: 10)
        let refusal = try #require(await orchestrator.admissionRefusal(for: long))
        #expect(refusal.httpStatus == 503)
        #expect(await Self.refusedForHeat(orchestrator, long))
        thermal.set(.serious)
        _ = try await orchestrator.generate(request: long)
        thermal.set(.critical)
        await engine.setMemory(kvBudget: 1 << 40, promptTokens: 100)
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "short", maxTokens: 10))
        await orchestrator.shutdown()
    }

    @Test("The engine gets a pacer that reads the live thermal state, and only when asked")
    func enginePacerFollowsSource() async throws {
        let engine = MockInferenceEngine()
        let thermal = FakeThermal(.nominal)
        let orchestrator = try await Self.orchestrator(engine, thermal)
        let pacer = try #require(await engine.pacer)
        #expect(thermal.readCount == 0)
        #expect(pacer.directive() == .normal)
        thermal.set(.serious)
        #expect(pacer.directive() == .paced)
        thermal.set(.critical)
        #expect(pacer.directive() == .refusing)
        await orchestrator.shutdown()
    }

    @Test("Without the policy the telemetry still reports the level, with the normal directive")
    func telemetryWithoutPolicy() async throws {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        let thermal = try #require(await orchestrator.telemetrySnapshot().thermal)
        #expect(thermal.directive == .normal)
    }

    @Test("/api/v1/stats and the --stats line show the thermal level and directive")
    func statsShowThermal() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = try await Self.orchestrator(engine, FakeThermal(.serious))
        let snapshot = await orchestrator.telemetrySnapshot()
        #expect(snapshot.thermal == ThermalTelemetry(level: .serious, directive: .paced))
        #expect(snapshot.formattedStatsLine.contains("Thermal: serious, paced"))
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "stub"))
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/api/v1/stats", method: .get) { res in
                let snap = try JSONDecoder().decode(ServerTelemetrySnapshot.self, from: Data(String(buffer: res.body).utf8))
                #expect(snap.thermal == ThermalTelemetry(level: .serious, directive: .paced))
            }
        }
        await orchestrator.shutdown()
    }
}
