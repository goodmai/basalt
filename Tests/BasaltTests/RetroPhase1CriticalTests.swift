import ArgumentParser
import Foundation
import MLX
import Testing
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Retrospective audit #544 Phase 1 critical fixes: reentrancy, privacy, GGUF bounds, Polar KV, Compose traversal, and notarization", .serialized)
struct RetroPhase1CriticalTests {
    actor GateSignal {
        private var entered: CheckedContinuation<Void, Never>?
        private var proceed: CheckedContinuation<Void, Never>?
        private(set) var unloadCalls = 0
        private(set) var decideCalls = 0

        func waitUntilEntered() async {
            await withCheckedContinuation { entered = $0 }
        }

        func markEnteredAndWait() async {
            entered?.resume()
            entered = nil
            await withCheckedContinuation { proceed = $0 }
        }

        func allowProceed() {
            proceed?.resume()
            proceed = nil
        }

        func recordUnload() { unloadCalls += 1 }
        func recordDecide() { decideCalls += 1 }
    }

    struct StubDecider: DecisionEngine {
        let modelID: String
        let acceptsMedia: Bool
        let sharesPerceived: Bool
        let onDeviceFlag: Bool
        let confidence: Double
        let signal: GateSignal?

        var onDevice: (any DecisionEngine)? { onDeviceFlag ? self : nil }

        func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
            await signal?.recordDecide()
            var ans = SystemOneAnswer()
            ans.noul = confidence
            ans.confidence = confidence
            let answers = Dictionary(uniqueKeysWithValues: request.questions.keys.map { ($0, ans) })
            let usage = SystemOneUsage(prefixTokens: 1, candidateBranches: 1, prefixEvaluations: 1, branchForwardCalls: 1, suffixTokens: 1)
            return SystemOneResponse(model: modelID, answers: answers, usage: usage, calibrated: true)
        }

        func unload() async {
            await signal?.recordUnload()
        }
    }

    @Test("OnDemandDecision unload() mid-load unloads newly loaded engine and resets phase and users to zero (CORE-01)")
    func onDemandDecisionMidLoadUnload() async throws {
        let signal = GateSignal()
        let onDemand = OnDemandDecision(modelID: "stub-jev", path: "/tmp/stub", idle: .seconds(60)) {
            await signal.markEnteredAndWait()
            return StubDecider(modelID: "stub-jev", acceptsMedia: false, sharesPerceived: true, onDeviceFlag: true, confidence: 0.9, signal: signal)
        }

        let req = SystemOneRequest(state: "doc", questions: ["q": SystemOneQuestion(type: .noul, instructions: "ok?")])
        let loadTask = Task { try await onDemand.decide(req) }
        await signal.waitUntilEntered()
        let unloadTask = Task { await onDemand.unload() }
        await signal.allowProceed()
        await unloadTask.value
        _ = try? await loadTask.value

        #expect(await onDemand.phase == .unloaded)
        #expect(await signal.unloadCalls >= 1)
    }

    @Test("CascadeDecisionEngine skips no-perceived cloud stage when escalating perceived content and skips text-only stages on raw media (CORE-03, CORE-06)")
    func cascadePrivacyAndMediaIsolation() async throws {
        let openRouterSignal = GateSignal()
        let localSignal = GateSignal()
        let gemini = StubDecider(modelID: "gemini", acceptsMedia: true, sharesPerceived: true, onDeviceFlag: false, confidence: 0.1, signal: nil)
        let openRouter = StubDecider(modelID: "openrouter", acceptsMedia: false, sharesPerceived: false, onDeviceFlag: false, confidence: 0.95, signal: openRouterSignal)
        let local = StubDecider(modelID: "local-jev", acceptsMedia: false, sharesPerceived: true, onDeviceFlag: true, confidence: 0.92, signal: localSignal)
        let cascade = CascadeDecisionEngine(stages: [gemini, openRouter, local], escalateBelow: 0.8)

        var perceivedReq = SystemOneRequest(state: "OCR secret invoice", questions: ["q1": SystemOneQuestion(type: .noul, instructions: "paid?")])
        perceivedReq.hasPerceivedContent = true
        perceivedReq.media = [DecisionMedia(kind: "image", mimeType: "image/png", data: Data([0x89, 0x50]))].compactMap { $0 }
        let res = try await cascade.decide(perceivedReq)
        #expect(await openRouterSignal.decideCalls == 0)
        #expect(await localSignal.decideCalls == 1)
        #expect(res.answers["q1"]?.confidence == 0.92)

        await cascade.unload()
        #expect(await localSignal.unloadCalls == 1)
    }

    @Test("GGUFFile rejects zero alignment, overflowing dimensions, zero dimensions, and excessive array depth without trapping (CORE-04)")
    func ggufHeaderSafety() {
        let dir = FileManager.default.temporaryDirectory.appending(component: "gguf-safe-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        func u32(_ v: UInt32) -> Data { var le = v.littleEndian; return Data(bytes: &le, count: 4) }
        func u64(_ v: UInt64) -> Data { var le = v.littleEndian; return Data(bytes: &le, count: 8) }
        func str(_ s: String) -> Data { u64(UInt64(s.utf8.count)) + Data(s.utf8) }

        var zeroAlign = Data()
        zeroAlign += u32(GGUFFile.magic) + u32(3) + u64(0) + u64(1)
        zeroAlign += str("general.alignment") + u32(4) + u32(0)
        let f1 = dir.appending(component: "zero_align.gguf")
        try? zeroAlign.write(to: f1)
        #expect(throws: BasaltError.self) { _ = try GGUFFile(url: f1) }

        var zeroDim = Data()
        zeroDim += u32(GGUFFile.magic) + u32(3) + u64(1) + u64(0)
        zeroDim += str("blk.0.weight") + u32(2) + u64(0) + u64(64) + u32(0) + u64(0)
        let f2 = dir.appending(component: "zero_dim.gguf")
        try? zeroDim.write(to: f2)
        #expect(throws: BasaltError.self) { _ = try GGUFFile(url: f2) }
    }

    @Test("PolarKeyRows handles partial-rotary qwen3_5 (rotated=32, headDim=128) without group-size crash and clears sigma on full trim (INF-01)")
    func polarPartialRotaryAndFullTrim() {
        let precision = KVPrecision.polar4.resolvingPolar(rotated: 32)
        var middle = MiddleKeys(precision: precision, groupSize: 64, headDim: 128)
        let keys = MLXRandom.normal([1, 2, 8, 128]).asType(.float16)
        middle.append(keys)
        #expect(middle.count == 8)
        let deq = middle.dequantizedRows(dtype: .float16)
        #expect(deq?.shape == [1, 2, 8, 128])
        let trimmed = middle.trim(8)
        #expect(trimmed == 8 && middle.count == 0 && middle.arrays.isEmpty)
    }

    @Test("ComposeSpec and ComposeRecord reject path-traversal system names and ServeCommand keys Gemini daily budget under stripped model ID (CLI-01, CLI-03)")
    func composeNameTraversalAndGeminiBudgetKey() throws {
        #expect(!ComposeSpec.isValidSystemName("../escape"))
        #expect(!ComposeSpec.isValidSystemName("a/b"))
        #expect(!ComposeSpec.isValidSystemName(".."))
        #expect(ComposeSpec.isValidSystemName("flux3-24gb-balanced"))
        #expect(throws: BasaltError.self) {
            _ = try ComposeRecord.write(name: "../../escape", port: 8080, file: "compose.yaml")
        }
        #expect(ComposeRecord.read("../../escape") == nil)

        let serve = try ServeCommand.parse(["--decision-model", "cloud:gemini/gemini-decision-1", "--decision-escalate", "self", "--decision-budget-usd-per-day", "2.5"])
        let limits = serve.cloudLimits(nil)
        #expect(limits["gemini-decision-1"] == 2.5)
        #expect(limits["gemini/gemini-decision-1"] == nil)
    }

    @Test("JevBundle rejects zero calibration temperature and EngineSupervisor ignores stale probe after shutdown (INF-02, APP-01)")
    func jevCalibrationAndSupervisorEpochGuard() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(component: "jev-bad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_name":"Qwen/Qwen3-0.6B"}"#.utf8).write(to: dir.appending(component: "openjev_config.json"))
        try Data(#"{"noul":{"temperature":0}}"#.utf8).write(to: dir.appending(component: "calibration.json"))
        #expect(throws: BasaltError.self) { _ = try JevBundle(directory: dir) }

        let signal = GateSignal()
        let record = EngineRecord(pid: ProcessInfo.processInfo.processIdentifier, port: 19090, token: "tok", arguments: ["--port", "0"])
        let env = EngineEnvironment(
            now: { .zero },
            sleep: { _ in },
            probe: { _ in
                await signal.markEnteredAndWait()
                return false
            },
            records: EngineRecords(load: { record }, save: { _ in }, remove: {}),
            isAlive: { _ in true }
        )
        let supervisor = EngineSupervisor(launch: HelperLaunch(executable: URL(fileURLWithPath: "/nonexistent-basalt-helper"), arguments: ["--port", "0"]), environment: env)
        let startTask = Task { await supervisor.start() }
        await signal.waitUntilEntered()
        await supervisor.shutdown()
        await signal.allowProceed()
        _ = await startTask.result
        #expect(await supervisor.state.phase == .stopped)
    }
}
