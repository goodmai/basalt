import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private let gibBytes = ExpertArenaPlanner.gib

private func gib(_ value: Double) -> ByteCount {
    ByteCount(Int64((value * Double(gibBytes)).rounded()))
}

@Suite("ExpertArenaPlanner tier budgets, 1.5 GB stepping, prompt chunks, and overrides (#639)")
struct ExpertArenaPlannerTests {
    struct TierCase: Sendable, CustomTestStringConvertible {
        let ramGB: Int
        let expertsGB: Double
        let denseGB: Double
        let kvGB: Double
        let scratchGB: Double
        let expectedMode: ExpertArenaPlan.Mode
        let expectedChunk: Int

        var testDescription: String { "\(ramGB)GB-\(expectedMode.rawValue)-E\(Int(expertsGB))GB" }
    }

    static let tierCases: [TierCase] = [
        TierCase(ramGB: 24, expertsGB: 34.0, denseGB: 3.6, kvGB: 0.5, scratchGB: 1.5, expectedMode: .streaming, expectedChunk: 1024),
        TierCase(ramGB: 32, expertsGB: 34.0, denseGB: 3.6, kvGB: 0.8, scratchGB: 1.5, expectedMode: .streaming, expectedChunk: 1024),
        TierCase(ramGB: 48, expertsGB: 34.0, denseGB: 3.6, kvGB: 0.8, scratchGB: 1.5, expectedMode: .streaming, expectedChunk: 1024),
        TierCase(ramGB: 64, expertsGB: 34.0, denseGB: 3.6, kvGB: 0.8, scratchGB: 2.0, expectedMode: .resident, expectedChunk: 2048),
        TierCase(ramGB: 64, expertsGB: 43.0, denseGB: 4.0, kvGB: 0.9, scratchGB: 2.0, expectedMode: .streaming, expectedChunk: 1024),
        TierCase(ramGB: 96, expertsGB: 50.0, denseGB: 4.8, kvGB: 1.6, scratchGB: 2.5, expectedMode: .resident, expectedChunk: 2048),
        TierCase(ramGB: 96, expertsGB: 67.9, denseGB: 4.0, kvGB: 1.6, scratchGB: 2.5, expectedMode: .streaming, expectedChunk: 4096),
        TierCase(ramGB: 128, expertsGB: 67.9, denseGB: 4.0, kvGB: 1.6, scratchGB: 2.5, expectedMode: .resident, expectedChunk: 2048),
        TierCase(ramGB: 128, expertsGB: 98.1, denseGB: 4.0, kvGB: 3.2, scratchGB: 3.0, expectedMode: .streaming, expectedChunk: 4096),
        TierCase(ramGB: 192, expertsGB: 98.1, denseGB: 4.0, kvGB: 3.2, scratchGB: 3.0, expectedMode: .resident, expectedChunk: 2048),
        TierCase(ramGB: 192, expertsGB: 140.0, denseGB: 4.8, kvGB: 3.2, scratchGB: 3.0, expectedMode: .streaming, expectedChunk: 4096),
        TierCase(ramGB: 256, expertsGB: 128.3, denseGB: 4.8, kvGB: 6.4, scratchGB: 3.0, expectedMode: .resident, expectedChunk: 2048),
    ]

    @Test("Plans 24-256 GB tiers with macOS reserve, tier caps, 1.5 GB stepping, and prompt chunk defaults", arguments: tierCases)
    func plansAcrossAllTiers(_ spec: TierCase) {
        let total = ByteCount(Int64(spec.ramGB) * gibBytes)
        let host = ExpertArenaHost(
            totalRAMBytes: total,
            availableMemoryBytes: total,
            metalWorkingSetBytes: ByteCount(Int64(Double(total.rawValue) * 0.75))
        )
        let workload = ExpertArenaWorkload(
            denseWeightsBytes: gib(spec.denseGB),
            kvCacheBytes: gib(spec.kvGB),
            scratchOverheadBytes: gib(spec.scratchGB),
            expertBytesTotal: gib(spec.expertsGB),
            ngramRowCacheBytes: ModelFitAnalyzer.defaultNGramRowCacheBytes
        )
        let plan = ExpertArenaPlanner.plan(ExpertArenaInput(host: host, workload: workload))

        let expectedReserve = max(ByteCount(7 * gibBytes), ByteCount(Int64((Double(total.rawValue) * 0.15).rounded(.towardZero))))
        #expect(plan.macOSReserveBytes == expectedReserve)
        #expect(plan.mode == spec.expectedMode)
        #expect(plan.recommendedPromptChunkTokens == spec.expectedChunk)
        #expect(plan.fitsBudget)
        #expect(plan.projectedFootprintBytes <= plan.workingSetCapBytes)
        #expect(plan.arenaBytes <= workload.expertBytesTotal)

        if spec.ramGB == 24 {
            #expect(plan.workingSetCapBytes == ByteCount(14 * gibBytes))
        } else if spec.ramGB >= 64 {
            let expectedShare = spec.expectedMode == .resident ? 0.70 : 0.60
            #expect(plan.workingSetCapBytes == ByteCount(Int64((Double(total.rawValue) * expectedShare).rounded(.towardZero))))
        }
        if spec.expectedMode == .streaming {
            #expect(plan.arenaBytes.rawValue % ExpertArenaPlanner.stepBytes.rawValue == 0)
            #expect(plan.arenaBytes > .zero)
        } else {
            #expect(plan.arenaBytes == workload.expertBytesTotal)
        }
    }

    @Test("On a 24 GB Mac the automatic expert cache is the 25% share unless the arena planner allows less: Coder-Next 8-bit would take 19 GiB and gets 10.5, the 4-bit pack and Qwen3.6-35B 8-bit keep their share (#642 P2)",
          arguments: [(2.34, 76.5, 10.5), (1.27, 40.5, nil), (3.45, 30.6, nil)])
    func automaticCacheOnTwentyFourGB(denseGB: Double, expertsGB: Double, plannedGB: Double?) {
        let host = ExpertArenaHost(totalRAMBytes: gib(24), availableMemoryBytes: gib(20), metalWorkingSetBytes: gib(18))
        let planned = ExpertArenaPlanner.plan(ExpertArenaInput(
            host: host, workload: ExpertArenaWorkload(denseWeightsBytes: gib(denseGB), expertBytesTotal: gib(expertsGB)))).arenaBytes
        let share = ExpertStreaming().cache(forExperts: gib(expertsGB))
        #expect(ExpertStreamingLoader.automaticCache(share: share, planned: planned) == plannedGB.map(gib) ?? share)
        #expect(ExpertStreamingLoader.automaticCache(share: gib(2), planned: .zero) == gib(2))
    }

    @Test("Explicit --expert-cache-gb override wins over caps and 1.5 GB steps while clamping safely above RAM")
    func explicitOverrideWinsAndClamps() {
        let host24 = ExpertArenaHost(totalRAMBytes: gib(24), availableMemoryBytes: gib(16), metalWorkingSetBytes: gib(18))
        let workload = ExpertArenaWorkload(
            denseWeightsBytes: gib(3.6), kvCacheBytes: gib(0.5), scratchOverheadBytes: gib(1.5), expertBytesTotal: gib(34.0)
        )
        let exact = ExpertArenaPlanner.plan(ExpertArenaInput(host: host24, workload: workload, explicitOverrideGB: 10.0))
        #expect(exact.usedExplicitOverride && !exact.overrideWasClamped)
        #expect(exact.arenaBytes == gib(10.0))
        #expect(exact.mode == .streaming)

        let overRAM = ExpertArenaPlanner.plan(ExpertArenaInput(host: host24, workload: workload, explicitOverrideGB: 40.0))
        #expect(overRAM.usedExplicitOverride && overRAM.overrideWasClamped)
        #expect(overRAM.arenaBytes == gib(24.0))

        let nanIgnored = ExpertArenaPlanner.plan(ExpertArenaInput(host: host24, workload: workload, explicitOverrideGB: .nan))
        #expect(!nanIgnored.usedExplicitOverride)
        #expect(nanIgnored.arenaBytes.rawValue % ExpertArenaPlanner.stepBytes.rawValue == 0)

        let hugeOverride = ExpertArenaPlanner.plan(
            ExpertArenaInput(host: host24, workload: workload, explicitOverrideGB: Double.greatestFiniteMagnitude)
        )
        #expect(hugeOverride.usedExplicitOverride && hugeOverride.overrideWasClamped && hugeOverride.arenaBytes == gib(24.0))

        let infIgnored = ExpertArenaPlanner.plan(ExpertArenaInput(host: host24, workload: workload, explicitOverrideGB: .infinity))
        #expect(!infIgnored.usedExplicitOverride)

        let host64 = ExpertArenaHost(totalRAMBytes: gib(64), availableMemoryBytes: gib(64), metalWorkingSetBytes: gib(48))
        let residentOverride = ExpertArenaPlanner.plan(ExpertArenaInput(host: host64, workload: workload, explicitOverrideGB: 40.0))
        #expect(residentOverride.mode == .resident && residentOverride.arenaBytes == gib(34.0))
        #expect(residentOverride.overrideWasClamped && residentOverride.fitsBudget)
        #expect(residentOverride.workingSetCapBytes == ByteCount(Int64((Double(gib(64).rawValue) * 0.70).rounded(.towardZero))))

        let footprint = ExpertStreamingFootprint(
            residentWeights: gib(3.6), expertBytes: gib(34.0), kv: gib(0.5),
            host: host24, streaming: ExpertStreaming(cacheBytes: gib(6.0))
        )
        #expect(footprint?.expertCache == gib(6.0))
    }

    @Test("Handles tiny RAM, zero available memory, negative inputs, fixed footprint exceeding budget, and zero experts without underflow")
    func edgeCasesClampSafely() async {
        let tinyHost = ExpertArenaHost(totalRAMBytes: gib(4), availableMemoryBytes: gib(2), metalWorkingSetBytes: gib(3))
        let heavy = ExpertArenaWorkload(denseWeightsBytes: gib(6), kvCacheBytes: gib(1), scratchOverheadBytes: gib(1), expertBytesTotal: gib(20))
        let tinyPlan = ExpertArenaPlanner.plan(ExpertArenaInput(host: tinyHost, workload: heavy))
        #expect(tinyPlan.arenaBytes == .zero && tinyPlan.workingSetCapBytes == .zero && !tinyPlan.fitsBudget)

        let zeroAvailHost = ExpertArenaHost(totalRAMBytes: gib(64), availableMemoryBytes: .zero, metalWorkingSetBytes: gib(48))
        let zeroAvailPlan = ExpertArenaPlanner.plan(ExpertArenaInput(host: zeroAvailHost, workload: heavy))
        #expect(zeroAvailPlan.arenaBytes == .zero && zeroAvailPlan.workingSetCapBytes == .zero && zeroAvailPlan.mode == .streaming)

        let negHost = ExpertArenaHost(totalRAMBytes: ByteCount(-1), availableMemoryBytes: ByteCount(-5), metalWorkingSetBytes: ByteCount(-1))
        let negPlan = ExpertArenaPlanner.plan(ExpertArenaInput(host: negHost, workload: heavy, explicitOverrideGB: -4.0))
        #expect(negPlan.arenaBytes == .zero && !negPlan.usedExplicitOverride)

        let overAvailHost = ExpertArenaHost(totalRAMBytes: gib(24), availableMemoryBytes: gib(100), metalWorkingSetBytes: gib(18))
        let overAvailPlan = ExpertArenaPlanner.plan(ExpertArenaInput(host: overAvailHost, workload: heavy))
        #expect(overAvailPlan.workingSetCapBytes == gib(14))

        let tightHost = ExpertArenaHost(totalRAMBytes: gib(24), availableMemoryBytes: gib(16), metalWorkingSetBytes: gib(18))
        let overFixed = ExpertArenaWorkload(denseWeightsBytes: gib(12), kvCacheBytes: gib(2), scratchOverheadBytes: gib(2), expertBytesTotal: gib(34))
        let overPlan = ExpertArenaPlanner.plan(ExpertArenaInput(host: tightHost, workload: overFixed))
        #expect(overPlan.arenaBytes == .zero && !overPlan.fitsBudget)

        let denseOnly = ExpertArenaWorkload(denseWeightsBytes: gib(6), kvCacheBytes: gib(1), scratchOverheadBytes: gib(1), expertBytesTotal: .zero)
        let densePlan = ExpertArenaPlanner.plan(ExpertArenaInput(host: tightHost, workload: denseOnly))
        #expect(densePlan.mode == .resident && densePlan.arenaBytes == .zero && densePlan.recommendedPromptChunkTokens == 2048)

        let results = await withTaskGroup(of: ByteCount.self, returning: [ByteCount].self) { group in
            for _ in 0 ..< 8 {
                group.addTask { ExpertArenaPlanner.plan(ExpertArenaInput(host: tightHost, workload: heavy)).arenaBytes }
            }
            var out: [ByteCount] = []
            for await val in group { out.append(val) }
            return out
        }
        #expect(Set(results).count == 1)
    }

    @Test("ComposeMemoryTier exposes native 96 GB and 192 GB tiers and qwen4_exp tier recommendations match the memory table")
    func qwen4ExpTierRecommendationsMatchTable() {
        #expect(ComposeMemoryTier.detect(ramGB: 96) == .gb96)
        #expect(ComposeMemoryTier.detect(ramGB: 192) == .gb192)
        #expect(ComposeMemoryTier.gb96.workingSetGB == 72.0)
        #expect(ComposeMemoryTier.gb192.workingSetGB == 144.0)

        let rec24 = MemoryEvaluator.qwen4ExpRecommendation(for: .gb24)
        #expect(rec24.mode == .streaming && !rec24.mtpEnabled && !rec24.ngramResident && rec24.recommendedPromptChunkTokens == 1024)
        #expect(rec24.plannedArenaGB >= 7.5 && rec24.workingSetCapGB == 14.0)

        let rec64 = MemoryEvaluator.qwen4ExpRecommendation(for: .gb64)
        #expect(rec64.mode == .resident && rec64.mtpEnabled && rec64.recommendedPromptChunkTokens == 2048)

        let rec96 = MemoryEvaluator.qwen4ExpRecommendation(for: .gb96)
        #expect(rec96.mode == .resident && rec96.mtpEnabled && rec96.primaryPack.contains("IQ3_S"))

        let rec192 = MemoryEvaluator.qwen4ExpRecommendation(for: .gb192)
        #expect(rec192.mode == .resident && rec192.mtpEnabled && rec192.primaryPack.contains("6bit"))

        for tier in ComposeMemoryTier.allCases {
            let rec = MemoryEvaluator.qwen4ExpRecommendation(for: tier)
            #expect(rec.tier == tier && !rec.primaryPack.isEmpty && !rec.alternativePack.isEmpty)
        }
        let hostResources = SystemProfiler.SystemResources(
            totalRAM: 96 * gibBytes, availableRAM: 80 * gibBytes, cpuCores: 14,
            gpuName: "Apple M4 Max", gpuMemory: 72 * gibBytes, diskSpace: 0, osVersion: "27", chipModel: "M4 Max"
        )
        #expect(MemoryEvaluator.qwen4ExpRecommendation(for: hostResources).tier == .gb96)

        let liveHost = ExpertArenaHost.current()
        #expect(liveHost.totalRAMBytes > .zero && liveHost.availableMemoryBytes >= .zero && liveHost.availableMemoryBytes <= liveHost.totalRAMBytes)
        let livePlan = ExpertArenaPlanner.planForCurrentHost(
            workload: ExpertArenaWorkload(denseWeightsBytes: gib(4), expertBytesTotal: gib(34))
        )
        #expect(livePlan.macOSReserveBytes >= ExpertArenaPlanner.macOSMinimumReserve)
    }
}

@Suite("ExpertResidencyPolicy decayed-frequency admission, routing profile JSON, and Zipf trace replay (#639)")
struct ExpertResidencyPolicyTests {
    @Test("One-hit cold expert bypasses a full hot cache while a score >= 2.0 expert replaces a decayed victim < 0.5")
    func strataThresholdAdmissionAndVictimReplacement() {
        var policy = ExpertResidencyPolicy(
            capacity: 2,
            config: .init(decayFactor: 0.5, decayWindow: 4, admitThreshold: 2.0, victimThreshold: 0.5)
        )
        _ = policy.step([0, 1])
        _ = policy.step([0])
        _ = policy.step([0])
        #expect(policy.slot(of: 0) != nil && policy.slot(of: 1) != nil)

        let oneHit = policy.step([99])
        #expect(oneHit.bypassed == [99] && oneHit.evictions == 0)
        #expect(policy.slot(of: 99) == nil && policy.slot(of: 0) != nil && policy.slot(of: 1) != nil)

        for _ in 0 ..< 8 {
            _ = policy.step([0])
        }
        #expect(policy.decayedScore(of: 1) < 0.5)

        _ = policy.step([42])
        let admit = policy.step([42])
        #expect(policy.decayedScore(of: 42) >= 2.0)
        #expect(admit.admissions == 1 && admit.evictions == 1)
        #expect(policy.slot(of: 42) != nil && policy.slot(of: 1) == nil && policy.slot(of: 0) != nil)
    }

    @Test("Falls back to cold LRU when no victim is below 0.5 and candidate is at least as hot, without evicting hotter slots")
    func coldLRUFallbackAndHotProtection() {
        var policy = ExpertResidencyPolicy(
            capacity: 2,
            config: .init(decayFactor: 0.5, decayWindow: 64, admitThreshold: 2.0, victimThreshold: 0.5)
        )
        _ = policy.step([10])
        _ = policy.step([10])
        _ = policy.step([20])
        _ = policy.step([20])
        #expect(policy.decayedScore(of: 10) == 2.0 && policy.decayedScore(of: 20) == 2.0)

        _ = policy.step([30])
        let swapped = policy.step([30])
        #expect(swapped.admissions == 1 && swapped.evictions == 1)
        #expect(policy.slot(of: 30) != nil && policy.slot(of: 10) == nil && policy.slot(of: 20) != nil)

        for _ in 0 ..< 5 {
            _ = policy.step([20, 30])
        }
        _ = policy.step([40])
        let blocked = policy.step([40])
        #expect(blocked.bypassed == [40] && blocked.evictions == 0)
        #expect(policy.slot(of: 40) == nil)
    }

    @Test("Offline replay of a skewed Zipf MoE trace achieves strictly higher hit rate than plain LRU and improves with warm start")
    func zipfTraceReplayBeatsPlainLRU() {
        let trace = Self.makeZipfTrace(steps: 1_500, numExperts: 128, topK: 4, seed: 0x6390_BEEF)
        let capacity = 24
        let lru = ExpertResidencyPolicy.replayLRU(trace: trace, capacity: capacity)
        let strata = ExpertResidencyPolicy.replay(trace: trace, capacity: capacity)

        #expect(strata.totalLookups == lru.totalLookups)
        #expect(strata.hitRate > lru.hitRate + 0.05)
        #expect(strata.evictions < lru.evictions)

        var routing = ExpertRouting()
        for step in trace.prefix(200) { routing.record(module: 0, step) }
        let profile = ExpertRoutingProfile.from(routing: routing, modelId: "qwen4_exp-test", totalTokens: 200)
        let warmed = ExpertResidencyPolicy.replay(
            trace: Array(trace.dropFirst(200)),
            capacity: capacity,
            warmStart: profile.rankedExpertIDs(forLayer: 0, limit: capacity)
        )
        let coldTail = ExpertResidencyPolicy.replay(trace: Array(trace.dropFirst(200)), capacity: capacity)
        #expect(warmed.hitRate >= coldTail.hitRate)
    }

    @Test("ExpertRoutingProfile JSON round-trips cleanly, warms policies deterministically, and rejects malformed data")
    func routingProfileRoundTripAndValidation() throws {
        var routing = ExpertRouting()
        routing.record(module: 2, [7, 3, 7, 7, 1, 3])
        routing.record(module: 5, [12, 12, 4])
        let profile = ExpertRoutingProfile.from(routing: routing, modelId: "Qwen/Qwen3.8-Flash-Next", totalTokens: 128)
        let encoded = try profile.encodeJSON()
        let decoded = try ExpertRoutingProfile.decodeJSON(encoded)
        #expect(decoded == profile)
        #expect(decoded.rankedExpertIDs(forLayer: 2) == [7, 3, 1])
        #expect(decoded.rankedExpertIDs(forLayer: 2, limit: 2) == [7, 3])

        var policy = ExpertResidencyPolicy(capacity: 2)
        let warmed = policy.warmStart(from: decoded, layer: 2)
        #expect(warmed.loads.map(\.expert) == [7, 3])
        #expect(policy.slot(of: 7) != nil && policy.slot(of: 3) != nil && policy.slot(of: 1) == nil)

        let badVersion = Data(#"{"version":99,"model_id":"m","total_tokens":1,"layers":[]}"#.utf8)
        #expect(throws: BasaltError.self) { _ = try ExpertRoutingProfile.decodeJSON(badVersion) }

        let badEntry = Data(#"{"version":1,"model_id":"m","total_tokens":1,"layers":[{"layer":0,"ranked_experts":[{"expert":-1,"count":1,"decayed_score":1.0}]}]}"#.utf8)
        #expect(throws: BasaltError.self) { _ = try ExpertRoutingProfile.decodeJSON(badEntry) }

        let dupLayer = Data(#"{"version":1,"model_id":"m","total_tokens":1,"layers":[{"layer":0,"ranked_experts":[]},{"layer":0,"ranked_experts":[]}]}"#.utf8)
        #expect(throws: BasaltError.self) { _ = try ExpertRoutingProfile.decodeJSON(dupLayer) }

        let decodedConfig = try JSONDecoder().decode(
            ExpertResidencyPolicy.Config.self,
            from: Data(#"{"decayFactor":0,"decayWindow":0,"admitThreshold":-1,"victimThreshold":-1}"#.utf8)
        )
        #expect(decodedConfig.decayFactor == 0.01 && decodedConfig.decayWindow == 1 && decodedConfig.admitThreshold == 0 && decodedConfig.victimThreshold == 0)
    }

    @Test("Window boundary decay, slow decay stability, warmStart self-preservation, and deterministic replay hold")
    func decayBoundariesAndDeterminism() {
        var bounded = ExpertResidencyPolicy(capacity: 3, config: .init(decayFactor: 0.5, decayWindow: 4))
        for _ in 0 ..< 4 { _ = bounded.step([7]) }
        #expect(bounded.currentEpoch == 0 && bounded.decayedScore(of: 7) == 4.0)
        _ = bounded.step([8])
        #expect(bounded.currentEpoch == 1 && bounded.decayedScore(of: 7) == 2.0)

        var warm = ExpertResidencyPolicy(capacity: 3)
        _ = warm.step([20, 50, 60])
        let warmed = warm.warmStart(from: [10, 20, 30])
        #expect(warmed.loads.map(\.expert) == [10, 30] && warmed.evictions == 2)
        #expect(warm.slot(of: 10) != nil && warm.slot(of: 20) != nil && warm.slot(of: 30) != nil)

        var slow = ExpertResidencyPolicy(capacity: 2, config: .init(decayFactor: 0.999, decayWindow: 1))
        _ = slow.step([1])
        for _ in 0 ..< 1_050 { _ = slow.step([2]) }
        #expect(slow.decayedScore(of: 1) > 0.3 && slow.decayedScore(of: 1) < 0.4)

        let trace = Self.makeZipfTrace(steps: 200, numExperts: 32, topK: 3, seed: 99)
        #expect(ExpertResidencyPolicy.replay(trace: trace, capacity: 8) == ExpertResidencyPolicy.replay(trace: trace, capacity: 8))
    }

    private static func makeZipfTrace(steps: Int, numExperts: Int, topK: Int, seed: UInt64) -> [[Int]] {
        var rng = SplitMix64(seed: seed)
        var cdf: [Double] = []
        cdf.reserveCapacity(numExperts)
        var total = 0.0
        for rank in 1 ... numExperts {
            total += 1.0 / Foundation.pow(Double(rank), 1.35)
            cdf.append(total)
        }
        var trace: [[Int]] = []
        trace.reserveCapacity(steps)
        let tailStart = min(numExperts - 1, max(1, numExperts / 3))
        let tailWidth = UInt64(max(1, numExperts - tailStart))
        for step in 0 ..< steps {
            var chosen: [Int] = []
            var seen: Set<Int> = []
            while chosen.count < topK {
                let expert: Int
                if step % 9 == 0 && chosen.count == topK - 1 {
                    expert = tailStart + Int(rng.next() % tailWidth)
                } else {
                    let u = (Double(rng.next() >> 11) / Double(1 << 53)) * total
                    expert = cdf.firstIndex(where: { $0 >= u }) ?? (numExperts - 1)
                }
                if seen.insert(expert).inserted { chosen.append(expert) }
            }
            trace.append(chosen)
        }
        return trace
    }
}

@Suite("ModelFitAnalyzer qwen4_exp SSD-resident PLE exclusion and shared expert accounting (#639)")
struct Qwen4ExpModelFitTests {
    private struct SyntheticFixture {
        let directory: URL
        let pleBytes: ByteCount
        let sharedBytes: ByteCount
        let routedBytes: ByteCount
        let denseBytes: ByteCount
    }

    private struct TensorSpec {
        let name: String
        let dtype: String
        let shape: [Int]
        let byteLength: Int
    }

    private static func writeSyntheticSafetensors(_ specs: [TensorSpec], to url: URL) throws {
        var header: [String: Any] = [:]
        var cursor = 0
        for spec in specs {
            header[spec.name] = [
                "dtype": spec.dtype,
                "shape": spec.shape,
                "data_offsets": [cursor, cursor + spec.byteLength],
            ]
            cursor += spec.byteLength
        }
        let headerBytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var lengthLE = UInt64(headerBytes.count).littleEndian
        var fileData = withUnsafeBytes(of: &lengthLE) { Data($0) }
        fileData.append(headerBytes)
        fileData.append(Data(repeating: 0, count: cursor))
        try fileData.write(to: url)
    }

    private static func makeSyntheticQwen4ExpFixture() throws -> SyntheticFixture {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qwen4-exp-fit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(#"{"model_type":"qwen4_exp"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))

        let denseBytes = 64 * 64 * 2
        let sharedBytes = 128 * 64 * 2
        let routedBytes = 8 * 64 * 64 * 2
        let pleShardBytes = 512 * 160 * 2

        let shard0URL = dir.appendingPathComponent("model-00001-of-00002.safetensors")
        let shard1URL = dir.appendingPathComponent("model-00002-of-00002.safetensors")
        try writeSyntheticSafetensors([
            TensorSpec(name: "model.embed_tokens.weight", dtype: "BF16", shape: [64, 64], byteLength: denseBytes),
            TensorSpec(name: "model.layers.0.mlp.shared_expert.gate_proj.weight", dtype: "BF16", shape: [128, 64], byteLength: sharedBytes),
            TensorSpec(name: "model.layers.0.mlp.switch_mlp.gate_proj.weight", dtype: "BF16", shape: [8, 64, 64], byteLength: routedBytes),
        ], to: shard0URL)
        try writeSyntheticSafetensors([
            TensorSpec(name: "model.ple.ple_embedding.ngram_embedding.shards.0.weight", dtype: "BF16", shape: [512, 160], byteLength: pleShardBytes),
            TensorSpec(name: "model.ple.ple_embedding.ngram_embedding.shards.1.weight", dtype: "BF16", shape: [512, 160], byteLength: pleShardBytes),
        ], to: shard1URL)

        let indexJSON: [String: Any] = [
            "weight_map": [
                "model.embed_tokens.weight": shard0URL.lastPathComponent,
                "model.layers.0.mlp.shared_expert.gate_proj.weight": shard0URL.lastPathComponent,
                "model.layers.0.mlp.switch_mlp.gate_proj.weight": shard0URL.lastPathComponent,
                "model.ple.ple_embedding.ngram_embedding.shards.0.weight": shard1URL.lastPathComponent,
                "model.ple.ple_embedding.ngram_embedding.shards.1.weight": shard1URL.lastPathComponent,
            ]
        ]
        try JSONSerialization.data(withJSONObject: indexJSON).write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        return SyntheticFixture(
            directory: dir,
            pleBytes: ByteCount(Int64(pleShardBytes * 2)),
            sharedBytes: ByteCount(Int64(sharedBytes)),
            routedBytes: ByteCount(Int64(routedBytes)),
            denseBytes: ByteCount(Int64(denseBytes))
        )
    }

    @Test("Excludes ple.ple_embedding.ngram_embedding.shards.* from working-set bytes while counting shared_expert as resident")
    func excludesPLEAndKeepsSharedExpertResident() async throws {
        let fixture = try Self.makeSyntheticQwen4ExpFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let summary = try #require(ModelFitAnalyzer.stackedExperts(in: fixture.directory))
        #expect(summary.ssdResidentBytes == fixture.pleBytes)
        #expect(summary.sharedExpertBytes == fixture.sharedBytes)
        #expect(summary.expertBytes == fixture.routedBytes)
        #expect(summary.workingSetShardBytes == summary.totalShardBytes - fixture.pleBytes)
        #expect(summary.residentWeights >= fixture.sharedBytes + fixture.denseBytes)
        #expect(summary.residentWeights < summary.totalShardBytes - fixture.pleBytes)

        let residentBudget = summary.workingSetShardBytes + ByteCount(4_096)
        #expect(residentBudget < summary.totalShardBytes)
        let residentFit = ModelFitAnalyzer.classifyDirectory(fixture.directory, workingSetBudget: residentBudget)
        #expect(residentFit.fitLevel.fitsResident)

        let streamingBudget = summary.residentWeights + ByteCount(Int64(Double(fixture.routedBytes.rawValue) * 0.30))
        #expect(streamingBudget < summary.workingSetShardBytes)
        let streamingFit = ModelFitAnalyzer.classifyDirectory(fixture.directory, workingSetBudget: streamingBudget)
        #expect(streamingFit.fitLevel == .moeTieredStreaming)
        #expect(streamingFit.streaming?.residentWeights == summary.residentWeights)

        let resources = SystemProfiler.SystemResources(
            totalRAM: residentBudget.rawValue * 2,
            availableRAM: residentBudget.rawValue,
            cpuCores: 12,
            gpuName: "Apple M4 Pro",
            gpuMemory: residentBudget.rawValue,
            diskSpace: 0,
            osVersion: "27",
            chipModel: "M4 Pro"
        )
        let evaluation = await MemoryEvaluator().evaluate(directory: fixture.directory, resources: resources, contextLength: 0)
        #expect(evaluation.requiredRAMBytes < summary.totalShardBytes.rawValue)
    }
}

@Suite("NGramRowStore sharded SSD row reads, dequantization parity, LRU cache, and header stripping (#639)", .serialized)
struct NGramRowStoreTests {
    @Test("Multi-shard 4-bit/g32 affine row reads match resident QuantizedEmbedding bit-for-bit across shard boundaries and duplicates")
    func quantizedAffineMultiShardParity() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ngram-q4-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let shardRowCounts = [17, 19, 13]
        let totalRows = shardRowCounts.reduce(0, +)
        let embedDim = 160
        let rawWeight = withRandomState(MLXRandom.RandomState(seed: 42)) {
            MLXRandom.normal([totalRows, embedDim]).asType(.bfloat16)
        }
        let resident = QuantizedEmbedding(weight: rawWeight, groupSize: 32, bits: 4, mode: .affine)
        eval(resident.weight, resident.scales)
        if let biases = resident.biases { eval(biases) }

        let shard0URL = dir.appendingPathComponent("model-00001-of-00002.safetensors")
        let shard1URL = dir.appendingPathComponent("model-00002-of-00002.safetensors")
        var file0: [String: MLXArray] = ["model.norm.weight": MLXArray.ones([embedDim], dtype: .bfloat16)]
        var file1: [String: MLXArray] = [:]
        let prefix = "model.ple.ple_embedding.ngram_embedding.shards"
        var offset = 0
        for (shardIdx, rows) in shardRowCounts.enumerated() {
            let range = offset ..< (offset + rows)
            var target = shardIdx < 2 ? file0 : file1
            target["\(prefix).\(shardIdx).weight"] = resident.weight[range]
            target["\(prefix).\(shardIdx).scales"] = resident.scales[range]
            if let biases = resident.biases { target["\(prefix).\(shardIdx).biases"] = biases[range] }
            if shardIdx < 2 { file0 = target } else { file1 = target }
            offset += rows
        }
        try MLX.save(arrays: file0, url: shard0URL)
        try MLX.save(arrays: file1, url: shard1URL)

        let store = try NGramRowStore(
            shards: [shard0URL, shard1URL],
            config: .init(quantization: .affine4g32, maxCachedRows: 8, layerID: 2)
        )
        defer { store.close() }

        #expect(store.index.totalRows == totalRows)
        #expect(store.index.embedDim == embedDim)
        #expect(store.index.shards.count == 3)

        let boundaryBatch = [0, 16, 17, 35, 36, 48, 17, 0, 35]
        let actual = try store.lookup(boundaryBatch)
        let expected = resident(MLXArray(boundaryBatch.map(Int32.init)))
        eval(actual, expected)
        #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
        #expect(MLX.arrayEqual(actual, expected).item(Bool.self))

        let gridIndices = MLXArray([Int32(0), 16, 17, 35, 36, 48]).reshaped(2, 3)
        let actualGrid = try store.lookup(gridIndices)
        let expectedGrid = resident(gridIndices)
        eval(actualGrid, expectedGrid)
        #expect(actualGrid.shape == [2, 3, embedDim])
        #expect(MLX.arrayEqual(actualGrid, expectedGrid).item(Bool.self))

        let empty = try store.lookup([Int]())
        #expect(empty.shape == [0, embedDim] && empty.dtype == .bfloat16)

        let emptyGrid = try store.lookup(MLXArray([Int32](), [0, 4]))
        #expect(emptyGrid.shape == [0, 4, embedDim])
    }

    @Test("Multi-shard BF16 and FP16 row reads match resident Embedding bit-for-bit", arguments: [DType.bfloat16, DType.float16])
    func denseFloatMultiShardParity(dtype: DType) throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ngram-float-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let totalRows = 30
        let embedDim = 64
        let rawWeight = withRandomState(MLXRandom.RandomState(seed: 19)) {
            MLXRandom.normal([totalRows, embedDim]).asType(dtype)
        }
        eval(rawWeight)
        let resident = Embedding(weight: rawWeight)
        let url = dir.appendingPathComponent("model.safetensors")
        let prefix = "ple.ple_embedding.ngram_embedding.shards"
        try MLX.save(arrays: [
            "\(prefix).0.weight": rawWeight[0 ..< 11],
            "\(prefix).1.weight": rawWeight[11 ..< 30],
        ], url: url)

        let store = try NGramRowStore(shards: [url], config: .init(maxCachedRows: 0))
        defer { store.close() }

        let ids = [0, 10, 11, 29, 10]
        let actual = try store.lookup(ids)
        let expected = resident(MLXArray(ids.map(Int32.init)))
        eval(actual, expected)
        #expect(MLX.arrayEqual(actual, expected).item(Bool.self))
        #expect(store.stats.cachedRows == 0 && store.stats.evictions == 0)
    }

    @Test("LRU row cache accounts hits, misses, evictions, prefetch warming, oversized batches, out-of-range refusal, and close")
    func lruCachePrefetchAndRefusals() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ngram-lru-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let rawWeight = withRandomState(MLXRandom.RandomState(seed: 7)) {
            MLXRandom.normal([20, 64]).asType(.bfloat16)
        }
        let resident = QuantizedEmbedding(weight: rawWeight, groupSize: 32, bits: 4, mode: .affine)
        eval(resident.weight, resident.scales)
        if let biases = resident.biases { eval(biases) }

        let url = dir.appendingPathComponent("model.safetensors")
        try MLX.save(arrays: [
            "model.ple.ple_embedding.ngram_embedding.shards.0.weight": resident.weight,
            "model.ple.ple_embedding.ngram_embedding.shards.0.scales": resident.scales,
            "model.ple.ple_embedding.ngram_embedding.shards.0.biases": resident.biases!,
        ], url: url)

        let store = try NGramRowStore(shards: [url], config: .init(maxCachedRows: 4))
        defer { store.close() }

        _ = try store.lookup([0, 1, 2, 3])
        #expect(store.stats.misses == 4 && store.stats.hits == 0 && store.stats.evictions == 0 && store.stats.cachedRows == 4)

        _ = try store.lookup([0, 1])
        #expect(store.stats.hits == 2 && store.stats.misses == 4 && store.stats.evictions == 0)

        let oversized = [0, 4, 5, 6, 7, 8]
        let actualOversized = try store.lookup(oversized)
        let expectedOversized = resident(MLXArray(oversized.map(Int32.init)))
        eval(actualOversized, expectedOversized)
        #expect(MLX.arrayEqual(actualOversized, expectedOversized).item(Bool.self))
        #expect(store.stats.evictions == 5 && store.stats.cachedRows == 4)

        store.resetStats()
        try store.prefetch([10, 11, 12])
        #expect(store.stats.hits == 0 && store.stats.misses == 0)
        _ = try store.lookup([10, 11, 12])
        #expect(store.stats.hits == 3 && store.stats.misses == 0)

        let beforeStats = store.stats
        try store.prefetch([10, 11])
        #expect(store.stats.bytesRead == beforeStats.bytesRead)

        #expect(throws: BasaltError.self) { _ = try store.lookup([-1]) }
        #expect(throws: BasaltError.self) { _ = try store.lookup([20]) }
        #expect(throws: BasaltError.self) { _ = try store.lookup([1, 99]) }
        #expect(throws: BasaltError.self) { _ = try store.lookup(MLXArray([Float(1.5)])) }
        #expect(throws: BasaltError.self) { _ = try store.lookup(MLXArray([Int64(Int32.max) + 1])) }
        #expect(store.stats == beforeStats)

        store.close()
        #expect(throws: BasaltError.self) { _ = try store.lookup([0]) }
        #expect(throws: BasaltError.self) { _ = try store.lookup(MLXArray([Int32](), [0])) }
    }
}

extension NGramRowStoreTests {
    @Test("Supports weight and scales/biases split across separate safetensors files and rejects missing shard gaps")
    func splitTensorFilesAndGapRefusal() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ngram-split-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let rawWeight = withRandomState(MLXRandom.RandomState(seed: 31)) {
            MLXRandom.normal([16, 64]).asType(.bfloat16)
        }
        let resident = QuantizedEmbedding(weight: rawWeight, groupSize: 32, bits: 4, mode: .affine)
        eval(resident.weight, resident.scales)
        if let biases = resident.biases { eval(biases) }

        let file0 = dir.appendingPathComponent("model-00001-of-00002.safetensors")
        let file1 = dir.appendingPathComponent("model-00002-of-00002.safetensors")
        let prefix = "model.ple.ple_embedding.ngram_embedding.shards"
        try MLX.save(arrays: ["\(prefix).0.weight": resident.weight], url: file0)
        try MLX.save(arrays: [
            "\(prefix).0.scales": resident.scales,
            "\(prefix).0.biases": resident.biases!,
        ], url: file1)

        let store = try NGramRowStore(directory: dir, config: .init(quantization: .affine4g32, maxCachedRows: 4))
        defer { store.close() }
        let ids = [0, 7, 15]
        let actual = try store.lookup(ids)
        let expected = resident(MLXArray(ids.map(Int32.init)))
        eval(actual, expected)
        #expect(MLX.arrayEqual(actual, expected).item(Bool.self))

        let gapFile = dir.appendingPathComponent("gap.safetensors")
        try MLX.save(arrays: [
            "\(prefix).0.weight": rawWeight[0 ..< 8],
            "\(prefix).2.weight": rawWeight[8 ..< 16],
        ], url: gapFile)
        #expect(throws: BasaltError.self) { _ = try NGramRowStore(shards: [gapFile]) }
    }

    @Test("Dropping PLE keys from resident safetensors header strips all n-gram shards while keeping dense and shared tensors intact")
    func residentHeaderStripsPLEKeysOnly() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ngram-drop-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let dense = withRandomState(MLXRandom.RandomState(seed: 3)) { MLXRandom.normal([8, 16]).asType(.bfloat16) }
        let shared = withRandomState(MLXRandom.RandomState(seed: 5)) { MLXRandom.normal([8, 16]).asType(.bfloat16) }
        let pleWeight = MLXArray.zeros([16, 20], dtype: .uint32)
        let pleScales = MLXArray.ones([16, 5], dtype: .bfloat16)
        eval(dense, shared, pleWeight, pleScales)

        let sourceURL = dir.appendingPathComponent("source.safetensors")
        let strippedURL = dir.appendingPathComponent("resident.safetensors")
        try MLX.save(arrays: [
            "model.embed_tokens.weight": dense,
            "model.layers.0.mlp.shared_expert.down_proj.weight": shared,
            "model.ple.ple_embedding.ngram_embedding.shards.0.weight": pleWeight,
            "model.ple.ple_embedding.ngram_embedding.shards.0.scales": pleScales,
        ], url: sourceURL)

        let dropped = try NGramRowStore.writeResidentShardDroppingPLE(from: sourceURL, to: strippedURL)
        #expect(dropped == [
            "model.ple.ple_embedding.ngram_embedding.shards.0.weight",
            "model.ple.ple_embedding.ngram_embedding.shards.0.scales",
        ])
        #expect(try NGramRowStore.droppedResidentKeys(at: [sourceURL])[0] == dropped)
        let strippedHeader = try NGramRowStore.residentHeaderDroppingPLE(at: sourceURL)
        let headerJSON = try #require(JSONSerialization.jsonObject(with: strippedHeader) as? [String: Any])
        #expect(headerJSON["model.ple.ple_embedding.ngram_embedding.shards.0.weight"] == nil)
        #expect(headerJSON["model.embed_tokens.weight"] != nil)

        let encodedQuant = try JSONEncoder().encode(NGramQuantization.affine4g32)
        let decodedQuant = try JSONDecoder().decode(NGramQuantization.self, from: encodedQuant)
        #expect(decodedQuant == .affine4g32)

        let loaded = try MLX.loadArrays(url: strippedURL)
        #expect(Set(loaded.keys) == [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
        ])
        let loadedDense = try #require(loaded["model.embed_tokens.weight"])
        let loadedShared = try #require(loaded["model.layers.0.mlp.shared_expert.down_proj.weight"])
        eval(loadedDense, loadedShared)
        #expect(MLX.arrayEqual(loadedDense, dense).item(Bool.self))
        #expect(MLX.arrayEqual(loadedShared, shared).item(Bool.self))
    }
}

@Suite("Qwen4Exp pure-CPU concurrent TSan verification of planner, policy, fit analyzer, and FD/LRU row-store path (#639)")
struct Qwen4ExpTSanSafeCoreTests {
    @Test("Concurrent planner, residency replay, and sharded header/FD lifecycle run race-free under TSan")
    func concurrentValueCoreAndHeaderLifecycle() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ngram-tsan-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let rowCount = 12
        let embedDim = 16
        let payloadBytes = rowCount * embedDim * 4
        let headerObj: [String: Any] = [
            "model.embed_tokens.weight": ["dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]],
            "model.ple.ple_embedding.ngram_embedding.shards.0.weight": [
                "dtype": "F32", "shape": [rowCount, embedDim], "data_offsets": [64, 64 + payloadBytes],
            ],
        ]
        let headerBytes = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        var lenLE = UInt64(headerBytes.count).littleEndian
        var blob = withUnsafeBytes(of: &lenLE) { Data($0) }
        blob.append(headerBytes)
        blob.append(Data(repeating: 0, count: 64 + payloadBytes))
        let shardURL = dir.appendingPathComponent("model.safetensors")
        try blob.write(to: shardURL)

        let results = try await withThrowingTaskGroup(of: Int.self, returning: [Int].self) { group in
            for worker in 0 ..< 4 {
                group.addTask {
                    let stripped = dir.appendingPathComponent("resident-\(worker).safetensors")
                    let dropped = try NGramRowStore.writeResidentShardDroppingPLE(from: shardURL, to: stripped)
                    let store = try NGramRowStore(shards: [shardURL], config: .init(maxCachedRows: 4))
                    defer { store.close() }
                    try store.prefetch([0, 3, 7, 11, 2])
                    let host = ExpertArenaHost(totalRAMBytes: gib(64), availableMemoryBytes: gib(48), metalWorkingSetBytes: gib(48))
                    let plan = ExpertArenaPlanner.plan(ExpertArenaInput(
                        host: host,
                        workload: ExpertArenaWorkload(denseWeightsBytes: gib(4), expertBytesTotal: gib(43))
                    ))
                    let replay = ExpertResidencyPolicy.replay(trace: [[0, 1], [0, 2], [0, 1], [3, 4]], capacity: 2)
                    return dropped.count + store.stats.cachedRows + (plan.arenaBytes > .zero ? 1 : 0) + replay.hits
                }
            }
            var out: [Int] = []
            for try await value in group { out.append(value) }
            return out
        }
        #expect(results.count == 4 && Set(results).count == 1)
    }
}
