import Foundation
import Testing
@testable import BasaltCore

@Suite("Serving SLO tail latency quantiles, ring-buffer eviction and throughput math (#855)")
struct ServingSLOTelemetryTests {
    @Test("Nearest-rank quantiles match hand-computed P50, P95 and P99 on unsorted 100-sample and edge inputs")
    func deterministicQuantiles() {
        var reducer = ServingSLOReducer()
        for step in 0 ..< 100 {
            let value = ((step * 37) % 100) + 1
            let record = ServingSLORecord(
                queueWaitSeconds: Double(value) * 0.001,
                ttftSeconds: Double(value) * 0.01,
                prefillTokens: 100,
                prefillSeconds: 0.1,
                decodeTokens: 10,
                decodeSeconds: 0.2
            ).withDecodeDetail(itlSamplesSeconds: [Double(value) * 0.001], activeWeightBytes: 4_000_000_000)
            reducer.record(record)
        }
        let snap = reducer.snapshot(peakBandwidthGBps: 273.0)
        #expect(snap.sampleCount == 100)
        #expect(snap.itlSampleCount == 100)
        #expect(abs(snap.ttftP50Seconds - 0.50) < 1e-12)
        #expect(abs(snap.ttftP95Seconds - 0.95) < 1e-12)
        #expect(abs(snap.ttftP99Seconds - 0.99) < 1e-12)
        #expect(abs(snap.itlP50Seconds - 0.050) < 1e-12)
        #expect(abs(snap.itlP95Seconds - 0.095) < 1e-12)
        #expect(abs(snap.itlP99Seconds - 0.099) < 1e-12)
        #expect(abs(snap.queueWaitP50Seconds - 0.050) < 1e-12)
        #expect(abs(snap.queueWaitP95Seconds - 0.095) < 1e-12)
        #expect(ServingSLOReducer.nearestRankQuantile([], quantile: 0.5) == 0)
        #expect(ServingSLOReducer.nearestRankQuantile([0.42], quantile: 0.99) == 0.42)
        #expect(ServingSLOReducer.nearestRankQuantile([1.0, 5.0, 9.0], quantile: 0.0) == 1.0)
        #expect(ServingSLOReducer.nearestRankQuantile([1.0, 5.0, 9.0], quantile: 1.5) == 9.0)
        #expect(ServingSLOReducer.nearestRankQuantile([1.0, 5.0], quantile: .nan) == 0)
    }

    @Test("Ring buffers evict oldest requests and ITL samples past capacity and reset cleanly")
    func ringBufferEvictionAndReset() {
        var small = ServingSLOReducer(requestCapacity: 4, itlCapacity: 6)
        for index in 1 ... 10 {
            let v = Double(index)
            let record = ServingSLORecord(
                queueWaitSeconds: v, ttftSeconds: v, prefillTokens: 10, prefillSeconds: 0.1,
                decodeTokens: 2, decodeSeconds: 0.1
            ).withDecodeDetail(itlSamplesSeconds: [v, v + 0.5], activeWeightBytes: 1_000_000_000)
            small.record(record)
        }
        let snap = small.snapshot(peakBandwidthGBps: 200.0)
        #expect(small.requestCount == 4)
        #expect(small.itlSampleCount == 6)
        #expect(snap.sampleCount == 4)
        #expect(snap.itlSampleCount == 6)
        #expect(snap.ttftP50Seconds == 8.0)
        #expect(snap.ttftP95Seconds == 10.0)
        #expect(snap.ttftP99Seconds == 10.0)
        #expect(snap.itlP50Seconds == 9.0)
        #expect(snap.itlP95Seconds == 10.5)
        #expect(snap.itlP99Seconds == 10.5)

        var fresh = ServingSLOReducer(requestCapacity: 4, itlCapacity: 6)
        for index in 7 ... 10 {
            let v = Double(index)
            let itls: [Double] = index == 7 ? [] : [v, v + 0.5]
            fresh.record(ServingSLORecord(
                queueWaitSeconds: v, ttftSeconds: v, prefillTokens: 10, prefillSeconds: 0.1,
                decodeTokens: 2, decodeSeconds: 0.1
            ).withDecodeDetail(itlSamplesSeconds: itls, activeWeightBytes: 1_000_000_000))
        }
        #expect(small == fresh)

        small.reset()
        #expect(small.isEmpty)
        #expect(small.requestCount == 0)
        #expect(small.itlSampleCount == 0)
        let afterReset = small.snapshot(peakBandwidthGBps: 200.0)
        #expect(afterReset.sampleCount == 0)
        #expect(afterReset.regime == .idle)
        #expect(afterReset.peakMemoryBandwidthGBps == 200.0)
    }

    @Test("Default ring buffer capacities enforce 512 requests and 4096 ITL samples")
    func defaultRingBufferCaps() {
        var bounded = ServingSLOReducer()
        let tenSamples = Array(repeating: 0.01, count: 10)
        for step in 0 ..< 600 {
            bounded.record(ServingSLORecord(ttftSeconds: Double(step)).withDecodeDetail(itlSamplesSeconds: tenSamples))
        }
        #expect(bounded.requestCount == ServingSLOReducer.defaultRequestCapacity)
        #expect(bounded.itlSampleCount == ServingSLOReducer.defaultITLCapacity)
    }

    @Test("Throughput, TPOT and unified memory bandwidth follow exact serving math")
    func bandwidthAndThroughputMath() {
        var reducer = ServingSLOReducer()
        let record = ServingSLORecord(
            queueWaitSeconds: 0.015,
            ttftSeconds: 0.5,
            prefillTokens: 1000,
            prefillSeconds: 0.5,
            decodeTokens: 100,
            decodeSeconds: 2.0
        ).withDecodeDetail(itlSamplesSeconds: [0.018, 0.020, 0.022], activeWeightBytes: 4_000_000_000)
        reducer.record(record)
        let snap = reducer.snapshot(peakBandwidthGBps: 273.0)
        #expect(abs(snap.prefillTokensPerSecond - 2000.0) < 1e-12)
        #expect(abs(snap.decodeTokensPerSecond - 50.0) < 1e-12)
        #expect(abs(snap.tpotMeanSeconds - 0.02) < 1e-12)
        #expect(abs(snap.achievedMemoryBandwidthGBps - 200.0) < 1e-12)
        #expect(abs(snap.peakMemoryBandwidthGBps - 273.0) < 1e-12)
        #expect(abs(snap.memoryBandwidthUtilization - (200.0 / 273.0)) < 1e-12)
        #expect(snap.regime == .memoryBound)
        #expect(abs(AppleSiliconBandwidthTable.achievedBandwidthGBps(activeWeightBytes: 4_000_000_000, decodeTokensPerSecond: 50.0) - 200.0) < 1e-12)
        #expect(AppleSiliconBandwidthTable.achievedBandwidthGBps(activeWeightBytes: 0, decodeTokensPerSecond: 50.0) == 0)
        #expect(AppleSiliconBandwidthTable.achievedBandwidthGBps(activeWeightBytes: 4_000_000_000, decodeTokensPerSecond: .nan) == 0)
        #expect(AppleSiliconBandwidthTable.utilization(achievedGBps: 100, peakGBps: 0) == 0)
        #expect(AppleSiliconBandwidthTable.utilization(achievedGBps: -5, peakGBps: 200) == 0)
    }
}

@Suite("Serving SLO non-finite safety and Codable backward compatibility (#855)")
struct ServingSLOSafetyAndCodableTests {
    @Test("Empty, zero, negative, NaN, infinite and subnormal inputs stay finite and safe")
    func zeroAndNonFiniteSafety() {
        var reducer = ServingSLOReducer(requestCapacity: 0, itlCapacity: -2)
        #expect(reducer.requestCapacity == 1)
        #expect(reducer.itlCapacity == 1)
        for invalidPeak in [0.0, -50.0, .nan, .infinity, -.infinity] {
            let empty = reducer.snapshot(peakBandwidthGBps: invalidPeak)
            #expect(empty.sampleCount == 0)
            #expect(empty.peakMemoryBandwidthGBps == 0)
            #expect(empty.memoryBandwidthUtilization == 0)
            #expect(empty.regime == .idle)
        }
        let dirty = ServingSLORecord(
            queueWaitSeconds: .nan,
            ttftSeconds: -.infinity,
            prefillTokens: -10,
            prefillSeconds: 0,
            decodeTokens: 10,
            decodeSeconds: Double.leastNonzeroMagnitude
        ).withDecodeDetail(itlSamplesSeconds: [.nan, .infinity, -.infinity, -0.5, 0.025], activeWeightBytes: 4_000_000_000)
        reducer.record(dirty)
        let snap = reducer.snapshot(peakBandwidthGBps: 273.0)
        #expect(snap.sampleCount == 1)
        #expect(snap.itlSampleCount == 1)
        #expect(snap.ttftP50Seconds == 0)
        #expect(snap.queueWaitP50Seconds == 0)
        #expect(snap.itlP50Seconds == 0.025)
        #expect(snap.itlP95Seconds == 0.025)
        #expect(snap.itlP99Seconds == 0.025)
        #expect(snap.prefillTokensPerSecond == 0)
        #expect(snap.decodeTokensPerSecond == 0)
        #expect(snap.tpotMeanSeconds.isFinite)
        #expect(snap.achievedMemoryBandwidthGBps == 0)
        #expect(snap.memoryBandwidthUtilization == 0)
        #expect(snap.regime == .idle)
    }

    @Test("ServingSLOSnapshot round-trips through JSON and decodes sparse legacy payloads with safe defaults")
    func codableRoundTripAndBackwardCompat() throws {
        var original = ServingSLOSnapshot()
        original.sampleCount = 12
        original.itlSampleCount = 240
        original.ttftP50Seconds = 0.15
        original.ttftP95Seconds = 0.40
        original.ttftP99Seconds = 0.65
        original.itlP50Seconds = 0.018
        original.itlP95Seconds = 0.029
        original.itlP99Seconds = 0.041
        original.tpotMeanSeconds = 0.020
        original.prefillTokensPerSecond = 1800
        original.decodeTokensPerSecond = 50
        original.queueWaitP50Seconds = 0.005
        original.queueWaitP95Seconds = 0.030
        original.achievedMemoryBandwidthGBps = 200
        original.peakMemoryBandwidthGBps = 273
        original.memoryBandwidthUtilization = 200.0 / 273.0
        original.regime = .memoryBound
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ServingSLOSnapshot.self, from: encoded)
        #expect(decoded == original)
        let raw = try #require(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(raw["sample_count"] as? Int == 12)
        #expect(raw["regime"] as? String == "memory_bound")

        let emptyDecoded = try JSONDecoder().decode(ServingSLOSnapshot.self, from: Data("{}".utf8))
        #expect(emptyDecoded == ServingSLOSnapshot())
        let sparse = Data(#"{"sample_count":-3,"ttft_p99_seconds":0.45,"itl_p50_seconds":-1.0,"regime":"compute_bound"}"#.utf8)
        let partial = try JSONDecoder().decode(ServingSLOSnapshot.self, from: sparse)
        #expect(partial.sampleCount == 0)
        #expect(partial.ttftP99Seconds == 0.45)
        #expect(partial.itlP50Seconds == 0)
        #expect(partial.regime == .computeBound)
    }
}

@Suite("Apple Silicon bandwidth table, compute regime classification and OpenMetrics SLO export (#855)")
struct ServingSLOBandwidthAndExportTests {
    struct RegimeCase: Sendable {
        let achieved: Double
        let peak: Double
        let prefill: Double
        let decode: Double
        let expected: ServingComputeRegime
    }

    @Test("AppleSiliconBandwidthTable maps M1 through M5 chips and binned Max variants with safe fallback", arguments: [
        ("Apple M1", nil, 68.0), ("Apple M1 Pro", nil, 200.0), ("Apple M1 Max", nil, 400.0), ("Apple M1 Ultra", nil, 800.0),
        ("Apple M2", nil, 100.0), ("Apple M2 Pro", nil, 200.0), ("Apple M2 Max", nil, 400.0), ("Apple M2 Ultra", nil, 800.0),
        ("Apple M3", nil, 100.0), ("Apple M3 Pro", nil, 150.0), ("Apple M3 Max", 40, 400.0), ("Apple M3 Max", 30, 300.0), ("Apple M3 Ultra", nil, 819.0),
        ("Apple M4", nil, 120.0), ("Apple M4 Pro", nil, 273.0), ("Apple M4 Max", 40, 546.0), ("Apple M4 Max", 32, 410.0), ("Apple M4 Ultra", nil, 1092.0),
        ("Apple M5", nil, 153.0), ("Apple M5 Pro", nil, 307.0), ("Apple M5 Max", 40, 614.0), ("Apple M5 Max", 32, 460.0), ("Apple M5 Ultra", nil, 1200.0),
        ("apple_m4_pro", nil, 273.0), ("Apple Silicon", nil, 100.0), ("Intel Core i9", nil, 100.0), ("", nil, 100.0),
    ] as [(String, Int?, Double)])
    func chipBandwidthLookup(chip: String, cores: Int?, expected: Double) {
        #expect(AppleSiliconBandwidthTable.peakBandwidthGBps(for: chip, gpuCores: cores) == expected)
    }

    @Test("Compute regime classification separates memory-bound, compute-bound and idle workloads", arguments: [
        RegimeCase(achieved: 0.0, peak: 273.0, prefill: 0.0, decode: 0.0, expected: .idle),
        RegimeCase(achieved: .nan, peak: 273.0, prefill: -5.0, decode: -.infinity, expected: .idle),
        RegimeCase(achieved: 200.0, peak: 273.0, prefill: 1200.0, decode: 50.0, expected: .memoryBound),
        RegimeCase(achieved: 136.5, peak: 273.0, prefill: 0.0, decode: 35.0, expected: .memoryBound),
        RegimeCase(achieved: 60.0, peak: 546.0, prefill: 2500.0, decode: 150.0, expected: .computeBound),
        RegimeCase(achieved: 0.0, peak: 273.0, prefill: 1800.0, decode: 0.0, expected: .computeBound),
        RegimeCase(achieved: 150.0, peak: 0.0, prefill: 0.0, decode: 40.0, expected: .memoryBound),
        RegimeCase(achieved: 0.0, peak: 273.0, prefill: 0.0, decode: 40.0, expected: .memoryBound),
    ])
    func regimeClassification(_ item: RegimeCase) {
        let actual = AppleSiliconBandwidthTable.classifyRegime(
            achievedGBps: item.achieved, peakGBps: item.peak,
            prefillTokPerSec: item.prefill, decodeTokPerSec: item.decode)
        #expect(actual == item.expected)
    }

    @Test("TelemetryExport.renderServingSLO emits OpenMetrics lines matching TelemetryExport conventions and skips non-finite values")
    func openMetricsSLOExport() {
        var reducer = ServingSLOReducer()
        reducer.record(ServingSLORecord(
            queueWaitSeconds: 0.02, ttftSeconds: 0.25, prefillTokens: 500, prefillSeconds: 0.25,
            decodeTokens: 100, decodeSeconds: 2.0
        ).withDecodeDetail(itlSamplesSeconds: [0.01, 0.02, 0.03], activeWeightBytes: 4_000_000_000))
        let slo = reducer.snapshot(peakBandwidthGBps: 200.0)
        let text = TelemetryExport.renderServingSLO(slo)
        #expect(text.hasSuffix("\n"))
        #expect(!text.contains("# EOF"))
        #expect(!text.contains("# TYPE basalt_ttft_seconds "))
        #expect(text.contains("# TYPE basalt_itl_seconds gauge\n"))
        #expect(text.contains("basalt_itl_seconds{quantile=\"0.5\"} 0.02\n"))
        #expect(text.contains("basalt_itl_seconds{quantile=\"0.95\"} 0.03\n"))
        #expect(text.contains("basalt_itl_seconds{quantile=\"0.99\"} 0.03\n"))
        #expect(text.contains("# TYPE basalt_tpot_seconds gauge\n# HELP basalt_tpot_seconds "))
        #expect(text.contains("basalt_tpot_seconds 0.02\n"))
        #expect(text.contains("basalt_prefill_tokens_per_second 2000\n"))
        #expect(text.contains("basalt_decode_tokens_per_second 50\n"))
        #expect(text.contains("basalt_queue_wait_seconds{quantile=\"0.5\"} 0.02\n"))
        #expect(text.contains("basalt_queue_wait_seconds{quantile=\"0.95\"} 0.02\n"))
        #expect(text.contains("basalt_memory_bandwidth_gbps 200\n"))
        #expect(text.contains("basalt_memory_bandwidth_peak_gbps 200\n"))
        #expect(text.contains("basalt_memory_bandwidth_utilization 1\n"))
        #expect(text.contains("basalt_slo_ttft_seconds{quantile=\"0.99\"} 0.25\n"))
        #expect(text.contains("basalt_serving_regime{regime=\"memory_bound\"} 1\n"))
        #expect(text.contains("basalt_serving_regime{regime=\"compute_bound\"} 0\n"))
        #expect(text.contains("basalt_serving_regime{regime=\"idle\"} 0\n"))

        var nonFinite = ServingSLOSnapshot()
        nonFinite.tpotMeanSeconds = .nan
        nonFinite.achievedMemoryBandwidthGBps = .infinity
        let filtered = TelemetryExport.renderServingSLO(nonFinite)
        #expect(!filtered.contains("basalt_tpot_seconds"))
        #expect(!filtered.contains("basalt_memory_bandwidth_gbps "))
    }
}
