import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("X-ray telemetry phase 2: routing histograms per MoE layer and streamed expert-cache counters in the snapshot and /api/v1/metrics (#450)", .serialized)
struct ExpertTelemetryTests {
    @Test("The histogram counts routings per layer, ranks the most used experts, and gives their share")
    func histogram() {
        var routing = ExpertRouting()
        routing.record(module: 3, [1, 1, 1, 2, 5])
        routing.record(module: 0, [7, 7])
        let layers = routing.layers(top: 2)
        #expect(layers.map(\.layer) == [0, 3] && layers[1].routed == 5 && layers[1].distinct == 3)
        #expect(layers[1].top.map(\.expert) == [1, 2] && layers[1].topShare == 0.8 && layers[0].topShare == 1)
    }

    @Test("OpenMetrics carries the expert cache counters and per-layer routing gauges")
    func metrics() {
        var snapshot = ServerTelemetrySnapshot()
        var cache = ExpertCacheStats()
        (cache.hits, cache.misses) = (30, 10)
        var routing = ExpertRouting()
        routing.record(module: 4, [0, 1, 1])
        snapshot.experts = ExpertTelemetry(cache: cache, capacityBytes: 1 << 30, layers: routing.layers())
        let text = TelemetryExport.openMetrics(snapshot)
        #expect(text.contains("basalt_expert_cache_hits 30\n") && text.contains("basalt_expert_cache_misses 10\n"))
        #expect(text.contains("basalt_expert_layer_distinct{layer=\"4\"} 2") && text.contains("basalt_expert_layer_top_share{layer=\"4\"} 1"))
    }

    @Test("A streamed tiny MoE records routings for every MoE layer it ran, at least one per token")
    func streamed() async throws {
        let container = try await ExpertStreamingTests.streamed(ExpertStreamingTests.fixtures.appendingPathComponent("qwen3_moe"), cache: ByteCount(1 << 30))
        _ = try await ExpertStreamingTests.outputs(container)
        let layers = try #require(await container.perform { ExpertStreamingLoader.store(in: $0.model)?.routing.layers() })
        #expect(!layers.isEmpty && layers.allSatisfy { $0.routed > 0 && $0.distinct > 0 && $0.topShare > 0 })
    }
}
