import Foundation
import Testing
import ArgumentParser
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltInference

@Suite("Expert LRU and the Streaming fit tier behind --experimental-expert-streaming (#225)")
struct ExpertLRUTests {
    static func slots(_ module: Int, _ experts: Int...) -> Set<ExpertSlot> {
        Set(experts.map { ExpertSlot(module: module, expert: $0) })
    }

    @Test("A repeat is a hit, a new expert a miss, and the least recently used one is evicted first")
    func leastRecentlyUsedGoesFirst() {
        var lru = ExpertLRU(capacity: ByteCount(30))
        #expect(lru.admit(Self.slots(0, 1, 2), bytes: ByteCount(10)).missing.count == 2)
        #expect(lru.admit(Self.slots(0, 3), bytes: ByteCount(10)).evicted.isEmpty)
        #expect(lru.admit(Self.slots(0, 1), bytes: ByteCount(10)).missing.isEmpty)
        let step = lru.admit(Self.slots(1, 1), bytes: ByteCount(10))
        #expect(step.missing == [ExpertSlot(module: 1, expert: 1)])
        #expect(step.evicted.count == 1)
        #expect(lru.holds(ExpertSlot(module: 0, expert: 1)))
        #expect(lru.holds(ExpertSlot(module: 0, expert: 3)))
        #expect(lru.resident == ByteCount(30))
        #expect(lru.stats.hits == 1)
        #expect(lru.stats.misses == 4)
        #expect(lru.stats.evictions == 1)
        #expect(lru.stats.bytesRead == ByteCount(40))
        #expect(abs(lru.stats.hitRate - 0.2) < 1e-9)
    }

    @Test("A call never evicts the experts it routes to, even when they alone exceed the capacity")
    func currentCallStaysResident() {
        var lru = ExpertLRU(capacity: ByteCount(5))
        _ = lru.admit(Self.slots(0, 1), bytes: ByteCount(10))
        let step = lru.admit(Self.slots(0, 2, 3), bytes: ByteCount(10))
        #expect(step.evicted == [ExpertSlot(module: 0, expert: 1)])
        #expect(lru.holds(ExpertSlot(module: 0, expert: 2)) && lru.holds(ExpertSlot(module: 0, expert: 3)))
        #expect(lru.resident == ByteCount(20))
    }

    @Test("The expert cache is the 25% placeholder of the expert weights unless --expert-cache-gb sets it")
    func cacheSize() throws {
        let weights = ByteCount(35_000)
        let share = try #require(ExpertStreamingFootprint(weights: weights, totalParameters: 35, activeParameters: 3, kv: .zero))
        #expect(share.expertCache == ByteCount(8_000))
        let fixed = try #require(ExpertStreamingFootprint(
            weights: weights, totalParameters: 35, activeParameters: 3, kv: .zero, streaming: ExpertStreaming(cacheBytes: ByteCount(1_000))))
        #expect(fixed.expertCache == ByteCount(1_000))
        #expect(fixed.required == ByteCount(4_000))
    }

    @Test("--expert-cache-gb sets the cache in bytes and is refused without --experimental-expert-streaming or when not positive")
    func options() async throws {
        #expect(try ExpertStreamingOptions.parse([]).setting == nil)
        #expect(try ExpertStreamingOptions.parse(["--experimental-expert-streaming"]).setting == ExpertStreaming())
        #expect(try ExpertStreamingOptions.parse(["--experimental-expert-streaming", "--expert-cache-gb", "2"]).setting == ExpertStreaming(cacheBytes: ByteCount(2 << 30)))
        #expect(throws: (any Error).self) { try ExpertStreamingOptions.parse(["--expert-cache-gb", "2"]) }
        #expect(throws: (any Error).self) { try ExpertStreamingOptions.parse(["--experimental-expert-streaming", "--expert-cache-gb", "0"]) }
        let engine = MLXInferenceEngine()
        await engine.streamExperts(ExpertStreaming())
        #expect(await engine.expertCacheStats() == nil)
    }

    @Test("models check and serve --dry-run admit a Streaming-tier MoE only with --experimental-expert-streaming, sizing the context after resident weights and cache")
    func admission() async throws {
        let mac24 = SystemProfiler.SystemResources(
            totalRAM: 24 << 30, availableRAM: 16 << 30, cpuCores: 12, gpuName: "Apple M4 Pro",
            gpuMemory: 18 << 30, diskSpace: 0, osVersion: "26", chipModel: "M4 Pro")
        let moe = ModelDefinition(id: "org/Sparse-35B-A3B-8bit", name: "Sparse", family: .qwen, task: .chat, modality: .text,
                                  spec: .init(ramMB: 37_000, contextWindow: 131072, quantization: "8bit", quality: 0.9))
        let refused = await MemoryEvaluator().evaluate(definition: moe, resources: mac24, contextLength: 4096)
        #expect(refused.fitLevel == .moeTieredStreaming)
        #expect(!refused.fitsInMemory)
        #expect(refused.maxContextBudgetTokens == 0)
        let admitted = await MemoryEvaluator(expertStreaming: ExpertStreaming(cacheBytes: ByteCount(2 << 30)))
            .evaluate(definition: moe, resources: mac24, contextLength: 4096)
        #expect(admitted.fitLevel == .moeTieredStreaming)
        #expect(admitted.fitsInMemory)
        #expect(admitted.warning == nil)
        #expect(admitted.expertStreaming?.expertCache == ByteCount(2 << 30))
        #expect(admitted.maxContextBudgetTokens > 0)
        #expect(admitted.reportLines(kvZones: false, kvEvict: nil, fits: "ok", exceeds: "no").contains { $0.contains("Loads with experimental expert streaming") })
    }
}

@Suite("A tiny MoE loaded with experts streamed from its safetensors matches the resident model (#225)", .serialized)
struct ExpertStreamingParityTests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/ReferenceLogits")
    static let prompt = (0 ..< 48).map { 1 + ($0 * 37 + 11) % 250 }

    static func outputs(_ container: ModelContainer) async throws -> (logits: [Float], tokens: [Int]) {
        try await container.perform { (context: ModelContext) throws -> ([Float], [Int]) in
            let inputs = MLXArray(prompt.map(Int32.init)).reshaped(1, -1)
            let logits = context.model(inputs, cache: try context.model.newCache(parameters: nil)).asType(.float32)
            return (logits.flattened().asArray(Float.self), try LookupFixture.baseline(context.model, prompt: prompt, maxTokens: 16))
        }
    }

    static func streamed(_ directory: URL, cache: ByteCount) async throws -> ModelContainer {
        try await ExpertStreamingLoader.loadContainer(
            from: directory, streaming: ExpertStreaming(cacheBytes: cache), tokenizerLoader: ByteTokenizerLoader())
    }

    static func quantized(_ family: String) async throws -> URL {
        let source = fixtures.appendingPathComponent(family)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("expert-streaming-\(family)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as? [String: Any])
        config["quantization"] = ["group_size": 32, "bits": 4]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        let container = try await loadModelContainer(from: source, using: ByteTokenizerLoader())
        try await container.perform { context in
            quantize(model: context.model, groupSize: 32, bits: 4)
            try save(arrays: Dictionary(uniqueKeysWithValues: context.model.parameters().flattened()), url: directory.appendingPathComponent("model.safetensors"))
        }
        return directory
    }

    @Test("Logits are bitwise equal and 16 greedy tokens identical with a cache that holds every expert, half of them, or none beyond the current call",
          arguments: ["lfm2_moe", "gemma4_moe", "qwen3_moe"], [false, true])
    func parity(family: String, quantized: Bool) async throws {
        let directory = quantized ? try await Self.quantized(family) : Self.fixtures.appendingPathComponent(family)
        defer { if quantized { try? FileManager.default.removeItem(at: directory) } }
        let resident = try await Self.outputs(try await loadModelContainer(from: directory, using: ByteTokenizerLoader()))
        let full = try await Self.streamed(directory, cache: ByteCount(1 << 30))
        let total = try #require(await full.perform { ExpertStreamingLoader.store(in: $0.model)?.expertBytes })
        let quantizedLayouts = await full.perform { ExpertStreamingLoader.store(in: $0.model)?.layouts.map { $0.quantization != nil } }
        #expect(quantizedLayouts?.isEmpty == false && quantizedLayouts?.allSatisfy { $0 == quantized } == true)
        for cache in [ByteCount(1 << 30), total / 2, ByteCount(1)] {
            let container = cache == ByteCount(1 << 30) ? full : try await Self.streamed(directory, cache: cache)
            let streamed = try await Self.outputs(container)
            #expect(streamed.tokens == resident.tokens, "\(family) quantized \(quantized) cache \(cache)")
            #expect(streamed.logits == resident.logits, "\(family) quantized \(quantized) cache \(cache)")
            let stats = try #require(await container.perform { ExpertStreamingLoader.store(in: $0.model)?.stats })
            #expect(stats.misses > 0)
            #expect(cache == ByteCount(1) ? stats.hits == 0 : stats.hits > 0)
            #expect(cache == ByteCount(1 << 30) ? stats.evictions == 0 : stats.evictions > 0)
        }
    }

    @Test("The streamed model holds no expert weights: each expert projection keeps a one-element placeholder")
    func expertsStayOnDisk() async throws {
        let container = try await Self.streamed(Self.fixtures.appendingPathComponent("qwen3_moe"), cache: ByteCount(1))
        let shapes = await container.perform { context in
            context.model.parameters().flattened().filter { $0.0.contains("switch_mlp") }.map(\.1.shape)
        }
        #expect(!shapes.isEmpty)
        #expect(shapes.allSatisfy { $0 == [1] })
    }

    @Test("A Hugging Face snapshot whose shards are relative symlinks into blobs/ streams like the files themselves")
    func symlinkedSnapshot() async throws {
        let source = Self.fixtures.appendingPathComponent("lfm2_moe")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("expert-streaming-hub-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let blobs = root.appendingPathComponent("blobs"), snapshot = root.appendingPathComponent("snapshots/rev")
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        for (index, name) in ["config.json", "model.safetensors"].enumerated() {
            try FileManager.default.copyItem(at: source.appendingPathComponent(name), to: blobs.appendingPathComponent("blob\(index)"))
            try FileManager.default.createSymbolicLink(atPath: snapshot.appendingPathComponent(name).path, withDestinationPath: "../../blobs/blob\(index)")
        }
        let resident = try await Self.outputs(try await loadModelContainer(from: source, using: ByteTokenizerLoader()))
        let streamed = try await Self.outputs(try await Self.streamed(snapshot, cache: ByteCount(1)))
        #expect(streamed.tokens == resident.tokens)
        #expect(streamed.logits == resident.logits)
    }

    @Test("The compacted expert tensor runs the quantized kernel the full one would: MLX picks gather_qmm_rhs by assignments per expert, so a sorted call over few of many experts passes sortedIndices only when the resident call would take that kernel")
    func residentKernelChoice() throws {
        let (x, weights, used) = withRandomState(MLXRandom.RandomState(seed: 7)) {
            (MLXRandom.normal([64, 1, 64]).asType(.bfloat16), MLXRandom.normal([32, 64, 64]).asType(.bfloat16), [3, 9, 17])
        }
        let (packed, scales, biases) = MLX.quantized(weights, groupSize: 32, bits: 4)
        let routed = (0 ..< 64).map { used[$0 * used.count / 64] }
        func multiply(_ w: MLXArray, _ s: MLXArray, _ b: MLXArray?, _ indices: [Int], sorted: Bool) -> MLXArray {
            MLX.gatherQuantizedMM(
                x, w, scales: s, biases: b, rhsIndices: MLXArray(indices.map(UInt32.init)), transpose: true,
                groupSize: 32, bits: 4, sortedIndices: sorted)
        }
        let resident = multiply(packed, scales, biases, routed, sorted: true)
        let pick = MLXArray(used.map(Int32.init))
        let local = routed.map { used.firstIndex(of: $0)! }
        let compact = { (sorted: Bool) in multiply(packed[pick], scales[pick], biases.map { $0[pick] }, local, sorted: sorted) }
        #expect(!StreamedSwitchLinear.residentKernel(sorted: true, assignments: 64, experts: 32))
        #expect(StreamedSwitchLinear.residentKernel(sorted: true, assignments: 128, experts: 32))
        #expect(!StreamedSwitchLinear.residentKernel(sorted: false, assignments: 128, experts: 32))
        #expect(MLX.arrayEqual(compact(StreamedSwitchLinear.residentKernel(sorted: true, assignments: 64, experts: 32)), resident).item(Bool.self))
        #expect(!MLX.arrayEqual(compact(true), resident).item(Bool.self))
    }

    @Test("A dense model is refused: there are no routed experts to stream")
    func denseRefused() async throws {
        await #expect(throws: BasaltError.self) {
            _ = try await Self.streamed(Self.fixtures.appendingPathComponent("qwen3"), cache: ByteCount(1))
        }
    }
}
