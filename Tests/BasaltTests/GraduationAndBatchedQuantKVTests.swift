import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("Graduation of SSD MoE expert streaming, ANE vision tower cache, and SpeechAnalyzer CLI flags (#226)")
struct GraduationFlagAndCacheTests {
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/ReferenceLogits")

    @Test("ExpertStreamingLoader.canStream detects stacked SwitchLinear MoE safetensors and rejects dense checkpoints",
          arguments: [
              ("qwen3_moe", true),
              ("qwen3_5_moe", true),
              ("lfm2_moe", true),
              ("gemma4_moe", true),
              ("qwen3", false),
              ("qwen3_5", false),
              ("lfm2", false),
              ("gemma4", false),
          ])
    func canStreamStackedMoEDirectories(name: String, expected: Bool) {
        let directory = Self.fixtures.appendingPathComponent(name)
        #expect(ExpertStreamingLoader.canStream(directory) == expected)
        #expect(ModelFitAnalyzer.hasStackedExperts(in: directory) == expected)
    }

    @Test("ModelFitAnalyzer and MemoryEvaluator auto-enable SSD expert streaming on Streaming-tier stacked MoE directories unless disabled")
    func streamingTierAutoAdmissionAndOverrides() async throws {
        let moeDir = Self.fixtures.appendingPathComponent("qwen3_moe")
        let denseDir = Self.fixtures.appendingPathComponent("qwen3")
        let summary = try #require(ModelFitAnalyzer.stackedExperts(in: moeDir))
        #expect(summary.totalShardBytes > summary.expertBytes && summary.expertBytes > .zero)

        let footprint = try #require(
            ExpertStreamingFootprint(residentWeights: summary.residentWeights, expertBytes: summary.expertBytes, kv: .zero)
        )
        let tightBudget = footprint.required + ByteCount(4_096)
        #expect(tightBudget < summary.totalShardBytes)

        let classified = ModelFitAnalyzer.classifyDirectory(moeDir, workingSetBudget: tightBudget)
        #expect(classified.fitLevel == .moeTieredStreaming)
        #expect(classified.streaming == footprint)
        #expect(ModelFitAnalyzer.autoExpertStreaming(directory: moeDir, workingSetBudget: tightBudget, environment: [:]) == ExpertStreaming())

        let geometry = try #require(KVCacheGeometry.load(modelDirectory: moeDir))
        let roomyBudget = summary.totalShardBytes * 2 + geometry.bytes(tokens: TokenCount(ModelFitAnalyzer.residentContextFloor)) * 2
        #expect(ModelFitAnalyzer.classifyDirectory(moeDir, workingSetBudget: roomyBudget).fitLevel.fitsResident)
        #expect(ModelFitAnalyzer.autoExpertStreaming(directory: moeDir, workingSetBudget: roomyBudget, environment: [:]) == nil)
        #expect(ModelFitAnalyzer.autoExpertStreaming(directory: denseDir, workingSetBudget: tightBudget, environment: [:]) == nil)
        #expect(ModelFitAnalyzer.autoExpertStreaming(directory: moeDir, workingSetBudget: tightBudget, disabled: true, environment: [:]) == nil)
        let conversion = EngineConfiguration.conversion
        #expect(ModelFitAnalyzer.autoExpertStreaming(
            directory: moeDir, workingSetBudget: tightBudget, explicit: conversion.expertStreaming,
            disabled: conversion.disableExpertStreaming, environment: [:]) == nil)
        #expect(ModelFitAnalyzer.autoExpertStreaming(
            directory: moeDir, workingSetBudget: tightBudget, environment: ["BASALT_NO_EXPERT_STREAMING": "1"]) == nil)
        #expect(ModelFitAnalyzer.autoExpertStreaming(
            directory: moeDir, workingSetBudget: tightBudget, environment: ["BASALT_EXPERT_STREAMING": "0"]) == nil)

        let resources = SystemProfiler.SystemResources(
            totalRAM: tightBudget.rawValue * 2, availableRAM: tightBudget.rawValue, cpuCores: 12,
            gpuName: "Apple M4 Pro", gpuMemory: tightBudget.rawValue, diskSpace: 0, osVersion: "27", chipModel: "M4 Pro"
        )
        let autoAdmitted = await MemoryEvaluator().evaluate(directory: moeDir, resources: resources, contextLength: 0)
        #expect(autoAdmitted.fitLevel == .moeTieredStreaming)
        #expect(autoAdmitted.fitsInMemory)

        let explicitlyDisabled = await MemoryEvaluator(disableExpertStreaming: true).evaluate(
            directory: moeDir, resources: resources, contextLength: 0
        )
        #expect(explicitlyDisabled.fitLevel == .moeTieredStreaming)
        #expect(!explicitlyDisabled.fitsInMemory)

        let denseTight = await MemoryEvaluator().evaluate(directory: denseDir, resources: resources, contextLength: 0)
        #expect(denseTight.fitLevel == .tooTight)
        #expect(!denseTight.fitsInMemory)
    }

    @Test("A MoE whose weights fit but leave resident KV room for fewer than residentContextFloor tokens streams its experts instead; one token more of room keeps it resident")
    func shortResidentContextStreams() async throws {
        let moeDir = Self.fixtures.appendingPathComponent("qwen3_moe")
        let denseDir = Self.fixtures.appendingPathComponent("qwen3")
        let geometry = try #require(KVCacheGeometry.load(modelDirectory: moeDir))
        let weights = try #require(ModelFitAnalyzer.stackedExperts(in: moeDir)).totalShardBytes
        let floorRoom = geometry.bytes(tokens: TokenCount(ModelFitAnalyzer.residentContextFloor)) * 2
        let short = weights + floorRoom / 2
        let roomy = weights + floorRoom + geometry.bytes(tokens: TokenCount(64)) * 2
        #expect(ModelFitAnalyzer.residentContext(moeDir, weights: weights, budget: short) < ModelFitAnalyzer.residentContextFloor)
        #expect(MemoryEvaluator.fitLevel(required: weights, budget: short) != .tooTight)
        #expect(ModelFitAnalyzer.classifyDirectory(moeDir, workingSetBudget: short).fitLevel == .moeTieredStreaming)
        #expect(ModelFitAnalyzer.autoExpertStreaming(directory: moeDir, workingSetBudget: short, environment: [:]) == ExpertStreaming())
        #expect(ModelFitAnalyzer.classifyDirectory(moeDir, workingSetBudget: roomy).fitLevel.fitsResident)
        #expect(ModelFitAnalyzer.autoExpertStreaming(directory: moeDir, workingSetBudget: roomy, environment: [:]) == nil)
        let resources = SystemProfiler.SystemResources(
            totalRAM: 1 << 40, availableRAM: 1 << 40, cpuCores: 8, gpuName: "test", gpuMemory: short.rawValue,
            diskSpace: 1 << 40, osVersion: "", chipModel: "")
        let resident = await MemoryEvaluator(disableExpertStreaming: true).evaluate(directory: moeDir, resources: resources)
        #expect(resident.fitLevel.fitsResident)
        #expect(await MemoryEvaluator(expertStreaming: ExpertStreaming()).evaluate(directory: moeDir, resources: resources).fitLevel == .moeTieredStreaming)
        let denseWeights = ByteCount(ModelFitAnalyzer.safetensorsShards(in: denseDir).reduce(Int64(0)) {
            $0 + ((try? FileManager.default.attributesOfItem(atPath: $1.path)[.size] as? Int64) ?? 0)
        })
        let denseShort = ModelFitAnalyzer.classifyDirectory(denseDir, workingSetBudget: denseWeights + ByteCount(1_024)).fitLevel
        #expect(denseShort != .moeTieredStreaming && denseShort != .tooTight)
    }

    @Test("Graduated --speech-analyzer, --ane-vision-tower, --expert-streaming, and --no-expert-streaming flags preserve --experimental-* aliases")
    func graduatedFlagAliases() async throws {
        let bundle = FileManager.default.temporaryDirectory.appending(path: "ane-grad-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: bundle) }
        try FileManager.default.createDirectory(at: bundle.appending(component: "tower.aimodel"), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: bundle.appending(component: "metadata.json"))

        let speechGraduated = try ServeCommand.parse(["--speech-analyzer"])
        let speechAlias = try ServeCommand.parse(["--experimental-speech-analyzer"])
        #expect(speechGraduated.experimental.speechAnalyzer && speechAlias.experimental.speechAnalyzer)

        let towerGraduated = try ServeCommand.parse(["--ane-vision-tower", bundle.path])
        let towerAlias = try ServeCommand.parse(["--experimental-ane-vision-tower", bundle.path])
        #expect(towerGraduated.experimental.aneVisionTower == bundle.path)
        #expect(towerAlias.experimental.aneVisionTower == bundle.path)
        let towerEngine = try towerGraduated.makeEngine()
        #expect(await towerEngine.configuration.visionTowerBundle == bundle.path)

        let streamGraduated = try ServeCommand.parse(["--expert-streaming", "--expert-cache-gb", "1.5"])
        let streamAlias = try ServeCommand.parse(["--experimental-expert-streaming", "--expert-cache-gb", "1.5"])
        #expect(streamGraduated.expertStreaming.setting == streamAlias.expertStreaming.setting)
        #expect(streamGraduated.expertStreaming.setting == ExpertStreaming(cacheBytes: ByteCount(Int64(1.5 * 1_073_741_824))))

        let streamOff = try ServeCommand.parse(["--no-expert-streaming"])
        #expect(streamOff.expertStreaming.disabled && streamOff.expertStreaming.setting == nil)
        let streamOffEngine = try streamOff.makeEngine()
        #expect(await streamOffEngine.configuration.disableExpertStreaming)
        #expect(throws: (any Error).self) {
            _ = try ServeCommand.parse(["--expert-streaming", "--no-expert-streaming"])
        }

        let fromYAML = try ServeConfigFileTests.serve(
            "version: 1\ndefaults:\n  serve:\n    speech_analyzer: true\n    ane_vision_tower: \(bundle.path)\n    expert_streaming: { enabled: false }\n"
        )
        #expect(fromYAML.experimental.speechAnalyzer)
        #expect(fromYAML.experimental.aneVisionTower == bundle.path)
        #expect(fromYAML.expertStreaming.disabled)
    }

    @Test("CoreAIVisionTowerCache computes deterministic SHA-256 keys across bundle contents and macOS build, and round-trips entries under ~/.basalt/cache/coreai-vision/")
    func visionTowerCacheKeyDeterminism() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "coreai-cache-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let bundleA = root.appending(path: "bundle-a")
        let bundleB = root.appending(path: "bundle-b")
        let cacheDir = root.appending(path: "cache")
        let metadata: [String: Any] = ["paddle_vision_tower": CoreAIVisionTowerBundleTests.tower]
        let metaBytes = try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys])

        for dir in [bundleA, bundleB] {
            try FileManager.default.createDirectory(at: dir.appending(path: "tower.aimodel/sub"), withIntermediateDirectories: true)
            try metaBytes.write(to: dir.appending(component: "metadata.json"))
            try Data([1, 2, 3, 4, 5, 6]).write(to: dir.appending(path: "tower.aimodel/sub/weights.bin"))
        }

        #expect(CoreAIVisionTowerCache.defaultDirectory.path.hasSuffix("/.basalt/cache/coreai-vision"))
        let digestA = try CoreAIVisionTowerCache.bundleDigest(bundleDirectory: bundleA)
        let digestB = try CoreAIVisionTowerCache.bundleDigest(bundleDirectory: bundleB)
        #expect(digestA.count == 64 && digestA == digestB)

        let keyBuild1 = try CoreAIVisionTowerCache.cacheKey(bundleDirectory: bundleA, osBuild: "27A100")
        let keyBuild1Copy = try CoreAIVisionTowerCache.cacheKey(bundleDirectory: bundleB, osBuild: "27A100")
        let keyBuild2 = try CoreAIVisionTowerCache.cacheKey(bundleDirectory: bundleA, osBuild: "27B200")
        #expect(keyBuild1.count == 64 && keyBuild1 == keyBuild1Copy)
        #expect(keyBuild1 != keyBuild2)

        try Data([1, 2, 3, 4, 5, 99]).write(to: bundleB.appending(path: "tower.aimodel/sub/weights.bin"))
        let mutatedKey = try CoreAIVisionTowerCache.cacheKey(bundleDirectory: bundleB, osBuild: "27A100")
        #expect(mutatedKey != keyBuild1)

        let entry = CoreAIVisionTowerCache.Entry(
            cacheKey: keyBuild1, bundleSHA256: digestA, osBuild: "27A100",
            asset: "tower", functions: ["n1024", "n5120"], bookmarkBase64: Data("bookmark".utf8).base64EncodedString()
        )
        try CoreAIVisionTowerCache.saveEntry(entry, in: cacheDir)
        #expect(CoreAIVisionTowerCache.entryURL(forKey: keyBuild1, in: cacheDir).lastPathComponent == "\(keyBuild1).json")
        #expect(CoreAIVisionTowerCache.loadEntry(forKey: keyBuild1, in: cacheDir) == entry)
        #expect(CoreAIVisionTowerCache.loadEntry(forKey: keyBuild2, in: cacheDir) == nil)
    }
}

@Suite("BatchQuantizedKVCache multi-seat insert, decode step, compaction, and BatchedDecoder --kv-bits parity (#243)", .serialized)
struct BatchQuantizedKVCacheParityTests {
    static func prefilledQuantized(_ model: any LanguageModel, _ prompt: [Int], bits: Int) throws -> [any KVCache] {
        var cache = try model.newCache(parameters: nil)
        eval(model(BatchFixture.row(prompt), cache: cache))
        maybeQuantizeKVCache(cache: &cache, kvBits: bits, kvGroupSize: 64, quantizedKVStart: 0)
        eval(cache.flatMap { $0.innerState() })
        return cache
    }

    static func stepAndVerifyDirect(
        batch: BatchQuantizedKVCache, solos: [QuantizedKVCache], active: [Int], stepSeed: UInt64
    ) {
        let scale: Float = 0.125
        let (qs, ks, vs) = withRandomState(MLXRandom.RandomState(seed: stepSeed)) {
            (
                active.map { _ in MLXRandom.normal([1, 4, 1, 64]).asType(.float16) },
                active.map { _ in MLXRandom.normal([1, 2, 1, 64]).asType(.float16) },
                active.map { _ in MLXRandom.normal([1, 2, 1, 64]).asType(.float16) }
            )
        }
        let mask = batch.makeMask(n: 1, windowSize: nil, returnArray: false)
        let (qk, qv) = batch.updateQuantized(keys: concatenated(ks, axis: 0), values: concatenated(vs, axis: 0))
        let batchedOut = quantizedScaledDotProductAttention(
            queries: concatenated(qs, axis: 0), quantizedKeys: qk, quantizedValues: qv,
            scale: scale, mask: mask, groupSize: batch.groupSize, bits: batch.bits, mode: batch.mode
        )
        for (slot, soloIdx) in active.enumerated() {
            let (sqk, sqv) = solos[soloIdx].updateQuantized(keys: ks[slot], values: vs[slot])
            let soloOut = quantizedScaledDotProductAttention(
                queries: qs[slot], quantizedKeys: sqk, quantizedValues: sqv,
                scale: scale, mask: .none, groupSize: 64, bits: batch.bits
            )[0]
            let diff = abs(batchedOut[slot].asType(.float32) - soloOut.asType(.float32)).max().item(Float.self)
            #expect(diff <= 2e-3, "slot \(slot) soloIdx \(soloIdx) bits \(batch.bits) diff=\(diff)")
        }
    }

    @Test("Direct groupSize 64 BatchQuantizedKVCache matches solo QuantizedKVCache across unequal-length insert, step, evict, late join, and dequantized update",
          arguments: [8, 4])
    func groupSize64DirectAttentionParity(bits: Int) throws {
        let lengths = [5, 17, 8, 11]
        let scale: Float = 0.125
        let solos: [QuantizedKVCache] = try lengths.enumerated().map { idx, len in
            let simple = KVCacheSimple()
            let (k, v) = withRandomState(MLXRandom.RandomState(seed: UInt64(100 + idx))) {
                (MLXRandom.normal([1, 2, len, 64]).asType(.float16), MLXRandom.normal([1, 2, len, 64]).asType(.float16))
            }
            _ = simple.update(keys: k, values: v)
            let q = try simple.toQuantized(groupSize: 64, bits: bits)
            eval(q.innerState())
            return q
        }

        let batch = BatchQuantizedKVCache(groupSize: 64, bits: bits)
        batch.append(Array(solos.prefix(3)))
        #expect(batch.groupSize == 64 && batch.bits == bits)
        #expect(batch.leftPadding == [12, 0, 9] && batch.offset == 17)

        Self.stepAndVerifyDirect(batch: batch, solos: solos, active: [0, 1, 2], stepSeed: 201)
        batch.filter(rows: [0, 2])
        #expect(batch.leftPadding == [3, 0] && batch.offset == 9)

        batch.append([solos[3]])
        #expect(batch.leftPadding == [5, 2, 0] && batch.offset == 11)
        Self.stepAndVerifyDirect(batch: batch, solos: solos, active: [0, 2, 3], stepSeed: 202)

        batch.filter(rows: [2])
        #expect(batch.leftPadding == [0] && batch.offset == 12)
        let (q1, k1, v1) = withRandomState(MLXRandom.RandomState(seed: 203)) {
            (
                MLXRandom.normal([1, 4, 1, 64]).asType(.float16),
                MLXRandom.normal([1, 2, 1, 64]).asType(.float16),
                MLXRandom.normal([1, 2, 1, 64]).asType(.float16)
            )
        }
        let (deqK, deqV) = batch.update(keys: k1, values: v1)
        let sdpaOut = MLXFast.scaledDotProductAttention(queries: q1, keys: deqK, values: deqV, scale: scale, mask: .none)[0]
        let (sqk, sqv) = solos[3].updateQuantized(keys: k1, values: v1)
        let soloOut = quantizedScaledDotProductAttention(
            queries: q1, quantizedKeys: sqk, quantizedValues: sqv, scale: scale, mask: .none, groupSize: 64, bits: bits
        )[0]
        #expect(allClose(sdpaOut, soloOut, atol: 1e-3).item(Bool.self))
    }

    @Test("TinyModel multi-seat BatchQuantizedKVCache insert, step, evict, and late join match solo quantized KV logits and next tokens",
          arguments: [TinyModel.Family.qwen3HeadDim32, .gemma4], [8, 4])
    func tinyModelMultiSeatStepAndEvictParity(family: TinyModel.Family, bits: Int) throws {
        let model = try TinyModel.make(family)
        let prompts = BatchFixture.prompts
        let soloCaches = try prompts.map { try Self.prefilledQuantized(model, $0, bits: bits) }
        let rowCaches = try prompts.map { try Self.prefilledQuantized(model, $0, bits: bits) }

        let layers = rowCaches[0].map(BatchCore.batchLayer)
        #expect(layers.contains { $0 is BatchQuantizedKVCache })
        for (idx, layer) in layers.enumerated() {
            BatchCore.join([rowCaches[0][idx], rowCaches[1][idx], rowCaches[2][idx]], into: layer)
        }

        func verifyStep(_ active: [Int], fed: [Int]) {
            let batched = BatchKVCacheTests.batchStep(model, layers, fed)
            for (slot, rowIdx) in active.enumerated() {
                let alone = model(MLXArray([Int32(fed[slot])]).reshaped(1, 1), cache: soloCaches[rowIdx])[0, -1]
                #expect(allClose(batched[slot], alone, atol: 2e-3).item(Bool.self), "row \(rowIdx) logits differ")
                #expect(argMax(batched[slot]).item(Int.self) == argMax(alone).item(Int.self), "row \(rowIdx) token differs")
            }
        }

        verifyStep([0, 1, 2], fed: [11, 12, 13])

        for layer in layers {
            (layer as? BatchQuantizedKVCache)?.filter(rows: [0, 2])
            (layer as? BatchRotatingKVCache)?.filter(rows: [0, 2])
            (layer as? BatchKVCache)?.filter(rows: [0, 2])
        }
        let qLayer = try #require(layers.compactMap { $0 as? BatchQuantizedKVCache }.first)
        #expect(qLayer.leftPadding == [3, 0] && qLayer.offset == 9)

        for (idx, layer) in layers.enumerated() {
            BatchCore.join([rowCaches[3][idx]], into: layer)
        }
        #expect(qLayer.leftPadding == [5, 2, 0] && qLayer.offset == 11)
        verifyStep([0, 2, 3], fed: [21, 23, 24])

        for layer in layers {
            (layer as? BatchQuantizedKVCache)?.filter(rows: [2])
            (layer as? BatchRotatingKVCache)?.filter(rows: [2])
            (layer as? BatchKVCache)?.filter(rows: [2])
        }
        #expect(qLayer.leftPadding == [0] && qLayer.offset == 12)
        verifyStep([3], fed: [31])
    }

    @Test("MLXInferenceEngine and ModelOrchestratorActor batch concurrent --kv-bits 8 and 4 requests with solo parity and isolate mismatched kvBits rungs",
          .timeLimit(.minutes(5)), arguments: [KVPrecision.q8, .q4])
    func orchestratorBatchesQuantizedKV(precision: KVPrecision) async throws {
        let prompts = (0 ..< 4).map { seed in
            let ids = BatchFixture.prompt(12, seed: seed + 5)
            return String(String.UnicodeScalarView(ids.compactMap(UnicodeScalar.init)))
        }
        let soloEngine = MLXInferenceEngine(EngineConfiguration(quantizedKVStart: 0, batchSlots: 1))
        await soloEngine.attach(try TinyModel.container(.qwen3HeadDim32))
        var expected: [String] = []
        for prompt in prompts {
            let reply = try await soloEngine.generate(
                request: GenerationRequest(prompt: prompt, maxTokens: 16, temperature: 0).withKVPrecision(precision)
            )
            expected.append(reply.generatedText)
        }

        let batchEngine = MLXInferenceEngine(EngineConfiguration(quantizedKVStart: 0, batchSlots: 4))
        await batchEngine.attach(try TinyModel.container(.qwen3HeadDim32))
        let orchestrator = ModelOrchestratorActor(
            engine: batchEngine, kv: OrchestratorKVOptions(precision: precision),
            telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 4)
        )
        let actual = try await withThrowingTaskGroup(of: (Int, GenerationResponse).self) { group in
            for (idx, prompt) in prompts.enumerated() {
                group.addTask {
                    let req = GenerationRequest(prompt: prompt, maxTokens: 16, temperature: 0).withKVPrecision(precision)
                    return (idx, try await orchestrator.generate(request: req))
                }
            }
            var map: [Int: GenerationResponse] = [:]
            for try await (idx, res) in group { map[idx] = res }
            return (0 ..< prompts.count).compactMap { map[$0] }
        }
        #expect(actual.map(\.generatedText) == expected)
        #expect(actual.allSatisfy { $0.completionTokens == 16 })
        #expect(await batchEngine.batchedRows == 4)
        #expect(await batchEngine.checkInvariants(idle: true) == [])
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])

        let core = BatchCore(model: try TinyModel.make(.qwen3HeadDim32), slots: 2)
        var seat4: BatchSeat? = core.decoder.seat(on: core, kvBits: 4)
        #expect(seat4 != nil)
        #expect(core.decoder.seat(on: core, kvBits: 8) == nil)
        #expect(core.decoder.seat(on: core, kvBits: nil) == nil)
        #expect(core.decoder.seat(on: core, kvBits: 4) != nil)
        seat4 = nil
        #expect(core.decoder.seat(on: core, kvBits: 8) != nil)
    }
}
