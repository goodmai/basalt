import Testing
import Foundation
@testable import BasaltCore

/// The deterministic parts of MemoryEvaluator and SystemProfiler. Coarse per-machine fits
/// for the model database live in DryRunAndMemoryIntegrationTests.
@Suite("MemoryEvaluator thresholds and SystemProfiler recommendations")
struct MemoryEvaluatorTests {

    private static let gib: Int64 = 1 << 30
    private static let mib: Int64 = 1 << 20

    private func resources(totalGB: Int64, availableGB: Int64) -> SystemProfiler.SystemResources {
        .init(totalRAM: totalGB * Self.gib, availableRAM: availableGB * Self.gib, cpuCores: 8,
              gpuName: "Apple M4", gpuMemory: 0, diskSpace: 0, osVersion: "15", chipModel: "M4")
    }

    private func model(ramMB: Int64, quantization: String = "4bit") -> ModelDefinition {
        ModelDefinition(id: "test/model", name: "Test", family: .qwen, task: .chat, modality: .text,
                        ramMB: ramMB, contextWindow: 4096, quantization: quantization, quality: 0.5)
    }

    @Test("Fit levels switch at 60%, 90% and 100% of the Metal working set (3/4 of 16 GiB = 12 GiB here)", arguments: [
        (7372,  FitLevel.perfect,  true),
        (7373,  FitLevel.good,     true),
        (11059, FitLevel.good,     true),
        (11060, FitLevel.marginal, true),
        (12288, FitLevel.marginal, true),
        (12289, FitLevel.tooTight, false),
    ] as [(Int64, FitLevel, Bool)])
    func fitThresholds(ramMB: Int64, level: FitLevel, fits: Bool) async {
        let a = await MemoryEvaluator().evaluate(definition: model(ramMB: ramMB),
                                                 resources: resources(totalGB: 16, availableGB: 10), contextLength: 0)
        #expect(a.requiredRAMBytes == ramMB * Self.mib)
        #expect(a.memoryBudgetBytes == 12 * Self.gib)
        #expect(a.fitLevel == level)
        #expect(a.fitsInMemory == fits)
        #expect((a.warning != nil) == (!fits || ramMB * Self.mib > 10 * Self.gib))
    }

    @Test("24 GB Mac with 9.4 GB free right now: LFM2.5-24B-A2B 4-bit fits the working set, with a note instead of a refusal")
    func momentaryFreeMemoryDoesNotRefuse() async {
        let mac24 = SystemProfiler.SystemResources(
            totalRAM: 24 * Self.gib, availableRAM: Int64(9.4 * Double(Self.gib)), cpuCores: 10, gpuName: "Apple M4",
            gpuMemory: 16 * Self.gib, diskSpace: 0, osVersion: "26", chipModel: "M4")
        let a = await MemoryEvaluator().evaluate(definition: model(ramMB: 12_943), resources: mac24, contextLength: 4096)
        #expect(a.fitsInMemory)
        #expect(a.memoryBudgetBytes == 16 * Self.gib)
        #expect(a.warning?.contains("free right now") == true)
        let kv = model(ramMB: 12_943).kvGeometry
        #expect(a.maxContextBudgetTokens == kv.maxTokens(fitting: (16 * Self.gib - 12_943 * Self.mib) / 2))
    }

    @Test("More than total RAM never fits, however much is 'available'")
    func totalRAMCaps() async {
        let a = await MemoryEvaluator().evaluate(definition: model(ramMB: 9 * 1024),
                                                 resources: resources(totalGB: 8, availableGB: 64), contextLength: 0)
        #expect(a.fitLevel == .tooTight)
        #expect(!a.fitsInMemory)
    }

    @Test("The KV estimate grows the requirement with the context length")
    func contextAddsKV() async {
        let e = MemoryEvaluator()
        let r = resources(totalGB: 64, availableGB: 48)
        let short = await e.evaluate(definition: model(ramMB: 1000), resources: r, contextLength: 0)
        let long = await e.evaluate(definition: model(ramMB: 1000), resources: r, contextLength: 32768)
        let kv = KVCacheGeometry(parameterProxy: model(ramMB: 1000).totalParameters)
        #expect(long.requiredRAMBytes - short.requiredRAMBytes == kv.bytes(tokens: 32768))
        let quantized = await e.evaluate(definition: model(ramMB: 1000), resources: r, contextLength: 32768, kvBits: 4)
        #expect(quantized.requiredRAMBytes - short.requiredRAMBytes == kv.bytes(tokens: 32768, kvBits: 4))
    }

    @Test("--kv-bits raises the context budget; the prompt budget is what its fp16 prefill and conversion copy leave")
    func quantizedPromptBudget() async {
        let e = MemoryEvaluator()
        let r = resources(totalGB: 16, availableGB: 12)
        let def = model(ramMB: 8_000)
        let fp16 = await e.evaluate(definition: def, resources: r)
        let q4 = await e.evaluate(definition: def, resources: r, kvBits: 4)
        let kv = KVCacheGeometry(parameterProxy: def.totalParameters)
        let room = (MemoryEvaluator.memoryBudget(r) - def.ramMB * Self.mib) / 2
        let cap = TokenBudgetCalculator.upperCapTokens
        #expect(fp16.maxPromptBudgetTokens == fp16.maxContextBudgetTokens)
        #expect(q4.maxContextBudgetTokens == min(cap, kv.maxTokens(fitting: room, kvBits: 4)))
        #expect(q4.maxPromptBudgetTokens == min(cap, kv.maxPromptTokens(fitting: room, kvBits: 4)))
        #expect(q4.maxContextBudgetTokens > fp16.maxContextBudgetTokens)
        #expect(q4.maxPromptBudgetTokens < fp16.maxPromptBudgetTokens)
    }

    @Test("--kv-zones: the prompt budget is the whole zoned context budget, the prompt is quantized as it is prefilled")
    func zonedPromptBudget() async {
        let e = MemoryEvaluator()
        let r = resources(totalGB: 16, availableGB: 12)
        let def = model(ramMB: 8_000)
        let uniform = await e.evaluate(definition: def, resources: r, kvBits: 4)
        let zoned = await e.evaluate(definition: def, resources: r, kvBits: 4, kvZones: .standard)
        let kv = KVCacheGeometry(parameterProxy: def.totalParameters)
        let room = (MemoryEvaluator.memoryBudget(r) - def.ramMB * Self.mib) / 2
        let cap = TokenBudgetCalculator.upperCapTokens
        #expect(zoned.maxContextBudgetTokens == min(cap, kv.maxTokens(fitting: room, kvBits: 4, zones: .standard)))
        #expect(zoned.maxPromptBudgetTokens == zoned.maxContextBudgetTokens)
        #expect(zoned.maxPromptBudgetTokens > uniform.maxPromptBudgetTokens)
        #expect(zoned.requiredRAMBytes > uniform.requiredRAMBytes)
        #expect(zoned.contextBudgetLine == "\(zoned.maxContextBudgetTokens) tokens")
    }

    @Test("Only an oversized full-precision or very large model gets the 4-bit recommendation")
    func recommendation() async {
        let e = MemoryEvaluator()
        let small = resources(totalGB: 8, availableGB: 4)
        #expect(await e.evaluate(definition: model(ramMB: 20_000, quantization: "bfloat16"), resources: small).recommendation != nil)
        #expect(await e.evaluate(definition: model(ramMB: 20_000, quantization: "4bit"), resources: small).recommendation == nil)
        #expect(await e.evaluate(definition: model(ramMB: 40_000, quantization: "4bit"), resources: small).recommendation != nil)
        #expect(await e.evaluate(definition: model(ramMB: 1_000, quantization: "bfloat16"), resources: small).recommendation == nil)
    }

    @Test("An unknown repo is sized from its name: 550 MB/B at 4-bit or DQ, 1000 at 8-bit, 2000 otherwise")
    func heuristicSizing() async {
        let e = MemoryEvaluator()
        // A fresh owner keeps the repo out of the model database and the local HF cache.
        let owner = "nobody-\(UUID().uuidString)"
        let cases: [(String, Int64)] = [
            ("Foo-7B-4bit", 7 * 550), ("Foo-26B-A4B-DQ", 26 * 550), ("Foo-8B-8bit", 8 * 1000),
            ("Foo-3B-bf16", 3 * 2000), ("no-size-in-name", 4000),
        ]
        for (name, mb) in cases {
            let a = await e.evaluate(modelId: "\(owner)/\(name)", contextLength: 0)
            #expect(a.requiredRAMBytes == mb * Self.mib, "\(name)")
            #expect(a.modelName == name)
        }
        let withContext = await e.evaluate(modelId: "\(owner)/Foo-7B-4bit", contextLength: 8192)
        #expect(withContext.requiredRAMBytes == 7 * 550 * Self.mib + KVCacheGeometry(parameterProxy: 7).bytes(tokens: 8192))
        // 800 GB needs no particular machine to be too big; only the unquantized one is told to quantize.
        let base = await e.evaluate(modelId: "\(owner)/Huge-400B-bf16")
        #expect(base.fitLevel == .tooTight)
        #expect(base.recommendation?.contains("4-bit") == true)
        let quantized = await e.evaluate(modelId: "\(owner)/Huge-2000B-4bit")
        #expect(quantized.fitLevel == .tooTight)
        #expect(quantized.recommendation == nil)
    }

    @Test("formatRAMBytes switches unit at 1 MiB and 1 GiB")
    func formatting() {
        #expect(formatRAMBytes(1_048_575) == "1024.0 KB")
        #expect(formatRAMBytes(1_048_576) == "1.0 MB")
        #expect(formatRAMBytes(Self.gib - 1).hasSuffix(" MB"))
        #expect(formatRAMBytes(Self.gib) == "1.00 GB")
        #expect(formatRAMBytes(24 * Self.gib + Self.gib / 2) == "24.50 GB")
    }

    @Test("Recommended model tiers switch at 8, 16, 32 and 64 GB of RAM", arguments: [
        (4, "0.5B"), (7, "0.5B"), (8, "3B"), (15, "3B"), (16, "7B"), (31, "7B"), (32, "14B"), (63, "14B"), (64, "32B"), (192, "32B"),
    ] as [(Int64, String)])
    func recommendationTiers(totalGB: Int64, size: String) async {
        let rec = await SystemProfiler().recommendModel(resources: resources(totalGB: totalGB, availableGB: totalGB / 2))
        #expect(rec.modelId.contains("-\(size)-"), "\(totalGB) GB → \(rec.modelId)")
    }

    @Test("Detected resources are self-consistent on this machine")
    func detectedResources() async {
        let profiler = SystemProfiler()
        let r = await profiler.detectResources()
        #expect(r.totalRAM == Int64(ProcessInfo.processInfo.physicalMemory))
        #expect(r.cpuCores == ProcessInfo.processInfo.processorCount)
        #expect(r.availableRAM > 0 && r.availableRAM <= r.totalRAM, "available \(r.availableRAM) of \(r.totalRAM)")

        let report = await profiler.generateReport()
        #expect(report.totalRAMBytes == r.totalRAM)
        #expect(report.compatible == (report.chip != "Intel"))
    }
}
