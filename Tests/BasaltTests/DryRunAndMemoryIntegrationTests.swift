import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("Dry-Run Memory Assessment and Hardware Fit Integration Tests")
struct DryRunAndMemoryIntegrationTests {

    // MARK: - Helper

    private func makeProfile(
        chip: String, totalGB: Int64, availableGB: Int64, cores: Int = 10
    ) -> SystemProfiler.SystemResources {
        SystemProfiler.SystemResources(
            totalRAM: totalGB * 1024 * 1024 * 1024,
            availableRAM: availableGB * 1024 * 1024 * 1024,
            cpuCores: cores,
            gpuName: "Apple \(chip)",
            gpuMemory: totalGB * 1024 * 1024 * 1024,
            diskSpace: 500 * 1024 * 1024 * 1024,
            osVersion: "15.0",
            chipModel: chip
        )
    }

    // MARK: - Real System Dry-Run Evaluation

    @Test("Dry-run evaluates Huihui Base BF16 vs MLX 4-bit on current machine")
    func testHuihuiModelDryRun() async {
        let evaluator = MemoryEvaluator()

        // 1. Base BF16 model (~54GB)
        let baseAssessment = await evaluator.evaluate(modelId: "huihui-ai/Huihui-Qwen3.8-27B-abliterated")
        #expect(baseAssessment.requiredRAMBytes > 50_000_000_000)

        // On 24GB or smaller machine, base model should NOT fit
        if baseAssessment.totalPhysicalRAMBytes <= 32_000_000_000 {
            #expect(!baseAssessment.fitsInMemory)
            #expect(baseAssessment.warning != nil)
            #expect(baseAssessment.recommendation != nil)
        }

        // 2. MLX 4-bit model (~15.2GB)
        let quantAssessment = await evaluator.evaluate(modelId: "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit")
        #expect(quantAssessment.requiredRAMBytes < 20_000_000_000)
        #expect(quantAssessment.requiredRAMBytes > 14_000_000_000)
    }

    @Test("Dry-run evaluates Ex0bit models on current machine")
    func testEx0bitModelsDryRun() async {
        let evaluator = MemoryEvaluator()

        // MYTHOS 26B Dynamic Quant (14.5 GB)
        let mythos = await evaluator.evaluate(modelId: "Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX")
        #expect(mythos.requiredRAMBytes < 18_000_000_000)
        #expect(mythos.modelName.contains("MYTHOS"))

        // Olmo 3 7B Instruct (4.5 GB)
        let olmo = await evaluator.evaluate(modelId: "Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated")
        #expect(olmo.requiredRAMBytes < 6_000_000_000)
        // fitsInMemory is a function of RAM free right now, so assert it only when
        // there is headroom. Unguarded, this failed purely because another process
        // held 13 GB — a machine-state failure that looks like a code regression.
        if olmo.availableRAMBytes > 8_000_000_000 {
            #expect(olmo.fitsInMemory == true)
        }

        // MiniMax 72GB MoE
        let minimax = await evaluator.evaluate(modelId: "Ex0bit/MiniMax-SLURPY-DQ-MLX")
        #expect(minimax.requiredRAMBytes > 70_000_000_000)
        if minimax.totalPhysicalRAMBytes <= 32_000_000_000 {
            #expect(!minimax.fitsInMemory)
            #expect(minimax.warning != nil)
        }
    }

    // MARK: - Parameterized Hardware Profiles (8GB, 24GB, 128GB)

    @Test("Simulated 8GB Mac Profile rejects large models and accepts small models")
    func testSimulated8GBProfile() async {
        let profile8GB = makeProfile(chip: "Apple M3", totalGB: 8, availableGB: 5)
        let evaluator = MemoryEvaluator()

        // 4B model (2.3GB) -> Fits
        let qwen4B = ModelDatabase.allModels.first(where: { $0.id == "mlx-community/Qwen3.5-4B-4bit" })!
        let score4B = await evaluator.evaluate(definition: qwen4B, resources: profile8GB)
        #expect(score4B.fitsInMemory == true)
        #expect(score4B.fitLevel == FitLevel.perfect || score4B.fitLevel == FitLevel.good)
        #expect(score4B.warning == nil)

        // Huihui 27B Base (54GB) -> Rejection
        let huihuiBase = ModelDatabase.allModels.first(where: { $0.id == "huihui-ai/Huihui-Qwen3.8-27B-abliterated" })!
        let scoreHuihui = await evaluator.evaluate(definition: huihuiBase, resources: profile8GB)
        #expect(scoreHuihui.fitsInMemory == false)
        #expect(scoreHuihui.fitLevel == FitLevel.tooTight)
        #expect(scoreHuihui.warning != nil)
        #expect(scoreHuihui.recommendation != nil)
    }

    @Test("Simulated 24GB Mac Profile accepts 4-bit 27B and rejects 54GB Base")
    func testSimulated24GBProfile() async {
        let profile24GB = makeProfile(chip: "Apple M4 Pro", totalGB: 24, availableGB: 16, cores: 16)
        let evaluator = MemoryEvaluator()

        // Huihui Base (54GB) -> Exceeds 24GB RAM
        let huihuiBase = ModelDatabase.allModels.first(where: { $0.id == "huihui-ai/Huihui-Qwen3.8-27B-abliterated" })!
        let scoreBase = await evaluator.evaluate(definition: huihuiBase, resources: profile24GB)
        #expect(scoreBase.fitsInMemory == false)
        #expect(scoreBase.fitLevel == FitLevel.tooTight)
        #expect(scoreBase.warning != nil)

        // Huihui 4-bit (15.2GB) -> Fits in 24GB RAM
        let huihui4Bit = ModelDatabase.allModels.first(where: { $0.id == "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit" })!
        let score4Bit = await evaluator.evaluate(definition: huihui4Bit, resources: profile24GB)
        #expect(score4Bit.fitsInMemory == true)

        // MYTHOS 26B DQ (14.5GB) -> Fits in 24GB RAM
        let mythos = ModelDatabase.allModels.first(where: { $0.id == "Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX" })!
        let scoreMythos = await evaluator.evaluate(definition: mythos, resources: profile24GB)
        #expect(scoreMythos.fitsInMemory == true)
    }

    @Test("Simulated 128GB Mac Studio Profile accepts all models")
    func testSimulated128GBProfile() async {
        let profile128GB = makeProfile(chip: "Apple M2 Ultra", totalGB: 128, availableGB: 110, cores: 24)
        let evaluator = MemoryEvaluator()

        for model in ModelDatabase.allModels {
            let assessment = await evaluator.evaluate(definition: model, resources: profile128GB)
            #expect(assessment.fitsInMemory == true)
            #expect(assessment.warning == nil)
            #expect(assessment.fitLevel == FitLevel.perfect || assessment.fitLevel == FitLevel.good)
        }
    }

    // MARK: - TokenBudgetCalculator with Large Models

    @Test("Token budget calculates appropriate context budget for large models")
    func testTokenBudgetLargeModels() {
        let budget = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: 24 * 1024 * 1024 * 1024,
            modelSizeMB: 15200,
            kv: KVCacheGeometry(parameterProxy: 27)
        )
        #expect(budget >= 4096)
        #expect(budget <= 131072)
    }

    static var kvProbeModelPath: String? {
        guard ProcessInfo.processInfo.environment["BASALT_E2E"] == "1" else { return nil }
        return ModelCache.resolve(repoId: "mlx-community/Qwen3-4B-4bit")?.path
    }

    @Test("Qwen3-4B: the fp16, 8- and 4-bit KV cache MLX holds after 16k tokens is within 3% of KVCacheGeometry, which ignores the 256-token allocation step",
          .enabled(if: DryRunAndMemoryIntegrationTests.kvProbeModelPath != nil,
                   "set BASALT_E2E=1 and cache mlx-community/Qwen3-4B-4bit"))
    func measuredKVMatchesGeometry() async throws {
        let path = try #require(Self.kvProbeModelPath)
        let engine = MLXInferenceEngine(gpuCacheLimit: 0)
        try await engine.load(modelPath: path)
        let container = try #require(await engine.container)
        let geometry = try #require(KVCacheGeometry.load(modelDirectory: URL(fileURLWithPath: path)))
        let (prompt, generated) = (16_380, 4)
        for bits in KVCacheGeometry.adaptiveKVBitsLadder {
            let held = try await container.perform { (context: ModelContext) in
                var parameters = GenerateParameters(maxTokens: generated, temperature: 0)
                parameters.kvBits = bits
                parameters.quantizedKVStart = KVCacheGeometry.defaultQuantizedKVStart
                Memory.clearCache()
                let before = Memory.activeMemory
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray((0..<prompt).map { 1_000 + $0 * 7_919 % 20_000 })),
                    model: context.model, cache: nil, parameters: parameters)
                while iterator.next() != nil {}
                Stream.gpu.synchronize()
                let held = Memory.activeMemory - before
                #expect(iterator.tokenCount == generated)
                return held
            }
            let estimate = geometry.bytes(tokens: prompt + generated, kvBits: bits)
            #expect(abs(Double(held) / Double(estimate) - 1) <= 0.03, "kvBits \(String(describing: bits)): MLX holds \(held) B, geometry says \(estimate) B")
        }
    }
}
