import Testing
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
@testable import GemCore

@Suite("Huihui and Ex0bit Model Tests")
struct HuihuiAndEx0bitModelTests {

    // MARK: - HFModelInfo Parsing Tests

    @Test("HFModelInfo parses Huihui and Ex0bit quantization formats")
    func testQuantizationParsing() {
        let huihui = HFModelInfo(id: "huihui-ai/Huihui-Qwen3.8-27B-abliterated", downloads: 5000, likes: 200, tags: ["safetensors"])
        #expect(huihui.parameterSize == "27B")
        #expect(huihui.org == "huihui-ai")
        #expect(huihui.name == "Huihui-Qwen3.8-27B-abliterated")

        let huihui4bit = HFModelInfo(id: "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit", downloads: 1000, likes: 50, tags: ["mlx"])
        #expect(huihui4bit.quantization == "4bit")
        #expect(huihui4bit.parameterSize == "27B")

        let mythos = HFModelInfo(id: "Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX", downloads: 637, likes: 30, tags: ["mlx"])
        #expect(mythos.quantization == "dq")
        #expect(mythos.parameterSize == "26B")

        let qwenMoENvfp4 = HFModelInfo(id: "Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4", downloads: 224, likes: 20, tags: ["mlx"])
        #expect(qwenMoENvfp4.quantization == "nvfp4")
        #expect(qwenMoENvfp4.parameterSize == "35B")

        let olmo = HFModelInfo(id: "Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated", downloads: 937, likes: 45, tags: ["transformers"])
        #expect(olmo.parameterSize == "7B")
    }

    // MARK: - ModelDatabase Tests

    @Test("ModelDatabase contains Huihui and Ex0bit model definitions")
    func testModelDatabaseEntries() {
        let all = ModelDatabase.allModels
        
        let huihuiBase = all.first(where: { $0.id == "huihui-ai/Huihui-Qwen3.8-27B-abliterated" })
        #expect(huihuiBase != nil)
        #expect(huihuiBase?.family == .qwen)
        #expect(huihuiBase?.contextWindow == 262144)

        let huihui4Bit = all.first(where: { $0.id == "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit" })
        #expect(huihui4Bit != nil)
        #expect(huihui4Bit?.quantization == "4-bit")

        let mythos = all.first(where: { $0.id == "Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX" })
        #expect(mythos != nil)
        #expect(mythos?.family == .gemma)

        let qwen35B = all.first(where: { $0.id == "Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4" })
        #expect(qwen35B != nil)
        #expect(qwen35B?.quantization == "NVFP4")
    }

    // MARK: - LLMTypeRegistry Aliases Tests

    @Test("MLXInferenceEngine registers type aliases in LLMTypeRegistry")
    func testTypeAliasesRegistration() async {
        await MLXInferenceEngine.registerAliases()

        let registry = LLMTypeRegistry.shared
        #expect(await registry.contains("qwen3_8"))
        #expect(await registry.contains("qwen3.8"))
        #expect(await registry.contains("qwen3_8_moe"))
        #expect(await registry.contains("qwen3_6"))
        #expect(await registry.contains("qwen3.6"))
        #expect(await registry.contains("qwen3_6_moe"))
        #expect(await registry.contains("minimax_m2"))
        #expect(await registry.contains("minimax-m2"))
        #expect(await registry.contains("qwen3_5"))
        #expect(await registry.contains("gemma4"))
        #expect(await registry.contains("olmo3"))
    }
}
