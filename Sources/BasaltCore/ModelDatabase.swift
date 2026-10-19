import Foundation

public enum ModelFamily: String, Codable, Sendable {
    case qwen, gemma, llama, mistral, deepseek
}

public enum ModelTask: String, Codable, Sendable {
    case chat, code, vision, audio, embedding, rerank
}

public enum Modality: String, Codable, Sendable {
    case text, image, audio, multimodal
}

public struct ModelDefinition: Codable, Sendable {
    public let id: String
    public let name: String
    public let family: ModelFamily
    public let task: ModelTask
    public let modality: Modality
    public let ramMB: Int64
    public let contextWindow: Int
    public let quantization: String
    public let quality: Double
    public let totalParameters: Double    // Billions of parameters, e.g. 26.0 for 26B
    public let activeParameters: Double   // Billions of active parameters, e.g. 4.0 for A4B, or 27.0 for dense

    public var isMoE: Bool {
        activeParameters < totalParameters
    }

    public init(
        id: String,
        name: String,
        family: ModelFamily,
        task: ModelTask,
        modality: Modality,
        ramMB: Int64,
        contextWindow: Int,
        quantization: String,
        quality: Double,
        totalParameters: Double? = nil,
        activeParameters: Double? = nil
    ) {
        self.id = id
        self.name = name
        self.family = family
        self.task = task
        self.modality = modality
        self.ramMB = ramMB
        self.contextWindow = contextWindow
        self.quantization = quantization
        self.quality = quality

        let total = totalParameters ?? Self.inferTotalParameters(id: id, name: name, ramMB: ramMB)
        self.totalParameters = total
        self.activeParameters = activeParameters ?? Self.inferActiveParameters(id: id, name: name, total: total)
    }

    /// Parameter count (billions) read from the size token of an id like "Qwen3-30B-A3B-4bit":
    /// a number followed by "b" that is not the start of "bit" and not preceded by "a" (that is
    /// the active count). Falls back to a size estimate from the 4-bit footprint.
    static func inferTotalParameters(id: String, name: String, ramMB: Int64) -> Double {
        let combined = (id + " " + name).lowercased()
        if let r = combined.range(of: #"(?<![a-z\d.])\d+(\.\d+)?(?=b(?!it)(?![a-z]))"#, options: .regularExpression),
           let value = Double(combined[r]) {
            return value
        }
        return max(1.0, Double(ramMB) / 600.0)
    }

    /// Active parameters of an MoE id ("…-A3B" → 3); the total for a dense model.
    static func inferActiveParameters(id: String, name: String, total: Double) -> Double {
        let combined = (id + " " + name).lowercased()
        if let r = combined.range(of: #"(?<=[-_ ]a)\d+(\.\d+)?(?=b(?![a-z]))"#, options: .regularExpression),
           let value = Double(combined[r]) {
            return value
        }
        return total
    }
}

public struct ModelDatabase: Sendable {
    /// Models embedded at compile time for Zero-I/O lookup
    public static let allModels: [ModelDefinition] = [
        ModelDefinition(
            id: "mlx-community/Qwen3.5-4B-4bit",
            name: "Qwen 3.5 4B (4-bit)",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 2300,
            contextWindow: 32768,
            quantization: "4-bit",
            quality: 0.85,
            totalParameters: 4.0,
            activeParameters: 4.0
        ),
        ModelDefinition(
            id: "mlx-community/Qwen2.5-Coder-7B",
            name: "Qwen2.5-Coder-7B",
            family: .qwen,
            task: .code,
            modality: .text,
            ramMB: 4100,
            contextWindow: 128000,
            quantization: "4-bit",
            quality: 0.89,
            totalParameters: 7.0,
            activeParameters: 7.0
        ),
        ModelDefinition(
            id: "mlx-community/Qwen3.6-27B-4bit",
            name: "Qwen 3.6 27B (4-bit)",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 14500,
            contextWindow: 32768,
            quantization: "4-bit",
            quality: 0.92,
            totalParameters: 27.0,
            activeParameters: 27.0
        ),
        ModelDefinition(
            id: "mlx-community/gemma-2-2b-it",
            name: "Gemma 4 E2B IT",
            family: .gemma,
            task: .chat,
            modality: .text,
            ramMB: 2700,
            contextWindow: 8192,
            quantization: "4-bit",
            quality: 0.83,
            totalParameters: 2.0,
            activeParameters: 2.0
        ),
        ModelDefinition(
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            name: "Llama 3.2 3B",
            family: .llama,
            task: .chat,
            modality: .text,
            ramMB: 2100,
            contextWindow: 128000,
            quantization: "4-bit",
            quality: 0.81,
            totalParameters: 3.0,
            activeParameters: 3.0
        ),
        // Huihui AI Models
        ModelDefinition(
            id: "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
            name: "Huihui Qwen 3.8 27B Abliterated (Base FP16)",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 54000,
            contextWindow: 262144,
            quantization: "bfloat16",
            quality: 0.95,
            totalParameters: 27.0,
            activeParameters: 27.0
        ),
        ModelDefinition(
            id: "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit",
            name: "Huihui Qwen 3.8 27B Abliterated (MLX 4-bit)",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 15200,
            contextWindow: 262144,
            quantization: "4-bit",
            quality: 0.94,
            totalParameters: 27.0,
            activeParameters: 27.0
        ),
        // Ex0bit Models
        ModelDefinition(
            id: "Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX",
            name: "MYTHOS 26B A4B PRISM PRO DQ (MLX)",
            family: .gemma,
            task: .chat,
            modality: .text,
            ramMB: 14500,
            contextWindow: 262144,
            quantization: "Dynamic Quant (DQ)",
            quality: 0.93,
            totalParameters: 26.0,
            activeParameters: 4.0
        ),
        ModelDefinition(
            id: "Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4",
            name: "Qwen 3.6 35B A3B PRISM (MLX NVFP4)",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 20500,
            contextWindow: 131072,
            quantization: "NVFP4",
            quality: 0.94,
            totalParameters: 35.0,
            activeParameters: 3.0
        ),
        ModelDefinition(
            id: "Ex0bit/MiniMax-SLURPY-DQ-MLX",
            name: "MiniMax SLURPY DQ MLX",
            family: .qwen,
            task: .chat,
            modality: .text,
            ramMB: 72000,
            contextWindow: 131072,
            quantization: "Dynamic Quant (DQ)",
            quality: 0.92,
            totalParameters: 72.0,
            activeParameters: 12.0
        ),
        // Ornith 1.5 — qwen3_5 / qwen3_5_moe architecture
        ModelDefinition(
            id: "ornith-ai/Ornith-1.5-9B-MLX-4bit",
            name: "Ornith 1.5 9B (MLX 4-bit)",
            family: .qwen,
            task: .code,
            modality: .text,
            ramMB: 5800,
            contextWindow: 262144,
            quantization: "4-bit",
            quality: 0.91,
            totalParameters: 9.0,
            activeParameters: 9.0
        ),
        ModelDefinition(
            id: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
            name: "Ornith 1.5 35B A3B MoE (MLX 4-bit)",
            family: .qwen,
            task: .code,
            modality: .text,
            ramMB: 21000,
            contextWindow: 262144,
            quantization: "4-bit",
            quality: 0.95,
            totalParameters: 35.0,
            activeParameters: 3.0
        ),
        ModelDefinition(
            id: "Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated",
            name: "Elbaz OLMo 3 7B Instruct Abliterated",
            family: .llama,
            task: .chat,
            modality: .text,
            ramMB: 4500,
            contextWindow: 65536,
            quantization: "4-bit",
            quality: 0.88,
            totalParameters: 7.0,
            activeParameters: 7.0
        )
    ]

    /// Zero-I/O compile-time lookup by exact ID
    public static func find(id: String) -> ModelDefinition? {
        allModels.first { $0.id == id }
    }

    /// Zero-I/O compile-time filter by model family
    public static func models(for family: ModelFamily) -> [ModelDefinition] {
        allModels.filter { $0.family == family }
    }

    /// Zero-I/O compile-time filter by task
    public static func models(for task: ModelTask) -> [ModelDefinition] {
        allModels.filter { $0.task == task }
    }
}
