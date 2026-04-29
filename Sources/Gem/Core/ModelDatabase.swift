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
    
    public init(id: String, name: String, family: ModelFamily, task: ModelTask, modality: Modality, ramMB: Int64, contextWindow: Int, quantization: String, quality: Double) {
        self.id = id
        self.name = name
        self.family = family
        self.task = task
        self.modality = modality
        self.ramMB = ramMB
        self.contextWindow = contextWindow
        self.quantization = quantization
        self.quality = quality
    }
}

public struct ModelDatabase: Sendable {
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
            quality: 0.85
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
            quality: 0.89
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
            quality: 0.92
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
            quality: 0.83
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
            quality: 0.81
        )
    ]
}
