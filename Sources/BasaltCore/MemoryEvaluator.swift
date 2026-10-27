import Foundation

public struct MemoryAssessment: Sendable, Codable, Equatable {
    public let modelId: String
    public let modelName: String
    public let requiredRAMBytes: Int64
    public let availableRAMBytes: Int64
    public let totalPhysicalRAMBytes: Int64
    public let fitsInMemory: Bool
    public let fitLevel: FitLevel
    public let maxContextBudgetTokens: Int
    public let maxPromptBudgetTokens: Int
    public let warning: String?
    public let recommendation: String?
    public let memoryBudgetBytes: Int64

    public var requiredRAMFormatted: String {
        formatRAMBytes(requiredRAMBytes)
    }

    public var availableRAMFormatted: String {
        formatRAMBytes(availableRAMBytes)
    }

    public var totalRAMFormatted: String {
        formatRAMBytes(totalPhysicalRAMBytes)
    }

    public init(
        modelId: String,
        modelName: String,
        requiredRAMBytes: Int64,
        availableRAMBytes: Int64,
        totalPhysicalRAMBytes: Int64,
        fitsInMemory: Bool,
        fitLevel: FitLevel,
        maxContextBudgetTokens: Int,
        maxPromptBudgetTokens: Int? = nil,
        warning: String? = nil,
        recommendation: String? = nil,
        memoryBudgetBytes: Int64 = 0
    ) {
        self.modelId = modelId
        self.modelName = modelName
        self.requiredRAMBytes = requiredRAMBytes
        self.availableRAMBytes = availableRAMBytes
        self.totalPhysicalRAMBytes = totalPhysicalRAMBytes
        self.fitsInMemory = fitsInMemory
        self.fitLevel = fitLevel
        self.maxContextBudgetTokens = maxContextBudgetTokens
        self.maxPromptBudgetTokens = maxPromptBudgetTokens ?? maxContextBudgetTokens
        self.warning = warning
        self.recommendation = recommendation
        self.memoryBudgetBytes = memoryBudgetBytes
    }

    public var contextBudgetLine: String {
        guard maxPromptBudgetTokens < maxContextBudgetTokens else { return "\(maxContextBudgetTokens) tokens" }
        return "\(maxContextBudgetTokens) tokens, the prompt at most \(maxPromptBudgetTokens) (prefilled in fp16, quantized after)"
    }

    public var memoryBudgetFormatted: String {
        formatRAMBytes(memoryBudgetBytes)
    }
}

public func formatRAMBytes(_ bytes: Int64) -> String {
    switch bytes {
    case 0..<1_048_576:    return String(format: "%.1f KB", Double(bytes) / 1_024)
    case 0..<1_073_741_824: return String(format: "%.1f MB", Double(bytes) / 1_048_576)
    default:               return String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
    }
}

public actor MemoryEvaluator {
    private let profiler: SystemProfiler

    public init(profiler: SystemProfiler = SystemProfiler()) {
        self.profiler = profiler
    }

    public func evaluate(modelId: String, contextLength: Int = 4096, kvBits: Int? = nil) async -> MemoryAssessment {
        let resources = await profiler.detectResources()
        
        // 1. Check ModelDatabase
        if let def = ModelDatabase.allModels.first(where: {
            $0.id.lowercased() == modelId.lowercased() ||
            $0.name.lowercased() == modelId.lowercased()
        }) {
            return evaluate(definition: def, resources: resources, contextLength: contextLength, kvBits: kvBits)
        }

        // 2. Check local cache directory
        let localDir = ModelCache.cacheDir(for: modelId)
        if FileManager.default.fileExists(atPath: localDir.path) {
            let size = directorySize(at: localDir)
            return evaluateLocal(
                modelId: modelId,
                localSize: size,
                resources: resources,
                contextLength: contextLength,
                kvBits: kvBits
            )
        }

        // 3. Fallback heuristic from repo ID
        return evaluateHeuristic(modelId: modelId, resources: resources, contextLength: contextLength, kvBits: kvBits)
    }

    public func evaluate(
        definition def: ModelDefinition,
        resources: SystemProfiler.SystemResources,
        contextLength: Int = 4096,
        kvBits: Int? = nil
    ) -> MemoryAssessment {
        let kv = def.kvGeometry
        let weights = def.ramMB * 1024 * 1024
        let required = weights + kv.bytes(tokens: contextLength, kvBits: kvBits)
        let oversized = def.quantization.lowercased().contains("bfloat16") || def.quantization.lowercased().contains("fp16") || def.ramMB > 30000
        return Self.assessment(
            id: def.id, name: def.name, required: required, weights: weights, kv: kv, kvBits: kvBits, resources: resources,
            subject: "Model",
            recommendation: oversized ? "Use an MLX 4-bit or Dynamic Quant (DQ) quantized version (e.g. AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit or Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX)." : nil)
    }

    public static func memoryBudget(_ resources: SystemProfiler.SystemResources) -> Int64 {
        resources.gpuMemory > 0 ? min(resources.gpuMemory, resources.totalRAM) : resources.totalRAM / 4 * 3
    }

    public static func fitLevel(required: Int64, budget: Int64) -> FitLevel {
        let share = budget > 0 ? Double(required) / Double(budget) : .infinity
        switch share {
        case ...0.60: return .perfect
        case ...0.90: return .good
        case ...1.00: return .marginal
        default:      return .tooTight
        }
    }

    public static func contextBudget(budget: Int64, weights: Int64, kv: KVCacheGeometry, kvBits: Int?) -> Int {
        min(TokenBudgetCalculator.upperCapTokens, max(0, kv.maxTokens(fitting: kvRoom(budget: budget, weights: weights), kvBits: kvBits)))
    }

    public static func promptBudget(budget: Int64, weights: Int64, kv: KVCacheGeometry, kvBits: Int?) -> Int {
        min(TokenBudgetCalculator.upperCapTokens, kv.maxPromptTokens(fitting: kvRoom(budget: budget, weights: weights), kvBits: kvBits))
    }

    static func kvRoom(budget: Int64, weights: Int64) -> Int64 {
        max(0, budget - weights) / 2
    }

    static func assessment(
        id: String, name: String, required: Int64, weights: Int64, kv: KVCacheGeometry, kvBits: Int?,
        resources: SystemProfiler.SystemResources, subject: String, recommendation: String?
    ) -> MemoryAssessment {
        let budget = memoryBudget(resources)
        let level = fitLevel(required: required, budget: budget)
        let fits = level != .tooTight
        let warning: String?
        if !fits {
            warning = "\(subject) requires ~\(formatRAMBytes(required)), more than the \(formatRAMBytes(budget)) Metal can use on this Mac (\(formatRAMBytes(resources.totalRAM)) physical)."
        } else if required > resources.availableRAM {
            warning = "Only \(formatRAMBytes(resources.availableRAM)) is free right now; loading the model makes macOS compress or swap other apps' memory."
        } else {
            warning = nil
        }
        return MemoryAssessment(
            modelId: id, modelName: name, requiredRAMBytes: required, availableRAMBytes: resources.availableRAM,
            totalPhysicalRAMBytes: resources.totalRAM, fitsInMemory: fits, fitLevel: level,
            maxContextBudgetTokens: contextBudget(budget: budget, weights: weights, kv: kv, kvBits: kvBits),
            maxPromptBudgetTokens: promptBudget(budget: budget, weights: weights, kv: kv, kvBits: kvBits),
            warning: warning, recommendation: fits ? nil : recommendation, memoryBudgetBytes: budget)
    }

    private func evaluateLocal(
        modelId: String,
        localSize: Int64,
        resources: SystemProfiler.SystemResources,
        contextLength: Int,
        kvBits: Int?
    ) -> MemoryAssessment {
        let modelMB = Int(localSize / (1024 * 1024))
        let kv = KVCacheGeometry.forModel(
            modelId, parameters: ModelDefinition.inferTotalParameters(id: modelId, name: "", ramMB: Int64(modelMB))
        )
        let required = Int64(Double(localSize) * 1.15) + kv.bytes(tokens: contextLength, kvBits: kvBits)
        return Self.assessment(
            id: modelId, name: modelId.components(separatedBy: "/").last ?? modelId, required: required,
            weights: localSize, kv: kv, kvBits: kvBits, resources: resources,
            subject: "Model files (\(formatRAMBytes(localSize)))", recommendation: nil)
    }

    private func evaluateHeuristic(
        modelId: String,
        resources: SystemProfiler.SystemResources,
        contextLength: Int,
        kvBits: Int?
    ) -> MemoryAssessment {
        let info = HFModelInfo(id: modelId, downloads: nil, likes: nil, tags: nil)
        var estimatedMB: Int64 = 4000
        let params = Double(info.parameterSize.replacingOccurrences(of: "B", with: ""))
        if let params {
            if info.quantization.contains("4bit") || info.quantization == "nvfp4" || info.quantization == "dq" {
                estimatedMB = Int64(params * 550)
            } else if info.quantization.contains("8bit") {
                estimatedMB = Int64(params * 1000)
            } else {
                // FP16 / base
                estimatedMB = Int64(params * 2000)
            }
        }

        let kv = KVCacheGeometry(
            parameterProxy: params ?? ModelDefinition.inferTotalParameters(id: modelId, name: "", ramMB: estimatedMB)
        )
        let weights = estimatedMB * 1024 * 1024
        let quantized = info.quantization.contains("4bit") || info.quantization.contains("dq") || info.quantization.contains("nvfp4")
        return Self.assessment(
            id: modelId, name: info.name, required: weights + kv.bytes(tokens: contextLength, kvBits: kvBits),
            weights: weights, kv: kv, kvBits: kvBits, resources: resources, subject: "Model (estimated)",
            recommendation: quantized ? nil : "Consider choosing a 4-bit quantized MLX model variant.")
    }

    /// Follows symlinks: the HuggingFace cache keeps weights in `blobs/` and
    /// fills `snapshots/` with links, so measuring the links reported a 16 GB
    /// model as a few megabytes and the fit assessment always said "fits".
    private func directorySize(at url: URL) -> Int64 {
        ModelOrchestratorActor.resolvedDirectorySize(at: url)
    }
}
