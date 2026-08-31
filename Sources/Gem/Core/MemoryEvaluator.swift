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
    public let warning: String?
    public let recommendation: String?

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
        warning: String? = nil,
        recommendation: String? = nil
    ) {
        self.modelId = modelId
        self.modelName = modelName
        self.requiredRAMBytes = requiredRAMBytes
        self.availableRAMBytes = availableRAMBytes
        self.totalPhysicalRAMBytes = totalPhysicalRAMBytes
        self.fitsInMemory = fitsInMemory
        self.fitLevel = fitLevel
        self.maxContextBudgetTokens = maxContextBudgetTokens
        self.warning = warning
        self.recommendation = recommendation
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

    /// Perform a dry-run memory assessment for a model given its repo ID or path.
    public func evaluate(modelId: String, contextLength: Int = 4096) async -> MemoryAssessment {
        let resources = await profiler.detectResources()
        
        // 1. Check ModelDatabase
        if let def = ModelDatabase.allModels.first(where: {
            $0.id.lowercased() == modelId.lowercased() ||
            $0.name.lowercased() == modelId.lowercased()
        }) {
            return evaluate(definition: def, resources: resources, contextLength: contextLength)
        }

        // 2. Check local cache directory
        let localDir = ModelCache.cacheDir(for: modelId)
        if FileManager.default.fileExists(atPath: localDir.path) {
            let size = directorySize(at: localDir)
            return evaluateLocal(
                modelId: modelId,
                localSize: size,
                resources: resources,
                contextLength: contextLength
            )
        }

        // 3. Fallback heuristic from repo ID
        return evaluateHeuristic(modelId: modelId, resources: resources, contextLength: contextLength)
    }

    public func evaluate(
        definition def: ModelDefinition,
        resources: SystemProfiler.SystemResources,
        contextLength: Int = 4096
    ) -> MemoryAssessment {
        let modelBytes = def.ramMB * 1024 * 1024
        let kvBytesPerToken: Int64 = 64
        let kvBytes = Int64(contextLength) * kvBytesPerToken * 128
        let totalRequired = modelBytes + (kvBytes / 1024)

        let fits = totalRequired <= resources.totalRAM && Double(totalRequired) <= Double(resources.availableRAM) * 1.25

        let fitLevel: FitLevel
        if totalRequired > resources.totalRAM || totalRequired > Int64(Double(resources.availableRAM) * 1.35) {
            fitLevel = .tooTight
        } else if totalRequired > Int64(Double(resources.availableRAM) * 0.90) {
            fitLevel = .marginal
        } else if totalRequired > Int64(Double(resources.availableRAM) * 0.60) {
            fitLevel = .good
        } else {
            fitLevel = .perfect
        }

        var warning: String? = nil
        var recommendation: String? = nil

        if !fits || fitLevel == .tooTight {
            warning = "Model requires ~\(formatRAMBytes(totalRequired)) RAM, exceeding available memory (\(formatRAMBytes(resources.availableRAM))) / total (\(formatRAMBytes(resources.totalRAM)))."
            if def.quantization.lowercased().contains("bfloat16") || def.quantization.lowercased().contains("fp16") || def.ramMB > 30000 {
                recommendation = "Use an MLX 4-bit or Dynamic Quant (DQ) quantized version (e.g. AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit or Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX)."
            }
        }

        let maxBudget = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: resources.availableRAM,
            modelSizeMB: Int(def.ramMB)
        )

        return MemoryAssessment(
            modelId: def.id,
            modelName: def.name,
            requiredRAMBytes: totalRequired,
            availableRAMBytes: resources.availableRAM,
            totalPhysicalRAMBytes: resources.totalRAM,
            fitsInMemory: fits,
            fitLevel: fitLevel,
            maxContextBudgetTokens: maxBudget,
            warning: warning,
            recommendation: recommendation
        )
    }

    private func evaluateLocal(
        modelId: String,
        localSize: Int64,
        resources: SystemProfiler.SystemResources,
        contextLength: Int
    ) -> MemoryAssessment {
        let totalRequired = Int64(Double(localSize) * 1.15) // weights + overhead
        let fits = totalRequired <= resources.totalRAM && Double(totalRequired) <= Double(resources.availableRAM) * 1.25

        let fitLevel: FitLevel
        if totalRequired > resources.totalRAM || totalRequired > Int64(Double(resources.availableRAM) * 1.35) {
            fitLevel = .tooTight
        } else if totalRequired > Int64(Double(resources.availableRAM) * 0.90) {
            fitLevel = .marginal
        } else if totalRequired > Int64(Double(resources.availableRAM) * 0.60) {
            fitLevel = .good
        } else {
            fitLevel = .perfect
        }

        var warning: String? = nil
        if !fits || fitLevel == .tooTight {
            warning = "Model files (\(formatRAMBytes(localSize))) require ~\(formatRAMBytes(totalRequired)) RAM, exceeding available memory (\(formatRAMBytes(resources.availableRAM)))."
        }

        let modelMB = Int(localSize / (1024 * 1024))
        let maxBudget = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: resources.availableRAM,
            modelSizeMB: modelMB
        )

        return MemoryAssessment(
            modelId: modelId,
            modelName: modelId.components(separatedBy: "/").last ?? modelId,
            requiredRAMBytes: totalRequired,
            availableRAMBytes: resources.availableRAM,
            totalPhysicalRAMBytes: resources.totalRAM,
            fitsInMemory: fits,
            fitLevel: fitLevel,
            maxContextBudgetTokens: maxBudget,
            warning: warning,
            recommendation: nil
        )
    }

    private func evaluateHeuristic(
        modelId: String,
        resources: SystemProfiler.SystemResources,
        contextLength: Int
    ) -> MemoryAssessment {
        let info = HFModelInfo(id: modelId, downloads: nil, likes: nil, tags: nil)
        var estimatedMB: Int64 = 4000
        if let paramsStr = info.parameterSize.replacingOccurrences(of: "B", with: "") as String?,
           let params = Double(paramsStr) {
            if info.quantization.contains("4bit") || info.quantization == "nvfp4" || info.quantization == "dq" {
                estimatedMB = Int64(params * 550)
            } else if info.quantization.contains("8bit") {
                estimatedMB = Int64(params * 1000)
            } else {
                // FP16 / base
                estimatedMB = Int64(params * 2000)
            }
        }

        let totalRequired = estimatedMB * 1024 * 1024
        let fits = totalRequired <= resources.totalRAM && Double(totalRequired) <= Double(resources.availableRAM) * 1.25

        let fitLevel: FitLevel
        if totalRequired > resources.totalRAM || totalRequired > Int64(Double(resources.availableRAM) * 1.35) {
            fitLevel = .tooTight
        } else if totalRequired > Int64(Double(resources.availableRAM) * 0.90) {
            fitLevel = .marginal
        } else if totalRequired > Int64(Double(resources.availableRAM) * 0.60) {
            fitLevel = .good
        } else {
            fitLevel = .perfect
        }

        var warning: String? = nil
        var rec: String? = nil
        if !fits || fitLevel == .tooTight {
            warning = "Model estimated to require ~\(formatRAMBytes(totalRequired)) RAM, exceeding available memory (\(formatRAMBytes(resources.availableRAM)))."
            if !info.quantization.contains("4bit") && !info.quantization.contains("dq") && !info.quantization.contains("nvfp4") {
                rec = "Consider choosing a 4-bit quantized MLX model variant."
            }
        }

        let maxBudget = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: resources.availableRAM,
            modelSizeMB: Int(estimatedMB)
        )

        return MemoryAssessment(
            modelId: modelId,
            modelName: info.name,
            requiredRAMBytes: totalRequired,
            availableRAMBytes: resources.availableRAM,
            totalPhysicalRAMBytes: resources.totalRAM,
            fitsInMemory: fits,
            fitLevel: fitLevel,
            maxContextBudgetTokens: maxBudget,
            warning: warning,
            recommendation: rec
        )
    }

    private func directorySize(at url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            total += (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { Int64($0) } ?? 0
        }
        return total
    }
}
