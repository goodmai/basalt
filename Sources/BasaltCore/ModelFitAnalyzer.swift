import Foundation

/// Epic 16.7 / Issue #31: Model Fit Analyzer
/// Multi-dimensional hardware fit scoring and dynamic quantization selection
public struct ModelFitAnalyzer: Sendable {
    
    // MARK: - Fit Category
    
    /// How well a model fits on the hardware
    public enum FitCategory: String, Sendable {
        case perfect   = "Perfect"   // 85-100: Runs great, highly recommended
        case good      = "Good"      // 65-84: Runs well, recommended
        case tight     = "Tight"     // 45-64: Runs but may be slow
        case tooTight  = "TooTight"  // 0-44: Not recommended, too slow or won't fit
        
        public var emoji: String {
            switch self {
            case .perfect: return "🟢"
            case .good: return "🟡"
            case .tight: return "🟠"
            case .tooTight: return "🔴"
            }
        }
        
        public var description: String {
            switch self {
            case .perfect: return "Perfect fit - highly recommended"
            case .good: return "Good fit - recommended"
            case .tight: return "Tight fit - may be slow"
            case .tooTight: return "Too tight - not recommended"
            }
        }
    }
    
    // MARK: - Quantization Option
    
    public struct QuantizationFitOption: Codable, Sendable {
        public let bits: Int
        public let name: String
        public let footprintMB: Int
        public let estimatedTPS: Int
        public let score: FourDimensionalScore
        public let fitsBudget: Bool
        public let isOptimal: Bool

        public init(
            bits: Int,
            name: String,
            footprintMB: Int,
            estimatedTPS: Int,
            score: FourDimensionalScore,
            fitsBudget: Bool,
            isOptimal: Bool
        ) {
            self.bits = bits
            self.name = name
            self.footprintMB = footprintMB
            self.estimatedTPS = estimatedTPS
            self.score = score
            self.fitsBudget = fitsBudget
            self.isOptimal = isOptimal
        }
    }

    // MARK: - Model Analysis Result
    
    /// Analysis result for a single model
    public struct ModelFitResult: Sendable {
        public let modelId: String
        public let modelName: String
        public let sizeMB: Int
        public let fitScore: Double
        public let fitCategory: FitCategory
        public let estimatedTPS: Int
        public let estimatedRAMUsage: Int  // MB
        public let recommendation: String
        
        public init(
            modelId: String,
            modelName: String,
            sizeMB: Int,
            fitScore: Double,
            fitCategory: FitCategory,
            estimatedTPS: Int,
            estimatedRAMUsage: Int,
            recommendation: String
        ) {
            self.modelId = modelId
            self.modelName = modelName
            self.sizeMB = sizeMB
            self.fitScore = fitScore
            self.fitCategory = fitCategory
            self.estimatedTPS = estimatedTPS
            self.estimatedRAMUsage = estimatedRAMUsage
            self.recommendation = recommendation
        }
    }
    
    // MARK: - Dynamic Quantization Selection (Grid: 8, 6, 4, 3, 2 bits)

    /// Evaluates model across an 8/6/4/3/2-bit grid and selects optimal q*(m):
    /// q*(m) = argmax_q S_Quality(m,q) subject to M_footprint(m,q,L_min) ≤ 0.85·M_avail and TPS(m,q) ≥ TPS_min
    public static func evaluateQuantizations(
        model: ModelDefinition,
        resources: SystemProfiler.SystemResources,
        tpsMin: Double = 10.0,
        lMin: Int = 4096,
        benchmarkProfile: HardwareBenchmarkProfile? = nil
    ) -> (optimal: QuantizationFitOption, options: [QuantizationFitOption]) {
        let gridBits = [8, 6, 4, 3, 2]
        let availableMB = Double(resources.availableRAM) / 1_048_576.0
        let modelBudgetMB = availableMB * 0.85
        
        let profile = benchmarkProfile ?? HardwareBenchmarkProfileStore.loadSync()
        
        let kvCacheMB = Double(model.kvGeometry.bytes(tokens: lMin)) / 1_048_576
        let overhead = 0.15 // 15% framework overhead

        var evaluated: [(bits: Int, name: String, footprintMB: Int, tps: Int, score: FourDimensionalScore, fits: Bool)] = []
        
        for q in gridBits {
            let weightGB = model.totalParameters * (Double(q) / 8.0) * (1.0 + overhead)
            // Parameters are in decimal billions, the budget is MiB: GB → MiB is ×1e9/2^20, not ×1024.
            let footprintMB = Int(weightGB * 1e9 / 1_048_576 + kvCacheMB)
            
            // Same speed model as `fit` scoring, evaluated at this grid width and at L_min.
            let speed = FitScorer.decodeSpeed(
                model: model, bits: Double(q), contextTokens: lMin,
                chip: resources.chipModel, gpuName: resources.gpuName, profile: profile
            )
            let tps = max(1, Int(speed.tps.rounded()))
            
            // Quality scale per quantization bit width
            let qMultiplier: Double
            switch q {
            case 8: qMultiplier = 1.00
            case 6: qMultiplier = 0.98
            case 4: qMultiplier = 0.93
            case 3: qMultiplier = 0.84
            default: qMultiplier = 0.68
            }
            let sQuality = min(1.0, max(0.0, model.quality * qMultiplier))
            
            // S_Fit
            let ramRatio = Double(footprintMB) / modelBudgetMB
            let sFit: Double
            if ramRatio <= 0.7 {
                sFit = 1.0
            } else if ramRatio <= 1.0 {
                sFit = 1.0 - (ramRatio - 0.7) * 1.66
            } else {
                sFit = max(0.0, 0.5 - (ramRatio - 1.0) * 0.5)
            }
            
            let sSpeed = min(1.0, Double(tps) / 100.0)
            let sContext = min(1.0, Double(model.contextWindow) / 65536.0)
            
            let fourD = FourDimensionalScore(quality: sQuality, speed: sSpeed, fit: sFit, context: sContext)
            let fits = Double(footprintMB) <= modelBudgetMB && Double(tps) >= tpsMin
            
            evaluated.append((bits: q, name: "\(q)-bit", footprintMB: footprintMB, tps: tps, score: fourD, fits: fits))
        }
        
        // Pick optimal: highest S_Quality among those satisfying the constraints
        let qualifying = evaluated.filter { $0.fits }
        let chosen = qualifying.max(by: { $0.score.quality < $1.score.quality })
        
        var options: [QuantizationFitOption] = []
        for e in evaluated {
            options.append(QuantizationFitOption(
                bits: e.bits,
                name: e.name,
                footprintMB: e.footprintMB,
                estimatedTPS: e.tps,
                score: e.score,
                fitsBudget: e.fits,
                isOptimal: chosen != nil && e.bits == chosen!.bits
            ))
        }
        
        let fallback = options.last ?? QuantizationFitOption(bits: 2, name: "2-bit", footprintMB: 0, estimatedTPS: 0, score: FourDimensionalScore(quality: 0, speed: 0, fit: 0, context: 0), fitsBudget: false, isOptimal: false)
        let optimal = options.first { $0.isOptimal } ?? fallback
        return (optimal: optimal, options: options)
    }

    // MARK: - Fit Score Calculation
    
    /// Calculate how well a model fits on the hardware (0-100 score)
    public static func calculateFitScore(
        model: ModelInfo,
        resources: SystemProfiler.SystemResources
    ) -> Double {
        let availableRAMGB = Double(resources.availableRAM) / 1_073_741_824.0
        let modelRAMGB = Double(model.sizeMB) / 1024.0
        let recommendedRAMGB = Double(model.recommendedRAMGB)
        let totalRAMGB = Double(resources.totalRAM) / 1_073_741_824.0
        
        var score = 100.0
        
        // Factor 1: RAM availability (40% weight)
        let ramRatio = modelRAMGB / availableRAMGB
        if ramRatio > 1.0 {
            score -= 40.0
        } else if ramRatio > 0.8 {
            score -= 30.0
        } else if ramRatio > 0.6 {
            score -= 15.0
        } else if ramRatio > 0.4 {
            score -= 5.0
        }
        
        // Factor 2: Total RAM vs recommended (30% weight)
        let totalRAMRatio = totalRAMGB / recommendedRAMGB
        if totalRAMRatio < 0.5 {
            score -= 30.0
        } else if totalRAMRatio < 0.75 {
            score -= 20.0
        } else if totalRAMRatio < 1.0 {
            score -= 10.0
        }
        
        // Factor 3: GPU memory (20% weight)
        let gpuMemoryGB = Double(resources.gpuMemory) / 1_073_741_824.0
        let gpuRatio = modelRAMGB / gpuMemoryGB
        if gpuRatio > 0.9 {
            score -= 20.0
        } else if gpuRatio > 0.7 {
            score -= 10.0
        } else if gpuRatio > 0.5 {
            score -= 5.0
        }
        
        // Factor 4: Disk space (10% weight)
        let diskSpaceGB = Double(resources.diskSpace) / 1_073_741_824.0
        let requiredDiskGB = modelRAMGB * 1.2
        if diskSpaceGB < requiredDiskGB {
            score -= 10.0
        }
        
        return max(0.0, min(100.0, score))
    }
    
    /// Categorize fit based on score
    public static func categorizeFit(score: Double) -> FitCategory {
        switch score {
        case 85...100:
            return .perfect
        case 65..<85:
            return .good
        case 45..<65:
            return .tight
        default:
            return .tooTight
        }
    }
    
    // MARK: - TPS Estimation
    
    /// Estimate tokens per second based on model size and hardware
    public static func estimateTPS(
        modelSizeMB: Int,
        resources: SystemProfiler.SystemResources
    ) -> Int {
        let baseTPS: Int
        switch modelSizeMB {
        case 0..<1000:          baseTPS = 150
        case 1000..<3000:       baseTPS = 90
        case 3000..<7000:       baseTPS = 60
        case 7000..<15000:      baseTPS = 30
        case 15000..<30000:     baseTPS = 15
        default:                baseTPS = 5
        }
        
        let availableRAMGB = Double(resources.availableRAM) / 1_073_741_824.0
        let modelRAMGB = Double(modelSizeMB) / 1024.0
        let ramMultiplier: Double
        
        if availableRAMGB > modelRAMGB * 3 {
            ramMultiplier = 1.2
        } else if availableRAMGB > modelRAMGB * 2 {
            ramMultiplier = 1.0
        } else if availableRAMGB > modelRAMGB * 1.5 {
            ramMultiplier = 0.8
        } else {
            ramMultiplier = 0.5
        }
        
        let chipMultiplier: Double
        if resources.chipModel.contains("M4") || resources.chipModel.contains("M5") {
            chipMultiplier = 1.3
        } else if resources.chipModel.contains("M3") {
            chipMultiplier = 1.15
        } else if resources.chipModel.contains("M2") {
            chipMultiplier = 1.0
        } else {
            chipMultiplier = 0.85
        }
        
        return Int(Double(baseTPS) * ramMultiplier * chipMultiplier)
    }
    
    // MARK: - Model Analysis
    
    /// Analyze multiple models and return sorted by fit score
    public static func analyzeModels(
        _ models: [ModelInfo],
        resources: SystemProfiler.SystemResources,
        minScore: Double = 0.0
    ) -> [ModelFitResult] {
        let results = models.map { model in
            let score = calculateFitScore(model: model, resources: resources)
            let category = categorizeFit(score: score)
            let tps = estimateTPS(modelSizeMB: model.sizeMB, resources: resources)
            let recommendation = generateRecommendation(
                category: category,
                tps: tps,
                modelSize: model.sizeMB
            )
            
            return ModelFitResult(
                modelId: model.id,
                modelName: model.name,
                sizeMB: model.sizeMB,
                fitScore: score,
                fitCategory: category,
                estimatedTPS: tps,
                estimatedRAMUsage: model.sizeMB,
                recommendation: recommendation
            )
        }
        
        return results
            .filter { $0.fitScore >= minScore }
            .sorted { $0.fitScore > $1.fitScore }
    }
    
    /// Get top N recommendations
    public static func getTopRecommendations(
        _ models: [ModelInfo],
        resources: SystemProfiler.SystemResources,
        limit: Int = 5
    ) -> [ModelFitResult] {
        let all = analyzeModels(models, resources: resources)
        return Array(all.prefix(limit))
    }
    
    // MARK: - Private Helpers
    
    private static func generateRecommendation(
        category: FitCategory,
        tps: Int,
        modelSize: Int
    ) -> String {
        switch category {
        case .perfect:
            return "Excellent choice! Fast performance expected (\(tps) TPS)"
        case .good:
            return "Good performance expected (\(tps) TPS)"
        case .tight:
            return "May run slowly (\(tps) TPS). Consider smaller model."
        case .tooTight:
            return "Not recommended. Choose a smaller model."
        }
    }
}

// MARK: - Model Info

/// Information about a model for fit analysis
public struct ModelInfo: Sendable {
    public let id: String
    public let name: String
    public let sizeMB: Int
    public let recommendedRAMGB: Int
    public let description: String
    
    public init(
        id: String,
        name: String,
        sizeMB: Int,
        recommendedRAMGB: Int,
        description: String
    ) {
        self.id = id
        self.name = name
        self.sizeMB = sizeMB
        self.recommendedRAMGB = recommendedRAMGB
        self.description = description
    }
}
