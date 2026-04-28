import Foundation

/// Epic 16.7: Model Fit Analyzer
/// Analyzes how well models fit on current hardware
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
            // Model won't fit in available RAM
            score -= 40.0
        } else if ramRatio > 0.8 {
            // Very tight on RAM
            score -= 30.0
        } else if ramRatio > 0.6 {
            // Moderately tight
            score -= 15.0
        } else if ramRatio > 0.4 {
            // Good fit
            score -= 5.0
        }
        // else: excellent fit, no penalty
        
        // Factor 2: Total RAM vs recommended (30% weight)
        let totalRAMRatio = totalRAMGB / recommendedRAMGB
        if totalRAMRatio < 0.5 {
            // Well below recommended
            score -= 30.0
        } else if totalRAMRatio < 0.75 {
            // Below recommended
            score -= 20.0
        } else if totalRAMRatio < 1.0 {
            // Close to recommended
            score -= 10.0
        }
        // else: meets or exceeds recommended, no penalty
        
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
        let requiredDiskGB = modelRAMGB * 1.2  // 20% overhead
        if diskSpaceGB < requiredDiskGB {
            score -= 10.0
        }
        
        // Ensure score is in valid range
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
        // Base TPS estimate based on model size
        // Smaller models = higher TPS
        let baseTPS: Int
        
        switch modelSizeMB {
        case 0..<1000:          // < 1GB
            baseTPS = 150
        case 1000..<3000:       // 1-3GB
            baseTPS = 90
        case 3000..<7000:       // 3-7GB
            baseTPS = 60
        case 7000..<15000:      // 7-15GB
            baseTPS = 30
        case 15000..<30000:     // 15-30GB
            baseTPS = 15
        default:                // 30GB+
            baseTPS = 5
        }
        
        // Adjust based on available RAM
        let availableRAMGB = Double(resources.availableRAM) / 1_073_741_824.0
        let modelRAMGB = Double(modelSizeMB) / 1024.0
        let ramMultiplier: Double
        
        if availableRAMGB > modelRAMGB * 3 {
            ramMultiplier = 1.2  // Plenty of RAM, +20%
        } else if availableRAMGB > modelRAMGB * 2 {
            ramMultiplier = 1.0  // Good RAM
        } else if availableRAMGB > modelRAMGB * 1.5 {
            ramMultiplier = 0.8  // Tight RAM, -20%
        } else {
            ramMultiplier = 0.5  // Very tight, -50%
        }
        
        // Adjust based on chip model
        let chipMultiplier: Double
        if resources.chipModel.contains("M4") || resources.chipModel.contains("M5") {
            chipMultiplier = 1.3
        } else if resources.chipModel.contains("M3") {
            chipMultiplier = 1.15
        } else if resources.chipModel.contains("M2") {
            chipMultiplier = 1.0
        } else {
            chipMultiplier = 0.85  // M1 or older
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
