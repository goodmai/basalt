import Foundation

public enum FitLevel: String, Codable, Sendable {
    case perfect = "Perfect"
    case good = "Good"
    case marginal = "Marginal"
    case tooTight = "TooTight"
    
    public var emoji: String {
        switch self {
        case .perfect: return "🟢"
        case .good: return "🟡"
        case .marginal: return "🟠"
        case .tooTight: return "🔴"
        }
    }
}

public struct FitScore: Codable, Sendable {
    public let modelName: String
    public let fitLevel: FitLevel
    public let score: Double
    public let estimatedRAM: Int64
    public let estimatedTPS: Int
    public let contextWindow: Int
    
    public init(modelName: String, fitLevel: FitLevel, score: Double, estimatedRAM: Int64, estimatedTPS: Int, contextWindow: Int) {
        self.modelName = modelName
        self.fitLevel = fitLevel
        self.score = score
        self.estimatedRAM = estimatedRAM
        self.estimatedTPS = estimatedTPS
        self.contextWindow = contextWindow
    }
}

public actor FitScorer {
    public let profile: SystemProfiler.SystemResources
    
    public init(profile: SystemProfiler.SystemResources) {
        self.profile = profile
    }
    
    public func score(_ model: ModelDefinition) -> FitScore {
        let ramFit = calculateRAMFit(model: model, available: profile.availableRAM)
        let contextFit = calculateContextFit(model: model)
        let speedFit = calculateSpeedFit(model: model, chip: profile.chipModel)
        
        let totalScore = ramFit * 0.70 + contextFit * 0.15 + speedFit * 0.10 + model.quality * 0.05
        
        let fitLevel: FitLevel
        switch ramFit {
        case 0.9...1.0: fitLevel = .perfect
        case 0.7..<0.9: fitLevel = .good
        case 0.5..<0.7: fitLevel = .marginal
        default:        fitLevel = .tooTight
        }
        
        return FitScore(
            modelName: model.name,
            fitLevel: fitLevel,
            score: totalScore * 100,
            estimatedRAM: model.ramMB,
            estimatedTPS: estimateTPS(model: model, chip: profile.chipModel),
            contextWindow: model.contextWindow
        )
    }
    
    private func calculateRAMFit(model: ModelDefinition, available: Int64) -> Double {
        let availableMB = Double(available) / 1048576.0
        let modelMB = Double(model.ramMB)
        let ratio = availableMB / modelMB
        if ratio >= 2.0 {
            return 1.0
        } else if ratio >= 1.5 {
            return 0.8
        } else if ratio >= 1.0 {
            return 0.6
        } else {
            return 0.2
        }
    }
    
    private func calculateContextFit(model: ModelDefinition) -> Double {
        if model.contextWindow >= 128000 {
            return 1.0
        } else if model.contextWindow >= 32768 {
            return 0.8
        } else if model.contextWindow >= 8192 {
            return 0.6
        } else {
            return 0.4
        }
    }
    
    private func calculateSpeedFit(model: ModelDefinition, chip: String) -> Double {
        let baseFit: Double
        if model.ramMB < 3000 {
            baseFit = 1.0
        } else if model.ramMB < 8000 {
            baseFit = 0.8
        } else if model.ramMB < 15000 {
            baseFit = 0.6
        } else {
            baseFit = 0.4
        }
        
        if chip.contains("M4") || chip.contains("M5") {
            return min(1.0, baseFit * 1.3)
        } else if chip.contains("M3") {
            return min(1.0, baseFit * 1.15)
        } else if chip.contains("M2") {
            return min(1.0, baseFit * 1.0)
        } else {
            return min(1.0, baseFit * 0.85)
        }
    }
    
    private func estimateTPS(model: ModelDefinition, chip: String) -> Int {
        let baseTPS: Double
        if model.ramMB < 3000 {
            baseTPS = 100.0
        } else if model.ramMB < 8000 {
            baseTPS = 60.0
        } else if model.ramMB < 15000 {
            baseTPS = 30.0
        } else {
            baseTPS = 10.0
        }
        
        let chipMultiplier: Double
        if chip.contains("M4") || chip.contains("M5") {
            chipMultiplier = 1.3
        } else if chip.contains("M3") {
            chipMultiplier = 1.15
        } else if chip.contains("M2") {
            chipMultiplier = 1.0
        } else {
            chipMultiplier = 0.85
        }
        
        return Int(baseTPS * chipMultiplier)
    }
}
