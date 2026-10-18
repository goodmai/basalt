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

public struct FourDimensionalScore: Codable, Sendable, Equatable {
    public let quality: Double   // S_Quality ∈ [0, 1]
    public let speed: Double     // S_Speed ∈ [0, 1]
    public let fit: Double       // S_Fit ∈ [0, 1]
    public let context: Double   // S_Context ∈ [0, 1]
    
    public var compositeScore: Double {
        (fit * 0.40 + quality * 0.30 + speed * 0.20 + context * 0.10) * 100.0
    }

    public init(quality: Double, speed: Double, fit: Double, context: Double) {
        self.quality = quality
        self.speed = speed
        self.fit = fit
        self.context = context
    }
}

public struct FitScore: Codable, Sendable {
    public let modelName: String
    public let fitLevel: FitLevel
    public let score: Double
    public let estimatedRAM: Int64
    public let estimatedTPS: Int
    public let contextWindow: Int
    public let isMeasured: Bool
    public let measuredTPS: Double?
    public let calibrationFactor: Double
    public let fourDimensionalScore: FourDimensionalScore?
    public let selectedQuantization: String
    
    public init(
        modelName: String,
        fitLevel: FitLevel,
        score: Double,
        estimatedRAM: Int64,
        estimatedTPS: Int,
        contextWindow: Int,
        isMeasured: Bool = false,
        measuredTPS: Double? = nil,
        calibrationFactor: Double = 1.0,
        fourDimensionalScore: FourDimensionalScore? = nil,
        selectedQuantization: String = "4-bit"
    ) {
        self.modelName = modelName
        self.fitLevel = fitLevel
        self.score = score
        self.estimatedRAM = estimatedRAM
        self.estimatedTPS = estimatedTPS
        self.contextWindow = contextWindow
        self.isMeasured = isMeasured
        self.measuredTPS = measuredTPS
        self.calibrationFactor = calibrationFactor
        self.fourDimensionalScore = fourDimensionalScore
        self.selectedQuantization = selectedQuantization
    }
}

public actor FitScorer {
    public let profile: SystemProfiler.SystemResources
    public let benchmarkProfile: HardwareBenchmarkProfile
    
    public init(
        profile: SystemProfiler.SystemResources,
        benchmarkProfile: HardwareBenchmarkProfile? = nil
    ) {
        self.profile = profile
        self.benchmarkProfile = benchmarkProfile ?? HardwareBenchmarkProfileStore.loadSync()
    }
    
    public func score(_ model: ModelDefinition) -> FitScore {
        let ramFit = calculateRAMFit(model: model, available: profile.availableRAM)
        let contextFit = calculateContextFit(model: model)
        
        let (speedFit, estTPS, isMeasured, measuredTPS, calibFactor) = calculateSpeedAndCalibration(
            model: model,
            chip: profile.chipModel,
            gpuName: profile.gpuName
        )
        
        let fourD = FourDimensionalScore(
            quality: model.quality,
            speed: speedFit,
            fit: ramFit,
            context: contextFit
        )
        
        let totalScore = fourD.compositeScore
        
        let availableBytes = profile.availableRAM
        let modelBytes = model.ramMB * 1_048_576
        let budgetBytes = Int64(Double(availableBytes) * 0.85)
        
        let fitLevel: FitLevel
        if modelBytes <= Int64(Double(availableBytes) * 0.70) && ramFit >= 0.8 {
            fitLevel = .perfect
        } else if modelBytes <= budgetBytes && ramFit >= 0.5 {
            fitLevel = .good
        } else if modelBytes <= availableBytes {
            fitLevel = .marginal
        } else {
            fitLevel = .tooTight
        }
        
        return FitScore(
            modelName: model.name,
            fitLevel: fitLevel,
            score: totalScore,
            estimatedRAM: model.ramMB,
            estimatedTPS: estTPS,
            contextWindow: model.contextWindow,
            isMeasured: isMeasured,
            measuredTPS: measuredTPS,
            calibrationFactor: calibFactor,
            fourDimensionalScore: fourD,
            selectedQuantization: model.quantization
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
    
    /// Analytical speed modeling with MoE accounting & hardware calibration:
    /// TPS_decode(m,q) = α_calib(m,chip) · η_Metal · BW_mem / ( P_active(m)·b(q)/8 + M_KV_read )
    private func calculateSpeedAndCalibration(
        model: ModelDefinition,
        chip: String,
        gpuName: String
    ) -> (speedFit: Double, estimatedTPS: Int, isMeasured: Bool, measuredTPS: Double?, calibrationFactor: Double) {
        let bw = Self.memoryBandwidthGBs(chip: chip, gpuName: gpuName)
        let etaMetal = 0.70 // 70% Metal memory bandwidth efficiency for decode
        let effectiveBW = bw * etaMetal
        
        let bits = Self.parseBits(from: model.quantization)
        let activeWeightsGB = model.activeParameters * (bits / 8.0)
        let kvReadGB = Self.kvCacheFootprintGB(params: model.totalParameters, contextTokens: model.contextWindow)
        let bytesPerToken = max(0.1, activeWeightsGB + kvReadGB)
        
        let theoreticalTPS = effectiveBW / bytesPerToken
        
        if let measurement = benchmarkProfile.measurement(for: model.id, currentChip: chip) {
            let measuredBits = measurement.quant.map { Self.parseBits(from: $0) } ?? bits
            let ratio = bits > 0 ? (measuredBits / bits) : 1.0
            let adjustedMeasuredTPS = measurement.measuredTPS * ratio
            let calib = theoreticalTPS > 0 ? (adjustedMeasuredTPS / theoreticalTPS) : 1.0
            let tps = max(1, Int(adjustedMeasuredTPS.rounded()))
            let speedFit = min(1.0, Double(tps) / 100.0)
            return (speedFit, tps, true, adjustedMeasuredTPS, calib)
        } else {
            let tps = max(1, Int(theoreticalTPS.rounded()))
            let speedFit = min(1.0, theoreticalTPS / 100.0)
            return (speedFit, tps, false, nil, 1.0)
        }
    }
    
    public static func kvCacheFootprintGB(params: Double, contextTokens: Int) -> Double {
        let referenceTokens = min(Double(contextTokens), 4096.0)
        let gbPer4k = (min(32.0, params) * 0.0015 + 0.01)
        return max(0.01, (referenceTokens / 4096.0) * gbPer4k)
    }

    public static func kvCacheMemoryMB(params: Double, contextTokens: Int) -> Double {
        let gbPer4k = (min(32.0, params) * 0.0015 + 0.01)
        let gb = (Double(contextTokens) / 4096.0) * gbPer4k
        return max(64.0, gb * 1024.0)
    }

    public static func memoryBandwidthGBs(chip: String, gpuName: String) -> Double {
        let combined = (chip + " " + gpuName).uppercased()
        if combined.contains("ULTRA") { return 800.0 }
        if combined.contains("MAX") { return 300.0 }
        if combined.contains("PRO") { return 150.0 }
        if combined.contains("M4") || combined.contains("M5") { return 120.0 }
        if combined.contains("M3") { return 100.0 }
        if combined.contains("M2") { return 100.0 }
        if combined.contains("M1") { return 68.0 }
        return 100.0
    }
    
    public static func parseBits(from quantization: String) -> Double {
        let q = quantization.lowercased()
        if q.contains("16") || q.contains("bfloat16") || q.contains("fp16") { return 16.0 }
        if q.contains("8") { return 8.0 }
        if q.contains("6") { return 6.0 }
        if q.contains("5") { return 5.0 }
        if q.contains("4") { return 4.0 }
        if q.contains("3") { return 3.0 }
        if q.contains("2") { return 2.0 }
        if let match = q.range(of: #"\b\d+(\.\d+)?\b"#, options: .regularExpression),
           let val = Double(q[match]) {
            return val
        }
        return 4.0 // default 4-bit
    }
}
