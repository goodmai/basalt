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
    public var isMeasured = false
    public var measuredTPS: Double?
    public var calibrationFactor = 1.0
    public var fourDimensionalScore: FourDimensionalScore?
    public var selectedQuantization = "4-bit"
    public var optimalQuantization: String?
    public var estimatedPrefillTPS: Double?

    public func estimatedTTFT(promptTokens: Int) -> Double? {
        estimatedPrefillTPS.map { Double(promptTokens) / $0 }
    }

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
        
        let (speedFit, estTPS, isMeasured, measuredTPS, calibFactor) = calculateSpeedAndCalibration(model: model)
        
        let fourD = FourDimensionalScore(
            quality: model.quality,
            speed: speedFit,
            fit: ramFit,
            context: contextFit
        )
        
        let totalScore = fourD.compositeScore
        
        let available = ByteCount(profile.availableRAM)
        let modelBytes = ByteCount(model.ramMB * 1_048_576)
        let budget = ByteCount(Int64(Double(available.rawValue) * 0.85))
        
        let fitLevel: FitLevel
        if modelBytes <= ByteCount(Int64(Double(available.rawValue) * 0.70)) && ramFit >= 0.8 {
            fitLevel = .perfect
        } else if modelBytes <= budget && ramFit >= 0.5 {
            fitLevel = .good
        } else if modelBytes <= available {
            fitLevel = .marginal
        } else {
            fitLevel = .tooTight
        }
        
        // q*(m): the highest-quality width of the 8/6/4/3/2 grid that fits 85% of available
        // memory at L_min and decodes at ≥ TPS_min; nil when none does.
        let grid = ModelFitAnalyzer.evaluateQuantizations(model: model, resources: profile, benchmarkProfile: benchmarkProfile)

        var result = FitScore(
            modelName: model.name, fitLevel: fitLevel, score: totalScore,
            estimatedRAM: model.ramMB, estimatedTPS: estTPS, contextWindow: model.contextWindow)
        result.isMeasured = isMeasured
        result.measuredTPS = measuredTPS
        result.calibrationFactor = calibFactor
        result.fourDimensionalScore = fourD
        result.selectedQuantization = model.quantization
        result.optimalQuantization = grid.optimal.fitsBudget ? grid.optimal.name : nil
        result.estimatedPrefillTPS = Self.prefillSpeed(model: model, resources: profile, profile: benchmarkProfile)?.tps
        return result
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
    
    private func calculateSpeedAndCalibration(
        model: ModelDefinition
    ) -> (speedFit: Double, estimatedTPS: Int, isMeasured: Bool, measuredTPS: Double?, calibrationFactor: Double) {
        let speed = Self.decodeSpeed(
            model: model, bits: Self.parseBits(from: model.quantization), contextTokens: model.contextWindow,
            resources: profile, profile: benchmarkProfile
        )
        let tps = max(1, Int(speed.tps.rounded()))
        let speedFit = min(1.0, speed.tps / 100.0)
        return (speedFit, tps, speed.measured, speed.measured ? speed.tps : nil, speed.calibrationFactor)
    }

    public static let etaMetal = 0.50

    public static func decodeSpeed(
        model: ModelDefinition,
        bits: Double,
        contextTokens: Int,
        resources: SystemProfiler.SystemResources,
        profile: HardwareBenchmarkProfile
    ) -> (tps: Double, measured: Bool, calibrationFactor: Double) {
        let bandwidth = memoryBandwidthGBs(chip: resources.chipModel, gpuName: resources.gpuName, gpuCores: resources.gpuCores)
        let kvReadGB = kvReadGB(model.kvGeometry, contextTokens: contextTokens)
        let theoretical = bandwidth * etaMetal / max(0.1, model.activeParameters * (bits / 8.0) + kvReadGB)

        guard bits > 0, let m = profile.measurement(for: model.id, currentChip: resources.chipModel) else {
            return (theoretical, false, 1.0)
        }
        let measuredBits = m.quant.map { parseBits(from: $0) } ?? parseBits(from: model.quantization)
        let tps = m.measuredTPS * measuredBits / bits
        return (tps, true, theoretical > 0 ? tps / theoretical : 1.0)
    }
    
    public static func kvReadGB(_ kv: KVCacheGeometry, contextTokens: Int) -> Double {
        max(0.01, Double(kv.bytes(tokens: TokenCount(min(contextTokens, 4096))).rawValue) / 1e9)
    }

    public static func memoryBandwidthGBs(chip: String, gpuName: String, gpuCores: Int? = nil) -> Double {
        AppleSiliconChip(chip + " " + gpuName)?.memoryBandwidthGBs(gpuCores: gpuCores) ?? 100.0
    }
    
    /// Bits per weight from a quantization label. An explicit "<n>-bit"/"<n>bit" wins, so digits
    /// elsewhere in the label (a group size, a version) cannot be mistaken for the width.
    public static func parseBits(from quantization: String) -> Double {
        let q = quantization.lowercased()
        if let match = q.range(of: #"\d+(\.\d+)?(?=\s*-?\s*bits?\b)"#, options: .regularExpression),
           let val = Double(q[match]) {
            return val
        }
        if q.contains("bf16") || q.contains("bfloat16") || q.contains("fp16") || q.contains("float16") { return 16.0 }
        if q.contains("fp8") { return 8.0 }      // mxfp8
        if q.contains("fp4") { return 4.0 }      // nvfp4, mxfp4
        return 4.0 // default 4-bit
    }
}
