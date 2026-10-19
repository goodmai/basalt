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
    /// Best grid width for this machine (see `ModelFitAnalyzer.evaluateQuantizations`), nil if none fits.
    public let optimalQuantization: String?

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
        selectedQuantization: String = "4-bit",
        optimalQuantization: String? = nil
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
        self.optimalQuantization = optimalQuantization
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
        
        // q*(m): the highest-quality width of the 8/6/4/3/2 grid that fits 85% of available
        // memory at L_min and decodes at ≥ TPS_min; nil when none does.
        let grid = ModelFitAnalyzer.evaluateQuantizations(model: model, resources: profile, benchmarkProfile: benchmarkProfile)

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
            selectedQuantization: model.quantization,
            optimalQuantization: grid.optimal.fitsBudget ? grid.optimal.name : nil
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
        let speed = Self.decodeSpeed(
            model: model, bits: Self.parseBits(from: model.quantization), contextTokens: model.contextWindow,
            chip: chip, gpuName: gpuName, profile: benchmarkProfile
        )
        let tps = max(1, Int(speed.tps.rounded()))
        let speedFit = min(1.0, speed.tps / 100.0)
        return (speedFit, tps, speed.measured, speed.measured ? speed.tps : nil, speed.calibrationFactor)
    }

    /// The one decode-speed model, shared by `fit` scoring and the quantization grid:
    /// TPS = η_Metal · BW_mem / (P_active·b/8 + M_KV_read). A benchmark of this model on this chip
    /// replaces the theory, rescaled from the bit width it was measured at (decode is bandwidth
    /// bound, so TPS ∝ 1/bits); a measurement without a recorded quant is taken as the model's own.
    public static func decodeSpeed(
        model: ModelDefinition,
        bits: Double,
        contextTokens: Int,
        chip: String,
        gpuName: String,
        profile: HardwareBenchmarkProfile
    ) -> (tps: Double, measured: Bool, calibrationFactor: Double) {
        let etaMetal = 0.70  // fraction of peak bandwidth Metal sustains during decode
        let effectiveBW = memoryBandwidthGBs(chip: chip, gpuName: gpuName) * etaMetal
        let kvReadGB = kvReadGB(model.kvGeometry, contextTokens: contextTokens)
        let theoretical = effectiveBW / max(0.1, model.activeParameters * (bits / 8.0) + kvReadGB)

        guard bits > 0, let m = profile.measurement(for: model.id, currentChip: chip) else {
            return (theoretical, false, 1.0)
        }
        let measuredBits = m.quant.map { parseBits(from: $0) } ?? parseBits(from: model.quantization)
        let tps = m.measuredTPS * measuredBits / bits
        return (tps, true, theoretical > 0 ? tps / theoretical : 1.0)
    }
    
    public static func kvReadGB(_ kv: KVCacheGeometry, contextTokens: Int) -> Double {
        max(0.01, Double(kv.bytes(tokens: min(contextTokens, 4096))) / 1e9)
    }

    /// Peak unified-memory bandwidth (GB/s) by generation and tier, from Apple's published specs
    /// (full-bin parts; binned Max/Pro parts are lower, which the benchmark calibration corrects).
    /// M5 Pro/Max/Ultra are not published yet: the M4 tier values stand in as a floor.
    public static func memoryBandwidthGBs(chip: String, gpuName: String) -> Double {
        let combined = (chip + " " + gpuName).uppercased()
        let table: [(gen: String, base: Double, pro: Double, max: Double, ultra: Double)] = [
            ("M5", 153, 273, 546, 1092),
            ("M4", 120, 273, 546, 1092),
            ("M3", 100, 150, 400, 819),
            ("M2", 100, 200, 400, 800),
            ("M1", 68, 200, 400, 800),
        ]
        guard let row = table.first(where: { combined.contains($0.gen) }) else { return 100.0 }
        if combined.contains("ULTRA") { return row.ultra }
        if combined.contains("MAX") { return row.max }
        if combined.contains("PRO") { return row.pro }
        return row.base
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
