import Foundation

public struct ModelFitAnalyzer: Sendable {

    public struct QuantizationFitOption: Codable, Sendable {
        public let bits: Int
        public let name: String
        public let footprintMB: Int
        public let estimatedTPS: Int
        public let score: FourDimensionalScore
        public let fitsBudget: Bool
        public let isOptimal: Bool
    }

    private static let gridBits = [8, 6, 4, 3, 2]
    private static let frameworkOverhead = 0.15

    public static func evaluateQuantizations(
        model: ModelDefinition,
        resources: SystemProfiler.SystemResources,
        tpsMin: Double = 10.0,
        lMin: Int = 4096,
        benchmarkProfile: HardwareBenchmarkProfile? = nil
    ) -> (optimal: QuantizationFitOption, options: [QuantizationFitOption]) {
        let modelBudgetMB = Double(resources.availableRAM) / 1_048_576.0 * 0.85
        let profile = benchmarkProfile ?? HardwareBenchmarkProfileStore.loadSync()
        let kvCacheMB = Double(model.kvGeometry.bytes(tokens: TokenCount(lMin)).rawValue) / 1_048_576

        let evaluated = gridBits.map { bits in
            let footprintMB = Int(weightMB(parametersBillions: model.totalParameters, bits: bits) + kvCacheMB)
            let speed = FitScorer.decodeSpeed(
                model: model, bits: Double(bits), contextTokens: lMin,
                chip: resources.chipModel, gpuName: resources.gpuName, profile: profile
            )
            let tps = max(1, Int(speed.tps.rounded()))
            let score = FourDimensionalScore(
                quality: min(1.0, max(0.0, model.quality * qualityMultiplier(bits: bits))),
                speed: min(1.0, Double(tps) / 100.0),
                fit: fitScore(ramRatio: Double(footprintMB) / modelBudgetMB),
                context: min(1.0, Double(model.contextWindow) / 65536.0)
            )
            let fits = Double(footprintMB) <= modelBudgetMB && Double(tps) >= tpsMin
            return (bits: bits, footprintMB: footprintMB, tps: tps, score: score, fits: fits)
        }

        let chosen = evaluated.filter { $0.fits }.max { $0.score.quality < $1.score.quality }
        let options = evaluated.map { candidate in
            QuantizationFitOption(
                bits: candidate.bits,
                name: "\(candidate.bits)-bit",
                footprintMB: candidate.footprintMB,
                estimatedTPS: candidate.tps,
                score: candidate.score,
                fitsBudget: candidate.fits,
                isOptimal: candidate.bits == chosen?.bits
            )
        }
        return (optimal: options.first { $0.isOptimal } ?? options[options.count - 1], options: options)
    }

    static func weightMB(parametersBillions: Double, bits: Int) -> Double {
        let weightGB = parametersBillions * (Double(bits) / 8.0) * (1.0 + frameworkOverhead)
        return weightGB * 1e9 / 1_048_576
    }

    static func qualityMultiplier(bits: Int) -> Double {
        switch bits {
        case 8: 1.00
        case 6: 0.98
        case 4: 0.93
        case 3: 0.84
        default: 0.68
        }
    }

    static func fitScore(ramRatio: Double) -> Double {
        if ramRatio <= 0.7 {
            1.0
        } else if ramRatio <= 1.0 {
            1.0 - (ramRatio - 0.7) * 1.66
        } else {
            max(0.0, 0.5 - (ramRatio - 1.0) * 0.5)
        }
    }
}
