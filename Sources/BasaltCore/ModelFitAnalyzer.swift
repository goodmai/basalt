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
                model: model, bits: Double(bits), contextTokens: lMin, resources: resources, profile: profile
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

extension ModelFitAnalyzer {
    public struct StackedExpertsSummary: Sendable, Equatable {
        public let totalShardBytes: ByteCount
        public let expertBytes: ByteCount
        public let ssdResidentBytes: ByteCount
        public let sharedExpertBytes: ByteCount

        public init(
            totalShardBytes: ByteCount,
            expertBytes: ByteCount,
            ssdResidentBytes: ByteCount = .zero,
            sharedExpertBytes: ByteCount = .zero
        ) {
            self.totalShardBytes = totalShardBytes
            self.expertBytes = expertBytes
            self.ssdResidentBytes = ssdResidentBytes
            self.sharedExpertBytes = sharedExpertBytes
        }

        public var workingSetShardBytes: ByteCount { max(.zero, totalShardBytes - ssdResidentBytes) }
        public var residentWeights: ByteCount { max(.zero, totalShardBytes - expertBytes - ssdResidentBytes) }
    }

    public static func hasStackedExperts(in directory: URL) -> Bool {
        stackedExperts(in: directory) != nil
    }

    public static func safetensorsShards(in directory: URL) -> [URL] {
        let index = directory.appending(component: "model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
           let map = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["weight_map"] as? [String: String] {
            let urls = Set(map.values).sorted().map { directory.appending(component: $0) }
            if !urls.isEmpty, urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) { return urls }
        }
        let entries = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return entries.filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    public static func stackedExperts(in directory: URL) -> StackedExpertsSummary? {
        guard let config = ModelCache.config(at: directory),
              let modelType = config["model_type"] as? String, !modelType.isEmpty,
              config["gguf_quantization"] == nil, config["qtip"] == nil,
              config["macaq_quantization"] == nil, config["flatquant_quantization"] == nil,
              config["lirquant_quantization"] == nil else {
            return nil
        }
        let shards = safetensorsShards(in: directory)
        guard !shards.isEmpty else { return nil }
        var totalShardBytes: Int64 = 0
        var expertBytes: Int64 = 0
        var ssdResidentBytes: Int64 = 0
        var sharedExpertBytes: Int64 = 0
        var weightCount = 0
        for url in shards {
            guard let parsed = inspectShardPartition(at: url) else { return nil }
            totalShardBytes += parsed.fileSize
            expertBytes += parsed.expertBytes
            ssdResidentBytes += parsed.ssdResidentBytes
            sharedExpertBytes += parsed.sharedExpertBytes
            weightCount += parsed.weightCount
        }
        return weightCount > 0 ? StackedExpertsSummary(
            totalShardBytes: ByteCount(totalShardBytes),
            expertBytes: ByteCount(expertBytes),
            ssdResidentBytes: ByteCount(ssdResidentBytes),
            sharedExpertBytes: ByteCount(sharedExpertBytes)
        ) : nil
    }

    public static func classifyDirectory(
        _ directory: URL, workingSetBudget: ByteCount, kvBytes: ByteCount = .zero, streaming: ExpertStreaming = ExpertStreaming()
    ) -> (fitLevel: FitLevel, streaming: ExpertStreamingFootprint?) {
        let shards = safetensorsShards(in: directory)
        let rawWeights = ByteCount(shards.reduce(Int64(0)) { $0 + resolvedFileSize(of: $1) })
        let weights = max(.zero, rawWeights - ssdResidentBytes(in: directory))
        guard rawWeights > .zero else { return (.tooTight, nil) }
        let level = MemoryEvaluator.fitLevel(required: weights + kvBytes, budget: workingSetBudget)
        guard level == .tooTight else { return (level, nil) }
        guard let summary = stackedExperts(in: directory),
              let footprint = ExpertStreamingFootprint(
                  residentWeights: summary.residentWeights, expertBytes: summary.expertBytes, kv: kvBytes, streaming: streaming),
              footprint.required <= workingSetBudget else {
            return (.tooTight, nil)
        }
        return (.moeTieredStreaming, footprint)
    }

    public static func autoExpertStreaming(
        directory: URL, workingSetBudget: ByteCount, kvBytes: ByteCount = .zero, explicit: ExpertStreaming? = nil,
        disabled: Bool = false, environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ExpertStreaming? {
        if let explicit { return disabled ? nil : explicit }
        guard !disabled, !ExpertStreaming.isDisabledByEnvironment(environment) else { return nil }
        return classifyDirectory(directory, workingSetBudget: workingSetBudget, kvBytes: kvBytes).fitLevel == .moeTieredStreaming
            ? ExpertStreaming() : nil
    }

    private static func resolvedFileSize(of url: URL) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)[.size] as? Int64) ?? 0
    }

    static func isStackedExpertKey(_ key: String) -> Bool {
        key.contains(".switch_mlp.") || key.contains(".switch_glu.")
            || key.contains(".experts.gate_proj.") || key.contains(".experts.up_proj.") || key.contains(".experts.down_proj.")
            || key.contains(".experts.gate_up_proj.") || key.contains(".experts.w13.") || key.contains(".experts.w2.")
    }

    static func stackedTensorBytes(
        key: String, value: Any, headerLength: Int, fileSize: Int64
    ) -> (bytes: Int64, isWeight: Bool)? {
        let payloadLimit = fileSize - 8 - Int64(headerLength)
        guard let entry = value as? [String: Any], let shape = entry["shape"] as? [Int],
              let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
              0 <= offsets[0], offsets[0] < offsets[1], Int64(offsets[1]) <= payloadLimit else {
            return nil
        }
        let spanBytes = offsets[1] - offsets[0]
        if key.hasSuffix(".weight") {
            guard shape.count == 3, shape[0] > 1, spanBytes % shape[0] == 0 else { return nil }
            return (Int64(spanBytes), true)
        }
        if key.hasSuffix(".scales") || key.hasSuffix(".biases") || key.hasSuffix(".bias") {
            guard !shape.isEmpty, shape[0] > 1, spanBytes % shape[0] == 0 else { return nil }
            return (Int64(spanBytes), false)
        }
        return (0, false)
    }
}
