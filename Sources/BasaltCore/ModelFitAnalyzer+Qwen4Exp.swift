import Foundation

public struct Qwen4ExpTierRecommendation: Sendable, Equatable, Codable {
    public let tier: ComposeMemoryTier
    public let primaryPack: String
    public let alternativePack: String
    public let mode: ExpertArenaPlan.Mode
    public let mtpEnabled: Bool
    public let ngramResident: Bool
    public let recommendedPromptChunkTokens: Int
    public let estimatedDenseGB: Double
    public let estimatedExpertsGB: Double
    public let estimatedKVGB: Double
    public let estimatedScratchGB: Double
    public let plannedArenaGB: Double
    public let workingSetCapGB: Double
}

extension ModelFitAnalyzer {
    public static let ssdResidentPLEKeyMarker = "ple.ple_embedding.ngram_embedding.shards."
    public static let defaultNGramRowCacheBytes = ByteCount(128 * 1_048_576)

    public struct ShardPartitionCounts: Sendable, Equatable {
        public let fileSize: Int64
        public let expertBytes: Int64
        public let ssdResidentBytes: Int64
        public let sharedExpertBytes: Int64
        public let weightCount: Int
    }

    public static func isSSDResidentPLEKey(_ key: String) -> Bool {
        key.hasPrefix(ssdResidentPLEKeyMarker) || key.contains("." + ssdResidentPLEKeyMarker)
    }

    public static func isSharedExpertKey(_ key: String) -> Bool {
        key.contains(".shared_expert.") || key.contains(".shared_experts.") || key.contains(".shared_expert_gate.")
    }

    public static func ssdResidentBytes(in directory: URL) -> ByteCount {
        let shards = safetensorsShards(in: directory)
        var total: Int64 = 0
        for url in shards {
            guard let parsed = inspectShardPartition(at: url) else { continue }
            total += parsed.ssdResidentBytes
        }
        return ByteCount(total)
    }

    public static func inspectShardPartition(at url: URL) -> ShardPartitionCounts? {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)[.size] as? Int64) ?? 0
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8),
              let headerLen = ModelInspector.safetensorsHeaderLength(prefix, fileSize: size),
              let headerData = try? handle.read(upToCount: headerLen), headerData.count == headerLen,
              let entries = (try? JSONSerialization.jsonObject(with: headerData)) as? [String: Any] else {
            return nil
        }
        return partitionHeaderEntries(entries, headerLength: headerLen, fileSize: size)
    }

    public static func partitionHeaderEntries(
        _ entries: [String: Any],
        headerLength: Int,
        fileSize: Int64
    ) -> ShardPartitionCounts? {
        var expertBytes: Int64 = 0
        var ssdResidentBytes: Int64 = 0
        var sharedExpertBytes: Int64 = 0
        var weightCount = 0

        for (key, value) in entries where key != "__metadata__" {
            if isSSDResidentPLEKey(key) {
                guard let span = rawTensorBytes(value: value, headerLength: headerLength, fileSize: fileSize) else {
                    return nil
                }
                ssdResidentBytes += span
            } else if isStackedExpertKey(key) {
                guard let span = stackedTensorBytes(key: key, value: value, headerLength: headerLength, fileSize: fileSize) else {
                    return nil
                }
                expertBytes += span.bytes
                if span.isWeight { weightCount += 1 }
            } else if isSharedExpertKey(key) {
                guard let span = rawTensorBytes(value: value, headerLength: headerLength, fileSize: fileSize) else {
                    return nil
                }
                sharedExpertBytes += span
            }
        }
        return ShardPartitionCounts(
            fileSize: fileSize,
            expertBytes: expertBytes,
            ssdResidentBytes: ssdResidentBytes,
            sharedExpertBytes: sharedExpertBytes,
            weightCount: weightCount
        )
    }

    private static func rawTensorBytes(value: Any, headerLength: Int, fileSize: Int64) -> Int64? {
        let payloadLimit = fileSize - 8 - Int64(headerLength)
        guard let entry = value as? [String: Any],
              let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
              0 <= offsets[0], offsets[0] <= offsets[1], Int64(offsets[1]) <= payloadLimit else {
            return nil
        }
        return Int64(offsets[1] - offsets[0])
    }
}

extension MemoryEvaluator {
    public static func qwen4ExpRecommendation(for tier: ComposeMemoryTier) -> Qwen4ExpTierRecommendation {
        let spec = qwen4ExpTierSpec(for: tier)
        let gib = Double(ExpertArenaPlanner.gib)
        let totalBytes = ByteCount(Int64(tier.nominalGB) * ExpertArenaPlanner.gib)
        let workingSetBytes = ByteCount(Int64((tier.workingSetGB * gib).rounded()))
        let host = ExpertArenaHost(
            totalRAMBytes: totalBytes,
            availableMemoryBytes: totalBytes,
            metalWorkingSetBytes: workingSetBytes
        )
        let workload = ExpertArenaWorkload(
            denseWeightsBytes: ByteCount(Int64((spec.denseGB * gib).rounded())),
            kvCacheBytes: ByteCount(Int64((spec.kvGB * gib).rounded())),
            scratchOverheadBytes: ByteCount(Int64((spec.scratchGB * gib).rounded())),
            expertBytesTotal: ByteCount(Int64((spec.expertsGB * gib).rounded())),
            ngramRowCacheBytes: ModelFitAnalyzer.defaultNGramRowCacheBytes
        )
        let plan = ExpertArenaPlanner.plan(ExpertArenaInput(host: host, workload: workload))
        return Qwen4ExpTierRecommendation(
            tier: tier,
            primaryPack: spec.primary,
            alternativePack: spec.alternative,
            mode: plan.mode,
            mtpEnabled: plan.mode == .resident,
            ngramResident: false,
            recommendedPromptChunkTokens: plan.recommendedPromptChunkTokens,
            estimatedDenseGB: spec.denseGB,
            estimatedExpertsGB: spec.expertsGB,
            estimatedKVGB: spec.kvGB,
            estimatedScratchGB: spec.scratchGB,
            plannedArenaGB: Double(plan.arenaBytes.rawValue) / gib,
            workingSetCapGB: Double(plan.workingSetCapBytes.rawValue) / gib
        )
    }

    public static func qwen4ExpRecommendation(for resources: SystemProfiler.SystemResources) -> Qwen4ExpTierRecommendation {
        qwen4ExpRecommendation(for: ComposeMemoryTier.detect(ramGB: resources.totalRAMGB))
    }

    private struct Qwen4ExpSpec {
        let primary: String
        let alternative: String
        let denseGB: Double
        let expertsGB: Double
        let kvGB: Double
        let scratchGB: Double
    }

    private static func qwen4ExpTierSpec(for tier: ComposeMemoryTier) -> Qwen4ExpSpec {
        switch tier {
        case .gb16:
            Qwen4ExpSpec(primary: "ISTA-DASLab/Qwen3.8-Flash-Next-Coder-IQ1_M", alternative: "mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit", denseGB: 3.6, expertsGB: 23.0, kvGB: 0.4, scratchGB: 1.5)
        case .gb24:
            Qwen4ExpSpec(primary: "ISTA-DASLab/Qwen3.8-Flash-Next-Q2_0", alternative: "mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit", denseGB: 3.6, expertsGB: 34.0, kvGB: 0.5, scratchGB: 1.5)
        case .gb36:
            Qwen4ExpSpec(primary: "ISTA-DASLab/Qwen3.8-Flash-Next-Q2_0", alternative: "ISTA-DASLab/Qwen3.8-Flash-Next-IQ2_XS", denseGB: 3.6, expertsGB: 34.0, kvGB: 0.9, scratchGB: 1.5)
        case .gb48:
            Qwen4ExpSpec(primary: "ISTA-DASLab/Qwen3.8-Flash-Next-Q2_0", alternative: "ISTA-DASLab/Qwen3.8-Flash-Next-IQ3_XXS", denseGB: 3.6, expertsGB: 34.0, kvGB: 0.9, scratchGB: 1.5)
        case .gb64:
            Qwen4ExpSpec(primary: "ISTA-DASLab/Qwen3.8-Flash-Next-Q2_0", alternative: "ISTA-DASLab/Qwen3.8-Flash-Next-IQ3_XXS", denseGB: 3.6, expertsGB: 34.0, kvGB: 0.9, scratchGB: 2.0)
        case .gb96:
            Qwen4ExpSpec(primary: "ISTA-DASLab/Qwen3.8-Flash-Next-IQ3_S", alternative: "mlx-community/Qwen3.8-Flash-Next-3bit", denseGB: 4.8, expertsGB: 50.0, kvGB: 1.7, scratchGB: 2.5)
        case .gb128:
            Qwen4ExpSpec(primary: "mlx-community/Qwen3.8-Flash-Next-4bit", alternative: "unsloth/Qwen3.8-Flash-Next-UD-Q4_K_XL", denseGB: 4.0, expertsGB: 67.9, kvGB: 1.7, scratchGB: 2.5)
        case .gb192:
            Qwen4ExpSpec(primary: "mlx-community/Qwen3.8-Flash-Next-6bit", alternative: "unsloth/Qwen3.8-Flash-Next-UD-Q6_K_XL", denseGB: 4.0, expertsGB: 98.1, kvGB: 3.2, scratchGB: 3.0)
        case .gb256:
            Qwen4ExpSpec(primary: "unsloth/Qwen3.8-Flash-Next-Q8_0", alternative: "mlx-community/Qwen3.8-Flash-Next-8bit", denseGB: 4.8, expertsGB: 128.3, kvGB: 6.4, scratchGB: 3.0)
        }
    }
}
