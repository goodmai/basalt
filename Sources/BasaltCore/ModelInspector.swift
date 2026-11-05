import Foundation

public actor ModelInspector {

    public init() {}

    public struct Report: Codable, Sendable {
        public let modelId: String
        public let path: String
        public let intact: Bool
        public let totalShards: Int
        public let totalSizeBytes: Int64
        public let totalSizeFormatted: String
        public let detectedBitWidth: Int
        public let bitWidthDescription: String
        public let quantizationMode: String?
        public let groupSize: Int?
        public let tensorDtype: String
        public let shardFiles: [ShardInfo]
        public let errors: [String]
        public var files: [HubFileVerdict]? = nil
    }

    public struct ShardInfo: Codable, Sendable {
        public let name: String
        public let sizeBytes: Int64
        public let sizeFormatted: String
        public let valid: Bool
    }

    public func inspect(modelIdOrPath: String, quant: String? = nil) async -> Report {
        await inspect(modelIdOrPath: modelIdOrPath, quant: quant, verifyingWith: nil)
    }

    package func inspect(modelIdOrPath: String, quant: String? = nil, verifyingWith hub: HuggingFaceHub?) async -> Report {
        let (dirURL, resolvedId) = resolveModelDirectory(modelIdOrPath: modelIdOrPath, quant: quant)
        guard let dirURL, FileManager.default.fileExists(atPath: dirURL.path) else {
            return Report(missingDirectory: dirURL?.path ?? modelIdOrPath, modelId: resolvedId)
        }

        let fileURLs = (try? FileManager.default.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: [.fileSizeKey], options: [])) ?? []
        let shardFiles = fileURLs.filter(Self.isWeightShard).sorted { $0.lastPathComponent < $1.lastPathComponent }

        var scan = ShardScan(errors: indexProblems(in: dirURL))
        if shardFiles.isEmpty {
            scan.errors.append("No weight shards (.safetensors, .npz, .bin) found in \(dirURL.path)")
        }
        for fileURL in shardFiles {
            scan.record(fileURL.lastPathComponent, shardOutcome(fileURL))
        }
        var files: [HubFileVerdict]?
        if let hub { files = await HubSnapshot(directory: dirURL).verify(hub: hub) }
        scan.errors += (files ?? []).filter { $0.verified == .mismatch }.map { "Digest mismatch: \($0.name)" }
        var report = Report(modelId: resolvedId, path: dirURL.path, scan: scan, config: ModelCache.config(at: dirURL))
        report.files = files
        return report
    }

    private func indexProblems(in dirURL: URL) -> [String] {
        let indexURL = dirURL.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return [] }
        guard let data = try? Data(contentsOf: indexURL), let required = Self.requiredShards(index: data) else {
            return ["Malformed model.safetensors.index.json"]
        }
        return required
            .filter { !FileManager.default.fileExists(atPath: dirURL.appendingPathComponent($0).path) }
            .map { "Missing shard referenced in index: \($0)" }
    }

    private func shardOutcome(_ fileURL: URL) -> ShardOutcome {
        let resolved = fileURL.resolvingSymlinksInPath()
        guard FileManager.default.fileExists(atPath: resolved.path) else { return .missingBlob }
        let size = (try? resolved.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { Int64($0) } ?? 0
        guard fileURL.pathExtension.lowercased() == "safetensors" else { return .present(sizeBytes: size, header: nil) }
        return .present(sizeBytes: size, header: Self.readSafetensors(at: resolved, fileSize: size))
    }

    private func resolveModelDirectory(modelIdOrPath: String, quant: String?) -> (URL?, String) {
        let expanded = (modelIdOrPath as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            let url = URL(fileURLWithPath: expanded)
            return (url, url.lastPathComponent)
        }
        let dir = ModelCache.resolve(repoId: modelIdOrPath, quant: quant)
            ?? ModelCache.cacheDir(for: modelIdOrPath).appendingPathComponent(quant ?? "")
        return (dir, modelIdOrPath)
    }
}
