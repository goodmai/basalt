import Foundation

// MARK: — Model Inspector (Shard Integrity & Bit Width Verification)

/// Analyzes local MLX/Safetensors weights to verify:
/// 1. Weight shard presence and integrity (safetensors headers, non-empty files, index completeness)
/// 2. Actual bit width and quantization configuration (4-bit, 8-bit, 16-bit, DQ, OptiQ, NVFP4)
/// 3. Absence of corrupted or truncated shards
public actor ModelInspector {

    public init() {}

    /// Detailed report for automated agent onboarding and diagnostics
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
    }

    public struct ShardInfo: Codable, Sendable {
        public let name: String
        public let sizeBytes: Int64
        public let sizeFormatted: String
        public let valid: Bool
    }

    /// Inspect a model given either a HuggingFace repo ID or a local directory path.
    public func inspect(modelIdOrPath: String) async -> Report {
        let (dirURL, resolvedId) = resolveModelDirectory(modelIdOrPath: modelIdOrPath)
        
        guard let dirURL, FileManager.default.fileExists(atPath: dirURL.path) else {
            return Report(
                modelId: resolvedId,
                path: dirURL?.path ?? modelIdOrPath,
                intact: false,
                totalShards: 0,
                totalSizeBytes: 0,
                totalSizeFormatted: "0 B",
                detectedBitWidth: 0,
                bitWidthDescription: "Not downloaded / missing directory",
                quantizationMode: nil,
                groupSize: nil,
                tensorDtype: "unknown",
                shardFiles: [],
                errors: ["Model directory not found at: \(dirURL?.path ?? modelIdOrPath)"]
            )
        }

        var errors: [String] = []
        var shardInfos: [ShardInfo] = []
        var totalSizeBytes: Int64 = 0
        var detectedDtypes = Set<String>()

        // 1. Scan directory for weight shards & indices
        let fileURLs = (try? FileManager.default.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: [.fileSizeKey], options: [])) ?? []
        
        let shardFiles = fileURLs.filter { url in
            let ext = url.pathExtension.lowercased()
            return ext == "safetensors" || ext == "npz" || ext == "bin"
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }

        let indexURL = dirURL.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            if let data = try? Data(contentsOf: indexURL),
               let indexJSON = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let weightMap = indexJSON["weight_map"] as? [String: String] {
                let requiredShards = Set(weightMap.values)
                for required in requiredShards {
                    let shardURL = dirURL.appendingPathComponent(required)
                    if !FileManager.default.fileExists(atPath: shardURL.path) {
                        errors.append("Missing shard referenced in index: \(required)")
                    }
                }
            } else {
                errors.append("Malformed model.safetensors.index.json")
            }
        }

        if shardFiles.isEmpty {
            errors.append("No weight shards (.safetensors, .npz, .bin) found in \(dirURL.path)")
        }

        // 2. Validate each shard
        for fileURL in shardFiles {
            let resolved = fileURL.resolvingSymlinksInPath()
            let name = fileURL.lastPathComponent
            let size = (try? resolved.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { Int64($0) } ?? 0
            totalSizeBytes += size

            if size == 0 {
                errors.append("Weight shard is empty (0 bytes): \(name)")
                shardInfos.append(ShardInfo(name: name, sizeBytes: 0, sizeFormatted: "0 B", valid: false))
                continue
            }

            var shardValid = true
            if fileURL.pathExtension.lowercased() == "safetensors" {
                let (headerValid, dtypes) = validateSafetensorsHeader(at: resolved, fileSize: size)
                if !headerValid {
                    errors.append("Invalid or corrupted safetensors header: \(name)")
                    shardValid = false
                }
                detectedDtypes.formUnion(dtypes)
            }

            shardInfos.append(ShardInfo(
                name: name,
                sizeBytes: size,
                sizeFormatted: formatBytes(size),
                valid: shardValid
            ))
        }

        // 3. Inspect config.json for bit width and quantization parameters
        var configBits: Int?
        var configGroupSize: Int?
        var configMode: String?
        var configDtype: String?

        let configURL = dirURL.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path),
           let configData = try? Data(contentsOf: configURL),
           let configJSON = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            
            if let qDict = configJSON["quantization"] as? [String: Any] {
                configBits = qDict["bits"] as? Int
                configGroupSize = qDict["group_size"] as? Int
                configMode = qDict["mode"] as? String
            } else if let qConfig = configJSON["quantization_config"] as? [String: Any] {
                configBits = qConfig["bits"] as? Int ?? qConfig["weight_bits"] as? Int
                configGroupSize = qConfig["group_size"] as? Int
                configMode = qConfig["quant_method"] as? String
            }
            configDtype = configJSON["torch_dtype"] as? String
        }

        // 4. Derive detected bit width
        let bitWidth: Int
        let bitDesc: String
        let primaryDtype = detectedDtypes.sorted().joined(separator: ", ")

        if let b = configBits {
            bitWidth = b
            let modeDesc = configMode.map { " (\($0))" } ?? ""
            bitDesc = "\(b)-bit quantized\(modeDesc)"
        } else if detectedDtypes.contains("uint32") || resolvedId.lowercased().contains("4bit") {
            bitWidth = 4
            bitDesc = "4-bit packed"
        } else if detectedDtypes.contains("I8") || detectedDtypes.contains("int8") || resolvedId.lowercased().contains("8bit") {
            bitWidth = 8
            bitDesc = "8-bit quantized"
        } else if detectedDtypes.contains("F16") || detectedDtypes.contains("BF16") || configDtype?.lowercased().contains("bfloat16") == true || configDtype?.lowercased().contains("float16") == true {
            bitWidth = 16
            bitDesc = "16-bit unquantized (FP16/BF16)"
        } else if detectedDtypes.contains("F32") {
            bitWidth = 32
            bitDesc = "32-bit floating point"
        } else {
            bitWidth = 16
            bitDesc = "16-bit (default)"
        }

        let intact = errors.isEmpty && !shardInfos.isEmpty

        return Report(
            modelId: resolvedId,
            path: dirURL.path,
            intact: intact,
            totalShards: shardInfos.count,
            totalSizeBytes: totalSizeBytes,
            totalSizeFormatted: formatBytes(totalSizeBytes),
            detectedBitWidth: bitWidth,
            bitWidthDescription: bitDesc,
            quantizationMode: configMode,
            groupSize: configGroupSize,
            tensorDtype: primaryDtype.isEmpty ? (configDtype ?? "unknown") : primaryDtype,
            shardFiles: shardInfos,
            errors: errors
        )
    }

    // MARK: — Helpers

    private func resolveModelDirectory(modelIdOrPath: String) -> (URL?, String) {
        let expanded = (modelIdOrPath as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            let url = URL(fileURLWithPath: expanded)
            return (url, url.lastPathComponent)
        }

        let cached = ModelCache.cacheDir(for: modelIdOrPath)
        if FileManager.default.fileExists(atPath: cached.path) {
            return (cached, modelIdOrPath)
        }

        // Direct check under models--
        let sanitized = modelIdOrPath.replacingOccurrences(of: "/", with: "--")
        let modelRoot = ModelCache.root.appendingPathComponent("models--\(sanitized)")
        if FileManager.default.fileExists(atPath: modelRoot.path) {
            let snapshots = modelRoot.appendingPathComponent("snapshots")
            if let entries = try? FileManager.default.contentsOfDirectory(atPath: snapshots.path),
               let first = entries.first {
                return (snapshots.appendingPathComponent(first), modelIdOrPath)
            }
        }

        return (cached, modelIdOrPath)
    }

    private func validateSafetensorsHeader(at url: URL, fileSize: Int64) -> (Bool, Set<String>) {
        guard fileSize >= 8, let handle = try? FileHandle(forReadingFrom: url) else {
            return (false, [])
        }
        defer { try? handle.close() }

        guard let lenData = try? handle.read(upToCount: 8), lenData.count == 8 else {
            return (false, [])
        }

        let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        guard headerLen > 0, headerLen <= UInt64(fileSize - 8) else {
            return (false, [])
        }

        guard let headerData = try? handle.read(upToCount: Int(headerLen)),
              headerData.count == Int(headerLen) else {
            return (false, [])
        }

        guard let json = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            return (false, [])
        }

        var dtypes = Set<String>()
        for (key, val) in json where key != "__metadata__" {
            if let tDict = val as? [String: Any], let dt = tDict["dtype"] as? String {
                dtypes.insert(dt)
            }
        }

        return (true, dtypes)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        switch bytes {
        case 0..<1_024:
            return "\(bytes) B"
        case 0..<1_048_576:
            return String(format: "%.1f KB", Double(bytes) / 1_024.0)
        case 0..<1_073_741_824:
            return String(format: "%.1f MB", Double(bytes) / 1_048_576.0)
        default:
            return String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
        }
    }
}
