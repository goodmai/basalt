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
    public func inspect(modelIdOrPath: String, quant: String? = nil) async -> Report {
        let (dirURL, resolvedId) = resolveModelDirectory(modelIdOrPath: modelIdOrPath, quant: quant)
        
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
        var tensors: [String: Tensor] = [:]

        // 1. Scan directory for weight shards & indices
        let fileURLs = (try? FileManager.default.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: [.fileSizeKey], options: [])) ?? []

        // Weights only: `.bin` is ambiguous (training_args.bin), so only pytorch_model*.bin counts.
        let shardFiles = fileURLs.filter { url in
            let ext = url.pathExtension.lowercased()
            return ext == "safetensors" || ext == "npz"
                || (ext == "bin" && url.lastPathComponent.hasPrefix("pytorch_model"))
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
            guard FileManager.default.fileExists(atPath: resolved.path) else {
                // HF cache shards are symlinks into blobs/; a dangling one is a missing download.
                errors.append("Weight shard points to a missing blob: \(name)")
                shardInfos.append(ShardInfo(name: name, sizeBytes: 0, sizeFormatted: "0 B", valid: false))
                continue
            }
            let size = (try? resolved.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { Int64($0) } ?? 0
            totalSizeBytes += size

            if size == 0 {
                errors.append("Weight shard is empty (0 bytes): \(name)")
                shardInfos.append(ShardInfo(name: name, sizeBytes: 0, sizeFormatted: "0 B", valid: false))
                continue
            }

            var shardValid = true
            if fileURL.pathExtension.lowercased() == "safetensors" {
                switch Self.readSafetensors(at: resolved, fileSize: size) {
                case .failure(let problem):
                    errors.append("\(problem.message): \(name)")
                    shardValid = false
                case .success(let header):
                    tensors.merge(header.tensors) { first, _ in first }
                    detectedDtypes.formUnion(header.tensors.values.map(\.dtype))
                }
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
        var layerGroupSizes: [String: Int] = [:]

        if let configJSON = ModelCache.config(at: dirURL) {
            
            if let qDict = configJSON["quantization"] as? [String: Any] {
                configBits = qDict["bits"] as? Int
                configGroupSize = qDict["group_size"] as? Int
                configMode = qDict["mode"] as? String
                // MLX mixed (DQ) models override bits/group_size per layer under the same dict.
                for (layer, value) in qDict {
                    if let override = value as? [String: Any], let gs = override["group_size"] as? Int {
                        layerGroupSizes[layer] = gs
                    }
                }
            } else if let qConfig = configJSON["quantization_config"] as? [String: Any] {
                configBits = qConfig["bits"] as? Int ?? qConfig["weight_bits"] as? Int
                configGroupSize = qConfig["group_size"] as? Int
                configMode = qConfig["quant_method"] as? String
            }
            configDtype = configJSON["torch_dtype"] as? String
        }
        // serve refuses a directory without a parseable config.json (ModelCache.resolve), so the
        // inspector must not call such a model intact — e.g. a download cut off at config.json.
        if !ModelCache.hasUsableConfig(at: dirURL) {
            errors.append("config.json is missing or not valid JSON")
        }

        // 4. Derive the bit width from the weights themselves. An MLX quantized layer stores
        // `X.weight` as uint32 words [out, in·bits/32] next to `X.scales` [out, in/group_size],
        // so bits = 32·weight[-1] / (scales[-1]·group_size). config.json is a claim checked
        // against that, not the answer.
        let primaryDtype = detectedDtypes.sorted().joined(separator: ", ")
        // MLX's default group size depends on the mode: 16 for nvfp4, 32 for mxfp4/mxfp8, else 64.
        let defaultGroupSize: Int
        switch configMode?.lowercased() {
        case "nvfp4": defaultGroupSize = 16
        case "mxfp4", "mxfp8": defaultGroupSize = 32
        default: defaultGroupSize = 64
        }
        let measured = Self.quantizedBits(tensors: tensors) { layer in
            layerGroupSizes[layer] ?? configGroupSize ?? defaultGroupSize
        }
        let measuredBits = Set(measured.values).sorted()
        let measuredList = measuredBits.map(String.init).joined(separator: "/")

        let bitWidth: Int
        let bitDesc: String
        if let common = Self.mostCommon(Array(measured.values)) {
            bitWidth = common
            bitDesc = measuredBits.count > 1
                ? "mixed \(measuredList)-bit quantized (measured over \(measured.count) layers)"
                : "\(common)-bit quantized (measured from weights\(configMode.map { ", \($0)" } ?? ""))"
            if let claimed = configBits, !measuredBits.contains(claimed) {
                errors.append("config.json claims \(claimed)-bit but the weights are \(measuredList)-bit")
            }
        } else if let b = configBits {
            bitWidth = b
            bitDesc = "\(b)-bit quantized (from config.json; no packed layers to measure)"
        } else if detectedDtypes.contains("I8") || detectedDtypes.contains("U8") {
            bitWidth = 8
            bitDesc = "8-bit quantized"
        } else if detectedDtypes.contains("F16") || detectedDtypes.contains("BF16") || configDtype?.lowercased().contains("bfloat16") == true || configDtype?.lowercased().contains("float16") == true {
            bitWidth = 16
            bitDesc = "16-bit unquantized (FP16/BF16)"
        } else if detectedDtypes.contains("F32") {
            bitWidth = 32
            bitDesc = "32-bit floating point"
        } else if resolvedId.lowercased().contains("4bit") {
            // Name heuristic only when neither weights nor config say anything.
            bitWidth = 4
            bitDesc = "4-bit (from repo name, unverified)"
        } else if resolvedId.lowercased().contains("8bit") {
            bitWidth = 8
            bitDesc = "8-bit (from repo name, unverified)"
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

    private func resolveModelDirectory(modelIdOrPath: String, quant: String?) -> (URL?, String) {
        let expanded = (modelIdOrPath as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            let url = URL(fileURLWithPath: expanded)
            return (url, url.lastPathComponent)
        }

        // Same resolution `serve` uses (refs/main → snapshot, then quant subfolders), so we
        // inspect the directory that will actually be loaded.
        let dir = ModelCache.resolve(repoId: modelIdOrPath, quant: quant)
            ?? ModelCache.cacheDir(for: modelIdOrPath).appendingPathComponent(quant ?? "")
        return (dir, modelIdOrPath)
    }

    package struct Tensor: Equatable, Sendable {
        package let dtype: String
        package let shape: [Int]
        package let end: Int
    }

    package struct SafetensorsHeader: Sendable {
        package let tensors: [String: Tensor]
        package let metadata: [String: String]
    }

    package struct ShardProblem: Error {
        package let message: String
    }

    /// Parses a safetensors header (8-byte LE length + JSON) and checks the file is complete:
    /// the data section must end exactly where the largest tensor offset says. A truncated
    /// download keeps an intact header, so only this check catches it.
    package static func readSafetensors(at url: URL, fileSize: Int64) -> Result<SafetensorsHeader, ShardProblem> {
        guard fileSize >= 8, let handle = try? FileHandle(forReadingFrom: url) else {
            return .failure(.init(message: "Unreadable safetensors file"))
        }
        defer { try? handle.close() }

        guard let lenData = try? handle.read(upToCount: 8), lenData.count == 8 else {
            return .failure(.init(message: "Invalid or corrupted safetensors header"))
        }
        let raw = RawSpan(elements: Array(lenData).span)
        let headerLen = raw.load(fromByteOffset: 0, as: UInt64.self).littleEndian
        // The format caps the JSON header at 100 MB; a corrupt length must not make us read
        // a multi-GB shard into memory.
        guard headerLen > 0, headerLen <= 100_000_000, headerLen <= UInt64(fileSize - 8),
              let headerData = try? handle.read(upToCount: Int(headerLen)),
              headerData.count == Int(headerLen),
              let json = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            return .failure(.init(message: "Invalid or corrupted safetensors header"))
        }

        var tensors: [String: Tensor] = [:]
        for (name, value) in json where name != "__metadata__" {
            guard let t = value as? [String: Any], let dtype = t["dtype"] as? String,
                  let shape = t["shape"] as? [Int], let offsets = t["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                return .failure(.init(message: "Malformed tensor entry '\(name)' in safetensors header"))
            }
            tensors[name] = Tensor(dtype: dtype, shape: shape, end: offsets[1])
        }
        guard !tensors.isEmpty else {
            return .failure(.init(message: "Safetensors header lists no tensors"))
        }
        let expected = 8 + Int64(headerLen) + Int64(tensors.values.map(\.end).max() ?? 0)
        guard expected == fileSize else {
            return .failure(.init(message: "Truncated or oversized shard (\(fileSize) bytes, header expects \(expected))"))
        }
        return .success(SafetensorsHeader(tensors: tensors, metadata: json["__metadata__"] as? [String: String] ?? [:]))
    }

    /// Bits per quantized layer, measured from each packed U32 weight and its scales.
    static func quantizedBits(tensors: [String: Tensor], groupSize: (String) -> Int) -> [String: Int] {
        var bits: [String: Int] = [:]
        for (name, weight) in tensors where name.hasSuffix(".weight") && weight.dtype == "U32" {
            let layer = String(name.dropLast(".weight".count))
            guard let scales = tensors[layer + ".scales"],
                  let packed = weight.shape.last, let groups = scales.shape.last else { continue }
            let denominator = groups * groupSize(layer)
            if denominator > 0, (32 * packed) % denominator == 0 {
                bits[layer] = 32 * packed / denominator
            }
        }
        return bits
    }

    /// The most frequent value; ties go to the larger one so the result is deterministic.
    static func mostCommon(_ values: [Int]) -> Int? {
        let counts = Dictionary(values.map { ($0, 1) }, uniquingKeysWith: +)
        return counts.max { a, b in a.value == b.value ? a.key < b.key : a.value < b.value }?.key
    }
}
