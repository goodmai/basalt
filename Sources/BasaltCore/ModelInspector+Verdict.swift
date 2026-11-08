import Foundation

extension ModelInspector {
    enum ShardOutcome: Sendable {
        case missingBlob
        case present(sizeBytes: Int64, header: Result<SafetensorsHeader, ShardProblem>?)
    }

    struct ShardScan: Sendable {
        var errors: [String] = []
        var shards: [ShardInfo] = []
        var totalSizeBytes: Int64 = 0
        var dtypes = Set<String>()
        var tensors: [String: Tensor] = [:]

        mutating func record(_ name: String, _ outcome: ShardOutcome) {
            guard case .present(let size, let header) = outcome else {
                errors.append("Weight shard points to a missing blob: \(name)")
                shards.append(ShardInfo(name: name, sizeBytes: 0, sizeFormatted: "0 B", valid: false))
                return
            }
            totalSizeBytes += size
            guard size > 0 else {
                errors.append("Weight shard is empty (0 bytes): \(name)")
                shards.append(ShardInfo(name: name, sizeBytes: 0, sizeFormatted: "0 B", valid: false))
                return
            }
            var valid = true
            switch header {
            case .failure(let problem):
                errors.append("\(problem.message): \(name)")
                valid = false
            case .success(let parsed):
                tensors.merge(parsed.tensors) { first, _ in first }
                dtypes.formUnion(parsed.tensors.values.map(\.dtype))
            case nil:
                break
            }
            shards.append(ShardInfo(name: name, sizeBytes: size, sizeFormatted: formatBytes(size), valid: valid))
        }
    }

    static func isWeightShard(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return ext == "safetensors" || ext == "npz"
            || (ext == "bin" && url.lastPathComponent.hasPrefix("pytorch_model"))
    }

    static func requiredShards(index data: Data) -> Set<String>? {
        guard let indexJSON = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = indexJSON["weight_map"] as? [String: String] else { return nil }
        return Set(weightMap.values)
    }
}

extension ModelInspector {
    struct QuantizationConfig: Sendable, Equatable {
        var bits: Int?
        var groupSize: Int?
        var mode: String?
        var dtype: String?
        var layerGroupSizes: [String: Int] = [:]

        init(json: [String: Any]?) {
            guard let json else { return }
            if let quantization = json["quantization"] as? [String: Any] {
                bits = quantization["bits"] as? Int
                groupSize = quantization["group_size"] as? Int
                mode = quantization["quant_method"] as? String == "nf4" ? "nf4" : quantization["mode"] as? String
                for (layer, value) in quantization {
                    if let override = value as? [String: Any], let size = override["group_size"] as? Int {
                        layerGroupSizes[layer] = size
                    }
                }
            } else if let quantization = json["quantization_config"] as? [String: Any] {
                bits = quantization["bits"] as? Int ?? quantization["weight_bits"] as? Int
                groupSize = quantization["group_size"] as? Int
                mode = quantization["quant_method"] as? String
            }
            dtype = json["torch_dtype"] as? String
        }

        var defaultGroupSize: Int {
            switch mode?.lowercased() {
            case "nvfp4": 16
            case "mxfp4", "mxfp8": 32
            default: 64
            }
        }

        func groupSize(forLayer layer: String) -> Int {
            layerGroupSizes[layer] ?? groupSize ?? defaultGroupSize
        }
    }

    struct BitWidth: Sendable, Equatable {
        let bits: Int
        let description: String
        var mismatch: String? = nil
    }

    static func bitWidth(tensors: [String: Tensor], dtypes: Set<String>, config: QuantizationConfig, modelId: String) -> BitWidth {
        let measured = quantizedBits(tensors: tensors, nf4: config.mode?.lowercased() == "nf4", groupSize: config.groupSize(forLayer:))
        guard let common = mostCommon(Array(measured.values)) else {
            return unmeasuredBitWidth(dtypes: dtypes, config: config, modelId: modelId)
        }
        let measuredBits = Set(measured.values).sorted()
        let measuredList = measuredBits.map(String.init).joined(separator: "/")
        let description = measuredBits.count > 1
            ? "mixed \(measuredList)-bit quantized (measured over \(measured.count) layers)"
            : "\(common)-bit quantized (measured from weights\(config.mode.map { ", \($0)" } ?? ""))"
        let mismatch = config.bits.flatMap { claimed in
            measuredBits.contains(claimed) ? nil : "config.json claims \(claimed)-bit but the weights are \(measuredList)-bit"
        }
        return BitWidth(bits: common, description: description, mismatch: mismatch)
    }

    static func unmeasuredBitWidth(dtypes: Set<String>, config: QuantizationConfig, modelId: String) -> BitWidth {
        let configDtype = config.dtype?.lowercased()
        if let bits = config.bits {
            return BitWidth(bits: bits, description: "\(bits)-bit quantized (from config.json; no packed layers to measure)")
        } else if dtypes.contains("I8") || dtypes.contains("U8") {
            return BitWidth(bits: 8, description: "8-bit quantized")
        } else if dtypes.contains("F16") || dtypes.contains("BF16") || configDtype?.contains("float16") == true {
            return BitWidth(bits: 16, description: "16-bit unquantized (FP16/BF16)")
        } else if dtypes.contains("F32") {
            return BitWidth(bits: 32, description: "32-bit floating point")
        } else if modelId.lowercased().contains("4bit") {
            return BitWidth(bits: 4, description: "4-bit (from repo name, unverified)")
        } else if modelId.lowercased().contains("8bit") {
            return BitWidth(bits: 8, description: "8-bit (from repo name, unverified)")
        }
        return BitWidth(bits: 16, description: "16-bit (default)")
    }

    static func quantizedBits(tensors: [String: Tensor], nf4: Bool = false, groupSize: (String) -> Int) -> [String: Int] {
        var bits: [String: Int] = [:]
        for (name, weight) in tensors where name.hasSuffix(".weight") && weight.dtype == "U32" {
            let layer = String(name.dropLast(".weight".count))
            guard let scales = tensors[layer + ".scales"],
                  let packed = weight.shape.last, let groups = scales.shape.last else { continue }
            if nf4, tensors[layer + ".biases"] == nil {
                bits[layer] = 4
                continue
            }
            let denominator = groups * groupSize(layer)
            if denominator > 0, (32 * packed) % denominator == 0 {
                bits[layer] = 32 * packed / denominator
            }
        }
        return bits
    }

    static func mostCommon(_ values: [Int]) -> Int? {
        let counts = Dictionary(values.map { ($0, 1) }, uniquingKeysWith: +)
        return counts.max { a, b in a.value == b.value ? a.key < b.key : a.value < b.value }?.key
    }
}

extension ModelInspector.Report {
    init(missingDirectory path: String, modelId: String) {
        self.init(
            modelId: modelId,
            path: path,
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
            errors: ["Model directory not found at: \(path)"]
        )
    }

    init(modelId: String, path: String, scan: ModelInspector.ShardScan, config: [String: Any]?) {
        let quantization = ModelInspector.QuantizationConfig(json: config)
        var errors = scan.errors
        if config == nil {
            errors.append("config.json is missing or not valid JSON")
        }
        let bitWidth = ModelInspector.bitWidth(tensors: scan.tensors, dtypes: scan.dtypes, config: quantization, modelId: modelId)
        if let mismatch = bitWidth.mismatch {
            errors.append(mismatch)
        }
        let primaryDtype = scan.dtypes.sorted().joined(separator: ", ")
        self.init(
            modelId: modelId,
            path: path,
            intact: errors.isEmpty && !scan.shards.isEmpty,
            totalShards: scan.shards.count,
            totalSizeBytes: scan.totalSizeBytes,
            totalSizeFormatted: formatBytes(scan.totalSizeBytes),
            detectedBitWidth: bitWidth.bits,
            bitWidthDescription: bitWidth.description,
            quantizationMode: quantization.mode,
            groupSize: quantization.groupSize,
            tensorDtype: primaryDtype.isEmpty ? (quantization.dtype ?? "unknown") : primaryDtype,
            shardFiles: scan.shards,
            errors: errors
        )
    }
}
