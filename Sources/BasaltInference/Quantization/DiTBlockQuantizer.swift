import BasaltCore
import Foundation
import MLX
import MLXNN

package enum DiTBlockQuantizer {
    private static let ditModelTypes: Set<String> = [
        "flux_klein", "flux3_klein", "flux3transformer", "fibo_perceiver",
        "s3_dit", "qwen_image_2_1", "step1x_reason_edit", "pixel_dit",
        "linear_sana", "seedvr2", "seedvr2_upscaler", "ace_step", "ace_step_1_5",
    ]

    private static let ditClassMarkers = [
        "flux", "qwenimage", "fibo", "seedvr2", "acestep", "s3dit", "pixeldit", "sana",
    ]

    package static func isDiTCheckpoint(_ directory: URL) -> Bool {
        let candidates = [
            directory.appending(component: "config.json"),
            directory.appending(path: "transformer/config.json"),
        ]
        for cfgURL in candidates {
            guard let data = try? Data(contentsOf: cfgURL),
                  let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
                continue
            }
            let modelType = (json["model_type"] as? String)?.lowercased() ?? ""
            if ditModelTypes.contains(modelType) { return true }
            let cls = (json["_class_name"] as? String)?.lowercased() ?? ""
            if ditClassMarkers.contains(where: { cls.contains($0) }) { return true }
        }
        return false
    }

    package static func convertMACaQDirectory(
        source: URL,
        output: URL,
        targetMode: MACaQTargetMode,
        hadamardBlock: Int = 32,
        alpha: Float = 0.75,
        onStep: (@Sendable (String) -> Void)? = nil
    ) throws -> (shards: Int, bytesAfter: Int64) {
        let block = [32, 64].contains(hadamardBlock) ? hadamardBlock : 32
        let (weightDir, cfgURL, rawCfg) = try resolveWeightRoot(in: source)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let shardURLs = try safetensorsFiles(in: weightDir)
        guard !shardURLs.isEmpty else {
            throw BasaltError.weightsCorrupted(path: weightDir.path, reason: "no .safetensors shards found")
        }
        let (bits, groupSize, mode): (Int, Int, QuantizationMode) = switch targetMode {
        case .mxfp4: (4, 32, .mxfp4)
        case .affine(let b, let g): (b, g, .affine)
        }
        let allKeys = Set(try buildShardIndex(for: shardURLs).keys)
        var writer = ShardWriter(directory: output, maxBytes: 4_500_000_000, metadata: ["format": "mlx"])
        var manifestModules: [String: MACaQLayerSpec] = [:]
        var bytesAfter: Int64 = 0
        for (sIdx, shard) in shardURLs.enumerated() {
            let tensors = try MLX.loadArrays(url: shard)
            for key in tensors.keys.sorted() {
                guard let tensor = tensors[key] else { continue }
                if let base = quantizableDiTBase(key: key, tensor: tensor, allKeys: allKeys, block: block, groupSize: groupSize) {
                    let hasBias = allKeys.contains(base + ".bias")
                    let quantized = quantizeDiTMatrix(
                        tensor,
                        bits: bits,
                        groupSize: groupSize,
                        mode: mode,
                        block: block,
                        alpha: alpha
                    )
                    var outDict: [String: MLXArray] = [
                        base + ".weight": quantized.weight,
                        base + ".scales": quantized.scales,
                        base + ".pre_scale": quantized.preScale,
                    ]
                    if let qBiases = quantized.biases { outDict[base + ".biases"] = qBiases }
                    var spec = MACaQLayerSpec(bits: bits, groupSize: groupSize, mode: mode, hadamardBlock: block)
                    spec.hasBias = hasBias
                    manifestModules[base] = spec
                    bytesAfter += outDict.values.reduce(0) { $0 + Int64($1.nbytes) }
                    try writer.add(outDict)
                } else {
                    eval(tensor)
                    bytesAfter += Int64(tensor.nbytes)
                    try writer.add([key: tensor])
                }
            }
            onStep?("DiT MACaQ shard \(sIdx + 1)/\(shardURLs.count) (\(manifestModules.count) blocks)")
        }
        let shards = try writer.finish()
        try writeDiTMACaQConfig(rawCfg: rawCfg, cfgSourceURL: cfgURL, sourceRoot: source, output: output, alpha: alpha, modules: manifestModules)
        return (shards, bytesAfter)
    }

    package static func dequantizeMACaQDirectory(
        source: URL,
        output: URL,
        dtype: DType = .bfloat16
    ) throws -> WeightConverter.Summary {
        guard let manifest = try MACaQNativeLoader.manifest(in: source) else {
            throw BasaltError.modelConfigurationInvalid(path: source.path, reason: "missing macaq_quantization manifest")
        }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let shardURLs = try safetensorsFiles(in: source)
        let keyToShard = try buildShardIndex(for: shardURLs)
        var writer = ShardWriter(directory: output, maxBytes: 4_500_000_000, metadata: ["format": "mlx"])
        var summary = WeightConverter.Summary()
        summary.bytesBefore = Int(UniversalSourceReader.directoryWeightBytes(source))
        for shard in shardURLs {
            let arrays = try MLX.loadArrays(url: shard)
            for key in arrays.keys.sorted() {
                if key.hasSuffix(".scales") || key.hasSuffix(".biases") || key.hasSuffix(".pre_scale") {
                    let prefix = key
                        .replacingOccurrences(of: ".scales", with: "")
                        .replacingOccurrences(of: ".biases", with: "")
                        .replacingOccurrences(of: ".pre_scale", with: "")
                    if manifest.modules[prefix] != nil { continue }
                }
                guard let tensor = arrays[key] else { continue }
                if key.hasSuffix(".weight"),
                   case let base = String(key.dropLast(".weight".count)),
                   let spec = manifest.modules[base],
                   let scales = try companionTensor(base + ".scales", in: arrays, currentShard: shard, index: keyToShard),
                   let preScale = try companionTensor(base + ".pre_scale", in: arrays, currentShard: shard, index: keyToShard) {
                    let biases = try companionTensor(base + ".biases", in: arrays, currentShard: shard, index: keyToShard)
                    let wRot = dequantized(
                        tensor, scales: scales, biases: biases,
                        groupSize: spec.groupSize, bits: spec.bits, mode: spec.quantizationMode
                    ).asType(.float32)
                    let unrot = MACaQMath.rotateBlocks(wRot, blockSize: spec.hadamardBlock)
                    let dense = (unrot * preScale.asType(.float32).expandedDimensions(axis: 0)).asType(dtype)
                    eval(dense)
                    summary.tensors += biases == nil ? 3 : 4
                    summary.outputTensors += 1
                    summary.bytesAfter += dense.nbytes
                    try writer.add([key: dense])
                } else {
                    let casted = tensor.dtype.isFloatingPoint ? tensor.asType(dtype) : tensor
                    eval(casted)
                    summary.tensors += 1
                    summary.outputTensors += 1
                    summary.bytesAfter += casted.nbytes
                    try writer.add([key: casted])
                }
            }
        }
        summary.shards = try writer.finish()
        try writeCleanDenseDiTConfig(from: source, to: output, dtype: dtype)
        return summary
    }
}

extension DiTBlockQuantizer {
    private static func resolveWeightRoot(in source: URL) throws -> (weightDir: URL, cfgURL: URL, rawCfg: [String: Any]) {
        let transformerDir = source.appending(component: "transformer")
        let rootCfg = source.appending(component: "config.json")
        let transCfg = transformerDir.appending(component: "config.json")
        if FileManager.default.fileExists(atPath: transCfg.path),
           let data = try? Data(contentsOf: transCfg),
           let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            return (transformerDir, transCfg, json)
        }
        guard let data = try? Data(contentsOf: rootCfg),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw BasaltError.modelConfigurationInvalid(path: rootCfg.path, reason: "missing or invalid config.json")
        }
        return (source, rootCfg, json)
    }

    private static func safetensorsFiles(in directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func buildShardIndex(for shardURLs: [URL]) throws -> [String: URL] {
        var index: [String: URL] = [:]
        for shard in shardURLs {
            for key in try MLX.loadArrays(url: shard).keys {
                index[key] = shard
            }
        }
        return index
    }

    private static func companionTensor(
        _ key: String,
        in current: [String: MLXArray],
        currentShard: URL,
        index: [String: URL]
    ) throws -> MLXArray? {
        if let local = current[key] { return local }
        guard let otherShard = index[key], otherShard != currentShard else { return nil }
        return try MLX.loadArrays(url: otherShard)[key]
    }

    private static func quantizableDiTBase(
        key: String,
        tensor: MLXArray,
        allKeys: Set<String>,
        block: Int,
        groupSize: Int
    ) -> String? {
        guard key.hasSuffix(".weight"), tensor.ndim == 2, tensor.dtype.isFloatingPoint else { return nil }
        let base = String(key.dropLast(".weight".count))
        guard !allKeys.contains(base + ".scales") else { return nil }
        let outF = tensor.dim(0)
        let inF = tensor.dim(1)
        guard outF >= 32, inF >= max(block, groupSize),
              inF.isMultiple(of: block), inF.isMultiple(of: groupSize) else {
            return nil
        }
        let lower = base.lowercased()
        let skip = ["time_in", "tempo_proj", "norm", "embed_tokens", "vocoder_harm"]
        if skip.contains(where: { lower.contains($0) }) { return nil }
        return base
    }

    private static func quantizeDiTMatrix(
        _ weight: MLXArray,
        bits: Int,
        groupSize: Int,
        mode: QuantizationMode,
        block: Int,
        alpha: Float
    ) -> (weight: MLXArray, scales: MLXArray, biases: MLXArray?, preScale: MLXArray) {
        let wf = weight.asType(.float32)
        let colEnergy = mean(wf * wf, axis: 0) + Float(1e-6)
        let preScale = MACaQMath.balancePreScale(weight: wf, meanSquareActivation: colEnergy, alpha: alpha)
        let scaled = wf / preScale.expandedDimensions(axis: 0)
        let rotated = MACaQMath.rotateBlocks(scaled, blockSize: block)
        let diagH = mean(rotated * rotated, axis: 0) + Float(1e-4)
        let isoH = diag(diagH)
        if mode == .mxfp4 {
            let screened = MACaQMath.screenMXFP4Exponents(weight: rotated, hessianDiag: diagH)
            var state = MACaQGridState(
                codes: screened.sidx,
                scaleElem: screened.scaleElem,
                biasElem: nil,
                levelTable: MLXArray(MACaQMath.mxfp4SignedLevels),
                maxLevel: 14
            )
            _ = MACaQMath.flipRefine(state: &state, targetWeight: rotated, hessian: isoH, maxIterations: 2)
            let packed = MACaQMath.packMXFP4(sidx: state.codes, e8m0: screened.e8m0)
            let preBF16 = preScale.asType(.bfloat16)
            eval(packed.weight, packed.scales, preBF16)
            return (packed.weight, packed.scales, nil, preBF16)
        }
        let (qW, qS, qB) = MLX.quantized(rotated.asType(.bfloat16), groupSize: groupSize, bits: bits, mode: .affine)
        let spec = PVCodeGrid.Spec(groupSize: groupSize, bits: bits, mode: .affine)
        var (grid, _) = MACaQMath.makeGridState(packedWeight: qW, scales: qS, biases: qB, spec: spec)
        let flipRes = MACaQMath.flipRefine(state: &grid, targetWeight: rotated, hessian: isoH, maxIterations: 2)
        let finalW = flipRes.flips > 0 ? PVCodeGrid.packCodes(grid.codes.asType(.uint8), bits: bits) : qW
        let preBF16 = preScale.asType(.bfloat16)
        eval(finalW, qS, preBF16)
        return (finalW, qS, qB, preBF16)
    }

    private static func writeDiTMACaQConfig(
        rawCfg: [String: Any],
        cfgSourceURL: URL,
        sourceRoot: URL,
        output: URL,
        alpha: Float,
        modules: [String: MACaQLayerSpec]
    ) throws {
        var json = rawCfg
        let manifest = MACaQNativeLoader.Manifest(version: 1, alpha: alpha, modules: modules)
        let manifestData = try JSONEncoder().encode(manifest)
        json[MACaQNativeLoader.configKey] = try JSONSerialization.jsonObject(with: manifestData)
        let encoded = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try encoded.write(to: output.appending(component: "config.json"), options: .atomic)
        copySidecarFiles(from: sourceRoot, excluding: cfgSourceURL, to: output)
    }

    private static func writeCleanDenseDiTConfig(from source: URL, to output: URL, dtype: DType) throws {
        let cfgURL = source.appending(component: "config.json")
        let raw = try Data(contentsOf: cfgURL)
        guard var json = (try JSONSerialization.jsonObject(with: raw)) as? [String: Any] else {
            throw BasaltError.modelConfigurationInvalid(path: cfgURL.path, reason: "invalid config.json")
        }
        for key in UniversalSourceReader.customConfigKeys { json[key] = nil }
        json["torch_dtype"] = dtype == .float16 ? "float16" : "bfloat16"
        let encoded = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try encoded.write(to: output.appending(component: "config.json"), options: .atomic)
        copySidecarFiles(from: source, excluding: cfgURL, to: output)
    }

    private static func copySidecarFiles(from source: URL, excluding excludedCfg: URL, to output: URL) {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: source.path) else { return }
        for name in names where !name.hasPrefix(".") {
            let lower = name.lowercased()
            if lower == "config.json" || lower.hasSuffix(".safetensors") || lower.hasSuffix(".safetensors.index.json") {
                continue
            }
            let src = source.appending(component: name)
            if src.standardizedFileURL == excludedCfg.standardizedFileURL { continue }
            guard (try? src.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            try? fm.copyItem(at: src, to: output.appending(component: name))
        }
    }
}
