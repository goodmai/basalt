import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct DetectedSourceCheckpoint: Sendable, Equatable {
    package var format: String
    package var effectiveBPW: Double
    package var totalBytes: Int64
    package var requiresCanonicalDecode: Bool
}

package enum UniversalSourceReader {
    static let customConfigKeys = [
        "quantization",
        "quantization_config",
        "macaq_quantization",
        "flatquant_quantization",
        "lirquant_quantization",
        "qtip_quantization",
        "gguf_quantization",
        "prism_hadamard",
    ]

    package static func detect(at directory: URL) throws(BasaltError) -> DetectedSourceCheckpoint {
        let fm = FileManager.default
        if directory.pathExtension.lowercased() == "gguf" {
            let size = (try? fm.attributesOfItem(atPath: directory.path)[.size] as? Int64) ?? 0
            return DetectedSourceCheckpoint(format: "gguf", effectiveBPW: 4.5, totalBytes: size, requiresCanonicalDecode: true)
        }
        let rootCfg = directory.appending(component: "config.json")
        let transCfg = directory.appending(path: "transformer/config.json")
        let cfgURL = FileManager.default.fileExists(atPath: rootCfg.path) ? rootCfg : transCfg
        guard let cfgData = try? Data(contentsOf: cfgURL),
              let json = (try? JSONSerialization.jsonObject(with: cfgData)) as? [String: Any] else {
            throw .modelConfigurationInvalid(path: rootCfg.path, reason: "missing or invalid config.json")
        }
        let bytes = directoryWeightBytes(directory)
        if let macaq = json["macaq_quantization"] as? [String: Any],
           let det = detectMACaQ(macaq, bytes: bytes) {
            return det
        }
        if let basis = detectBasisManifest(json, bytes: bytes) {
            return basis
        }
        if let qtip = json["qtip_quantization"] as? [String: Any] {
            let bpw = (qtip["bpw"] as? NSNumber)?.doubleValue ?? 2.5
            return DetectedSourceCheckpoint(format: String(format: "qtip:%.2f", bpw), effectiveBPW: bpw, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        if json["gguf_quantization"] != nil {
            return DetectedSourceCheckpoint(format: "gguf-native", effectiveBPW: 4.5, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        if (json["model_type"] as? String) == PrismHadamard.modelType || json["prism_hadamard"] != nil {
            return DetectedSourceCheckpoint(format: "prism-hadamard", effectiveBPW: 4.5, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        if let quant = (json["quantization"] as? [String: Any]) ?? (json["quantization_config"] as? [String: Any]) {
            return detectStandardQuant(quant, bytes: bytes)
        }
        let dtype = ((json["torch_dtype"] as? String) ?? "bfloat16").lowercased()
        let label = dtype.contains("float16") && !dtype.contains("bfloat") ? "dense:f16" : "dense:bf16"
        return DetectedSourceCheckpoint(format: label, effectiveBPW: 16.0, totalBytes: bytes, requiresCanonicalDecode: false)
    }

    private static func detectBasisManifest(_ json: [String: Any], bytes: Int64) -> DetectedSourceCheckpoint? {
        if let fq = json["flatquant_quantization"] as? [String: Any] {
            let first = (fq["modules"] as? [String: [String: Any]])?.values.first
            let bits = (fq["bits"] as? Int) ?? (first?["bits"] as? Int) ?? 4
            let group = (fq["group_size"] as? Int) ?? (first?["group_size"] as? Int) ?? 64
            let bpw = Double(bits) + 32.0 / Double(max(1, group)) + 0.02
            return DetectedSourceCheckpoint(format: "flatquant:\(bits):\(group)", effectiveBPW: bpw, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        if let lir = json["lirquant_quantization"] as? [String: Any] {
            let first = (lir["modules"] as? [String: [String: Any]])?.values.first
            let bits = (lir["bits"] as? Int) ?? (first?["bits"] as? Int) ?? 4
            let group = (lir["group_size"] as? Int) ?? (first?["group_size"] as? Int) ?? 64
            let sub = (lir["subspace_dim"] as? Int) ?? (first?["subspace_dim"] as? Int) ?? 128
            let bpw = Double(bits) + 32.0 / Double(max(1, group)) + 0.03
            return DetectedSourceCheckpoint(format: "lirquant:\(bits):\(group):\(sub)", effectiveBPW: bpw, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        return nil
    }

    private static func detectMACaQ(_ macaq: [String: Any], bytes: Int64) -> DetectedSourceCheckpoint? {
        guard let modules = macaq["modules"] as? [String: [String: Any]],
              let first = modules.values.first else { return nil }
        if let moeBPW = (macaq["moe_bpw"] as? NSNumber)?.doubleValue, moeBPW > 0 {
            let label = QuantizationTarget.macaqMoEBudget(bpw: moeBPW).specString
            return DetectedSourceCheckpoint(format: label, effectiveBPW: moeBPW, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        let allBits = modules.values.compactMap { $0["bits"] as? Int }
        if Set(allBits).count > 1, !allBits.isEmpty {
            let avgBits = Double(allBits.reduce(0, +)) / Double(allBits.count)
            let label = QuantizationTarget.macaqMoEBudget(bpw: avgBits).specString
            return DetectedSourceCheckpoint(format: label, effectiveBPW: avgBits, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        let mode = (first["mode"] as? String) ?? "affine"
        let bits = (first["bits"] as? Int) ?? 4
        let group = (first["group_size"] as? Int) ?? 64
        let label = mode == "mxfp4" ? "macaq:mxfp4" : "macaq:affine:\(bits):\(group)"
        let bpw = mode == "mxfp4" ? 4.27 : (Double(bits) + 32.0 / Double(max(1, group)))
        return DetectedSourceCheckpoint(format: label, effectiveBPW: bpw, totalBytes: bytes, requiresCanonicalDecode: true)
    }

    private static func detectStandardQuant(_ quant: [String: Any], bytes: Int64) -> DetectedSourceCheckpoint {
        let mode = ((quant["mode"] as? String) ?? "affine").lowercased()
        let bits = (quant["bits"] as? Int) ?? 4
        let group = (quant["group_size"] as? Int) ?? 64
        if mode == "mxfp4" {
            return DetectedSourceCheckpoint(format: "mxfp4", effectiveBPW: 4.25, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        if mode == "nf4" {
            return DetectedSourceCheckpoint(format: "nf4:\(group)", effectiveBPW: 4.0 + 16.0 / Double(max(1, group)), totalBytes: bytes, requiresCanonicalDecode: true)
        }
        if mode == "nvfp4" {
            return DetectedSourceCheckpoint(format: "nvfp4", effectiveBPW: 4.5, totalBytes: bytes, requiresCanonicalDecode: true)
        }
        return DetectedSourceCheckpoint(
            format: "affine:\(bits):\(group)",
            effectiveBPW: Double(bits) + 32.0 / Double(max(1, group)),
            totalBytes: bytes,
            requiresCanonicalDecode: false)
    }

    static func directoryWeightBytes(_ directory: URL) -> Int64 {
        let fm = FileManager.default
        var total: Int64 = 0
        for sub in [directory, directory.appending(component: "transformer")] {
            guard let names = try? fm.contentsOfDirectory(atPath: sub.path) else { continue }
            for name in names where name.hasSuffix(".safetensors") || name.hasSuffix(".gguf") {
                let path = sub.appending(component: name).path
                total += (try? fm.attributesOfItem(atPath: path)[.size] as? Int64) ?? 0
            }
        }
        return total
    }
}

extension UniversalSourceReader {
    package static func exportCanonicalDense(
        from source: URL,
        to output: URL,
        dtype: DType = .bfloat16,
        onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> WeightConverter.Summary {
        let detected = try detect(at: source)
        if detected.format.hasPrefix("affine:") || detected.format == "mxfp4" || detected.format.hasPrefix("dense:") {
            let summary = try WeightConverter(precision: .dense(dtype)).run(source: source, output: output)
            onStep?("canonical decode wrote \(summary.outputTensors) tensor(s) in \(summary.shards) shard(s)")
            return summary
        }
        if detected.format.hasPrefix("nf4:") {
            let group = Int(detected.format.dropFirst("nf4:".count)) ?? 64
            let summary = try exportNF4Shards(from: source, to: output, groupSize: group, dtype: dtype)
            onStep?("canonical NF4 decode wrote \(summary.outputTensors) tensor(s) in \(summary.shards) shard(s)")
            return summary
        }
        if detected.format.hasPrefix("flatquant:") || detected.format.hasPrefix("lirquant:") {
            let summary = try BasisTransformConverter.exportCanonicalShards(
                from: source, to: output, dtype: dtype, prepareDenseSidecars: prepareCleanDenseConfigAndSidecars)
            onStep?("canonical basis decode wrote \(summary.outputTensors) tensor(s) in \(summary.shards) shard(s)")
            return summary
        }
        if detected.format.hasPrefix("macaq:"), DiTBlockQuantizer.isDiTCheckpoint(source) {
            let summary = try DiTBlockQuantizer.dequantizeMACaQDirectory(source: source, output: output, dtype: dtype)
            onStep?("canonical DiT MACaQ decode wrote \(summary.outputTensors) tensor(s) in \(summary.shards) shard(s)")
            return summary
        }
        return try await exportContainerDense(from: source, to: output, dtype: dtype, onStep: onStep)
    }

    private static func exportNF4Shards(
        from source: URL,
        to output: URL,
        groupSize: Int,
        dtype: DType
    ) throws -> WeightConverter.Summary {
        try prepareCleanDenseConfigAndSidecars(from: source, to: output, dtype: dtype)
        let fm = FileManager.default
        let shardNames = try fm.contentsOfDirectory(atPath: source.path)
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
        var writer = ShardWriter(directory: output, maxBytes: 5 << 30, metadata: ["format": "mlx"])
        var summary = WeightConverter.Summary()
        summary.bytesBefore = Int(directoryWeightBytes(source))
        for name in shardNames {
            let arrays = try loadArrays(url: source.appending(component: name))
            for key in arrays.keys.sorted() where !key.hasSuffix(".scales") {
                guard let tensor = arrays[key] else { continue }
                let outTensor: MLXArray
                if key.hasSuffix(".weight"),
                   let scales = arrays[String(key.dropLast(".weight".count)) + ".scales"] {
                    outTensor = NF4Metal.dequantized(tensor, scales: scales, groupSize: groupSize, dtype: dtype)
                    summary.tensors += 2
                } else {
                    outTensor = tensor.dtype.isFloatingPoint ? tensor.asType(dtype) : tensor
                    summary.tensors += 1
                }
                eval(outTensor)
                summary.outputTensors += 1
                summary.bytesAfter += outTensor.nbytes
                try writer.add([key: outTensor])
            }
        }
        summary.shards = try writer.finish()
        return summary
    }

    fileprivate static func prepareCleanDenseConfigAndSidecars(from source: URL, to output: URL, dtype: DType) throws {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let cfgURL = source.appending(component: "config.json")
        let rawCfg = try Data(contentsOf: cfgURL)
        guard var json = (try JSONSerialization.jsonObject(with: rawCfg)) as? [String: Any] else {
            throw BasaltError.modelConfigurationInvalid(path: cfgURL.path, reason: "invalid config.json")
        }
        if (json["model_type"] as? String) == PrismHadamard.modelType,
           let prism = json["prism_hadamard"] as? [String: Any],
           let baseType = prism["base_model_type"] as? String {
            json["model_type"] = baseType
        }
        for key in customConfigKeys { json[key] = nil }
        json["torch_dtype"] = dtype == .float16 ? "float16" : "bfloat16"
        try copyNonWeightSidecars(from: source, to: output)
        try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(component: "config.json"))
    }
}

extension UniversalSourceReader {
    fileprivate static func exportContainerDense(
        from source: URL,
        to output: URL,
        dtype: DType,
        onStep: (@Sendable (String) -> Void)?
    ) async throws -> WeightConverter.Summary {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0, disableExpertStreaming: true))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: source.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        try prepareCleanDenseConfigAndSidecars(from: source, to: output, dtype: dtype)
        let bytesBefore = Int(directoryWeightBytes(source))
        return try await container.perform { context -> WeightConverter.Summary in
            let root = context.model as Module
            var writer = ShardWriter(directory: output, maxBytes: 5 << 30, metadata: ["format": "mlx"])
            var summary = WeightConverter.Summary()
            summary.bytesBefore = bytesBefore
            var handledParamKeys: Set<String> = []
            for (path, mod) in root.namedModules() {
                guard let decoded = decodeModule(path: path, module: mod, dtype: dtype, handledKeys: &handledParamKeys) else {
                    continue
                }
                summary.tensors += decoded.count
                summary.outputTensors += decoded.count
                summary.bytesAfter += decoded.values.reduce(0) { $0 + $1.nbytes }
                try writer.add(decoded)
            }
            for (paramKey, arr) in root.parameters().flattened() where !handledParamKeys.contains(paramKey) {
                let casted = arr.dtype.isFloatingPoint ? arr.asType(dtype) : arr
                eval(casted)
                summary.tensors += 1
                summary.outputTensors += 1
                summary.bytesAfter += casted.nbytes
                try writer.add([paramKey: casted])
            }
            summary.shards = try writer.finish()
            onStep?("canonical decode wrote \(summary.outputTensors) tensor(s) in \(summary.shards) shard(s)")
            return summary
        }
    }

    fileprivate static func copyNonWeightSidecars(from source: URL, to output: URL) throws {
        let fm = FileManager.default
        for name in try fm.contentsOfDirectory(atPath: source.path) where !name.hasPrefix(".") {
            let lower = name.lowercased()
            guard lower != "config.json",
                  !lower.hasSuffix(".safetensors"),
                  !lower.hasSuffix(".safetensors.index.json"),
                  !lower.hasSuffix(".gguf") else { continue }
            let file = source.appending(component: name).resolvingSymlinksInPath()
            guard (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            try fm.copyItem(at: file, to: output.appending(component: name))
        }
    }
}

extension UniversalSourceReader {
    static func decodeModule(
        path: String,
        module: Module,
        dtype: DType,
        handledKeys: inout Set<String>
    ) -> [String: MLXArray]? {
        let prefix = path.isEmpty ? "" : "\(path)."
        if let out = decodeRotatedModule(prefix: prefix, module: module, dtype: dtype, handledKeys: &handledKeys) {
            return out
        }
        return decodePackedModule(prefix: prefix, module: module, dtype: dtype, handledKeys: &handledKeys)
    }

    private static func weightBiasDict(prefix: String, weight: MLXArray, bias: MLXArray?, dtype: DType) -> [String: MLXArray] {
        var out: [String: MLXArray] = ["\(prefix)weight": weight.asType(dtype)]
        if let bias { out["\(prefix)bias"] = bias.asType(dtype) }
        eval(Array(out.values))
        return out
    }

    private static func decodeRotatedModule(
        prefix: String,
        module: Module,
        dtype: DType,
        handledKeys: inout Set<String>
    ) -> [String: MLXArray]? {
        if let macaq = module as? MACaQQuantizedLinear {
            markAllParameters(of: macaq, prefix: prefix, into: &handledKeys)
            return weightBiasDict(prefix: prefix, weight: macaq.unrotatedDenseWeight(), bias: macaq.bias, dtype: dtype)
        }
        if let msw = module as? MACaQSwitchLinear {
            markAllParameters(of: msw, prefix: prefix, into: &handledKeys)
            return decodeMACaQSwitch(msw, prefix: prefix, dtype: dtype)
        }
        if let fq = module as? FlatQuantLinear {
            markAllParameters(of: fq, prefix: prefix, into: &handledKeys)
            return weightBiasDict(prefix: prefix, weight: fq.unrotatedDenseWeight(), bias: fq.bias, dtype: dtype)
        }
        if let lir = module as? LIRQuantLinear {
            markAllParameters(of: lir, prefix: prefix, into: &handledKeys)
            return weightBiasDict(prefix: prefix, weight: lir.unrotatedDenseWeight(), bias: lir.bias, dtype: dtype)
        }
        if let qtip = module as? QTIPLinear {
            markAllParameters(of: qtip, prefix: prefix, into: &handledKeys)
            return weightBiasDict(prefix: prefix, weight: qtip.decodedWeight(dtype: dtype), bias: qtip.bias, dtype: dtype)
        }
        if let hlin = module as? HadamardQuantizedLinear {
            markAllParameters(of: hlin, prefix: prefix, into: &handledKeys)
            let wRot = dequantized(
                hlin.weight, scales: hlin.scales, biases: hlin.biases,
                groupSize: hlin.groupSize, bits: hlin.bits, mode: hlin.mode).asType(.float32)
            return weightBiasDict(prefix: prefix, weight: hlin.rotation.inverse(wRot, signs: hlin.signs), bias: hlin.bias, dtype: dtype)
        }
        if let hemb = module as? HadamardQuantizedEmbedding {
            markAllParameters(of: hemb, prefix: prefix, into: &handledKeys)
            let wRot = dequantized(
                hemb.weight, scales: hemb.scales, biases: hemb.biases,
                groupSize: hemb.groupSize, bits: hemb.bits, mode: hemb.mode).asType(.float32)
            return weightBiasDict(prefix: prefix, weight: hemb.rotation.inverse(wRot, signs: hemb.signs), bias: nil, dtype: dtype)
        }
        return nil
    }

    private static func decodePackedModule(
        prefix: String,
        module: Module,
        dtype: DType,
        handledKeys: inout Set<String>
    ) -> [String: MLXArray]? {
        if let nf4 = module as? NF4Linear {
            markAllParameters(of: nf4, prefix: prefix, into: &handledKeys)
            let w = NF4Metal.dequantized(nf4.weight, scales: nf4.scales, groupSize: nf4.groupSize, dtype: dtype)
            return weightBiasDict(prefix: prefix, weight: w, bias: nf4.bias, dtype: dtype)
        }
        if let nf4sw = module as? NF4SwitchLinear {
            markAllParameters(of: nf4sw, prefix: prefix, into: &handledKeys)
            return decodeNF4Switch(nf4sw, prefix: prefix, dtype: dtype)
        }
        if let gguf = module as? GGUFLinear {
            markAllParameters(of: gguf, prefix: prefix, into: &handledKeys)
            return decodeGGUFLinear(gguf, prefix: prefix, dtype: dtype)
        }
        if let gemb = module as? GGUFEmbedding {
            markAllParameters(of: gemb, prefix: prefix, into: &handledKeys)
            let w = GGUFKernels.dequantize(gemb.weight, type: gemb.type, columns: gemb.dimensions, dtype: dtype)
            return weightBiasDict(prefix: prefix, weight: w, bias: nil, dtype: dtype)
        }
        if let ql = module as? QuantizedLinear {
            markAllParameters(of: ql, prefix: prefix, into: &handledKeys)
            let w = dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode,
                globalScale: ql.globalScale, dtype: dtype)
            return weightBiasDict(prefix: prefix, weight: w, bias: ql.bias, dtype: dtype)
        }
        if let qsw = module as? QuantizedSwitchLinear {
            markAllParameters(of: qsw, prefix: prefix, into: &handledKeys)
            let p = Dictionary(uniqueKeysWithValues: qsw.parameters().flattened())
            guard let wPack = p["weight"], let sPack = p["scales"] else { return nil }
            let w = dequantized(
                wPack, scales: sPack, biases: p["biases"],
                groupSize: qsw.groupSize, bits: qsw.bits, mode: qsw.mode, dtype: dtype)
            return weightBiasDict(prefix: prefix, weight: w, bias: p["bias"], dtype: dtype)
        }
        if let qemb = module as? QuantizedEmbedding {
            markAllParameters(of: qemb, prefix: prefix, into: &handledKeys)
            let w = dequantized(
                qemb.weight, scales: qemb.scales, biases: qemb.biases,
                groupSize: qemb.groupSize, bits: qemb.bits, mode: qemb.mode,
                globalScale: qemb.globalScale, dtype: dtype)
            return weightBiasDict(prefix: prefix, weight: w, bias: nil, dtype: dtype)
        }
        return nil
    }
}

extension UniversalSourceReader {
    fileprivate static func markAllParameters(of module: Module, prefix: String, into set: inout Set<String>) {
        for (k, _) in module.parameters().flattened() {
            set.insert("\(prefix)\(k)")
        }
    }

    private static func decodeMACaQSwitch(_ msw: MACaQSwitchLinear, prefix: String, dtype: DType) -> [String: MLXArray]? {
        let p = Dictionary(uniqueKeysWithValues: msw.parameters().flattened())
        guard let wPack = p["weight"] else { return nil }
        let experts = wPack.dim(0)
        let pre = msw.preScale.asType(.float32)
        var pieces: [MLXArray] = []
        for e in 0 ..< experts {
            let deq = dequantized(
                wPack[e], scales: msw.scales[e], biases: msw.biases?[e],
                groupSize: msw.groupSize, bits: msw.bits, mode: msw.mode).asType(.float32)
            let unrot = MACaQMath.rotateBlocks(deq, blockSize: msw.hadamardBlock)
            let scaleVec = pre.ndim == 2 ? pre[e] : pre
            pieces.append((unrot * scaleVec.expandedDimensions(axis: 0)).asType(dtype))
        }
        let stackedW = stacked(pieces, axis: 0)
        var out: [String: MLXArray] = ["\(prefix)weight": stackedW]
        if let b = p["bias"] { out["\(prefix)bias"] = b.asType(dtype) }
        eval(Array(out.values))
        return out
    }

    private static func decodeNF4Switch(_ sw: NF4SwitchLinear, prefix: String, dtype: DType) -> [String: MLXArray]? {
        let p = Dictionary(uniqueKeysWithValues: sw.parameters().flattened())
        guard let codes = p["weight"] else { return nil }
        let experts = codes.dim(0)
        let pieces = (0 ..< experts).map { e in
            NF4Metal.dequantized(codes[e], scales: sw.scales[e], groupSize: sw.groupSize, dtype: dtype)
        }
        let stackedW = stacked(pieces, axis: 0)
        var out: [String: MLXArray] = ["\(prefix)weight": stackedW]
        if let b = p["bias"] { out["\(prefix)bias"] = b.asType(dtype) }
        eval(Array(out.values))
        return out
    }

    private static func decodeGGUFLinear(_ gguf: GGUFLinear, prefix: String, dtype: DType) -> [String: MLXArray] {
        var w = GGUFKernels.dequantize(gguf.weight, type: gguf.type, columns: gguf.inputs, dtype: .float32)
        if let block = gguf.hadamardBlock, block > 1 {
            w = MACaQMath.rotateBlocks(w, blockSize: block)
        }
        if let pre = gguf.preScale {
            w = w * pre.asType(.float32).expandedDimensions(axis: 0)
        }
        if let order = gguf.inputOrder {
            let invOrder = argSort(order, axis: 0)
            w = w.take(invOrder, axis: -1)
        }
        var out: [String: MLXArray] = ["\(prefix)weight": w.asType(dtype)]
        if let b = gguf.bias { out["\(prefix)bias"] = b.asType(dtype) }
        eval(Array(out.values))
        return out
    }
}
