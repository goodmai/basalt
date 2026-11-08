import BasaltCore
import Foundation
import MLX
import MLXLMCommon

package struct WeightConverter: Sendable {
    package enum Format: String, Sendable, CaseIterable { case mlx, hf }

    package enum Precision: Sendable, Equatable {
        case source
        case affine(bits: Int, groupSize: Int)
        case mixed(MixedBits)
        case nf4(groupSize: Int)
        case dense(DType?)
    }

    package enum Failure: LocalizedError {
        case options(String)
        case refused(String)
        case malformed(String)

        package var errorDescription: String? {
            switch self {
            case .options(let message), .refused(let message), .malformed(let message): message
            }
        }
    }

    package struct Summary: Sendable {
        package var tensors = 0
        package var outputTensors = 0
        package var bytesBefore = 0
        package var bytesAfter = 0
        package var shards = 0
        package var plan: BitPlan?
    }

    package var precision: Precision
    package var format: Format
    package var adapter: URL?
    package var maxShardBytes = 5 << 30

    package init(precision: Precision, format: Format = .mlx, adapter: URL? = nil) {
        self.precision = precision
        self.format = format
        self.adapter = adapter
    }

    package static func precision(
        quantize: String?, dequantize: String?, format: Format, budgetGB: Double? = nil
    ) throws(Failure) -> Precision {
        guard budgetGB == nil || quantize?.hasPrefix("mixed:") == true else { throw .options("--budget-gb goes with --quantize mixed:<low>:<high>:<group>") }
        switch (quantize, dequantize) {
        case (_?, _?):
            throw .options("--quantize and --dequantize exclude each other")
        case (let spec?, nil):
            guard format == .mlx else { throw .options("--format hf writes dense weights; drop --quantize") }
            return try quantization(spec, budgetGB: budgetGB)
        case (nil, let name?):
            guard let dtype = ["bf16": DType.bfloat16, "f16": DType.float16][name] else {
                throw .options("--dequantize takes bf16 or f16, got '\(name)'")
            }
            return .dense(dtype)
        case (nil, nil):
            return format == .hf ? .dense(nil) : .source
        }
    }

    static func quantization(_ spec: String, budgetGB: Double?) throws(Failure) -> Precision {
        let parts = spec.split(separator: ":")
        let numbers = parts.dropFirst().compactMap { Int($0) }
        let bitsValid = numbers.dropLast().allSatisfy { [2, 3, 4, 5, 6, 8].contains($0) } && [32, 64, 128].contains(numbers.last ?? 0)
        switch parts.first.map(String.init) {
        case "nf4" where parts.count == 2 && numbers.count == 1 && [32, 64, 128].contains(numbers[0]):
            return .nf4(groupSize: numbers[0])
        case "affine" where parts.count == 3 && numbers.count == 2 && bitsValid:
            return .affine(bits: numbers[0], groupSize: numbers[1])
        case "mixed" where parts.count == 4 && numbers.count == 3 && bitsValid && numbers[0] < numbers[1]:
            guard let budgetGB, budgetGB.isFinite, budgetGB > 0, budgetGB * 1e9 < Double(Int.max) else { throw .options("--quantize mixed needs --budget-gb, a positive size in GB (10^9 bytes)") }
            return .mixed(MixedBits(low: numbers[0], high: numbers[1], groupSize: numbers[2], budgetBytes: Int(budgetGB * 1e9)))
        default:
            throw .options(
                "--quantize takes affine:<bits>:<group> or mixed:<low>:<high>:<group> (bits 2-6 or 8, low below high, group 32, 64 or 128) or nf4:<group> (group 32, 64 or 128), got '\(spec)'")
        }
    }

    @discardableResult
    package func run(source: URL, output: URL) throws -> Summary {
        try Self.checkOutput(output, source: source)
        let configURL = source.appending(component: "config.json")
        guard let config = ModelCache.config(at: source) else { throw Failure.malformed("\(configURL.path) is missing or unreadable") }
        let layout = try SourceLayout(directory: source, config: config, configData: Data(contentsOf: configURL))
        guard !layout.shardOf.keys.contains(where: { $0.hasSuffix(".global_scale") }) else {
            throw Failure.refused("\(source.path) is an nvfp4 checkpoint with global scales; converting it is not supported yet")
        }
        try checkNF4Source(layout, source: source)
        let lora = try adapter.map { try LoRAUpdate(directory: $0, keys: layout.shardOf) }
        var converter = self
        var bitPlan: BitPlan?
        if case .mixed(var mixed) = precision {
            let made = try plan(layout, mixed)
            mixed.highLayers = made.highLayers
            converter.precision = .mixed(mixed)
            bitPlan = made
        }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        do {
            try converter.copySidecars(from: source, to: output, config: config)
            var summary = try converter.stream(layout, lora: lora, to: output)
            summary.plan = bitPlan
            return summary
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }
    }

    func stream(_ layout: SourceLayout, lora: LoRAUpdate?, to output: URL) throws -> Summary {
        var writer = ShardWriter(directory: output, maxBytes: maxShardBytes, metadata: ["format": format == .hf ? "pt" : "mlx"])
        var summary = Summary()
        for shard in layout.shards {
            var arrays = try loadArrays(url: shard)
            for key in arrays.keys.sorted() where !layout.isCompanion(key) {
                guard let tensor = arrays.removeValue(forKey: key) else { continue }
                let packed = try layout.packed(key, taking: &arrays)
                let outputs = try convert(key, tensor, packed: packed, lora: lora, sourceQuantized: layout.quantization != nil)
                let inputs = [tensor, packed?.scales, packed?.biases].compactMap { $0 }
                summary.tensors += inputs.count
                summary.bytesBefore += inputs.reduce(0) { $0 + $1.nbytes }
                summary.outputTensors += outputs.count
                summary.bytesAfter += outputs.values.reduce(0) { $0 + $1.nbytes }
                try writer.add(outputs)
            }
        }
        summary.shards = try writer.finish()
        return summary
    }

    func convert(
        _ key: String, _ tensor: MLXArray, packed: PackedWeight?, lora: LoRAUpdate?, sourceQuantized: Bool
    ) throws -> [String: MLXArray] {
        let isWeight = key.hasSuffix(".weight")
        let base = isWeight ? String(key.dropLast(".weight".count)) : key
        let pair = isWeight ? lora?.pairs[base] : nil
        let nf4: Int? = if case .nf4(let groupSize) = precision,
            Self.takesNF4(key, tensor, packed: packed, sourceQuantized: sourceQuantized, groupSize: groupSize) { groupSize } else { nil }
        let target = targetQuantization(key, tensor, packed: packed, sourceQuantized: sourceQuantized)
        if nf4 == nil, pair == nil, let packed, let target, packed.quantization.asTuple == target.asTuple {
            return packed.arrays(base, weight: tensor)
        }
        if nf4 == nil, pair == nil, packed == nil, target == nil { return [key: cast(tensor)] }
        var weight = Self.dense(tensor, packed)
        if let pair, let lora { weight = try LoRAUpdate.merged(weight, pair, scale: lora.scale, layer: base) }
        if let nf4 {
            let (codes, scales, _) = MLX.quantized(weight, groupSize: nf4, bits: 4, mode: .nf4)
            return [base + ".weight": codes, base + ".scales": scales]
        }
        guard let target else { return [key: cast(weight)] }
        let (wq, scales, biases) = MLX.quantized(weight, groupSize: target.groupSize, bits: target.bits, mode: target.mode)
        return PackedWeight(scales: scales, biases: biases, quantization: target).arrays(base, weight: wq)
    }

    func targetQuantization(
        _ key: String, _ tensor: MLXArray, packed: PackedWeight?, sourceQuantized: Bool
    ) -> BaseConfiguration.Quantization? {
        switch precision {
        case .dense:
            return nil
        case .source, .nf4:
            return packed?.quantization
        case .affine(let bits, let groupSize):
            guard Self.quantizable(key, tensor, packed: packed, sourceQuantized: sourceQuantized, groupSize: groupSize) else { return nil }
            return BaseConfiguration.Quantization(groupSize: groupSize, bits: bits)
        case .mixed(let mixed):
            guard Self.quantizable(key, tensor, packed: packed, sourceQuantized: sourceQuantized, groupSize: mixed.groupSize) else { return nil }
            return BaseConfiguration.Quantization(groupSize: mixed.groupSize, bits: mixed.bits(layer: String(key.dropLast(".weight".count))))
        }
    }

    static func quantizable(_ key: String, _ tensor: MLXArray, packed: PackedWeight?, sourceQuantized: Bool, groupSize: Int) -> Bool {
        if let packed { return (packed.scales.dim(-1) * packed.quantization.groupSize) % groupSize == 0 }
        return !sourceQuantized && isLinearLike(key, tensor) && tensor.dim(-1) % groupSize == 0
    }

    static func dense(_ tensor: MLXArray, _ packed: PackedWeight?) -> MLXArray {
        packed.map {
            MLX.dequantized(
                tensor, scales: $0.scales, biases: $0.biases,
                groupSize: $0.quantization.groupSize, bits: $0.quantization.bits, mode: $0.quantization.mode)
        } ?? tensor
    }

    static func isLinearLike(_ key: String, _ tensor: MLXArray) -> Bool {
        (2 ... 3).contains(tensor.ndim) && tensor.dtype.isFloatingPoint && isLinearPath(key)
    }

    static func isLinearPath(_ key: String) -> Bool {
        let path = key.split(separator: ".")
        guard path.last == "weight" else { return false }
        return !path.dropLast().contains { part in
            part == "gate" || ["embed", "norm", "router", "vision", "visual", "audio"].contains { part.contains($0) }
        }
    }

    static func takesNF4(_ key: String, _ tensor: MLXArray, packed: PackedWeight?, sourceQuantized: Bool, groupSize: Int) -> Bool {
        guard (2 ... 3).contains(tensor.ndim), isLinearPath(key) else { return false }
        guard let packed else { return !sourceQuantized && tensor.dtype.isFloatingPoint && tensor.dim(-1) % groupSize == 0 }
        return packed.quantization.bits == 4 && packed.quantization.mode == .affine
            && (packed.scales.dim(-1) * packed.quantization.groupSize) % groupSize == 0
    }

    func cast(_ tensor: MLXArray) -> MLXArray {
        guard case .dense(let dtype?) = precision, tensor.dtype.isFloatingPoint, tensor.dtype != dtype else { return tensor }
        return tensor.asType(dtype)
    }
}

extension WeightConverter {
    static func checkOutput(_ output: URL, source: URL) throws(Failure) {
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw .refused("\(output.path) exists; --out must name a new directory")
        }
        guard !canonical(output).starts(with: canonical(source)) else {
            throw .refused("\(output.path) is inside the source \(source.path), which is never written")
        }
    }

    func checkNF4Source(_ layout: SourceLayout, source: URL) throws(Failure) {
        guard case .nf4 = precision else { return }
        let unbiased = layout.shardOf.keys.filter { $0.hasSuffix(".scales") && layout.shardOf[String($0.dropLast(".scales".count)) + ".biases"] == nil }
        guard let example = unbiased.min() else { return }
        throw .refused("\(source.path) has quantized layers without biases (\(example)); nf4 conversion needs a dense or affine source, since the loader reads scales without biases as nf4")
    }

    static func canonical(_ url: URL) -> [String] {
        var existing = url.standardizedFileURL
        var missing: [String] = []
        while !FileManager.default.fileExists(atPath: existing.path), existing.pathComponents.count > 1 {
            missing.insert(existing.lastPathComponent, at: 0)
            existing = existing.deletingLastPathComponent()
        }
        return existing.resolvingSymlinksInPath().pathComponents + missing
    }

    func copySidecars(from source: URL, to output: URL, config: [String: Any]) throws {
        let fileManager = FileManager.default
        for name in try fileManager.contentsOfDirectory(atPath: source.path) where !name.hasPrefix(".") {
            let lowercased = name.lowercased()
            guard lowercased != "config.json", !lowercased.hasSuffix(".safetensors"), !lowercased.hasSuffix(".safetensors.index.json") else { continue }
            let file = source.appending(component: name).resolvingSymlinksInPath()
            guard (try file.resourceValues(forKeys: [.isRegularFileKey])).isRegularFile == true else { continue }
            try fileManager.copyItem(at: file, to: output.appending(component: name))
        }
        var config = config
        switch precision {
        case .source:
            break
        case .dense:
            config["quantization"] = nil
            config["quantization_config"] = nil
        case .affine(let bits, let groupSize):
            let block: [String: Any] = ["bits": bits, "group_size": groupSize, "mode": "affine"]
            config["quantization"] = block
            config["quantization_config"] = block
        case .mixed(let mixed):
            config["quantization"] = mixed.configBlock
            config["quantization_config"] = mixed.configBlock
        case .nf4(let groupSize):
            var block = config["quantization"] as? [String: Any] ?? ["bits": 4, "group_size": groupSize, "mode": "affine"]
            block["mode"] = "nf4"
            block["quant_method"] = nil
            config["quantization"] = block
            config["quantization_config"] = block
        }
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(component: "config.json"))
    }
}

struct PackedWeight {
    var scales: MLXArray
    var biases: MLXArray?
    var quantization: BaseConfiguration.Quantization

    func arrays(_ base: String, weight: MLXArray) -> [String: MLXArray] {
        var arrays = [base + ".weight": weight, base + ".scales": scales]
        arrays[base + ".biases"] = biases
        return arrays
    }
}

struct SourceLayout {
    let shards: [URL]
    let shardOf: [String: URL]
    let quantization: BaseConfiguration.PerLayerQuantization?

    init(directory: URL, config: [String: Any], configData: Data) throws {
        let index = (try? Data(contentsOf: directory.appending(component: "model.safetensors.index.json")))
            .flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] }
        let files = Set(try FileManager.default.contentsOfDirectory(atPath: directory.path))
        shards = ModelDeepCheck.weightFiles(in: files, prefix: "", weightMap: index?["weight_map"] as? [String: String])
            .map { directory.appending(component: $0) }
        guard !shards.isEmpty else { throw WeightConverter.Failure.malformed("no .safetensors weights in \(directory.path)") }
        var shardOf: [String: URL] = [:]
        for shard in shards {
            for key in try loadArrays(url: shard).keys { shardOf[key] = shard }
        }
        self.shardOf = shardOf
        quantization = config["quantization"] == nil
            ? nil : try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData).perLayerQuantization
        for key in shardOf.keys where key.hasSuffix(".scales") {
            let base = String(key.dropLast(".scales".count))
            if shardOf[base + ".weight"] != nil, quantization?.quantization(layer: base) == nil {
                throw WeightConverter.Failure.malformed("\(base) is quantized but config.json has no quantization for it")
            }
        }
    }

    func isCompanion(_ key: String) -> Bool {
        guard key.hasSuffix(".scales") || key.hasSuffix(".biases") else { return false }
        let base = String(key.dropLast(".scales".count))
        return shardOf[base + ".weight"] != nil && shardOf[base + ".scales"] != nil
    }

    func packed(_ key: String, taking arrays: inout [String: MLXArray]) throws -> PackedWeight? {
        guard key.hasSuffix(".weight") else { return nil }
        let base = String(key.dropLast(".weight".count))
        guard shardOf[base + ".scales"] != nil, let quantization = quantization?.quantization(layer: base) else { return nil }
        func take(_ name: String) throws -> MLXArray? {
            guard let shard = shardOf[name] else { return nil }
            return try arrays.removeValue(forKey: name) ?? loadArrays(url: shard)[name]
        }
        guard let scales = try take(base + ".scales") else { return nil }
        return PackedWeight(scales: scales, biases: try take(base + ".biases"), quantization: quantization)
    }
}

struct LoRAUpdate {
    let scale: Float
    let pairs: [String: (a: MLXArray, b: MLXArray)]

    init(directory: URL, keys: [String: URL]) throws {
        let configuration = try JSONDecoder().decode(
            LoRAConfiguration.self, from: Data(contentsOf: directory.appending(component: "adapter_config.json")))
        guard configuration.fineTuneType == .lora else {
            throw WeightConverter.Failure.malformed("only LoRA adapters merge; \(directory.path) is \(configuration.fineTuneType.rawValue)")
        }
        let arrays = try loadArrays(url: directory.appending(component: "adapters.safetensors"))
        var pairs: [String: (a: MLXArray, b: MLXArray)] = [:]
        for (key, a) in arrays where key.hasSuffix(".lora_a") {
            let base = String(key.dropLast(".lora_a".count))
            guard let b = arrays[base + ".lora_b"], keys[base + ".weight"] != nil else {
                throw WeightConverter.Failure.malformed("adapter layer \(base) has no lora_b or is not in the model")
            }
            pairs[base] = (a, b)
        }
        guard !pairs.isEmpty else { throw WeightConverter.Failure.malformed("\(directory.path) holds no lora_a/lora_b pairs") }
        scale = configuration.loraParameters.scale
        self.pairs = pairs
    }

    static func merged(_ weight: MLXArray, _ pair: (a: MLXArray, b: MLXArray), scale: Float, layer: String) throws -> MLXArray {
        let a = pair.a.asType(.float32)
        let b = pair.b.asType(.float32)
        let delta = a.ndim == 3 ? matmul(b, a) : matmul(a, b).T
        guard delta.shape == weight.shape else {
            throw WeightConverter.Failure.malformed("adapter \(layer): update \(delta.shape) does not fit weight \(weight.shape)")
        }
        return (weight.asType(.float32) + scale * delta).asType(weight.dtype)
    }
}

struct ShardWriter {
    let directory: URL
    let maxBytes: Int
    let metadata: [String: String]
    private var pending: [String: MLXArray] = [:]
    private var pendingBytes = 0
    private var written: [[String]] = []
    private var totalBytes = 0

    init(directory: URL, maxBytes: Int, metadata: [String: String]) {
        self.directory = directory
        self.maxBytes = maxBytes
        self.metadata = metadata
    }

    mutating func add(_ arrays: [String: MLXArray]) throws {
        let bytes = arrays.values.reduce(0) { $0 + $1.nbytes }
        if !pending.isEmpty, pendingBytes + bytes > maxBytes { try flush() }
        eval(Array(arrays.values))
        pending.merge(arrays) { _, new in new }
        pendingBytes += bytes
    }

    mutating func finish() throws -> Int {
        if !pending.isEmpty { try flush() }
        var weightMap: [String: String] = [:]
        for (index, keys) in written.enumerated() {
            let name = written.count == 1
                ? "model.safetensors" : String(format: "model-%05d-of-%05d.safetensors", index + 1, written.count)
            try FileManager.default.moveItem(at: part(index), to: directory.appending(component: name))
            for key in keys { weightMap[key] = name }
        }
        let index: [String: Any] = ["metadata": ["total_size": totalBytes], "weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appending(component: "model.safetensors.index.json"))
        return written.count
    }

    private mutating func flush() throws {
        try save(arrays: pending, metadata: metadata, url: part(written.count))
        written.append(Array(pending.keys))
        totalBytes += pendingBytes
        pending = [:]
        pendingBytes = 0
        Memory.clearCache()
    }

    private func part(_ index: Int) -> URL {
        directory.appending(component: "part-\(index).safetensors")
    }
}
