import BasaltCore
import Foundation
import MLX
import MLXNN

package enum VideoDiTCheckpointCompat {
    package static let supportedModelTypes: Set<String> = [
        "video_dit",
        "wan_video_dit",
        "ltx2_video",
        "ltx_2_5_av",
        "kandinsky_cross_dit",
        "moe_video_dit",
        "hunyuan_video",
        "cogvideox",
        "mochi_video",
        "sana_video",
    ]

    package static func isVideoDiTDirectory(_ directory: URL) -> Bool {
        let cfgURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return false
        }
        if (json["video_dit"] as? Bool) == true { return true }
        guard let modelType = (json["model_type"] as? String)?.lowercased() else { return false }
        return supportedModelTypes.contains(modelType)
    }

    package static func loadWeightsIfPresent(in directory: URL, into module: CausalVideoDiTModule) throws(BasaltError) {
        let candidates = [directory, directory.appending(component: "transformer")]
        guard let weightDir = candidates.first(where: { !safetensorsFiles(in: $0).isEmpty }) else { return }
        if let macaq = try MACaQNativeLoader.manifest(in: weightDir) {
            try MACaQNativeLoader.install(macaq.modules, in: module)
        } else if let macaqRoot = try MACaQNativeLoader.manifest(in: directory) {
            try MACaQNativeLoader.install(macaqRoot.modules, in: module)
        } else if let qtip = QTIPNativeLoader.manifest(in: weightDir) ?? QTIPNativeLoader.manifest(in: directory) {
            try QTIPNativeLoader.install(qtip, in: module)
        } else {
            installStandardQuantizedIfPresent(in: directory, module: module)
        }
        let existing = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        var matched: [(String, MLXArray)] = []
        do {
            for file in safetensorsFiles(in: weightDir) {
                let tensors = try MLX.loadArrays(url: file)
                for (key, value) in tensors {
                    if let target = existing[key], target.shape == value.shape {
                        matched.append((key, value.asType(target.dtype)))
                    }
                }
            }
            if !matched.isEmpty {
                module.update(parameters: ModuleParameters.unflattened(matched))
            }
            eval(module)
        } catch {
            throw .weightsCorrupted(path: directory.path, reason: error.localizedDescription)
        }
    }

    private static func installStandardQuantizedIfPresent(in directory: URL, module: Module) {
        guard let data = try? Data(contentsOf: directory.appending(component: "config.json")),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let quant = (json["quantization"] as? [String: Any]) ?? (json["quantization_config"] as? [String: Any]) else {
            return
        }
        let bits = (quant["bits"] as? Int) ?? 4
        let groupSize = (quant["group_size"] as? Int) ?? 64
        let mode: QuantizationMode = ((quant["mode"] as? String)?.lowercased() == "mxfp4") ? .mxfp4 : .affine
        quantize(model: module, groupSize: groupSize, bits: bits, mode: mode) { _, mod in
            guard let lin = mod as? Linear else { return false }
            return lin.weight.dim(-1).isMultiple(of: groupSize)
        }
    }

    static func safetensorsFiles(in directory: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" && !$0.lastPathComponent.hasPrefix(".") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}

extension VideoDiTCheckpointCompat {
    package static func convertCalibrated(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) throws -> (shards: Int, bytesAfter: Int64) {
        let (_, cfg) = try VideoDiTConfig.load(from: workingSource, identifier: request.source)
        let module = CausalVideoDiTModule(config: cfg)
        try loadWeightsIfPresent(in: workingSource, into: module)
        switch request.target {
        case .macaqMXFP4:
            return try convertToMACaQ(
                module: module, mode: .mxfp4, request: request, source: workingSource, output: outputURL, onProgress: onProgress
            )
        case .macaqAffine(let bits, let groupSize):
            return try convertToMACaQ(
                module: module, mode: .affine(bits: bits, groupSize: groupSize),
                request: request, source: workingSource, output: outputURL, onProgress: onProgress
            )
        case .qtip(let bpw), .qtipBudget(let bpw), .gsqRCO(let bpw):
            let resolvedBPW = QTIPRate(bpw: bpw) != nil ? bpw : 2.0
            return try convertToQTIP(
                module: module, bpw: resolvedBPW, source: workingSource, output: outputURL, onProgress: onProgress
            )
        default:
            throw BasaltError.invalidRequestStructure(details: "video-dit convert: unsupported calibrated target \(request.target.specString)")
        }
    }

    private static func convertToMACaQ(
        module: CausalVideoDiTModule,
        mode: MACaQTargetMode,
        request: ModelConversionRequest,
        source: URL,
        output: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) throws -> (shards: Int, bytesAfter: Int64) {
        let align = max(mode.groupSize, request.hadamardBlock ?? 32)
        let hadamard = request.hadamardBlock ?? 32
        let alpha = request.alpha ?? 0.5
        var manifest: [String: MACaQLayerSpec] = [:]
        for (path, sub) in module.namedModules() {
            guard let lin = denseLinear(from: sub),
                  lin.shape.1.isMultiple(of: align) else { continue }
            let built = try quantizeLinearToMACaQ(lin, mode: mode, hadamardBlock: hadamard, alpha: alpha)
            try PVTuner.swapModule(path, with: built.layer, in: module)
            manifest[path] = built.spec
        }
        onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: "MACaQ quantized \(manifest.count) Video DiT layer(s)"))
        let written = try MACaQRefiner.writeMACaQDirectory(
            model: module, manifest: manifest, alpha: alpha, source: source, output: output
        )
        return (written.shards, Int64(written.bytes))
    }

    private static func quantizeLinearToMACaQ(
        _ linear: Linear,
        mode: MACaQTargetMode,
        hadamardBlock: Int,
        alpha: Float
    ) throws -> (layer: MACaQQuantizedLinear, spec: MACaQLayerSpec) {
        let dense = linear.weight.asType(.float32)
        let inF = dense.dim(1)
        let rawH = eye(inF, dtype: .float32)
        let pre = MACaQMath.balancePreScale(weight: dense, meanSquareActivation: diagonal(rawH), alpha: alpha).asType(.bfloat16)
        eval(pre)
        let frame = MACaQMath.transformFrame(weight: dense, hessian: rawH, preScale: pre, blockSize: hadamardBlock, ridge: 1e-4)
        let packed: MACaQPackedWeights
        switch mode {
        case .mxfp4:
            let scr = MACaQMath.screenMXFP4Exponents(weight: frame.weight, hessianDiag: diagonal(frame.hessian))
            var grid = MACaQGridState(
                codes: scr.sidx, scaleElem: scr.scaleElem, biasElem: nil,
                levelTable: MLXArray(MACaQMath.mxfp4SignedLevels), maxLevel: 14
            )
            _ = MACaQMath.flipRefine(state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: 2)
            let pk = MACaQMath.packMXFP4(sidx: grid.codes, e8m0: scr.e8m0)
            packed = MACaQPackedWeights(weight: pk.weight, scales: pk.scales, biases: nil)
        case .affine(let bits, let groupSize):
            let (wq, sc, bi) = MLX.quantized(frame.weight.asType(.bfloat16), groupSize: groupSize, bits: bits, mode: .affine)
            packed = MACaQPackedWeights(weight: wq, scales: sc, biases: bi)
        }
        var spec = MACaQLayerSpec(bits: mode.bits, groupSize: mode.groupSize, mode: mode.quantizationMode, hadamardBlock: hadamardBlock)
        spec.scaleDType = "\(packed.scales.dtype)"
        spec.hasBias = linear.bias != nil
        let qLayer = MACaQQuantizedLinear(packed: packed, bias: linear.bias, preScale: pre, spec: spec)
        eval(qLayer)
        return (qLayer, spec)
    }

    private static func convertToQTIP(
        module: CausalVideoDiTModule,
        bpw: Double,
        source: URL,
        output: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) throws -> (shards: Int, bytesAfter: Int64) {
        guard let rate = QTIPRate(bpw: bpw) else {
            throw BasaltError.invalidRequestStructure(details: "video-dit qtip: invalid bpw \(bpw)")
        }
        let seed: UInt64 = 0x5170_2026
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        eval(tlut)
        var entries: [String: QTIPNativeLoader.LayerEntry] = [:]
        for (path, sub) in module.namedModules() {
            guard let lin = denseLinear(from: sub) else { continue }
            let (outF, inF) = lin.shape
            let inKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: inF, sharedMoERHT: false)
            let outKey = QTIPQuantizer.sharedOutputKey(path: path, sharedMoERHT: false)
            guard let rhtIn = KroneckerRHT(dimension: inF, seed: QTIPQuantizer.hashString(inKey, seed: seed)),
                  let rhtOut = KroneckerRHT(dimension: outF, seed: QTIPQuantizer.hashString(outKey, seed: seed)) else {
                continue
            }
            let spec = QTIPLayerSpec(rate: rate, rhtIn: rhtIn, rhtOut: rhtOut)
            let qLayer = QTIPQuantizer.quantizeMatrix(
                weight: lin.weight.asType(.float32), bias: lin.bias, spec: spec, choleskyFactors: nil, tlut: tlut
            )
            eval(qLayer)
            try PVTuner.swapModule(path, with: qLayer, in: module)
            entries[path] = QTIPNativeLoader.LayerEntry(outFeatures: outF, inFeatures: inF, hasBias: lin.bias != nil, bpw: rate.bpw)
        }
        onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: "QTIP quantized \(entries.count) Video DiT layer(s)"))
        return try writeQTIPVideoDirectory(module: module, rate: rate, seed: seed, entries: entries, source: source, output: output)
    }

    private static func writeQTIPVideoDirectory(
        module: CausalVideoDiTModule,
        rate: QTIPRate,
        seed: UInt64,
        entries: [String: QTIPNativeLoader.LayerEntry],
        source: URL,
        output: URL
    ) throws -> (shards: Int, bytesAfter: Int64) {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var writer = ShardWriter(directory: output, maxBytes: 4_500_000_000, metadata: ["format": "mlx"])
        var bytesAfter: Int64 = 0
        for (key, arr) in module.parameters().flattened().sorted(by: { $0.0 < $1.0 }) {
            bytesAfter += Int64(arr.nbytes)
            try writer.add([key: arr])
        }
        let shards = try writer.finish()
        let cfgURL = source.appending(component: "config.json")
        var cfg = (try? JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as? [String: Any]) ?? [:]
        cfg.removeValue(forKey: "quantization")
        cfg.removeValue(forKey: "quantization_config")
        let manifest = QTIPNativeLoader.Manifest(bpw: rate.bpw, seed: seed, layers: entries, sharedMoERHT: false)
        cfg["qtip_quantization"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
        try JSONSerialization.data(withJSONObject: cfg, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(component: "config.json"))
        return (shards, bytesAfter)
    }
}

private extension VideoDiTCheckpointCompat {
    static func denseLinear(from sub: Module) -> Linear? {
        if type(of: sub) == Linear.self, let lin = sub as? Linear {
            return lin
        }
        if let ql = sub as? QuantizedLinear {
            let deq = MLX.dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode, dtype: .float32)
            eval(deq)
            return Linear(weight: deq, bias: ql.bias)
        }
        return nil
    }
}

extension VideoDiTCheckpointCompat {
    package static func exportCanonicalDense(
        from source: URL,
        to output: URL,
        dtype: DType,
        onStep: (@Sendable (String) -> Void)? = nil
    ) throws -> WeightConverter.Summary {
        let (_, cfg) = try VideoDiTConfig.load(from: source, identifier: source.lastPathComponent)
        let module = CausalVideoDiTModule(config: cfg)
        try loadWeightsIfPresent(in: source, into: module)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let cfgURL = source.appending(component: "config.json")
        var json = (try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as? [String: Any]) ?? [:]
        for key in UniversalSourceReader.customConfigKeys { json[key] = nil }
        json["torch_dtype"] = dtype == .float16 ? "float16" : "bfloat16"
        try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(component: "config.json"))
        var writer = ShardWriter(directory: output, maxBytes: 5 << 30, metadata: ["format": "mlx"])
        var summary = WeightConverter.Summary()
        summary.bytesBefore = Int(UniversalSourceReader.directoryWeightBytes(source))
        var handled: Set<String> = []
        for (path, sub) in module.namedModules() {
            guard let decoded = UniversalSourceReader.decodeModule(path: path, module: sub, dtype: dtype, handledKeys: &handled) else {
                continue
            }
            summary.tensors += decoded.count
            summary.outputTensors += decoded.count
            summary.bytesAfter += decoded.values.reduce(0) { $0 + $1.nbytes }
            try writer.add(decoded)
        }
        for (paramKey, arr) in module.parameters().flattened() where !handled.contains(paramKey) {
            let casted = arr.dtype.isFloatingPoint ? arr.asType(dtype) : arr
            eval(casted)
            summary.tensors += 1
            summary.outputTensors += 1
            summary.bytesAfter += casted.nbytes
            try writer.add([paramKey: casted])
        }
        summary.shards = try writer.finish()
        onStep?("canonical Video DiT decode wrote \(summary.outputTensors) tensor(s)")
        return summary
    }

    package static func pvTuneDirectory(
        output: URL,
        teacher: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) throws -> ModelConversionPVSummary {
        let (_, cfg) = try VideoDiTConfig.load(from: teacher, identifier: teacher.lastPathComponent)
        let teacherMod = CausalVideoDiTModule(config: cfg)
        try loadWeightsIfPresent(in: teacher, into: teacherMod)
        let studentMod = CausalVideoDiTModule(config: cfg)
        try loadWeightsIfPresent(in: output, into: studentMod)

        let noise = PhasedVideoRunnerActor.makeInitialNoise(seed: 0xC0FFEE, seqLen: 16, channels: cfg.latentChannels)
        let cond = PhasedVideoRunnerActor.makeInitialNoise(seed: 0xBAD5EED, seqLen: 1, channels: cfg.hiddenSize).reshaped(1, cfg.hiddenSize)
        let teacherVel = teacherMod.predictVelocity(
            teacherMod.modulatedInput(noise, timestep: 0.5, pooledCond: cond), pooledCond: cond, staWindow: nil
        ).velocity
        eval(teacherVel)

        let lossBefore = velocityDivergence(student: studentMod, noise: noise, cond: cond, target: teacherVel)
        var bestLoss = lossBefore
        var updatedCount = 0
        let evalLoss = { velocityDivergence(student: studentMod, noise: noise, cond: cond, target: teacherVel) }
        for (path, sub) in studentMod.namedModules() {
            if refineContinuousScale(path: path, module: sub, root: studentMod, evalLoss: evalLoss, currentBest: &bestLoss) {
                updatedCount += 1
            }
        }
        _ = try PVTuner.writeTunedShards(from: studentMod, into: output)
        onProgress(.init(
            phase: "pv_tune", completed: 3, total: 4,
            message: "Video DiT PV-Tuning refined \(updatedCount) scale group(s)",
            valKLBefore: Double(lossBefore), valKLCurrent: Double(bestLoss)
        ))
        return ModelConversionPVSummary(
            valKLBefore: Double(lossBefore),
            valKLAfter: Double(bestLoss),
            updatedCodes: updatedCount
        )
    }

    private static func velocityDivergence(
        student: CausalVideoDiTModule,
        noise: MLXArray,
        cond: MLXArray,
        target: MLXArray
    ) -> Float {
        let pred = student.predictVelocity(
            student.modulatedInput(noise, timestep: 0.5, pooledCond: cond), pooledCond: cond, staWindow: nil
        ).velocity
        let diff = pred.asType(.float32) - target.asType(.float32)
        let mse = mean(diff * diff)
        eval(mse)
        let val = mse.item(Float.self)
        return val.isFinite ? max(0.0, val) : 1.0
    }

    private static func refineContinuousScale(
        path: String,
        module: Module,
        root: CausalVideoDiTModule,
        evalLoss: () -> Float,
        currentBest: inout Float
    ) -> Bool {
        let paramName: String
        let original: MLXArray
        if let macaq = module as? MACaQQuantizedLinear {
            paramName = "\(path).pre_scale"
            original = macaq.preScale + MLXArray(Float(0), dtype: macaq.preScale.dtype)
        } else if let qtip = module as? QTIPLinear {
            paramName = "\(path).row_scales"
            original = qtip.rowScales + MLXArray(Float(0), dtype: qtip.rowScales.dtype)
        } else {
            return false
        }
        eval(original)
        var improved = false
        for factor in [Float(0.985), Float(1.015), Float(0.97), Float(1.03)] {
            let candidate = (original.asType(.float32) * factor).asType(original.dtype)
            eval(candidate)
            root.update(parameters: ModuleParameters.unflattened([(paramName, candidate)]))
            let candidateLoss = evalLoss()
            if candidateLoss + 1e-7 < currentBest {
                currentBest = candidateLoss
                improved = true
                return true
            }
        }
        if !improved {
            root.update(parameters: ModuleParameters.unflattened([(paramName, original)]))
        }
        return false
    }
}
