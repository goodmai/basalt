import BasaltCore
import Foundation
import MLX
import MLXLMCommon

struct SINQStreamInputs {
    let config: [String: Any]
    let lora: LoRAUpdate?
    let activationRMS: [String: MLXArray]
    let calibrationTokens: Int
}

extension WeightConverter {
    @discardableResult
    package func runWithSINQ(source: URL, output: URL, calibrationText: String? = nil) async throws -> Summary {
        guard recipe == .asinq || recipe == .moeRW else { return try run(source: source, output: output) }
        guard precision != .nvfp4 else { throw Failure.refused("nvfp4 takes the rtn recipe; \(recipe.rawValue) has no nvfp4 normalizer yet (#677)") }
        try Self.checkOutput(output, source: source)
        let configURL = source.appending(component: "config.json")
        guard let config = ModelCache.config(at: source) else {
            throw Failure.malformed("\(configURL.path) is missing or unreadable")
        }
        let layout = try SourceLayout(directory: source, config: config, configData: Data(contentsOf: configURL))
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
        let groups = SINQAbsorption.discoverGroups(keys: Set(layout.shardOf.keys))
        let resolvedCal = (calibrationText?.isEmpty == false ? calibrationText : nil)
            ?? (recipe == .moeRW ? MultilingualMoECalibrationCorpus.defaultText : nil)
        let calibrated = try await SINQAbsorption.captureActivationRMS(
            source: source, groups: groups, calibrationText: resolvedCal)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        do {
            try converter.copySidecars(from: source, to: output, config: config)
            let inInfo = SINQStreamInputs(
                config: config, lora: lora,
                activationRMS: calibrated.rmsByNorm, calibrationTokens: calibrated.tokens)
            var (summary, _) = try converter.streamWithSINQ(layout, inputs: inInfo, to: output)
            summary.plan = bitPlan
            if recipe == .moeRW {
                _ = try await UniversalModelConverter.refineStandardDirectoryWithMoERW(
                    output: output, teacher: source, calibrationText: resolvedCal)
                summary.bytesAfter = Int(UniversalSourceReader.directoryWeightBytes(output))
            }
            return summary
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }
    }

    func streamWithSINQ(
        _ layout: SourceLayout,
        inputs: SINQStreamInputs,
        to output: URL
    ) throws -> (Summary, SINQSummary) {
        let startClock = ContinuousClock.now
        let groups = SINQAbsorption.discoverGroups(keys: Set(layout.shardOf.keys))
        let addsUnitOffset = SINQAbsorption.addsUnitOffset(config: inputs.config)
        let groupByKey = Self.indexSINQGroups(groups)
        var evaluatedGroups: Set<Int> = []
        var stagedOutputs: [String: [String: MLXArray]] = [:]
        var shardCache = SingleShardCache()
        var sinqAcc = SINQAccumulator(calibrationTokens: inputs.calibrationTokens)
        var writer = ShardWriter(directory: output, maxBytes: maxShardBytes, metadata: ["format": format == .hf ? "pt" : "mlx"])
        var summary = Summary()

        for shard in layout.shards {
            var arrays = try loadArrays(url: shard)
            shardCache.current = (shard, arrays)
            for key in arrays.keys.sorted() where !layout.isCompanion(key) {
                if let gIdx = groupByKey[key], !evaluatedGroups.contains(gIdx) {
                    evaluatedGroups.insert(gIdx)
                    let ctx = SINQGroupEvalContext(
                        layout: layout, lora: inputs.lora,
                        addsUnitOffset: addsUnitOffset, actStd: inputs.activationRMS[groups[gIdx].normWeightKey])
                    if let solved = try evaluateSINQGroup(groups[gIdx], ctx: ctx, shardCache: &shardCache) {
                        sinqAcc.record(solved, into: &stagedOutputs)
                    }
                }
                guard let tensor = arrays.removeValue(forKey: key) else { continue }
                let packed = try layout.packed(key, taking: &arrays)
                let outputs = try stagedOutputs.removeValue(forKey: key)
                    ?? convert(key, tensor, packed: packed, lora: inputs.lora, sourceQuantized: layout.quantization != nil)
                noteGroupChoice(key, tensor, packed: packed, sourceQuantized: layout.quantization != nil, into: &summary)
                let rawInputs = [tensor, packed?.scales, packed?.biases].compactMap { $0 }
                summary.tensors += rawInputs.count
                summary.bytesBefore += rawInputs.reduce(0) { $0 + $1.nbytes }
                summary.outputTensors += outputs.count
                summary.bytesAfter += outputs.values.reduce(0) { $0 + $1.nbytes }
                try writer.add(outputs)
            }
        }
        summary.shards = try writer.finish()
        try finishGroupChoices(summary, output: output)
        let finalSINQ = sinqAcc.finalize(elapsed: ContinuousClock.now - startClock)
        summary.sinq = finalSINQ
        return (summary, finalSINQ)
    }

    private static func indexSINQGroups(_ groups: [SINQAbsorptionGroup]) -> [String: Int] {
        var groupByKey: [String: Int] = [:]
        for (idx, g) in groups.enumerated() {
            groupByKey[g.normWeightKey] = idx
            if let bKey = g.normBiasKey { groupByKey[bKey] = idx }
            for base in g.quantizedBases { groupByKey[base + ".weight"] = idx }
            for dKey in g.denseCompensatedKeys { groupByKey[dKey] = idx }
        }
        return groupByKey
    }
}

struct SINQAccumulator {
    var summary = SINQSummary()
    var imbalanceBeforeSum: Float = 0
    var imbalanceAfterSum: Float = 0

    init(calibrationTokens: Int) {
        summary.calibrationTokens = calibrationTokens
    }

    mutating func record(_ solved: EvaluatedSINQGroup, into stagedOutputs: inout [String: [String: MLXArray]]) {
        for (k, outDict) in solved.outputs { stagedOutputs[k] = outDict }
        summary.absorbedGroups += 1
        summary.normalizedTensors += solved.normalizedCount
        summary.routerProjectionsAdjusted += solved.routerCount
        imbalanceBeforeSum += solved.imbalanceBefore
        imbalanceAfterSum += solved.imbalanceAfter
    }

    func finalize(elapsed: Duration) -> SINQSummary {
        var out = summary
        if out.absorbedGroups > 0 {
            let denom = Float(out.absorbedGroups)
            out.meanImbalanceBefore = imbalanceBeforeSum / denom
            out.meanImbalanceAfter = imbalanceAfterSum / denom
        }
        out.wallSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        return out
    }
}

struct SingleShardCache {
    var current: (url: URL, arrays: [String: MLXArray])?
    var extra: (url: URL, arrays: [String: MLXArray])?

    mutating func tensor(_ key: String, in layout: SourceLayout) throws -> MLXArray? {
        guard let url = layout.shardOf[key] else { return nil }
        if current?.url == url, let val = current?.arrays[key] { return val }
        if extra?.url == url, let val = extra?.arrays[key] { return val }
        let loaded = try loadArrays(url: url)
        extra = (url, loaded)
        return loaded[key]
    }
}

struct SINQGroupEvalContext {
    let layout: SourceLayout
    let lora: LoRAUpdate?
    let addsUnitOffset: Bool
    let actStd: MLXArray?
}

struct EvaluatedSINQGroup {
    let outputs: [String: [String: MLXArray]]
    let normalizedCount: Int
    let routerCount: Int
    let imbalanceBefore: Float
    let imbalanceAfter: Float
}

extension WeightConverter {
    func evaluateSINQGroup(
        _ group: SINQAbsorptionGroup,
        ctx: SINQGroupEvalContext,
        shardCache: inout SingleShardCache
    ) throws -> EvaluatedSINQGroup? {
        guard let normWeight = try shardCache.tensor(group.normWeightKey, in: ctx.layout),
              normWeight.ndim == 1, normWeight.dtype.isFloatingPoint else {
            return nil
        }
        let width = normWeight.dim(0)
        guard width > 0,
              let entries = try collectSINQEntries(group: group, width: width, ctx: ctx, shardCache: &shardCache),
              let denseCompensated = try collectDenseConsumers(group: group, width: width, ctx: ctx, shardCache: &shardCache)
        else { return nil }

        let stacked2D = concatenated(entries.map { $0.dense.asType(.float32).reshaped($0.rows2D, width) }, axis: 0)
        let factors = SINQNormalizer.solve(stacked2D, activationStd: ctx.actStd, config: sinqConfig)

        var outputs: [String: [String: MLXArray]] = [:]
        outputs[group.normWeightKey] = [
            group.normWeightKey: cast(SINQNormalizer.scaleNormWeight(
                normWeight, colScale: factors.colScale, addsUnitOffset: ctx.addsUnitOffset)),
        ]
        if let biasKey = group.normBiasKey, let normBias = try shardCache.tensor(biasKey, in: ctx.layout),
           normBias.size == width, normBias.dtype.isFloatingPoint {
            let scaledBias = (normBias.asType(.float32) * factors.colScale.reshaped(normBias.shape)).asType(normBias.dtype)
            eval(scaledBias)
            outputs[biasKey] = [biasKey: cast(scaledBias)]
        }
        var cursor = 0
        for entry in entries {
            let rowSlice = factors.rowScale[cursor ..< (cursor + entry.rows2D)]
            cursor += entry.rows2D
            outputs[entry.base + ".weight"] = SINQNormalizer.quantizeNormalized(
                base: entry.base, weight: entry.dense,
                rowScale: rowSlice, colScale: factors.colScale, format: entry.format)
        }
        for item in denseCompensated {
            let comp = SINQNormalizer.compensateDenseConsumer(item.weight, colScale: factors.colScale)
            outputs[item.key] = [item.key: cast(comp)]
        }
        return EvaluatedSINQGroup(
            outputs: outputs, normalizedCount: entries.count, routerCount: denseCompensated.count,
            imbalanceBefore: factors.imbalanceBefore, imbalanceAfter: factors.imbalanceAfter)
    }

    private func collectSINQEntries(
        group: SINQAbsorptionGroup,
        width: Int,
        ctx: SINQGroupEvalContext,
        shardCache: inout SingleShardCache
    ) throws -> [(base: String, dense: MLXArray, rows2D: Int, format: SINQNormalizer.QuantFormat)]? {
        var entries: [(base: String, dense: MLXArray, rows2D: Int, format: SINQNormalizer.QuantFormat)] = []
        for base in group.quantizedBases {
            let wKey = base + ".weight"
            guard let rawW = try shardCache.tensor(wKey, in: ctx.layout) else { return nil }
            let packed = try loadPackedForSINQ(base: base, layout: ctx.layout, shardCache: &shardCache)
            guard let qFormat = sinqFormat(
                key: wKey, tensor: rawW, packed: packed, sourceQuantized: ctx.layout.quantization != nil
            ) else { return nil }
            var dense = Self.dense(rawW, packed)
            if let pair = ctx.lora?.pairs[base], let lora = ctx.lora {
                dense = try LoRAUpdate.merged(dense, pair, scale: lora.scale, layer: base)
            }
            guard (2 ... 3).contains(dense.ndim), dense.dim(-1) == width else { return nil }
            let flatRows = dense.size / width
            guard flatRows > 0 else { return nil }
            entries.append((base, dense, flatRows, qFormat))
        }
        return entries.isEmpty ? nil : entries
    }

    private func collectDenseConsumers(
        group: SINQAbsorptionGroup,
        width: Int,
        ctx: SINQGroupEvalContext,
        shardCache: inout SingleShardCache
    ) throws -> [(key: String, weight: MLXArray)]? {
        var list: [(key: String, weight: MLXArray)] = []
        for dKey in group.denseCompensatedKeys {
            let dBase = String(dKey.dropLast(".weight".count))
            guard ctx.layout.shardOf[dBase + ".scales"] == nil,
                  var dTensor = try shardCache.tensor(dKey, in: ctx.layout),
                  dTensor.ndim >= 2, dTensor.dim(-1) == width, dTensor.dtype.isFloatingPoint else {
                return nil
            }
            if let pair = ctx.lora?.pairs[dBase], let lora = ctx.lora {
                dTensor = try LoRAUpdate.merged(dTensor, pair, scale: lora.scale, layer: dBase)
            }
            list.append((dKey, dTensor))
        }
        return list
    }

    private func loadPackedForSINQ(
        base: String, layout: SourceLayout, shardCache: inout SingleShardCache
    ) throws -> PackedWeight? {
        guard layout.shardOf[base + ".scales"] != nil,
              let quant = layout.quantization?.quantization(layer: base),
              let scales = try shardCache.tensor(base + ".scales", in: layout) else {
            return nil
        }
        let biases = try shardCache.tensor(base + ".biases", in: layout)
        return PackedWeight(scales: scales, biases: biases, quantization: quant)
    }

    private func sinqFormat(
        key: String, tensor: MLXArray, packed: PackedWeight?, sourceQuantized: Bool
    ) -> SINQNormalizer.QuantFormat? {
        switch precision {
        case .source, .dense:
            return nil
        case .mxfp4:
            return Self.quantizable(key, tensor, packed: packed, sourceQuantized: sourceQuantized, groupSize: 32) ? .mxfp4 : nil
        case .nvfp4:
            return nil
        case .nf4(let groupSize):
            return Self.takesNF4(key, tensor, packed: packed, sourceQuantized: sourceQuantized, groupSize: groupSize) ? .nf4(groupSize: groupSize) : nil
        case .affine(let bits, let groupSize):
            return Self.fittingGroup(key, tensor, packed: packed, sourceQuantized: sourceQuantized, requested: groupSize).map { .affine(bits: bits, groupSize: $0) }
        case .mixed(let mixed):
            guard let group = Self.fittingGroup(key, tensor, packed: packed, sourceQuantized: sourceQuantized, requested: mixed.groupSize) else { return nil }
            return .affine(bits: mixed.bits(layer: String(key.dropLast(".weight".count))), groupSize: group)
        }
    }
}
