import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension PVTuner {
    package static let defaultCalibrationCorpus: String = {
        let block = """
        Swift 6.4 strict concurrency enforces actor isolation and Sendable value cores across module boundaries.
        func decodeQuantizedRow(packed: Span<UInt32>, scales: Span<Float16>, groupSize: Int) -> Float {
            var total: Float = 0
            for idx in packed.indices { total += Float(packed[idx] & 0x7) * Float(scales[idx / groupSize]) }
            return total
        }
        {"model":"gemma-4-26b-a4b","quantization":{"bits":3,"group_size":64,"mode":"affine"},"status":"ready"}
        When converting a quantized transformer checkpoint to three bits per weight, uniform round-to-nearest rounding
        shifts output logit distributions on outlier channels unless per-group scales and discrete codes are refined
        against top-K teacher log-probabilities on held-out sequences with monotone Kullback-Leibler rollback.
        """
        return Array(repeating: block, count: 64).joined(separator: "\n\n")
    }()

    package static func tune(
        _ model: any LanguageModel,
        train: [PVTeacherBatch],
        val: [PVTeacherBatch],
        config: PVTuneConfig,
        onStep: ((String) -> Void)? = nil
    ) throws -> PVTuneSummary {
        let startClock = ContinuousClock.now
        let root = model as Module
        let inProjTouch = root.parameters().flattened().filter { $0.0.hasSuffix("in_proj_qkv.weight") }
        if !inProjTouch.isEmpty {
            root.update(parameters: ModuleParameters.unflattened(inProjTouch))
            root.invalidateCompiledTraces()
        }
        let reversed = Array(Set(root.namedModules().compactMap { textBlockIndex($0.0) }).sorted().reversed())
        let stepSize = max(1, config.groupSizeBlocks)
        var groups = stride(from: 0, to: reversed.count, by: stepSize).map {
            Array(reversed[$0 ..< min(reversed.count, $0 + stepSize)])
        }
        if let cap = config.maxLayerGroups { groups = Array(groups.prefix(cap)) }
        let (trainBefore, valBefore) = (meanKL(model, Array(train.prefix(4))), meanKL(model, val))
        var (totalUpdated, completed) = (0, 0)
        defer {
            root.freeze()
            root.train(false)
            MLXInferenceEngine.settle(root)
        }
        if config.globalContinuousSteps > 0, !reversed.isEmpty {
            var globalCfg = config
            globalCfg.continuousStepsPerGroup = max(2, config.globalContinuousSteps / max(1, (reversed.count + 9) / 10))
            for (wIdx, start) in stride(from: 0, to: reversed.count, by: 10).enumerated() {
                let winBlocks = Set(reversed[start ..< min(reversed.count, start + 10)])
                try runContinuousGroup(model, blocks: winBlocks, train: train, val: val, stepSeed: wIdx, config: globalCfg)
                Memory.clearCache()
            }
            onStep?("global-P done, valKL=\(String(format: "%.4f", meanKL(model, val)))")
        }
        for (groupIdx, blockGroup) in groups.enumerated() {
            try Task.checkCancellation()
            let blockSet = Set(blockGroup)
            let flipped = try runDiscreteGroup(model, blocks: blockSet, train: train, val: val, stepSeed: groupIdx + 1, config: config)
            totalUpdated += flipped
            try Task.checkCancellation()
            try runContinuousGroup(model, blocks: blockSet, train: train, val: val, stepSeed: groupIdx + 1, config: config)
            completed += 1
            Memory.clearCache()
            onStep?("group \(groupIdx + 1)/\(groups.count) flipped=\(flipped) valKL=\(String(format: "%.4f", meanKL(model, val)))")
        }
        let (trainAfter, valAfter) = (meanKL(model, Array(train.prefix(4))), meanKL(model, val))
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let teacherTokens = (train.count + val.count) * (train.first?.seqLen ?? 0)
        return PVTuneSummary(
            trainKLBefore: trainBefore, trainKLAfter: trainAfter.isFinite ? trainAfter : trainBefore,
            valKLBefore: valBefore, valKLAfter: valAfter.isFinite ? valAfter : valBefore,
            totalUpdatedCodes: totalUpdated, completedGroups: completed, wallSeconds: secs, teacherTokens: teacherTokens)
    }

    package static func writeTunedShards(from model: Module, into directory: URL) throws -> Int {
        let updated = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let fm = FileManager.default
        let shardFiles = try fm.contentsOfDirectory(atPath: directory.path)
            .filter { $0.lowercased().hasSuffix(".safetensors") && !$0.hasPrefix(".") }
            .sorted()
            .map { directory.appending(component: $0) }
        var rewritten = 0
        for shardURL in shardFiles {
            var arrays = try loadArrays(url: shardURL)
            var changed = false
            for (key, existing) in arrays {
                let stripped = key.hasPrefix("language_model.") ? String(key.dropFirst("language_model.".count)) : nil
                guard let candidate = updated[key] ?? stripped.flatMap({ updated[$0] }),
                      candidate.shape == existing.shape, candidate.dtype == existing.dtype else { continue }
                arrays[key] = candidate
                rewritten += 1
                changed = true
            }
            guard changed else { continue }
            eval(Array(arrays.values))
            let tempURL = directory.appending(component: ".\(shardURL.lastPathComponent).pvtune.safetensors")
            try? fm.removeItem(at: tempURL)
            try save(arrays: arrays, metadata: ["format": "mlx"], url: tempURL)
            _ = try fm.replaceItemAt(shardURL, withItemAt: tempURL)
            Memory.clearCache()
        }
        return rewritten
    }

    package static func tuneLoadedStudent(
        _ studentContainer: ModelContainer,
        into output: URL,
        allBatches: [PVTeacherBatch],
        config: PVTuneConfig = PVTuneConfig(),
        onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> PVTuneSummary {
        guard allBatches.count >= 2 else {
            throw BasaltError.invalidRequestStructure(details: "pv-tune: calibration text is too short (need at least 2 teacher batches)")
        }
        let valCount = max(1, allBatches.count / 5)
        let trainBatches = Array(allBatches.dropLast(valCount))
        let valBatches = Array(allBatches.suffix(valCount))
        return try await studentContainer.perform { ctx in
            var tuned = try tune(ctx.model, train: trainBatches, val: valBatches, config: config, onStep: onStep)
            tuned.rewrittenTensors = try writeTunedShards(from: ctx.model as Module, into: output)
            return tuned
        }
    }

    package static func tuneConvertedDirectory(
        output: URL, teacher: URL, calibrationText: String? = nil, config: PVTuneConfig = PVTuneConfig(), onStep: (@Sendable (String) -> Void)? = nil
    ) async throws -> PVTuneSummary {
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? defaultCalibrationCorpus
        let allBatches: [PVTeacherBatch]
        do {
            let teacherEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
            defer { await withTaskCancellationShield { await teacherEngine.unload(); Memory.clearCache() } }
            try await teacherEngine.load(modelPath: teacher.path)
            guard let teacherContainer = await teacherEngine.container else { throw BasaltError.modelNotLoaded }
            allBatches = await collectTeacherBatches(teacherContainer, texts: corpus, count: 40, seqLen: 128, topK: 64)
        }
        let studentEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await studentEngine.unload(); Memory.clearCache() } }
        try await studentEngine.load(modelPath: output.path)
        guard let studentContainer = await studentEngine.container else { throw BasaltError.modelNotLoaded }
        return try await tuneLoadedStudent(studentContainer, into: output, allBatches: allBatches, config: config, onStep: onStep)
    }
}
