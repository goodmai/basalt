import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct PVQuantTarget {
    package var path: String
    package var module: Module
    package var packedWeight: MLXArray
    package var scales: MLXArray
    package var biases: MLXArray?
    package var bias: MLXArray?
    package var spec: PVCodeGrid.Spec
    package var isSwitch: Bool
    package var elementCount: Int
    package var macaqPreScale: MLXArray? = nil
    package var macaqBlock: Int? = nil
}

extension PVTuner {
    package static func swapModule(_ path: String, with module: Module, in root: Module) throws {
        var parts = path.split(separator: ".").map(String.init)
        guard let key = parts.popLast(),
              let parent = root.namedModules().first(where: { $0.0 == parts.joined(separator: ".") })?.1 else {
            throw BasaltError.invalidRequestStructure(details: "pv: missing parent for \(path)")
        }
        try parent.update(modules: ModuleChildren.unflattened([(key, module)]), verify: .noUnusedKeys)
        root.invalidateCompiledTraces()
    }

    static func collectTargets(_ root: Module, blocks: Set<Int>, maxSwitchExperts: Int) -> [PVQuantTarget] {
        var targets: [PVQuantTarget] = []
        for (path, mod) in root.namedModules() {
            guard let idx = textBlockIndex(path), blocks.contains(idx), !path.contains("router") else { continue }
            if let item = target(path: path, module: mod, maxSwitchExperts: maxSwitchExperts) {
                targets.append(item)
            }
        }
        return targets
    }

    private static func target(path: String, module: Module, maxSwitchExperts: Int) -> PVQuantTarget? {
        if let mac = macaqTarget(path: path, module: module, maxSwitchExperts: maxSwitchExperts) {
            return mac
        }
        if let nf4 = module as? NF4Linear {
            let wCopy = nf4.weight + MLXArray(UInt32(0))
            let sCopy = nf4.scales + MLXArray(0, dtype: nf4.scales.dtype)
            eval(wCopy, sCopy)
            return PVQuantTarget(
                path: path, module: nf4, packedWeight: wCopy, scales: sCopy, biases: nil, bias: nf4.bias,
                spec: PVCodeGrid.Spec(nf4GroupSize: nf4.groupSize), isSwitch: false,
                elementCount: nf4.weight.dim(0) * (nf4.weight.dim(1) * 8))
        }
        if let ql = module as? QuantizedLinear, ql.globalScale == nil, [2, 3, 4].contains(ql.bits), ql.mode == .affine {
            let wCopy = ql.weight + MLXArray(UInt32(0))
            let sCopy = ql.scales + MLXArray(0, dtype: ql.scales.dtype)
            let bCopy = ql.biases.map { $0 + MLXArray(0, dtype: $0.dtype) }
            eval(wCopy, sCopy)
            if let bCopy { eval(bCopy) }
            return PVQuantTarget(
                path: path, module: ql, packedWeight: wCopy, scales: sCopy, biases: bCopy, bias: ql.bias,
                spec: PVCodeGrid.Spec(groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode), isSwitch: false,
                elementCount: ql.weight.dim(0) * (ql.weight.dim(1) * 32 / ql.bits))
        }
        return switchTarget(path: path, module: module, maxSwitchExperts: maxSwitchExperts)
    }

    private static func switchTarget(path: String, module: Module, maxSwitchExperts: Int) -> PVQuantTarget? {
        if let nf4s = module as? NF4SwitchLinear {
            let p = Dictionary(uniqueKeysWithValues: nf4s.parameters().flattened())
            guard let w = p["weight"], let s = p["scales"], w.dim(0) <= maxSwitchExperts else { return nil }
            let (wCopy, sCopy) = (w + MLXArray(UInt32(0)), s + MLXArray(0, dtype: s.dtype))
            eval(wCopy, sCopy)
            return PVQuantTarget(
                path: path, module: nf4s, packedWeight: wCopy, scales: sCopy, biases: nil, bias: p["bias"],
                spec: PVCodeGrid.Spec(nf4GroupSize: nf4s.groupSize), isSwitch: true,
                elementCount: w.dim(0) * w.dim(1) * (w.dim(2) * 8))
        }
        guard let qs = module as? QuantizedSwitchLinear, [2, 3, 4].contains(qs.bits), qs.mode == .affine else { return nil }
        let p = Dictionary(uniqueKeysWithValues: qs.parameters().flattened())
        guard let w = p["weight"], let s = p["scales"], w.dim(0) <= maxSwitchExperts else { return nil }
        let (wCopy, sCopy) = (w + MLXArray(UInt32(0)), s + MLXArray(0, dtype: s.dtype))
        let bCopy = p["biases"].map { $0 + MLXArray(0, dtype: $0.dtype) }
        eval(wCopy, sCopy)
        if let bCopy { eval(bCopy) }
        return PVQuantTarget(
            path: path, module: qs, packedWeight: wCopy, scales: sCopy, biases: bCopy, bias: p["bias"],
            spec: PVCodeGrid.Spec(groupSize: qs.groupSize, bits: qs.bits, mode: qs.mode), isSwitch: true,
            elementCount: w.dim(0) * w.dim(1) * (w.dim(2) * 32 / qs.bits))
    }

    static func chunkTargets(_ targets: [PVQuantTarget], budget: Int) -> [[PVQuantTarget]] {
        var chunks: [[PVQuantTarget]] = []
        var current: [PVQuantTarget] = []
        var used = 0
        for t in targets {
            if !current.isEmpty && used + t.elementCount > budget {
                chunks.append(current)
                current = []
                used = 0
            }
            current.append(t)
            used += t.elementCount
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }

    static func setTrainingSuffix(_ root: Module, from minBlock: Int) {
        root.train(false)
        for (path, mod) in root.namedModules() {
            if let idx = textBlockIndex(path), idx >= minBlock { mod.train(true) }
        }
    }

    static func denseGradients(
        _ model: any LanguageModel, chunk: [PVQuantTarget], minBlock: Int, batches: [PVTeacherBatch]
    ) throws -> [String: MLXArray] {
        let root = model as Module
        root.freeze()
        defer {
            for t in chunk { try? swapModule(t.path, with: t.module, in: root) }
            root.freeze()
            root.train(false)
        }
        for t in chunk {
            let dense: MLXArray
            switch t.spec.encoding {
            case .nf4:
                dense = NF4Metal.dequantized(t.packedWeight, scales: t.scales, groupSize: t.spec.groupSize)
            case .affine(let mode):
                let outType: DType = t.scales.dtype.isFloatingPoint ? t.scales.dtype : .float32
                dense = dequantized(
                    t.packedWeight, scales: t.scales, biases: t.biases,
                    groupSize: t.spec.groupSize, bits: t.spec.bits, mode: mode, dtype: outType)
            }
            eval(dense)
            let replacement = makeDenseProbe(for: t, dense: dense)
            replacement.freeze()
            try replacement.unfreeze(keys: ["weight"], strict: false)
            try swapModule(t.path, with: replacement, in: root)
        }
        setTrainingSuffix(root, from: minBlock)
        var accum: [String: MLXArray] = [:]
        let gradFn = valueAndGrad(model: root) { _, arrays in
            [klLossArrays(model, inputs: arrays[0], ids: arrays[1], tLogp: arrays[2])]
        }
        for batch in batches {
            try Task.checkCancellation()
            let tok = MLXArray(batch.tokens)[.newAxis]
            let ids = MLXArray(batch.topIDs).reshaped(batch.seqLen, batch.topK)
            let lp = MLXArray(batch.topLogprobs).reshaped(batch.seqLen, batch.topK)
            let (_, grads) = gradFn(root, [tok, ids, lp])
            let flat = Dictionary(uniqueKeysWithValues: grads.flattened())
            for t in chunk {
                guard let g = flat["\(t.path).weight"]?.asType(.float32) else { continue }
                accum[t.path] = accum[t.path].map { $0 + g } ?? g
            }
            eval(Array(accum.values))
        }
        let scale = Float(1.0 / Double(max(batches.count, 1)))
        let averaged = accum.mapValues { $0 * scale }
        eval(Array(averaged.values))
        return averaged
    }
}

extension PVTuner {
    static func runDiscreteGroup(
        _ model: any LanguageModel, blocks: Set<Int>, train: [PVTeacherBatch], val: [PVTeacherBatch], stepSeed: Int, config: PVTuneConfig
    ) throws -> Int {
        guard !train.isEmpty, !config.discreteFractions.isEmpty else { return 0 }
        let root = model as Module
        let targets = collectTargets(root, blocks: blocks, maxSwitchExperts: config.maxContinuousSwitchExperts)
        let chunks = chunkTargets(targets, budget: config.maxActiveElements)
        let minBlock = blocks.min() ?? 0
        let bCount = min(config.batchesPerDiscreteStep, train.count)
        let slice = (0 ..< bCount).map { train[(stepSeed * bCount + $0) % train.count] }
        let gateBatches = val.isEmpty ? slice : val
        var totalFlipped = 0
        for chunk in chunks {
            try Task.checkCancellation()
            let grads = try denseGradients(model, chunk: chunk, minBlock: minBlock, batches: slice)
            let baseTrainKL = meanKL(model, Array(slice.prefix(min(4, slice.count))))
            let baseValKL = meanKL(model, gateBatches)
            let unpacked = Dictionary(uniqueKeysWithValues: chunk.map { ($0.path, PVMACaQGrid.unpackCodes($0.packedWeight, spec: $0.spec)) })
            eval(Array(unpacked.values))
            var accepted = false
            for fraction in config.discreteFractions {
                var candidateCount = 0
                var updates: [(String, MLXArray)] = []
                for t in chunk {
                    guard let g = grads[t.path], let codes = unpacked[t.path] else { continue }
                    let step = PVMACaQGrid.discreteStep(codes: codes, scales: t.scales, grad: g, spec: t.spec, fraction: fraction)
                    updates.append(("\(t.path).weight", PVMACaQGrid.packCodes(step.codes, spec: t.spec)))
                    candidateCount += step.updated
                }
                guard candidateCount > 0 else { continue }
                root.update(parameters: ModuleParameters.unflattened(updates))
                eval(root)
                let candValKL = meanKL(model, gateBatches)
                let candTrainKL = meanKL(model, Array(slice.prefix(min(4, slice.count))))
                if candValKL.isFinite && candTrainKL.isFinite && candValKL < baseValKL - 1e-7 && candTrainKL < baseTrainKL - 1e-7 {
                    totalFlipped += candidateCount
                    accepted = true
                    break
                }
            }
            if !accepted {
                root.update(parameters: ModuleParameters.unflattened(chunk.map { ("\($0.path).weight", $0.packedWeight) }))
                eval(root)
                root.invalidateCompiledTraces()
            }
        }
        totalFlipped += try runQTIPDiscreteGroup(model, blocks: blocks, train: train, val: val, stepSeed: stepSeed, config: config)
        return totalFlipped
    }

    private static func unfreezeContinuousTarget(
        path: String,
        mod: Module,
        maxSwitchExperts: Int,
        nf4Swaps: inout [(String, PVScaleLinear)],
        qtipSwaps: inout [(String, PVQTIPLinear)]
    ) throws {
        if try unfreezeMACaQContinuous(mod: mod, maxSwitchExperts: maxSwitchExperts) {
            return
        } else if let qt = mod as? QTIPLinear {
            qtipSwaps.append((path, PVQTIPLinear(from: qt)))
        } else if let nf4 = mod as? NF4Linear {
            nf4Swaps.append((path, PVScaleLinear(from: nf4)))
        } else if let ql = mod as? QuantizedLinear, ql.globalScale == nil, ql.mode == .affine {
            try ql.unfreeze(keys: ["scales", "biases"], strict: false)
        } else if let qs = mod as? QuantizedSwitchLinear, qs.mode == .affine {
            let nExp = qs.parameters().flattened().first(where: { $0.0 == "weight" })?.1.dim(0) ?? 0
            if nExp > 0 && nExp <= maxSwitchExperts { try qs.unfreeze(keys: ["scales", "biases"], strict: false) }
        } else if mod is RMSNorm || mod is LayerNorm || path.hasSuffix("norm") {
            mod.unfreeze()
        }
    }

    static func prepareContinuousUnfreeze(
        _ root: Module, blocks: Set<Int>, maxSwitchExperts: Int
    ) throws -> (nf4: [(String, PVScaleLinear)], qtip: [(String, PVQTIPLinear)]) {
        root.freeze()
        var nf4Swaps: [(String, PVScaleLinear)] = []
        var qtipSwaps: [(String, PVQTIPLinear)] = []
        for (path, mod) in root.namedModules() {
            guard let idx = textBlockIndex(path), blocks.contains(idx), !path.contains("router") else { continue }
            try unfreezeContinuousTarget(
                path: path, mod: mod, maxSwitchExperts: maxSwitchExperts,
                nf4Swaps: &nf4Swaps, qtipSwaps: &qtipSwaps)
        }
        for (path, wrapper) in nf4Swaps { try swapModule(path, with: wrapper, in: root) }
        for (path, wrapper) in qtipSwaps { try swapModule(path, with: wrapper, in: root) }
        return (nf4Swaps, qtipSwaps)
    }

    static func runContinuousGroup(
        _ model: any LanguageModel, blocks: Set<Int>, train: [PVTeacherBatch], val: [PVTeacherBatch], stepSeed: Int, config: PVTuneConfig
    ) throws {
        guard config.continuousStepsPerGroup > 0, !train.isEmpty else { return }
        let root = model as Module
        let swaps = try prepareContinuousUnfreeze(root, blocks: blocks, maxSwitchExperts: config.maxContinuousSwitchExperts)
        defer {
            for (path, wrapper) in swaps.nf4 { try? swapModule(path, with: wrapper.packedLayer(), in: root) }
            for (path, wrapper) in swaps.qtip { try? swapModule(path, with: wrapper.packedLayer(), in: root) }
            root.freeze()
            root.train(false)
        }
        let flatInit = root.trainableParameters().flattened()
        guard !flatInit.isEmpty else { return }
        let origDTypes = Dictionary(uniqueKeysWithValues: flatInit.map { ($0.0, $0.1.dtype) })
        let saved = Dictionary(uniqueKeysWithValues: flatInit.map { ($0.0, ($0.1 + MLXArray(0, dtype: $0.1.dtype)).asType($0.1.dtype)) })
        eval(Array(saved.values))
        var accepted = false
        defer {
            if !accepted {
                root.update(parameters: ModuleParameters.unflattened(Array(saved)))
                eval(root)
                root.invalidateCompiledTraces()
            }
        }
        let evalTrain = Array(train.prefix(min(4, train.count)))
        let gateVal = val.isEmpty ? evalTrain : val
        let (beforeTrainKL, beforeValKL) = (meanKL(model, evalTrain), meanKL(model, gateVal))
        var master = Dictionary(uniqueKeysWithValues: flatInit.map { ($0.0, $0.1.asType(.float32)) })
        var mState = Dictionary(uniqueKeysWithValues: flatInit.map { ($0.0, MLXArray.zeros($0.1.shape, dtype: .float32)) })
        var vState = Dictionary(uniqueKeysWithValues: flatInit.map { ($0.0, MLXArray.zeros($0.1.shape, dtype: .float32)) })
        eval(Array(master.values) + Array(mState.values) + Array(vState.values))
        setTrainingSuffix(root, from: blocks.min() ?? 0)
        let stepFn = valueAndGrad(model: root) { _, arrays in [klLossArrays(model, inputs: arrays[0], ids: arrays[1], tLogp: arrays[2])] }
        for s in 0 ..< config.continuousStepsPerGroup {
            try Task.checkCancellation()
            let batch = train[(stepSeed * config.continuousStepsPerGroup + s) % train.count]
            let (vals, grads) = stepFn(root, [MLXArray(batch.tokens)[.newAxis], MLXArray(batch.topIDs).reshaped(batch.seqLen, batch.topK), MLXArray(batch.topLogprobs).reshaped(batch.seqLen, batch.topK)])
            let lrCorr = config.continuousLR * sqrt(1.0 - pow(Float(0.999), Float(s + 1))) / (1.0 - pow(Float(0.9), Float(s + 1)))
            var castedUpdates: [(String, MLXArray)] = []
            for (k, gRaw) in grads.flattened() {
                guard let p = master[k], let mOld = mState[k], let vOld = vState[k], let dt = origDTypes[k] else { continue }
                let g = gRaw.asType(.float32)
                let (mNew, vNew) = (Float(0.9) * mOld + Float(0.1) * g, Float(0.999) * vOld + Float(0.001) * (g * g))
                let pNew = clampContinuousUpdate(
                    key: k, value: p - lrCorr * mNew / (sqrt(vNew) + Float(1e-8)))
                (master[k], mState[k], vState[k]) = (pNew, mNew, vNew)
                castedUpdates.append((k, pNew.asType(dt)))
            }
            root.update(parameters: ModuleParameters.unflattened(castedUpdates))
            eval(Array(master.values) + Array(mState.values) + Array(vState.values) + castedUpdates.map(\.1) + vals)
        }
        let (afterTrainKL, afterValKL) = (meanKL(model, evalTrain), meanKL(model, gateVal))
        if afterValKL.isFinite && afterTrainKL.isFinite && afterValKL <= beforeValKL && afterTrainKL <= beforeTrainKL {
            accepted = true
        }
    }
}
