import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension MACaQRefiner {
    package static func refineModel(
        _ student: any LanguageModel,
        calibrationTokenBatches: [[Int32]],
        config: MACaQConfig = MACaQConfig(),
        teacherWeight: (String) throws -> MLXArray?,
        onStep: ((String) -> Void)? = nil
    ) throws -> MACaQSummary {
        let startClock = ContinuousClock.now
        let root = student as Module
        root.train(false)
        let blockGroups = groupedBlocks(
            in: root, stepSize: config.blockGroupSize, maxHessianBytes: config.maxHessianBytes)
        let calTokens = calibrationTokenBatches.reduce(0) { $0 + $1.count }
        var (refinedCount, totalFlips) = (0, 0)
        var ratioAccum: Float = 0
        for (gIdx, group) in blockGroups.enumerated() {
            try Task.checkCancellation()
            let targets = collectRefinableTargets(root, blocks: Set(group))
            guard !targets.linears.isEmpty || !targets.switches.isEmpty else { continue }
            let captured = try captureBlockHessians(
                model: student, root: root,
                targets: BlockTargets(linears: targets.linears, switches: targets.switches),
                batches: calibrationTokenBatches, maxSwitchRows: config.maxSwitchSamples)
            for item in targets.linears {
                guard let rawH = captured.linear[item.groupKey],
                      let targetW = try teacherWeight(item.path) else { continue }
                if let res = try refineLinearTarget(item, targetW: targetW, rawH: rawH, config: config, root: root) {
                    refinedCount += 1
                    totalFlips += res.flips
                    ratioAccum += res.lossBefore > 0 ? (res.lossAfter / res.lossBefore) : 1.0
                }
            }
            for item in targets.switches {
                guard let resBox = captured.switches[item.path],
                      let targetW = try teacherWeight(item.path) else { continue }
                let rWeight = routerWeight(in: root, for: item.path)
                if let res = try refineSwitchTarget(item, targetW: targetW, reservoir: resBox, routerW: rWeight, config: config, root: root) {
                    refinedCount += 1
                    totalFlips += res.flips
                    ratioAccum += res.ratio
                }
            }
            Memory.clearCache()
            onStep?("macaq-refine group \(gIdx + 1)/\(blockGroups.count) layers=\(refinedCount) flips=\(totalFlips)")
        }
        MLXInferenceEngine.settle(root)
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let meanRatio = refinedCount > 0 ? (ratioAccum / Float(refinedCount)) : 1.0
        return MACaQSummary(
            refinedLayers: refinedCount, totalFlips: totalFlips,
            calibrationTokens: calTokens, meanLossRatio: meanRatio, wallSeconds: secs)
    }

    private static func collectRefinableTargets(
        _ root: Module, blocks: Set<Int>
    ) -> (linears: [LinearEntry], switches: [SwitchEntry]) {
        var linears: [LinearEntry] = []
        var switches: [SwitchEntry] = []
        for (path, mod) in root.namedModules() {
            guard let bIdx = PVTuner.textBlockIndex(path), blocks.contains(bIdx),
                  !path.contains("router"), !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a")
            else { continue }
            if let nf4 = mod as? NF4Linear {
                let (outF, inF) = nf4.shape
                let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: inF, sharedMoERHT: false)
                linears.append(LinearEntry(path: path, groupKey: gKey, module: nf4, outFeatures: outF, inFeatures: inF))
            } else if let ql = mod as? QuantizedLinear, !(mod is MACaQQuantizedLinear), ql.globalScale == nil,
                      ([2, 3, 4].contains(ql.bits) && ql.mode == .affine) || (ql.bits == 4 && ql.mode == .mxfp4) {
                let outF = ql.weight.dim(0)
                let inF = (ql.weight.dim(1) * 32) / ql.bits
                let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: inF, sharedMoERHT: false)
                linears.append(LinearEntry(path: path, groupKey: gKey, module: ql, outFeatures: outF, inFeatures: inF))
            } else if let qsw = mod as? QuantizedSwitchLinear,
                      ([2, 3, 4].contains(qsw.bits) && qsw.mode == .affine) || (qsw.bits == 4 && qsw.mode == .mxfp4) {
                let p = Dictionary(uniqueKeysWithValues: qsw.parameters().flattened())
                if let w = p["weight"], w.ndim == 3 {
                    let inF = (w.dim(2) * 32) / qsw.bits
                    switches.append(SwitchEntry(
                        path: path, routerPath: path, module: qsw,
                        numExperts: w.dim(0), outFeatures: w.dim(1), inFeatures: inF))
                }
            }
        }
        return (linears, switches)
    }

    private static func refineLinearTarget(
        _ item: LinearEntry,
        targetW: MLXArray,
        rawH: MLXArray,
        config: MACaQConfig,
        root: Module
    ) throws -> MACaQFlipResult? {
        guard targetW.ndim == 2, targetW.shape == [item.outFeatures, item.inFeatures] else { return nil }
        let frame = MACaQMath.transformFrame(
            weight: targetW, hessian: rawH, preScale: nil, blockSize: nil, ridge: config.ridge)
        if let nf4 = item.module as? NF4Linear {
            let spec = PVCodeGrid.Spec(nf4GroupSize: nf4.groupSize)
            var (grid, _) = MACaQMath.makeGridState(
                packedWeight: nf4.weight, scales: nf4.scales, biases: nil, spec: spec)
            let res = MACaQMath.flipRefine(
                state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: config.maxFlipIterations)
            guard res.flips > 0, res.lossAfter.isFinite, res.lossAfter <= res.lossBefore * 1.00001 + 1e-10 else {
                return res
            }
            let packed = PVCodeGrid.packCodes(grid.codes.asType(.uint8), bits: 4)
            root.update(parameters: ModuleParameters.unflattened([("\(item.path).weight", packed)]))
            eval(packed)
            return res
        }
        guard let ql = item.module as? QuantizedLinear else { return nil }
        let spec = PVCodeGrid.Spec(groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
        var grid: MACaQGridState
        var newScales = ql.scales
        if ql.mode == .mxfp4, config.screenMXFP4 {
            let scr = MACaQMath.screenMXFP4Exponents(weight: frame.weight, hessianDiag: diagonal(frame.hessian))
            grid = MACaQGridState(
                codes: scr.sidx, scaleElem: scr.scaleElem, biasElem: nil,
                levelTable: MLXArray(MACaQMath.mxfp4SignedLevels), maxLevel: 14)
            newScales = scr.e8m0
        } else {
            grid = MACaQMath.makeGridState(
                packedWeight: ql.weight, scales: ql.scales, biases: ql.biases, spec: spec).state
        }
        let res = MACaQMath.flipRefine(
            state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: config.maxFlipIterations)
        guard res.lossAfter.isFinite, res.lossAfter <= res.lossBefore * 1.00001 + 1e-10 else { return nil }
        if ql.mode == .mxfp4 {
            let packed = MACaQMath.packMXFP4(sidx: grid.codes, e8m0: newScales)
            root.update(parameters: ModuleParameters.unflattened([
                ("\(item.path).weight", packed.weight),
                ("\(item.path).scales", packed.scales),
            ]))
            eval(packed.weight, packed.scales)
        } else if res.flips > 0 {
            let packed = PVCodeGrid.packCodes(grid.codes.asType(.uint8), bits: ql.bits)
            root.update(parameters: ModuleParameters.unflattened([("\(item.path).weight", packed)]))
            eval(packed)
        }
        return res
    }
}

extension MACaQRefiner {
    private static func refineSwitchTarget(
        _ item: SwitchEntry,
        targetW: MLXArray,
        reservoir: MACaQSwitchReservoir,
        routerW: MLXArray?,
        config: MACaQConfig,
        root: Module
    ) throws -> (flips: Int, ratio: Float)? {
        guard let qsw = item.module as? QuantizedSwitchLinear,
              targetW.ndim == 3,
              targetW.shape == [item.numExperts, item.outFeatures, item.inFeatures]
        else { return nil }
        let params = Dictionary(uniqueKeysWithValues: qsw.parameters().flattened())
        guard let wAll = params["weight"], let sAll = params["scales"] else { return nil }
        let bAll = params["biases"]
        let spec = PVCodeGrid.Spec(groupSize: qsw.groupSize, bits: qsw.bits, mode: qsw.mode)
        var (wSlices, sSlices) = ([MLXArray](), [MLXArray]())
        var (flipsSum, lossBeforeSum, lossAfterSum) = (0, Float(0), Float(0))
        for e in 0 ..< item.numExperts {
            guard let rawH = reservoir.expertHessian(expert: e, minTokens: config.minExpertTokens, routerWeight: routerW) else {
                wSlices.append(wAll[e])
                sSlices.append(sAll[e])
                continue
            }
            let frame = MACaQMath.transformFrame(
                weight: targetW[e], hessian: rawH, preScale: nil, blockSize: nil, ridge: config.ridge)
            let packed = MACaQPackedWeights(weight: wAll[e], scales: sAll[e], biases: bAll.map { $0[e] })
            let step = refineSingleSwitchExpert(packed: packed, frame: frame, spec: spec, config: config)
            lossBeforeSum += step.res.lossBefore
            lossAfterSum += step.accepted ? step.res.lossAfter : step.res.lossBefore
            if step.accepted { flipsSum += step.res.flips }
            wSlices.append(step.weight)
            sSlices.append(step.scales)
        }
        let newW = stacked(wSlices, axis: 0)
        var updates: [(String, MLXArray)] = [("\(item.path).weight", newW)]
        if qsw.mode == .mxfp4 { updates.append(("\(item.path).scales", stacked(sSlices, axis: 0))) }
        root.update(parameters: ModuleParameters.unflattened(updates))
        eval(updates.map(\.1))
        let ratio = lossBeforeSum > 0 ? (lossAfterSum / lossBeforeSum) : 1.0
        return (flipsSum, ratio)
    }

    private static func refineSingleSwitchExpert(
        packed: MACaQPackedWeights,
        frame: (weight: MLXArray, hessian: MLXArray),
        spec: PVCodeGrid.Spec,
        config: MACaQConfig
    ) -> (weight: MLXArray, scales: MLXArray, res: MACaQFlipResult, accepted: Bool) {
        var grid: MACaQGridState
        var sExpert = packed.scales
        let isMXFP4 = spec.encoding == .affine(.mxfp4)
        if isMXFP4, config.screenMXFP4 {
            let scr = MACaQMath.screenMXFP4Exponents(weight: frame.weight, hessianDiag: diagonal(frame.hessian))
            grid = MACaQGridState(
                codes: scr.sidx, scaleElem: scr.scaleElem, biasElem: nil,
                levelTable: MLXArray(MACaQMath.mxfp4SignedLevels), maxLevel: 14)
            sExpert = scr.e8m0
        } else {
            grid = MACaQMath.makeGridState(
                packedWeight: packed.weight, scales: packed.scales, biases: packed.biases, spec: spec).state
        }
        let res = MACaQMath.flipRefine(
            state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: config.maxFlipIterations)
        guard res.lossAfter.isFinite, res.lossAfter <= res.lossBefore * 1.00001 + 1e-10 else {
            return (packed.weight, packed.scales, res, false)
        }
        if isMXFP4 {
            let pk = MACaQMath.packMXFP4(sidx: grid.codes, e8m0: sExpert)
            return (pk.weight, pk.scales, res, true)
        }
        return (PVCodeGrid.packCodes(grid.codes.asType(.uint8), bits: spec.bits), sExpert, res, true)
    }

    package static func quantizeModel(
        _ model: any LanguageModel,
        targetMode: MACaQTargetMode,
        calibrationTokenBatches: [[Int32]],
        config: MACaQConfig = MACaQConfig(),
        onStep: ((String) -> Void)? = nil
    ) throws -> (summary: MACaQSummary, manifest: [String: MACaQLayerSpec]) {
        let startClock = ContinuousClock.now
        let root = model as Module
        root.train(false)
        let blockGroups = groupedBlocks(
            in: root, stepSize: config.blockGroupSize, maxHessianBytes: config.maxHessianBytes)
        let calTokens = calibrationTokenBatches.reduce(0) { $0 + $1.count }
        var manifest: [String: MACaQLayerSpec] = [:]
        var (quantCount, totalFlips, ratioAccum) = (0, 0, Float(0))
        for (gIdx, group) in blockGroups.enumerated() {
            try Task.checkCancellation()
            let targets = collectQuantizableTargets(root, blocks: Set(group), mode: targetMode, blockSize: config.hadamardBlock)
            guard !targets.linears.isEmpty || !targets.switches.isEmpty else { continue }
            let captured = try captureBlockHessians(
                model: model, root: root,
                targets: BlockTargets(linears: targets.linears, switches: targets.switches),
                batches: calibrationTokenBatches, maxSwitchRows: config.maxSwitchSamples)
            for item in targets.linears {
                guard let rawH = captured.linear[item.groupKey] else { continue }
                let built = try buildMACaQLinear(item, rawH: rawH, mode: targetMode, config: config)
                try PVTuner.swapModule(item.path, with: built.layer, in: root)
                manifest[item.path] = built.spec
                quantCount += 1
                totalFlips += built.result.flips
                ratioAccum += built.result.lossBefore > 0 ? (built.result.lossAfter / built.result.lossBefore) : 1.0
            }
            for item in targets.switches {
                guard let resBox = captured.switches[item.path] else { continue }
                let rWeight = routerWeight(in: root, for: item.path)
                if let built = try MACaQSwitchBuilder.build(
                    item, reservoir: resBox, routerW: rWeight, mode: targetMode, config: config
                ) {
                    try PVTuner.swapModule(item.path, with: built.layer, in: root)
                    manifest[item.path] = built.spec
                    quantCount += 1
                    totalFlips += built.flips
                    ratioAccum += built.ratio
                }
            }
            Memory.clearCache()
            onStep?("macaq group \(gIdx + 1)/\(blockGroups.count) quantized=\(quantCount) flips=\(totalFlips)")
        }
        MLXInferenceEngine.settle(root)
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let meanRatio = quantCount > 0 ? (ratioAccum / Float(quantCount)) : 1.0
        let summary = MACaQSummary(
            refinedLayers: quantCount, totalFlips: totalFlips,
            calibrationTokens: calTokens, meanLossRatio: meanRatio, wallSeconds: secs)
        return (summary, manifest)
    }
}

extension MACaQRefiner {
    private static func collectQuantizableTargets(
        _ root: Module, blocks: Set<Int>, mode: MACaQTargetMode, blockSize: Int
    ) -> (linears: [LinearEntry], switches: [SwitchEntry]) {
        var (linears, switches) = ([LinearEntry](), [SwitchEntry]())
        let align = max(blockSize, mode.groupSize)
        for (path, mod) in root.namedModules() {
            guard let bIdx = PVTuner.textBlockIndex(path), blocks.contains(bIdx),
                  !path.contains("router"), !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a")
            else { continue }
            if let ql = mod as? QuantizedLinear, !(mod is MACaQQuantizedLinear), ql.globalScale == nil {
                let outF = ql.weight.dim(0)
                let inF = (ql.weight.dim(1) * 32) / ql.bits
                guard inF.isMultiple(of: align) else { continue }
                let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: inF, sharedMoERHT: false)
                linears.append(LinearEntry(path: path, groupKey: gKey, module: ql, outFeatures: outF, inFeatures: inF))
            } else if type(of: mod) == Linear.self, let lin = mod as? Linear, lin.shape.1.isMultiple(of: align) {
                let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: lin.shape.1, sharedMoERHT: false)
                linears.append(LinearEntry(path: path, groupKey: gKey, module: lin, outFeatures: lin.shape.0, inFeatures: lin.shape.1))
            } else if let sw = mod as? SwitchLinear, !(mod is MACaQSwitchLinear) {
                let p = Dictionary(uniqueKeysWithValues: sw.parameters().flattened())
                guard let w = p["weight"], w.ndim == 3 else { continue }
                let inF: Int
                if let qsw = mod as? QuantizedSwitchLinear {
                    inF = (w.dim(2) * 32) / qsw.bits
                } else {
                    inF = w.dim(2)
                }
                guard inF.isMultiple(of: align) else { continue }
                switches.append(SwitchEntry(
                    path: path, routerPath: path, module: sw,
                    numExperts: w.dim(0), outFeatures: w.dim(1), inFeatures: inF))
            }
        }
        return (linears, switches)
    }

    private static func buildMACaQLinear(
        _ item: LinearEntry,
        rawH: MLXArray,
        mode: MACaQTargetMode,
        config: MACaQConfig
    ) throws -> (layer: MACaQQuantizedLinear, spec: MACaQLayerSpec, result: MACaQFlipResult) {
        let (dense, bias) = QTIPQuantizer.extractDenseWeight(item.module)
        let saveDType: DType = dense.dtype == .float16 ? .float16 : .bfloat16
        let pre = MACaQMath.balancePreScale(
            weight: dense, meanSquareActivation: diagonal(rawH), alpha: config.alpha)
        let preSaved = pre.asType(saveDType)
        eval(preSaved)
        let frame = MACaQMath.transformFrame(
            weight: dense, hessian: rawH, preScale: preSaved, blockSize: config.hadamardBlock, ridge: config.ridge)
        let packedW: MLXArray
        let packedS: MLXArray
        let packedB: MLXArray?
        let res: MACaQFlipResult
        switch mode {
        case .mxfp4:
            let scr = MACaQMath.screenMXFP4Exponents(weight: frame.weight, hessianDiag: diagonal(frame.hessian))
            var grid = MACaQGridState(
                codes: scr.sidx, scaleElem: scr.scaleElem, biasElem: nil,
                levelTable: MLXArray(MACaQMath.mxfp4SignedLevels), maxLevel: 14)
            res = MACaQMath.flipRefine(
                state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: config.maxFlipIterations)
            let pk = MACaQMath.packMXFP4(sidx: grid.codes, e8m0: scr.e8m0)
            (packedW, packedS, packedB) = (pk.weight, pk.scales, nil)
        case .affine(let bits, let groupSize):
            let (wqInit, sInit, bInit) = MLX.quantized(
                frame.weight.asType(saveDType), groupSize: groupSize, bits: bits, mode: .affine)
            let pSpec = PVCodeGrid.Spec(groupSize: groupSize, bits: bits, mode: .affine)
            var grid = MACaQMath.makeGridState(packedWeight: wqInit, scales: sInit, biases: bInit, spec: pSpec).state
            res = MACaQMath.flipRefine(
                state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: config.maxFlipIterations)
            let useRefined = res.lossAfter.isFinite && res.lossAfter <= res.lossBefore * 1.00001 + 1e-10
            packedW = useRefined ? PVCodeGrid.packCodes(grid.codes.asType(.uint8), bits: bits) : wqInit
            (packedS, packedB) = (sInit, bInit)
        }
        var spec = MACaQLayerSpec(
            bits: mode.bits, groupSize: mode.groupSize, mode: mode.quantizationMode,
            hadamardBlock: config.hadamardBlock)
        spec.scaleDType = "\(packedS.dtype)"
        spec.hasBias = bias != nil
        let layer = MACaQQuantizedLinear(
            packed: MACaQPackedWeights(weight: packedW, scales: packedS, biases: packedB),
            bias: bias, preScale: preSaved, spec: spec)
        eval(layer)
        return (layer, spec, res)
    }
}

private enum MACaQSwitchBuilder {
    static func build(
        _ item: MACaQRefiner.SwitchEntry,
        reservoir: MACaQSwitchReservoir,
        routerW: MLXArray?,
        mode: MACaQTargetMode,
        config: MACaQConfig
    ) throws -> (layer: MACaQSwitchLinear, spec: MACaQLayerSpec, flips: Int, ratio: Float)? {
        let p = Dictionary(uniqueKeysWithValues: item.module.parameters().flattened())
        guard let rawW = p["weight"] else { return nil }
        let denseAll: MLXArray
        if let qsw = item.module as? QuantizedSwitchLinear, let s = p["scales"] {
            denseAll = dequantized(rawW, scales: s, biases: p["biases"], groupSize: qsw.groupSize, bits: qsw.bits, mode: qsw.mode)
        } else {
            denseAll = rawW
        }
        let saveDType: DType = denseAll.dtype == .float16 ? .float16 : .bfloat16
        var (wList, sList, bList, preList) = ([MLXArray](), [MLXArray](), [MLXArray](), [MLXArray]())
        var (flipsSum, beforeSum, afterSum) = (0, Float(0), Float(0))
        for e in 0 ..< item.numExperts {
            let hExp = reservoir.expertHessian(expert: e, minTokens: config.minExpertTokens, routerWeight: routerW)
                ?? eye(item.inFeatures, dtype: .float32)
            let wExp = denseAll[e]
            let pre = MACaQMath.balancePreScale(
                weight: wExp, meanSquareActivation: diagonal(hExp), alpha: config.alpha).asType(saveDType)
            eval(pre)
            preList.append(pre)
            let frame = MACaQMath.transformFrame(
                weight: wExp, hessian: hExp, preScale: pre, blockSize: config.hadamardBlock, ridge: config.ridge)
            let enc = quantizeExpertMatrix(frame: frame, mode: mode, saveDType: saveDType, maxIters: config.maxFlipIterations)
            wList.append(enc.w)
            sList.append(enc.s)
            if let b = enc.b { bList.append(b) }
            flipsSum += enc.res.flips
            beforeSum += enc.res.lossBefore
            afterSum += enc.res.lossAfter
        }
        let bias = p["bias"]
        var spec = MACaQLayerSpec(
            bits: mode.bits, groupSize: mode.groupSize, mode: mode.quantizationMode,
            hadamardBlock: config.hadamardBlock)
        spec.perExpertPreScale = true
        spec.scaleDType = sList.first.map { "\($0.dtype)" }
        spec.hasBias = bias != nil
        let packed = MACaQPackedWeights(
            weight: stacked(wList, axis: 0),
            scales: stacked(sList, axis: 0),
            biases: bList.isEmpty ? nil : stacked(bList, axis: 0))
        let layer = MACaQSwitchLinear(
            geometry: MACaQSwitchGeometry(experts: item.numExperts, outputs: item.outFeatures, inputs: item.inFeatures),
            packed: packed,
            bias: bias,
            preScale: stacked(preList, axis: 0),
            spec: spec)
        eval(layer)
        return (layer, spec, flipsSum, beforeSum > 0 ? (afterSum / beforeSum) : 1.0)
    }

    private static func quantizeExpertMatrix(
        frame: (weight: MLXArray, hessian: MLXArray),
        mode: MACaQTargetMode,
        saveDType: DType,
        maxIters: Int
    ) -> (w: MLXArray, s: MLXArray, b: MLXArray?, res: MACaQFlipResult) {
        switch mode {
        case .mxfp4:
            let scr = MACaQMath.screenMXFP4Exponents(weight: frame.weight, hessianDiag: diagonal(frame.hessian))
            var grid = MACaQGridState(
                codes: scr.sidx, scaleElem: scr.scaleElem, biasElem: nil,
                levelTable: MLXArray(MACaQMath.mxfp4SignedLevels), maxLevel: 14)
            let res = MACaQMath.flipRefine(state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: maxIters)
            let pk = MACaQMath.packMXFP4(sidx: grid.codes, e8m0: scr.e8m0)
            return (pk.weight, pk.scales, nil, res)
        case .affine(let bits, let groupSize):
            let (wq, sc, bi) = MLX.quantized(frame.weight.asType(saveDType), groupSize: groupSize, bits: bits, mode: .affine)
            let pSpec = PVCodeGrid.Spec(groupSize: groupSize, bits: bits, mode: .affine)
            var grid = MACaQMath.makeGridState(packedWeight: wq, scales: sc, biases: bi, spec: pSpec).state
            let res = MACaQMath.flipRefine(state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: maxIters)
            let ok = res.lossAfter.isFinite && res.lossAfter <= res.lossBefore * 1.00001 + 1e-10
            let finalW = ok ? PVCodeGrid.packCodes(grid.codes.asType(.uint8), bits: bits) : wq
            return (finalW, sc, bi, res)
        }
    }
}
