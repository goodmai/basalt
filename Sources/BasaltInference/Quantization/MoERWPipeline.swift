import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum MoERWPipeline {
    struct RouterSite {
        var block: Int
        var path: String
        var module: Module
        var numExperts: Int
        var inFeatures: Int
        var expertScale: MLXArray? = nil
    }

    struct CaptureBundle {
        var linears: [String: MLXArray]
        var switches: [String: MoERWSwitchReservoir]
    }

    struct ExecutionContext {
        let model: any LanguageModel
        let root: Module
        let routers: [Int: RouterSite]
        let batches: [[Int32]]
        let estimator: MoERouterWeightedHessian
        let config: MACaQConfig
    }

    static let routerSuffixes = [".mlp.gate", ".mlp.router", ".feed_forward.gate", ".feed_forward.router", ".router.proj"]

    static func discoverRouters(in root: Module) -> [Int: RouterSite] {
        var out: [Int: RouterSite] = [:]
        let modules = root.namedModules()
        let byPath = Dictionary(modules, uniquingKeysWith: { first, _ in first })
        for (path, mod) in modules {
            guard let bIdx = PVTuner.textBlockIndex(path), out[bIdx] == nil, routerSuffixes.contains(where: path.hasSuffix) else { continue }
            let dims: (out: Int, inp: Int)? = if let nf4 = mod as? NF4Linear {
                nf4.shape
            } else if let ql = mod as? QuantizedLinear {
                (ql.weight.dim(0), (ql.weight.dim(1) * 32) / max(1, ql.bits))
            } else if let lin = mod as? Linear {
                lin.shape
            } else {
                nil
            }
            if let dims, dims.out >= 2, dims.inp >= 16 {
                let parent = path.hasSuffix(".proj") ? byPath[String(path.dropLast(".proj".count))] : nil
                let scale = parent.flatMap { Dictionary(uniqueKeysWithValues: $0.parameters().flattened())["per_expert_scale"] }
                out[bIdx] = RouterSite(
                    block: bIdx, path: path, module: mod, numExperts: dims.out, inFeatures: dims.inp,
                    expertScale: scale?.size == dims.out ? scale?.asType(.float32) : nil)
            }
        }
        if out.isEmpty, modules.contains(where: { PVTuner.textBlockIndex($0.0) != nil && $0.1 is SwitchLinear }) {
            BasaltLogger(module: "MoERW").error("moe-rw found switch layers but no router (looked for \(routerSuffixes.joined(separator: ", "))); expert Hessians fall back to unweighted tokens")
        }
        return out
    }

    static func collectCandidateTargets(
        _ root: Module, blocks: Set<Int>
    ) -> (linears: [MACaQRefiner.LinearEntry], switches: [MACaQRefiner.SwitchEntry]) {
        var (linears, switches) = ([MACaQRefiner.LinearEntry](), [MACaQRefiner.SwitchEntry]())
        for (path, mod) in root.namedModules() {
            guard let bIdx = PVTuner.textBlockIndex(path), blocks.contains(bIdx),
                  !path.contains("router"), !path.hasSuffix(".mlp.gate"), !path.hasSuffix(".feed_forward.gate"),
                  !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a")
            else { continue }
            if let ql = mod as? QuantizedLinear, !(mod is MACaQQuantizedLinear), ql.globalScale == nil {
                let inF = (ql.weight.dim(1) * 32) / ql.bits
                guard inF >= 32, inF.isMultiple(of: 32) else { continue }
                let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: inF, sharedMoERHT: false)
                linears.append(.init(path: path, groupKey: gKey, module: ql, outFeatures: ql.weight.dim(0), inFeatures: inF))
            } else if type(of: mod) == Linear.self, let lin = mod as? Linear, lin.shape.1 >= 32, lin.shape.1.isMultiple(of: 32) {
                let gKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: lin.shape.1, sharedMoERHT: false)
                linears.append(.init(path: path, groupKey: gKey, module: lin, outFeatures: lin.shape.0, inFeatures: lin.shape.1))
            } else if let sw = mod as? SwitchLinear, !(mod is MACaQSwitchLinear) {
                let p = Dictionary(uniqueKeysWithValues: sw.parameters().flattened())
                guard let w = p["weight"], w.ndim == 3 else { continue }
                let inF = (mod as? QuantizedSwitchLinear).map { (w.dim(2) * 32) / $0.bits } ?? w.dim(2)
                guard inF >= 32, inF.isMultiple(of: 32) else { continue }
                switches.append(.init(
                    path: path, routerPath: path, module: sw,
                    numExperts: w.dim(0), outFeatures: w.dim(1), inFeatures: inF))
            }
        }
        return (linears, switches)
    }

    static func captureMoEGroup(
        model: any LanguageModel,
        root: Module,
        targets: MACaQRefiner.BlockTargets,
        routers: [Int: RouterSite],
        batches: [[Int32]],
        maxSwitchRows: Int
    ) throws -> CaptureBundle {
        var linearBoxes: [String: MACaQHessianBox] = [:]
        var gateBoxes: [Int: MoERouterGateBox] = [:]
        var switchBoxes: [String: MoERWSwitchReservoir] = [:]
        var swapped: [(String, Module)] = []
        defer {
            for (path, orig) in swapped { try? PVTuner.swapModule(path, with: orig, in: root) }
        }
        let activeBlocks = Set(targets.switches.compactMap { PVTuner.textBlockIndex($0.path) })
        for bIdx in activeBlocks.sorted() {
            guard let site = routers[bIdx] else { continue }
            let gBox = MoERouterGateBox(numExperts: site.numExperts, inFeatures: site.inFeatures, expertScale: site.expertScale)
            gateBoxes[bIdx] = gBox
            try PVTuner.swapModule(site.path, with: MoERouterGateTap(wrapping: site.module, box: gBox), in: root)
            swapped.append((site.path, site.module))
        }
        var tappedGroups: Set<String> = []
        for item in targets.linears where tappedGroups.insert(item.groupKey).inserted {
            let box = MACaQHessianBox(inFeatures: item.inFeatures)
            linearBoxes[item.groupKey] = box
            try PVTuner.swapModule(item.path, with: MACaQLinearTap(wrapping: item.module, box: box), in: root)
            swapped.append((item.path, item.module))
        }
        for item in targets.switches {
            let gBox = PVTuner.textBlockIndex(item.path).flatMap { gateBoxes[$0] }
            let res = MoERWSwitchReservoir(
                inFeatures: item.inFeatures, numExperts: item.numExperts, maxRows: maxSwitchRows, gateBox: gBox)
            switchBoxes[item.path] = res
            try PVTuner.swapModule(item.path, with: MoERWSwitchTap(wrapping: item.module, reservoir: res), in: root)
            swapped.append((item.path, item.module))
        }
        for tokens in batches where !tokens.isEmpty {
            let out = model(LMInput.Text(tokens: MLXArray(tokens)[.newAxis]), cache: nil, state: nil)
            eval(out.logits)
        }
        var resolved: [String: MLXArray] = [:]
        for (k, box) in linearBoxes { if let h = box.meanHessian() { resolved[k] = h } }
        return CaptureBundle(linears: resolved, switches: switchBoxes)
    }
}

extension MoERWPipeline {
    static func measureGlobalSensitivities(
        model: any LanguageModel,
        root: Module,
        routers: [Int: RouterSite],
        batches: [[Int32]],
        targetBPW: Double
    ) throws -> [DeadZoneBitAllocator.ExpertSensitivity] {
        let allBlocks = Set(root.namedModules().compactMap { PVTuner.textBlockIndex($0.0) })
        let candidates = collectCandidateTargets(root, blocks: allBlocks)
        var gateBoxes: [Int: MoERouterGateBox] = [:]
        var actBoxes: [String: ActivationMeanSquareBox] = [:]
        var swapped: [(String, Module)] = []
        defer {
            for (path, orig) in swapped { try? PVTuner.swapModule(path, with: orig, in: root) }
        }
        for (bIdx, site) in routers.sorted(by: { $0.key < $1.key }) {
            let gBox = MoERouterGateBox(numExperts: site.numExperts, inFeatures: site.inFeatures, expertScale: site.expertScale)
            gateBoxes[bIdx] = gBox
            try PVTuner.swapModule(site.path, with: MoERouterGateTap(wrapping: site.module, box: gBox), in: root)
            swapped.append((site.path, site.module))
        }
        var tappedKeys: Set<String> = []
        for entry in candidates.linears where tappedKeys.insert(entry.groupKey).inserted {
            let box = ActivationMeanSquareBox()
            actBoxes[entry.groupKey] = box
            try PVTuner.swapModule(entry.path, with: SensitivityTraceTap(wrapping: entry.module, box: box), in: root)
            swapped.append((entry.path, entry.module))
        }
        for tokens in batches.prefix(4) where !tokens.isEmpty {
            let out = model(LMInput.Text(tokens: MLXArray(tokens)[.newAxis]), cache: nil, state: nil)
            eval(out.logits)
        }
        var items: [DeadZoneBitAllocator.ExpertSensitivity] = []
        for lin in candidates.linears {
            items.append(scoreLinearCandidate(lin, actBoxes: actBoxes, routers: routers, gateBoxes: gateBoxes, targetBPW: targetBPW))
        }
        for sw in candidates.switches {
            items.append(scoreSwitchCandidate(sw, routers: routers, gateBoxes: gateBoxes))
        }
        return items
    }

    private static func scoreLinearCandidate(
        _ item: MACaQRefiner.LinearEntry,
        actBoxes: [String: ActivationMeanSquareBox],
        routers: [Int: RouterSite],
        gateBoxes: [Int: MoERouterGateBox],
        targetBPW: Double
    ) -> DeadZoneBitAllocator.ExpertSensitivity {
        let (dense, _) = QTIPQuantizer.extractDenseWeight(item.module)
        let wf = dense.asType(.float32)
        let wEnergy = mean(wf * wf)
        eval(wEnergy)
        let xEnergy = actBoxes[item.groupKey]?.meanSquare ?? 1.0
        let role = RCOSensitivityAllocator.LayerRole.infer(from: item.path)
        let minB = (role.requiresSensitiveFloor && targetBPW >= 2.65) ? 3 : 2
        let block = PVTuner.textBlockIndex(item.path) ?? 0
        let cKL = probeCascadeKL(weight2D: wf, block: block, routers: routers, gateBoxes: gateBoxes)
        var sens = DeadZoneBitAllocator.ExpertSensitivity(
            id: item.path, weightCount: item.outFeatures * item.inFeatures,
            routerWeightedTrace: max(1e-10, Double(wEnergy.item(Float.self)) * xEnergy),
            cascadeRouterKL: Double(cKL), routingMass: role.requiresSensitiveFloor ? 1.35 : 1.0)
        sens.minBits = minB
        return sens
    }

    private static func scoreSwitchCandidate(
        _ item: MACaQRefiner.SwitchEntry,
        routers: [Int: RouterSite],
        gateBoxes: [Int: MoERouterGateBox]
    ) -> DeadZoneBitAllocator.ExpertSensitivity {
        let denseAll = extractSwitchDense(item.module)
        let block = PVTuner.textBlockIndex(item.path) ?? 0
        var rwEnergy = 0.0
        var dominantExpert = 0
        if let zList = gateBoxes[block]?.pooledLogits, !zList.isEmpty {
            let zAll = concatenated(zList, axis: 0)
            let probs = softmax(zAll, axis: -1, precise: true)
            let meanProbs = mean(probs, axis: 0)
            let sqProbs = mean(probs * probs, axis: 0)
            let wPerExp = mean(denseAll * denseAll, axes: [1, 2])
            eval(meanProbs, sqProbs, wPerExp)
            let pArr = meanProbs.asArray(Float.self)
            let p2Arr = sqProbs.asArray(Float.self)
            let wArr = wPerExp.asArray(Float.self)
            dominantExpert = pArr.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            for e in 0 ..< min(wArr.count, p2Arr.count) {
                rwEnergy += Double(p2Arr[e] * wArr[e])
            }
        } else {
            let m = mean(denseAll * denseAll)
            eval(m)
            rwEnergy = Double(m.item(Float.self))
        }
        let cKL = probeCascadeKL(weight2D: denseAll[dominantExpert], block: block, routers: routers, gateBoxes: gateBoxes)
        return DeadZoneBitAllocator.ExpertSensitivity(
            id: item.path, weightCount: item.numExperts * item.outFeatures * item.inFeatures,
            routerWeightedTrace: max(1e-10, rwEnergy), cascadeRouterKL: Double(cKL), routingMass: 1.0)
    }

    private static func probeCascadeKL(
        weight2D: MLXArray,
        block: Int,
        routers: [Int: RouterSite],
        gateBoxes: [Int: MoERouterGateBox]
    ) -> Float {
        guard let nextBlock = routers.keys.filter({ $0 > block }).min(),
              let nextSite = routers[nextBlock],
              let nextBox = gateBoxes[nextBlock],
              !nextBox.pooledInputs.isEmpty, !nextBox.pooledLogits.isEmpty,
              weight2D.dim(0) == nextSite.inFeatures, weight2D.dim(1).isMultiple(of: 32)
        else { return 0.0 }
        let nextX = concatenated(nextBox.pooledInputs, axis: 0)
        let teacherZ = concatenated(nextBox.pooledLogits, axis: 0)
        let (wq, sc, bi) = MLX.quantized(weight2D.asType(.bfloat16), groupSize: 32, bits: 2, mode: .affine)
        let deq = MLX.dequantized(wq, scales: sc, biases: bi, groupSize: 32, bits: 2, mode: .affine, dtype: .float32)
        let dw = deq - weight2D
        let probeIn: MLXArray = if weight2D.dim(1) == nextSite.inFeatures {
            nextX
        } else {
            MLXRandom.normal([nextX.dim(0), weight2D.dim(1)], scale: 0.25, key: MLXRandom.key(UInt64(56900 + block)))
        }
        let perturbed = nextX + matmul(probeIn, dw.T)
        let (rW, rB) = QTIPQuantizer.extractDenseWeight(nextSite.module)
        var studentZ = matmul(perturbed, rW.asType(.float32).T)
        if let rB { studentZ = studentZ + rB.asType(.float32) }
        return MoERouterWeightedHessian.cascadeRouterKL(teacherLogits: teacherZ, studentLogits: studentZ)
    }

    static func extractSwitchDense(_ module: Module) -> MLXArray {
        let p = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        guard let rawW = p["weight"] else { return MLXArray.zeros([1, 1, 32], dtype: .float32) }
        if let qsw = module as? QuantizedSwitchLinear, let s = p["scales"] {
            return dequantized(rawW, scales: s, biases: p["biases"], groupSize: qsw.groupSize, bits: qsw.bits, mode: qsw.mode, dtype: .float32)
        }
        return rawW.asType(.float32)
    }
}

extension MoERWPipeline {
    static func quantizeBudgetModel(
        _ model: any LanguageModel,
        batches: [[Int32]],
        targetBPW: Double,
        estimator: MoERouterWeightedHessian,
        config: MACaQConfig,
        onStep: ((String) -> Void)?
    ) throws -> (summary: MoERWConversionSummary, manifest: [String: MACaQLayerSpec]) {
        let startClock = ContinuousClock.now
        let root = model as Module
        root.train(false)
        let routers = discoverRouters(in: root)
        let sens = try measureGlobalSensitivities(
            model: model, root: root, routers: routers, batches: batches, targetBPW: targetBPW)
        let plan = DeadZoneBitAllocator(targetBPW: targetBPW, deadZoneEpsilon: 0.02).allocate(sens)
        let bitsByPath = Dictionary(uniqueKeysWithValues: plan.allocations.map { ($0.id, $0.bits) })
        let ctx = ExecutionContext(
            model: model, root: root, routers: routers, batches: batches, estimator: estimator, config: config)
        let encoded = try applyAllocatedMACaQ(
            ctx, modeForPath: { path, inF in
                let b = bitsByPath[path] ?? 3
                let g = inF.isMultiple(of: 64) ? 64 : 32
                return .affine(bits: b, groupSize: g)
            }, onStep: onStep)
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let summary = MoERWConversionSummary(
            plan: plan, quantizedLayers: encoded.count, totalFlips: encoded.flips,
            meanLossRatio: encoded.meanRatio,
            calibrationTokens: batches.reduce(0) { $0 + $1.count }, wallSeconds: secs)
        return (summary, encoded.manifest)
    }

    static func quantizeUniformMACaQ(
        _ model: any LanguageModel,
        targetMode: MACaQTargetMode,
        batches: [[Int32]],
        estimator: MoERouterWeightedHessian,
        config: MACaQConfig,
        onStep: ((String) -> Void)?
    ) throws -> (summary: MACaQSummary, manifest: [String: MACaQLayerSpec]) {
        let startClock = ContinuousClock.now
        let root = model as Module
        root.train(false)
        let routers = discoverRouters(in: root)
        let ctx = ExecutionContext(
            model: model, root: root, routers: routers, batches: batches, estimator: estimator, config: config)
        let encoded = try applyAllocatedMACaQ(
            ctx, modeForPath: { _, inF in
                switch targetMode {
                case .mxfp4: return .mxfp4
                case .affine(let b, let g): return .affine(bits: b, groupSize: inF.isMultiple(of: g) ? g : 32)
                }
            }, onStep: onStep)
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let sum = MACaQSummary(
            refinedLayers: encoded.count, totalFlips: encoded.flips,
            calibrationTokens: batches.reduce(0) { $0 + $1.count },
            meanLossRatio: encoded.meanRatio, wallSeconds: secs)
        return (sum, encoded.manifest)
    }

    private static func applyAllocatedMACaQ(
        _ ctx: ExecutionContext,
        modeForPath: (String, Int) -> MACaQTargetMode,
        onStep: ((String) -> Void)?
    ) throws -> (manifest: [String: MACaQLayerSpec], count: Int, flips: Int, meanRatio: Float) {
        let blockGroups = MACaQRefiner.groupedBlocks(
            in: ctx.root, stepSize: ctx.config.blockGroupSize, maxHessianBytes: ctx.config.maxHessianBytes)
        var manifest: [String: MACaQLayerSpec] = [:]
        var (quantCount, totalFlips, ratioAccum) = (0, 0, Float(0))
        for (gIdx, group) in blockGroups.enumerated() {
            try Task.checkCancellation()
            let targets = collectCandidateTargets(ctx.root, blocks: Set(group))
            guard !targets.linears.isEmpty || !targets.switches.isEmpty else { continue }
            let captured = try captureMoEGroup(
                model: ctx.model, root: ctx.root,
                targets: .init(linears: targets.linears, switches: targets.switches),
                routers: ctx.routers, batches: ctx.batches, maxSwitchRows: ctx.config.maxSwitchSamples)
            for item in targets.linears {
                guard let rawH = captured.linears[item.groupKey] else { continue }
                let mode = modeForPath(item.path, item.inFeatures)
                let built = buildLinearLayer(item, rawH: rawH, mode: mode, config: ctx.config)
                try PVTuner.swapModule(item.path, with: built.layer, in: ctx.root)
                manifest[item.path] = built.spec
                quantCount += 1
                totalFlips += built.flips
                ratioAccum += built.ratio
            }
            for item in targets.switches {
                guard let resBox = captured.switches[item.path] else { continue }
                let mode = modeForPath(item.path, item.inFeatures)
                let rW = MACaQRefiner.routerWeight(in: ctx.root, for: item.path)
                if let built = buildSwitchLayer(item, reservoir: resBox, routerW: rW, mode: mode, estimator: ctx.estimator, config: ctx.config) {
                    try PVTuner.swapModule(item.path, with: built.layer, in: ctx.root)
                    manifest[item.path] = built.spec
                    quantCount += 1
                    totalFlips += built.flips
                    ratioAccum += built.ratio
                }
            }
            Memory.clearCache()
            onStep?("moe-rw group \(gIdx + 1)/\(blockGroups.count) quantized=\(quantCount) flips=\(totalFlips)")
        }
        MLXInferenceEngine.settle(ctx.root)
        let meanRatio = quantCount > 0 ? (ratioAccum / Float(quantCount)) : 1.0
        return (manifest, quantCount, totalFlips, meanRatio)
    }
}

extension MoERWPipeline {
    fileprivate static func buildLinearLayer(
        _ item: MACaQRefiner.LinearEntry,
        rawH: MLXArray,
        mode: MACaQTargetMode,
        config: MACaQConfig
    ) -> (layer: MACaQQuantizedLinear, spec: MACaQLayerSpec, flips: Int, ratio: Float) {
        let (dense, bias) = QTIPQuantizer.extractDenseWeight(item.module)
        let saveDType: DType = dense.dtype == .float16 ? .float16 : .bfloat16
        let blk = item.inFeatures.isMultiple(of: config.hadamardBlock) ? config.hadamardBlock : 32
        let pre = MACaQMath.balancePreScale(
            weight: dense, meanSquareActivation: diagonal(rawH), alpha: config.alpha
        ).asType(saveDType)
        eval(pre)
        let frame = MACaQMath.transformFrame(
            weight: dense, hessian: rawH, preScale: pre, blockSize: blk, ridge: max(0.02, config.ridge))
        let enc = encodeFrameMatrix(frame: frame, mode: mode, saveDType: saveDType, maxIters: config.maxFlipIterations)
        var spec = MACaQLayerSpec(bits: mode.bits, groupSize: mode.groupSize, mode: mode.quantizationMode, hadamardBlock: blk)
        spec.scaleDType = "\(enc.s.dtype)"
        spec.hasBias = bias != nil
        let layer = MACaQQuantizedLinear(
            packed: MACaQPackedWeights(weight: enc.w, scales: enc.s, biases: enc.b),
            bias: bias, preScale: pre, spec: spec)
        eval(layer)
        let ratio = enc.res.lossBefore > 0 ? (enc.res.lossAfter / enc.res.lossBefore) : 1.0
        return (layer, spec, enc.res.flips, ratio)
    }

    fileprivate static func buildSwitchLayer(
        _ item: MACaQRefiner.SwitchEntry,
        reservoir: MoERWSwitchReservoir,
        routerW: MLXArray?,
        mode: MACaQTargetMode,
        estimator: MoERouterWeightedHessian,
        config: MACaQConfig
    ) -> (layer: MACaQSwitchLinear, spec: MACaQLayerSpec, flips: Int, ratio: Float)? {
        let p = Dictionary(uniqueKeysWithValues: item.module.parameters().flattened())
        guard p["weight"] != nil else { return nil }
        let denseAll = extractSwitchDense(item.module)
        let saveDType: DType = .bfloat16
        let blk = item.inFeatures.isMultiple(of: config.hadamardBlock) ? config.hadamardBlock : 32
        var (wList, sList, bList, preList) = ([MLXArray](), [MLXArray](), [MLXArray](), [MLXArray]())
        var (flipsSum, beforeSum, afterSum) = (0, Float(0), Float(0))
        for e in 0 ..< item.numExperts {
            let hExp = reservoir.expertHessian(expert: e, estimator: estimator, routerWeight: routerW)?.hessian
                ?? eye(item.inFeatures, dtype: .float32)
            let wExp = denseAll[e]
            let pre = MACaQMath.balancePreScale(
                weight: wExp, meanSquareActivation: diagonal(hExp), alpha: config.alpha
            ).asType(saveDType)
            eval(pre)
            preList.append(pre)
            let frame = MACaQMath.transformFrame(
                weight: wExp, hessian: hExp, preScale: pre, blockSize: blk, ridge: config.ridge)
            let enc = encodeFrameMatrix(frame: frame, mode: mode, saveDType: saveDType, maxIters: config.maxFlipIterations)
            wList.append(enc.w)
            sList.append(enc.s)
            if let b = enc.b { bList.append(b) }
            flipsSum += enc.res.flips
            beforeSum += enc.res.lossBefore
            afterSum += enc.res.lossAfter
        }
        var spec = MACaQLayerSpec(bits: mode.bits, groupSize: mode.groupSize, mode: mode.quantizationMode, hadamardBlock: blk)
        spec.perExpertPreScale = true
        spec.scaleDType = sList.first.map { "\($0.dtype)" }
        spec.hasBias = p["bias"] != nil
        let packed = MACaQPackedWeights(
            weight: stacked(wList, axis: 0),
            scales: stacked(sList, axis: 0),
            biases: bList.isEmpty ? nil : stacked(bList, axis: 0))
        let layer = MACaQSwitchLinear(
            geometry: .init(experts: item.numExperts, outputs: item.outFeatures, inputs: item.inFeatures),
            packed: packed, bias: p["bias"], preScale: stacked(preList, axis: 0), spec: spec)
        eval(layer)
        return (layer, spec, flipsSum, beforeSum > 0 ? (afterSum / beforeSum) : 1.0)
    }

    private static func encodeFrameMatrix(
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

extension MoERWPipeline {
    static func refineStandardModel(
        _ student: any LanguageModel,
        batches: [[Int32]],
        estimator: MoERouterWeightedHessian,
        config: MACaQConfig,
        teacherWeight: (String) throws -> MLXArray?
    ) throws -> MACaQSummary {
        let startClock = ContinuousClock.now
        let root = student as Module
        root.train(false)
        let routers = discoverRouters(in: root)
        let blockGroups = MACaQRefiner.groupedBlocks(
            in: root, stepSize: config.blockGroupSize, maxHessianBytes: config.maxHessianBytes)
        var (refinedCount, totalFlips, ratioAccum) = (0, 0, Float(0))
        for group in blockGroups {
            try Task.checkCancellation()
            let targets = collectCandidateTargets(root, blocks: Set(group))
            guard !targets.switches.isEmpty else { continue }
            let captured = try captureMoEGroup(
                model: student, root: root, targets: .init(linears: [], switches: targets.switches),
                routers: routers, batches: batches, maxSwitchRows: config.maxSwitchSamples)
            for item in targets.switches {
                guard let qsw = item.module as? QuantizedSwitchLinear,
                      let resBox = captured.switches[item.path],
                      let targetW = try teacherWeight(item.path) else { continue }
                let rW = MACaQRefiner.routerWeight(in: root, for: item.path)
                if let res = refineQuantizedSwitch(item, qsw: qsw, targetW: targetW, reservoir: resBox, routerW: rW, estimator: estimator) {
                    root.update(parameters: ModuleParameters.unflattened(res.updates))
                    eval(res.updates.map(\.1))
                    refinedCount += 1
                    totalFlips += res.flips
                    ratioAccum += res.ratio
                }
            }
            Memory.clearCache()
        }
        MLXInferenceEngine.settle(root)
        let elapsed = ContinuousClock.now - startClock
        let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        return MACaQSummary(
            refinedLayers: refinedCount, totalFlips: totalFlips,
            calibrationTokens: batches.reduce(0) { $0 + $1.count },
            meanLossRatio: refinedCount > 0 ? (ratioAccum / Float(refinedCount)) : 1.0, wallSeconds: secs)
    }

    private static func refineQuantizedSwitch(
        _ item: MACaQRefiner.SwitchEntry,
        qsw: QuantizedSwitchLinear,
        targetW: MLXArray,
        reservoir: MoERWSwitchReservoir,
        routerW: MLXArray?,
        estimator: MoERouterWeightedHessian
    ) -> (updates: [(String, MLXArray)], flips: Int, ratio: Float)? {
        let validMode = ([2, 3, 4].contains(qsw.bits) && qsw.mode == .affine) || (qsw.bits == 4 && qsw.mode == .mxfp4)
        guard validMode, targetW.shape == [item.numExperts, item.outFeatures, item.inFeatures] else { return nil }
        let p = Dictionary(uniqueKeysWithValues: qsw.parameters().flattened())
        guard let wAll = p["weight"], let sAll = p["scales"] else { return nil }
        let bAll = p["biases"]
        let spec = PVCodeGrid.Spec(groupSize: qsw.groupSize, bits: qsw.bits, mode: qsw.mode)
        var (wSlices, sSlices) = ([MLXArray](), [MLXArray]())
        var (flipsSum, beforeSum, afterSum) = (0, Float(0), Float(0))
        for e in 0 ..< item.numExperts {
            guard let hExp = reservoir.expertHessian(expert: e, estimator: estimator, routerWeight: routerW)?.hessian else {
                wSlices.append(wAll[e])
                sSlices.append(sAll[e])
                continue
            }
            let packed = MACaQPackedWeights(weight: wAll[e], scales: sAll[e], biases: bAll.map { $0[e] })
            let step = refineSingleExpertSlice(packed: packed, target: targetW[e], hessian: hExp, spec: spec)
            wSlices.append(step.w)
            sSlices.append(step.s)
            beforeSum += step.lossBefore
            afterSum += step.lossAfter
            flipsSum += step.flips
        }
        var updates: [(String, MLXArray)] = [("\(item.path).weight", stacked(wSlices, axis: 0))]
        if qsw.mode == .mxfp4 { updates.append(("\(item.path).scales", stacked(sSlices, axis: 0))) }
        return (updates, flipsSum, beforeSum > 0 ? (afterSum / beforeSum) : 1.0)
    }

    private static func refineSingleExpertSlice(
        packed: MACaQPackedWeights,
        target: MLXArray,
        hessian: MLXArray,
        spec: PVCodeGrid.Spec
    ) -> (w: MLXArray, s: MLXArray, flips: Int, lossBefore: Float, lossAfter: Float) {
        let frame = MACaQMath.transformFrame(weight: target, hessian: hessian, preScale: nil, blockSize: nil, ridge: 0.0)
        var grid = MACaQMath.makeGridState(
            packedWeight: packed.weight, scales: packed.scales, biases: packed.biases, spec: spec).state
        let res = MACaQMath.flipRefine(state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: 96)
        guard res.lossAfter.isFinite, res.lossAfter <= res.lossBefore * 1.00001 + 1e-10 else {
            return (packed.weight, packed.scales, 0, res.lossBefore, res.lossBefore)
        }
        if spec.encoding == .affine(.mxfp4) {
            let pk = MACaQMath.packMXFP4(sidx: grid.codes, e8m0: packed.scales)
            return (pk.weight, pk.scales, res.flips, res.lossBefore, res.lossAfter)
        }
        let wOut = PVCodeGrid.packCodes(grid.codes.asType(.uint8), bits: spec.bits)
        return (wOut, packed.scales, res.flips, res.lossBefore, res.lossAfter)
    }
}
