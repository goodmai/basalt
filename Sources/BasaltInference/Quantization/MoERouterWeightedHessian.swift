import BasaltCore
import Foundation
import MLX

package enum GatingWeightMode: String, Sendable, CaseIterable, Codable {
    case quadratic
    case linear
    case unweighted
}

package struct MoERouterWeightedHessian: Sendable, Equatable {
    package var mode: GatingWeightMode
    package var damping: Float
    package var minRoutedTokens: Int

    package init(
        mode: GatingWeightMode = .quadratic,
        damping: Float = 0.02,
        minRoutedTokens: Int = 4
    ) {
        self.mode = mode
        self.damping = max(0, damping)
        self.minRoutedTokens = max(1, minRoutedTokens)
    }

    package func hessian(activations: MLXArray, gatingWeights: MLXArray) -> MLXArray {
        Self.computeHessian(
            activations: activations,
            gatingWeights: gatingWeights,
            mode: mode,
            damping: damping)
    }

    package static func topKGating(
        logits: MLXArray,
        topK: Int,
        normTopKProb: Bool = true
    ) -> (indices: MLXArray, weights: MLXArray, fullProbs: MLXArray) {
        let numExperts = max(1, logits.dim(-1))
        let rows = max(1, logits.size / numExperts)
        let flatLogits = logits.reshaped(rows, numExperts).asType(.float32)
        let probs = softmax(flatLogits, axis: -1, precise: true)
        let k = min(max(1, topK), numExperts)
        let indices = argSort(-flatLogits, axis: -1)[0..., 0 ..< k].asType(.int32)
        var selected = takeAlong(probs, indices, axis: -1)
        if normTopKProb {
            selected = selected / maximum(sum(selected, axis: -1, keepDims: true), Float(1e-12))
        }
        eval(indices, selected, probs)
        return (indices, selected, probs)
    }

    package static func computeHessian(
        activations: MLXArray,
        gatingWeights: MLXArray,
        mode: GatingWeightMode = .quadratic,
        damping: Float = 0.02
    ) -> MLXArray {
        let cols = activations.dim(-1)
        let rows = cols > 0 ? activations.size / cols : 0
        guard rows > 0, cols > 0 else {
            let ident = eye(max(1, cols), dtype: .float32) * max(damping, Float(1e-6))
            eval(ident)
            return ident
        }
        let xf = activations.reshaped(rows, cols).asType(.float32)
        let cf = gatingWeights.reshaped(rows).asType(.float32)
        let lambda = max(Float(0.0), damping)
        let out: MLXArray
        switch mode {
        case .quadratic:
            let xw = xf * abs(cf).expandedDimensions(axis: 1)
            let base = matmul(xw.T, xw) * Float(2.0)
            let reg = lambda * maximum(mean(diagonal(base)), Float(1e-12))
            out = base + eye(cols, dtype: .float32) * reg
        case .linear:
            let xw = xf * sqrt(maximum(cf, Float(0.0))).expandedDimensions(axis: 1)
            let base = matmul(xw.T, xw) * Float(2.0)
            out = base + eye(cols, dtype: .float32) * lambda
        case .unweighted:
            let base = matmul(xf.T, xf) * Float(2.0)
            let reg = lambda * maximum(mean(diagonal(base)), Float(1e-12))
            out = base + eye(cols, dtype: .float32) * reg
        }
        eval(out)
        return out
    }
}

extension MoERouterWeightedHessian {
    package static func cascadeRouterKL(teacherLogits: MLXArray, studentLogits: MLXArray) -> Float {
        let numExperts = max(1, teacherLogits.dim(-1))
        let rows = max(1, teacherLogits.size / numExperts)
        let tLogits = teacherLogits.reshaped(rows, numExperts).asType(.float32)
        let sLogits = studentLogits.reshaped(rows, numExperts).asType(.float32)
        let logP = tLogits - logSumExp(tLogits, axis: -1, keepDims: true)
        let logQ = sLogits - logSumExp(sLogits, axis: -1, keepDims: true)
        let p = exp(logP)
        let kl = maximum(mean(sum(p * (logP - logQ), axis: -1)), Float(0.0))
        eval(kl)
        let val = kl.item(Float.self)
        return val.isFinite ? val : 0.0
    }

    package static func topKFlipRate(
        teacherLogits: MLXArray,
        studentLogits: MLXArray,
        topK: Int
    ) -> Float {
        let numExperts = max(1, teacherLogits.dim(-1))
        let rows = max(1, teacherLogits.size / numExperts)
        let k = min(max(1, topK), numExperts)
        let tTop = argSort(-teacherLogits.reshaped(rows, numExperts).asType(.float32), axis: -1)[0..., 0 ..< k].asType(.int32)
        let sTop = argSort(-studentLogits.reshaped(rows, numExperts).asType(.float32), axis: -1)[0..., 0 ..< k].asType(.int32)
        eval(tTop, sTop)
        let tArr = tTop.asArray(Int32.self)
        let sArr = sTop.asArray(Int32.self)
        var flips = 0
        for row in 0 ..< rows {
            let base = row * k
            for slot in 0 ..< k {
                let target = tArr[base + slot]
                var matched = false
                for cand in 0 ..< k where sArr[base + cand] == target {
                    matched = true
                    break
                }
                if !matched { flips += 1 }
            }
        }
        return Float(flips) / Float(max(1, rows * k))
    }

    package static func routedReconstructionMSE(
        activations: MLXArray,
        gatingWeights: MLXArray,
        targetWeight: MLXArray,
        quantizedWeight: MLXArray
    ) -> Float {
        let cols = max(1, targetWeight.dim(-1))
        let rows = max(1, activations.size / cols)
        let xf = activations.reshaped(rows, cols).asType(.float32)
        let cf = gatingWeights.reshaped(rows, 1).asType(.float32)
        let diff = targetWeight.asType(.float32) - quantizedWeight.asType(.float32)
        let err = matmul(xf * cf, diff.T)
        let mse = mean(err * err)
        eval(mse)
        return mse.item(Float.self)
    }

    package struct ExpertQuantResult {
        package var dequantized: MLXArray
        package var packedWeight: MLXArray
        package var scales: MLXArray
        package var biases: MLXArray?
        package var columnBalance: MLXArray
        package var flips: Int
        package var lossBefore: Float
        package var lossAfter: Float
    }

    package func quantizeExpert(
        weight: MLXArray,
        activations: MLXArray,
        gatingWeights: MLXArray,
        bits: Int,
        groupSize: Int = 32,
        maxFlipIterations: Int = 96
    ) -> ExpertQuantResult {
        let wf = weight.asType(.float32)
        let hMat = hessian(activations: activations, gatingWeights: gatingWeights)
        let diagH = maximum(diagonal(hMat), Float(1e-12))
        let colImportance = sqrt(diagH / maximum(mean(diagH), Float(1e-12)))
        let balance = clip(pow(colImportance, MLXArray(Float(0.5))), min: Float(0.25), max: Float(4.0))
        let wBalanced = wf * balance.reshaped(1, wf.dim(1))
        let (wqInit, sInit, bInit) = MLX.quantized(
            wBalanced.asType(.bfloat16), groupSize: groupSize, bits: bits, mode: .affine)
        let invBal = Float(1.0) / balance
        let rows = wf.dim(0)
        let cols = wf.dim(1)
        let groups = cols / groupSize
        let baseScaleElem = broadcast(
            sInit.asType(.float32).expandedDimensions(axis: -1),
            to: [rows, groups, groupSize]
        ).reshaped(rows, cols) * invBal.reshaped(1, cols)
        let baseBiasElem = bInit.map {
            broadcast(
                $0.asType(.float32).expandedDimensions(axis: -1),
                to: [rows, groups, groupSize]
            ).reshaped(rows, cols) * invBal.reshaped(1, cols)
        }
        let codesInit = PVCodeGrid.unpackCodes(wqInit, bits: bits).asType(.int32)
        var grid = MACaQGridState(
            codes: codesInit, scaleElem: baseScaleElem, biasElem: baseBiasElem,
            levelTable: nil, maxLevel: Int32((1 << bits) - 1))
        let res = MACaQMath.flipRefine(
            state: &grid, targetWeight: wf, hessian: hMat, maxIterations: maxFlipIterations)
        let accept = res.lossAfter.isFinite && res.lossAfter <= res.lossBefore * 1.00001 + 1e-10
        let finalCodes = accept ? grid.codes : codesInit
        let packed = PVCodeGrid.packCodes(finalCodes.asType(.uint8), bits: bits)
        let deq = accept
            ? grid.dequantizedValues()
            : (MLX.dequantized(wqInit, scales: sInit, biases: bInit, groupSize: groupSize, bits: bits, mode: .affine, dtype: .float32) * invBal.reshaped(1, cols))
        eval(deq, packed, balance)
        return ExpertQuantResult(
            dequantized: deq, packedWeight: packed, scales: sInit, biases: bInit,
            columnBalance: balance,
            flips: accept ? res.flips : 0, lossBefore: res.lossBefore, lossAfter: accept ? res.lossAfter : res.lossBefore)
    }
}

package struct DeadZoneBitAllocator: Sendable, Equatable {
    package struct ExpertSensitivity: Sendable, Equatable {
        package var id: String
        package var weightCount: Int
        package var routerWeightedTrace: Double
        package var cascadeRouterKL: Double
        package var routingMass: Double
        package var minBits: Int = 2
        package var maxBits: Int = 4
        package var overheadBPW: Double = 0.0

        package init(
            id: String,
            weightCount: Int,
            routerWeightedTrace: Double,
            cascadeRouterKL: Double = 0.0,
            routingMass: Double = 1.0
        ) {
            self.id = id
            self.weightCount = max(1, weightCount)
            self.routerWeightedTrace = max(0.0, routerWeightedTrace)
            self.cascadeRouterKL = max(0.0, cascadeRouterKL)
            self.routingMass = max(0.0, routingMass)
        }
    }

    package struct Allocation: Sendable, Equatable {
        package var id: String
        package var bits: Int
        package var weightCount: Int
        package var effectiveScore: Double
        package var effectiveBPW: Double
    }

    package struct Plan: Sendable, Equatable {
        package var allocations: [Allocation]
        package var targetBPW: Double
        package var achievedBPW: Double
        package var deadZoneLowBPW: Double
        package var deadZoneHighBPW: Double
        package var withinDeadZone: Bool
        package var deadZonePenalty: Double
        package var oscillationCount: Int
        package var bitHistogram: [Int: Int]
    }

    package var targetBPW: Double
    package var deadZoneEpsilon: Double
    package var cascadeKLWeight: Double
    package var allowedBits: [Int]

    package init(
        targetBPW: Double = 2.8,
        deadZoneEpsilon: Double = 0.02,
        cascadeKLWeight: Double = 8.0,
        allowedBits: [Int] = [2, 3, 4]
    ) {
        self.targetBPW = targetBPW
        self.deadZoneEpsilon = min(max(0.001, deadZoneEpsilon), 0.25)
        self.cascadeKLWeight = max(0.0, cascadeKLWeight)
        let cleaned = Array(Set(allowedBits)).sorted()
        self.allowedBits = cleaned.isEmpty ? [2, 3, 4] : cleaned
    }

    package func deadZonePenalty(achievedBPW: Double) -> Double {
        let lo = (1.0 - deadZoneEpsilon) * targetBPW
        let hi = (1.0 + deadZoneEpsilon) * targetBPW
        let over = max(0.0, achievedBPW - hi)
        let under = max(0.0, lo - achievedBPW)
        return over * over + under * under
    }
}

extension DeadZoneBitAllocator {
    fileprivate struct SolverState {
        let items: [ExpertSensitivity]
        let ladders: [[Int]]
        let scores: [Double]
        let totalWeights: Double
        var indices: [Int]
        var usedBits: Double
        var directions: [Int]
        var oscillations = 0
    }

    package func allocate(_ items: [ExpertSensitivity]) -> Plan {
        let lo = (1.0 - deadZoneEpsilon) * targetBPW
        let hi = (1.0 + deadZoneEpsilon) * targetBPW
        guard !items.isEmpty else {
            return Plan(
                allocations: [], targetBPW: targetBPW, achievedBPW: 0.0,
                deadZoneLowBPW: lo, deadZoneHighBPW: hi, withinDeadZone: false,
                deadZonePenalty: deadZonePenalty(achievedBPW: 0.0), oscillationCount: 0, bitHistogram: [:])
        }
        let ladders = items.map { item -> [Int] in
            let valid = allowedBits.filter { $0 >= item.minBits && $0 <= item.maxBits }
            return valid.isEmpty ? [min(max(allowedBits[0], item.minBits), item.maxBits)] : valid
        }
        let scores = items.map { item -> Double in
            let mass = max(1e-4, item.routingMass)
            let base = max(1e-12, item.routerWeightedTrace * mass)
            return base * (1.0 + cascadeKLWeight * item.cascadeRouterKL)
        }
        let totalWeights = Double(items.reduce(0) { $0 + $1.weightCount })
        let initIdx = initialDualPicks(items: items, ladders: ladders, scores: scores, totalWeights: totalWeights)
        var state = SolverState(
            items: items, ladders: ladders, scores: scores, totalWeights: totalWeights,
            indices: initIdx, usedBits: totalBitSum(items: items, ladders: ladders, indices: initIdx),
            directions: Array(repeating: 0, count: items.count))
        if state.usedBits / totalWeights > hi {
            downgradeIntoDeadZone(&state, maxBitsBudget: hi * totalWeights)
        } else {
            upgradeIntoDeadZone(
                &state, targetBits: targetBPW * totalWeights,
                minBitsBudget: lo * totalWeights, maxBitsBudget: hi * totalWeights)
        }
        return finalizePlan(state, lo: lo, hi: hi)
    }

    private func initialDualPicks(
        items: [ExpertSensitivity],
        ladders: [[Int]],
        scores: [Double],
        totalWeights: Double
    ) -> [Int] {
        let minAvg = totalBitSum(items: items, ladders: ladders, indices: Array(repeating: 0, count: items.count)) / totalWeights
        if minAvg >= targetBPW { return Array(repeating: 0, count: items.count) }
        let maxIdx = ladders.map { $0.count - 1 }
        if totalBitSum(items: items, ladders: ladders, indices: maxIdx) / totalWeights <= targetBPW { return maxIdx }
        var (lamLo, lamHi) = (0.0, 1.0)
        while dualBitSum(items: items, ladders: ladders, scores: scores, lambda: lamHi) / totalWeights > targetBPW {
            lamHi *= 2.0
        }
        for _ in 0 ..< 56 {
            let mid = 0.5 * (lamLo + lamHi)
            if dualBitSum(items: items, ladders: ladders, scores: scores, lambda: mid) / totalWeights <= targetBPW {
                lamHi = mid
            } else {
                lamLo = mid
            }
        }
        return items.indices.map { dualPick(ladder: ladders[$0], score: scores[$0], lambda: lamHi) }
    }

    private func dualPick(ladder: [Int], score: Double, lambda: Double) -> Int {
        var best = 0
        var bestCost = score * pow(4.0, -Double(ladder[0])) + lambda * Double(ladder[0])
        for idx in 1 ..< ladder.count {
            let cost = score * pow(4.0, -Double(ladder[idx])) + lambda * Double(ladder[idx])
            if cost <= bestCost {
                bestCost = cost
                best = idx
            }
        }
        return best
    }

    private func dualBitSum(
        items: [ExpertSensitivity], ladders: [[Int]], scores: [Double], lambda: Double
    ) -> Double {
        var sum = 0.0
        for i in items.indices {
            let pick = dualPick(ladder: ladders[i], score: scores[i], lambda: lambda)
            sum += Double(items[i].weightCount) * (Double(ladders[i][pick]) + items[i].overheadBPW)
        }
        return sum
    }

    fileprivate func totalBitSum(items: [ExpertSensitivity], ladders: [[Int]], indices: [Int]) -> Double {
        var sum = 0.0
        for i in items.indices {
            sum += Double(items[i].weightCount) * (Double(ladders[i][indices[i]]) + items[i].overheadBPW)
        }
        return sum
    }
}

extension DeadZoneBitAllocator {
    fileprivate func downgradeIntoDeadZone(_ state: inout SolverState, maxBitsBudget: Double) {
        while state.usedBits > maxBitsBudget + 1e-9 {
            var bestIdx = -1
            var minLossPerBit = Double.infinity
            for i in state.items.indices where state.indices[i] > 0 {
                let curB = Double(state.ladders[i][state.indices[i]])
                let prevB = Double(state.ladders[i][state.indices[i] - 1])
                let lossPerBit = state.scores[i] * (pow(4.0, -prevB) - pow(4.0, -curB)) / max(1e-9, curB - prevB)
                if lossPerBit < minLossPerBit {
                    minLossPerBit = lossPerBit
                    bestIdx = i
                }
            }
            guard bestIdx >= 0 else { break }
            let curBits = state.ladders[bestIdx][state.indices[bestIdx]]
            let prevBits = state.ladders[bestIdx][state.indices[bestIdx] - 1]
            let delta = Double(state.items[bestIdx].weightCount * (curBits - prevBits))
            state.indices[bestIdx] -= 1
            state.usedBits -= delta
            if state.directions[bestIdx] > 0 { state.oscillations += 1 }
            state.directions[bestIdx] = -1
        }
    }

    fileprivate func upgradeIntoDeadZone(
        _ state: inout SolverState,
        targetBits: Double,
        minBitsBudget: Double,
        maxBitsBudget: Double
    ) {
        while state.usedBits < targetBits - 1e-9 {
            var bestIdx = -1
            var bestGainPerBit = -1.0
            for i in state.items.indices where state.indices[i] + 1 < state.ladders[i].count {
                let curB = Double(state.ladders[i][state.indices[i]])
                let nextB = Double(state.ladders[i][state.indices[i] + 1])
                let deltaBits = Double(state.items[i].weightCount) * (nextB - curB)
                if state.usedBits + deltaBits > maxBitsBudget + 1e-9 && state.usedBits >= minBitsBudget - 1e-9 {
                    continue
                }
                let gainPerBit = state.scores[i] * (pow(4.0, -curB) - pow(4.0, -nextB)) / max(1e-9, nextB - curB)
                if gainPerBit > bestGainPerBit {
                    bestGainPerBit = gainPerBit
                    bestIdx = i
                }
            }
            guard bestIdx >= 0 else { break }
            let nextBits = state.ladders[bestIdx][state.indices[bestIdx] + 1]
            let curBits = state.ladders[bestIdx][state.indices[bestIdx]]
            let delta = Double(state.items[bestIdx].weightCount * (nextBits - curBits))
            if state.usedBits >= minBitsBudget - 1e-9 && state.usedBits + delta > maxBitsBudget + 1e-9 { break }
            state.indices[bestIdx] += 1
            state.usedBits += delta
            if state.directions[bestIdx] < 0 { state.oscillations += 1 }
            state.directions[bestIdx] = 1
        }
    }

    fileprivate func finalizePlan(_ state: SolverState, lo: Double, hi: Double) -> Plan {
        var histogram: [Int: Int] = [:]
        let allocations = state.items.indices.map { i -> Allocation in
            let b = state.ladders[i][state.indices[i]]
            histogram[b, default: 0] += 1
            return Allocation(
                id: state.items[i].id,
                bits: b,
                weightCount: state.items[i].weightCount,
                effectiveScore: state.scores[i],
                effectiveBPW: Double(b) + state.items[i].overheadBPW)
        }
        let achieved = state.usedBits / max(1.0, state.totalWeights)
        let penalty = deadZonePenalty(achievedBPW: achieved)
        let inside = achieved + 1e-9 >= lo && achieved - 1e-9 <= hi
        return Plan(
            allocations: allocations, targetBPW: targetBPW, achievedBPW: achieved,
            deadZoneLowBPW: lo, deadZoneHighBPW: hi, withinDeadZone: inside,
            deadZonePenalty: penalty, oscillationCount: state.oscillations, bitHistogram: histogram)
    }
}

package enum MultilingualMoECalibrationCorpus {
    package struct Probe: Sendable, Equatable {
        package let domain: String
        package let text: String
    }

    package static let probes: [Probe] = [
        Probe(
            domain: "en",
            text: "Sparse Mixture-of-Experts transformers route each token to top-K specialized feed-forward networks while preserving constant active FLOPs per decoded symbol."),
        Probe(
            domain: "ru",
            text: "Взвешенный весами маршрутизатора гессиан учитывает квадратичный вклад экспертов H_i = 2 X_i^T diag(c_i^2) X_i и предотвращает деградацию маршрутизации на следующем слое."),
        Probe(
            domain: "ko",
            text: "전문가 혼합 모델의 양자화에서는 라우터 게이트 가중치와 다음 계층의 쿨백-라이블러 발산을 함께 고려해야 다국어 추론 정확도를 유지할 수 있습니다."),
        Probe(
            domain: "ja",
            text: "混合エキスパートモデルの混合精度量子化では、各エキスパートの活性頻度とゲート重みに応じて2ビットから4ビットを適応的に割り当てます。"),
        Probe(
            domain: "math",
            text: "Theorem: Let H_RW = 2 X^T diag(c^2) X + lambda I > 0 and D_KL(P || Q) = sum_j P_j (log P_j - log Q_j) >= 0; for b_i in {2,3,4}, Phi_eps(b, B) = max(0, |b - B| - eps B)^2 = 0."),
        Probe(
            domain: "code",
            text: #"func routeTopK(logits: [Float], k: Int) -> [(id: Int, gate: Float)] { logits.enumerated().sorted { $0.element > $1.element }.prefix(k).map { ($0.offset, $0.element) } } {"moe_bpw":2.8,"recipe":"moe-rw"}"#),
    ]

    package static let defaultText: String = {
        let cycle = probes.map(\.text).joined(separator: "\n")
        return Array(repeating: cycle, count: 24).joined(separator: "\n\n")
    }()

    package static func routingEntropy(gatingProbs: MLXArray) -> Float {
        let numExperts = max(1, gatingProbs.dim(-1))
        guard numExperts > 1 else { return 1.0 }
        let rows = max(1, gatingProbs.size / numExperts)
        let flat = gatingProbs.reshaped(rows, numExperts).asType(.float32)
        let marginal = mean(flat, axis: 0)
        let normalized = marginal / maximum(sum(marginal), Float(1e-12))
        let ent = -sum(normalized * log(maximum(normalized, Float(1e-12))))
        eval(ent)
        let raw = ent.item(Float.self)
        let denom = Foundation.log(Float(numExperts))
        return denom > 0 ? min(1.0, max(0.0, raw / denom)) : 0.0
    }
}
