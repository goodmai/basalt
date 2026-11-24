import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

struct GroupLatentGrouping: Sendable, Equatable {
    let groups: Int
    let labels: [Int]
}

struct GroupLatentRow: Sendable {
    let layer: Int
    let side: String
    let dim: Int
    let energies: [[Float]]

    var lines: [String] {
        zip(GroupLatentSpectrum.groupCounts, energies).map { groups, kept in
            "\(layer) | \(side) | \(dim) | \(groups) | " + kept.map { String(format: "%.4f", $0) }.joined(separator: " | ")
        }
    }
}

enum GroupLatentSpectrum {
    static let groupCounts = [1, 4, 8]
    static let latentRatios = [0.5, 0.7]
    static let subspaceRank = 16

    static func latentRanks(_ dim: Int) -> [Int] {
        latentRatios.map { Int((Double(dim) * $0).rounded()) }
    }

    static func energies(count: Int, dim: Int, key: MLXArray, rows: (Int) -> MLXArray) -> [[Float]] {
        let found = groupings(count: count, dim: dim, rank: subspaceRank, key: key, rows: rows)
        return retainedEnergy(found, dim: dim, ranks: latentRanks(dim), rows: rows)
    }

    static func groupings(count: Int, dim: Int, rank: Int, key: MLXArray, rows: (Int) -> MLXArray) -> [GroupLatentGrouping] {
        let r = min(max(1, rank), dim)
        let probe = MLXRandom.normal([dim, min(r + 8, dim)], key: key)
        let bases = (0 ..< count).map { expert in
            let basis = subspaceBasis(rows(expert), rank: r, probe: probe)
            eval(basis)
            return basis
        }
        let similarity = subspaceSimilarity(bases)
        eval(similarity)
        return groupCounts.map { GroupLatentGrouping(groups: $0, labels: equalSizeKMeans(similarity: similarity, groups: $0)) }
    }

    static func subspaceBasis(_ rows: MLXArray, rank: Int, probe: MLXArray) -> MLXArray {
        let first = MLXLinalg.qr(matmul(rows.T, matmul(rows, probe)), stream: .cpu).0
        let q = MLXLinalg.qr(matmul(rows.T, matmul(rows, first)), stream: .cpu).0
        let projected = matmul(rows, q)
        let vectors = MLXLinalg.eigh(matmul(projected.T, projected), stream: .cpu).1
        let k = vectors.dim(1)
        return matmul(q, vectors[0..., (k - rank) ..< k]).T
    }

    static func subspaceSimilarity(_ bases: [MLXArray]) -> MLXArray {
        let (count, rank) = (bases.count, bases[0].dim(0))
        let stacked = concatenated(bases, axis: 0)
        let overlap = matmul(stacked, stacked.T)
        return (overlap * overlap).reshaped(count, rank, count, rank).sum(axes: [1, 3])
    }

    static func equalSizeKMeans(similarity: MLXArray, groups: Int, iterations: Int = 16) -> [Int] {
        let count = similarity.dim(0)
        guard groups > 1, count >= groups else { return [Int](repeating: 0, count: count) }
        var centers = [0]
        while centers.count < groups {
            let nearest = similarity.take(MLXArray(centers.map { Int32($0) }), axis: 1).max(axis: 1)
            centers.append(argMin(nearest).item(Int.self))
        }
        let toCenters = -similarity.take(MLXArray(centers.map { Int32($0) }), axis: 1)
        var labels = balancedAssign(cost: toCenters.asArray(Float.self), groups: groups)
        for _ in 0 ..< iterations {
            let next = balancedAssign(cost: kernelCost(similarity, labels: labels, groups: groups), groups: groups)
            if next == labels { break }
            labels = next
        }
        return labels
    }

    static func kernelCost(_ similarity: MLXArray, labels: [Int], groups: Int) -> [Float] {
        let ids = MLXArray(labels.map { Int32($0) }).reshaped(-1, 1)
        let members = (ids .== MLXArray((0 ..< groups).map { Int32($0) }).reshaped(1, -1)).asType(.float32)
        let sizes = maximum(members.sum(axis: 0), Float(1))
        let toGroup = matmul(similarity, members)
        let within = (members * toGroup).sum(axis: 0) / (sizes * sizes)
        return (within.reshaped(1, -1) - toGroup * Float(2) / sizes.reshaped(1, -1)).asArray(Float.self)
    }

    static func balancedAssign(cost: [Float], groups: Int) -> [Int] {
        let count = cost.count / groups
        var capacity = (0 ..< groups).map { count / groups + ($0 < count % groups ? 1 : 0) }
        let order = (0 ..< count).map { item -> (item: Int, ranked: [Int], margin: Float) in
            let ranked = (0 ..< groups).sorted { cost[item * groups + $0] < cost[item * groups + $1] }
            let margin = groups > 1 ? cost[item * groups + ranked[1]] - cost[item * groups + ranked[0]] : 0
            return (item, ranked, margin)
        }.sorted { $0.margin > $1.margin }
        var labels = [Int](repeating: 0, count: count)
        for entry in order {
            guard let group = entry.ranked.first(where: { capacity[$0] > 0 }) else { continue }
            labels[entry.item] = group
            capacity[group] -= 1
        }
        return labels
    }

    static func retainedEnergy(_ groupings: [GroupLatentGrouping], dim: Int, ranks: [Int], rows: (Int) -> MLXArray) -> [[Float]] {
        var grams = groupings.map { grouping in
            (0 ..< grouping.groups).map { _ in MLXArray.zeros([dim, dim], dtype: .float32) }
        }
        for expert in 0 ..< (groupings.first?.labels.count ?? 0) {
            let r = rows(expert)
            let gram = matmul(r.T, r)
            for (index, grouping) in groupings.enumerated() {
                let group = grouping.labels[expert]
                grams[index][group] = grams[index][group] + gram
            }
            eval(grams.flatMap { $0 })
        }
        return grams.map { spectrumEnergy($0, ranks: ranks) }
    }

    static func spectrumEnergy(_ grams: [MLXArray], ranks: [Int]) -> [Float] {
        var kept = [Float](repeating: 0, count: ranks.count)
        var total: Float = 0
        for gram in grams {
            let values = MLXLinalg.eigvalsh(gram, stream: .cpu).asArray(Float.self)
            total += values.reduce(0, +)
            for (index, rank) in ranks.enumerated() {
                kept[index] += values.suffix(rank).reduce(0, +)
            }
        }
        return kept.map { $0 / max(total, 1e-12) }
    }
}

enum GroupLatentCapture {
    static func sweep(_ model: any LanguageModel, batches: [[Int32]], maxLayers: Int, log: ModelRunLog) throws -> [GroupLatentRow] {
        let root = model as Module
        root.train(false)
        let routers = MoERWPipeline.discoverRouters(in: root)
        let blocks = Set(root.namedModules().compactMap { PVTuner.textBlockIndex($0.0) }).sorted().prefix(maxLayers)
        var rows: [GroupLatentRow] = []
        for block in blocks {
            guard let measured = try measureBlock(block, model: model, routers: routers, batches: batches) else { continue }
            for row in measured {
                row.lines.forEach { log.record("GROUPLATENT", $0) }
            }
            rows += measured
            Memory.clearCache()
        }
        return rows
    }

    static func measureBlock(
        _ block: Int, model: any LanguageModel, routers: [Int: MoERWPipeline.RouterSite], batches: [[Int32]]
    ) throws -> [GroupLatentRow]? {
        let root = model as Module
        let switches = MoERWPipeline.collectCandidateTargets(root, blocks: [block]).switches
        let inputs = switches.filter { !$0.path.hasSuffix("down_proj") }
        let inputReaders = inputs.compactMap { expertReader($0.module) }
        guard let first = inputs.first, inputReaders.count == inputs.count,
              let down = switches.first(where: { $0.path.hasSuffix("down_proj") }),
              let downReader = expertReader(down.module) else { return nil }
        let captured = try MoERWPipeline.captureMoEGroup(
            model: model, root: root, targets: .init(linears: [], switches: [first, down]),
            routers: routers, batches: batches, maxSwitchRows: .max)
        let estimator = MoERouterWeightedHessian()
        guard let inReservoir = captured.switches[first.path], let outReservoir = captured.switches[down.path],
              let inWhitening = inputWhitening(inReservoir, estimator: estimator) else { return nil }
        let key = MLXRandom.key(UInt64(6440 + block))
        let mass = (0 ..< first.numExperts).map { gateMass(inReservoir, expert: $0, estimator: estimator) }
        let input = GroupLatentSpectrum.energies(count: first.numExperts, dim: first.inFeatures, key: key) { expert in
            matmul(concatenated(inputReaders.map { $0(expert) }, axis: 0), inWhitening) * mass[expert].squareRoot()
        }
        let whiteners = (0 ..< down.numExperts).map { expert in
            routedHessian(outReservoir, expert: expert, estimator: estimator).map {
                InputWhitenedSVD.computeWhiteningPair(hessian: $0).s
            }
        }
        let output = GroupLatentSpectrum.energies(count: down.numExperts, dim: down.outFeatures, key: key) { expert in
            whiteners[expert].map { matmul($0.T, downReader(expert).T) }
                ?? MLXArray.zeros([down.inFeatures, down.outFeatures], dtype: .float32)
        }
        return [
            GroupLatentRow(layer: block, side: "in", dim: first.inFeatures, energies: input),
            GroupLatentRow(layer: block, side: "out", dim: down.outFeatures, energies: output),
        ]
    }

    static func expertReader(_ module: Module) -> ((Int) -> MLXArray)? {
        let params = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        guard let weight = params["weight"], weight.ndim == 3 else { return nil }
        if let quantized = module as? QuantizedSwitchLinear, let scales = params["scales"] {
            let biases = params["biases"]
            return { expert in
                dequantized(
                    weight[expert], scales: scales[expert], biases: biases.map { $0[expert] },
                    groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode, dtype: .float32)
            }
        }
        guard type(of: module) == SwitchLinear.self else { return nil }
        return { weight[$0].asType(.float32) }
    }

    static func routedHessian(_ reservoir: MoERWSwitchReservoir, expert: Int, estimator: MoERouterWeightedHessian) -> MLXArray? {
        guard reservoir.countsByExpert[expert] >= estimator.minRoutedTokens else { return nil }
        return estimator.hessian(
            activations: concatenated(reservoir.routedX[expert], axis: 0),
            gatingWeights: MLXArray(reservoir.routedC[expert]))
    }

    static func gateMass(_ reservoir: MoERWSwitchReservoir, expert: Int, estimator: MoERouterWeightedHessian) -> Float {
        guard reservoir.countsByExpert[expert] >= estimator.minRoutedTokens else { return 0 }
        return reservoir.routedC[expert].reduce(0) { $0 + $1 * $1 }
    }

    static func inputWhitening(_ reservoir: MoERWSwitchReservoir, estimator: MoERouterWeightedHessian) -> MLXArray? {
        var total: MLXArray?
        for expert in 0 ..< reservoir.numExperts {
            guard let hessian = routedHessian(reservoir, expert: expert, estimator: estimator) else { continue }
            let sum = total.map { $0 + hessian } ?? hessian
            eval(sum)
            total = sum
        }
        return total.map { InputWhitenedSVD.computeWhiteningPair(hessian: $0).s }
    }

    static func summarize(_ rows: [GroupLatentRow], log: ModelRunLog) {
        let format = { (value: Float) in String(format: "%.4f", value) }
        for side in ["in", "out"] {
            let sideRows = rows.filter { $0.side == side }
            for (index, groups) in GroupLatentSpectrum.groupCounts.enumerated() {
                let half = sideRows.map { $0.energies[index][0] }.sorted()
                guard !half.isEmpty else { continue }
                let passing = half.filter { $0 >= 0.97 }.count
                log.record("GROUPLATENT summary", "side \(side) G \(groups) e(0.5d) >= 0.97 on \(passing)/\(half.count) layers, min \(format(half[0])) median \(format(half[half.count / 2])) max \(format(half[half.count - 1]))")
                if side == "in", groups == 8 {
                    let go = Double(passing) >= 0.9 * Double(half.count)
                    log.record("GROUPLATENT verdict", "\(go ? "GO" : "NO-GO") to S1: input side, G 8, l 0.5d, bar 97% on 90% of layers")
                }
            }
        }
    }
}

@Suite("Group-latent MoE S0: router-weighted energy retained by grouped latent projections, from the moe-rw Hessian (#644)", .serialized)
struct GroupLatentMoESpikeTests {
    static func recovers(_ labels: [Int], groups: Int, perGroup: Int) -> Bool {
        let assigned = (0 ..< groups).map { Set(labels[($0 * perGroup) ..< (($0 + 1) * perGroup)]) }
        return assigned.allSatisfy { $0.count == 1 } && Set(assigned.flatMap { $0 }).count == groups
    }

    @Test("Input side: subspace k-means recovers planted expert groups, G = 4 keeps >= 97% at 0.5d where G = 1 does not, and the Gram spectrum matches the SVD")
    func inputSidePlantedGroups() throws {
        try withRandomState(MLXRandom.RandomState(seed: 6440)) {
            let (d, rowsPerExpert, groups, perGroup) = (32, 24, 4, 4)
            let scales = MLXArray((0 ..< d).map { Float(exp(Double($0) * 0.05)) }).reshaped(1, d)
            let x = MLXRandom.normal([96, d]) * scales
            let whitening = InputWhitenedSVD.computeWhiteningPair(hessian: matmul(x.T, x)).s
            let rows = (0 ..< groups).flatMap { _ in
                let basis = MLXRandom.normal([d / 2, d])
                return (0 ..< perGroup).map { _ in
                    let expert = matmul(MLXRandom.normal([rowsPerExpert, d / 2]), basis) + MLXRandom.normal([rowsPerExpert, d]) * Float(0.02)
                    return matmul(expert, whitening)
                }
            }
            eval(rows)
            let found = GroupLatentSpectrum.groupings(count: rows.count, dim: d, rank: d / 2, key: MLXRandom.key(6440)) { rows[$0] }
            let energies = GroupLatentSpectrum.retainedEnergy(found, dim: d, ranks: [d / 2]) { rows[$0] }
            let g4 = try #require(found.firstIndex { $0.groups == groups })
            let sigma: MLXArray = MLXLinalg.svd(concatenated(rows, axis: 0), stream: .cpu)
            let squares = (sigma * sigma).asArray(Float.self)
            let direct = squares.prefix(d / 2).reduce(0, +) / squares.reduce(0, +)

            #expect(Self.recovers(found[g4].labels, groups: groups, perGroup: perGroup))
            #expect(energies[g4][0] >= 0.97)
            #expect(energies[0][0] < energies[g4][0] - 0.05)
            #expect(abs(energies[0][0] - direct) < 1e-3)
        }
    }

    @Test("Output side: per-expert router-weighted whitening of the down rows keeps planted output subspaces, so G = 4 recovers the groups and keeps >= 97% at 0.5d")
    func outputSidePlantedGroups() throws {
        try withRandomState(MLXRandom.RandomState(seed: 6441)) {
            let (d, inter, groups, perGroup) = (32, 8, 4, 3)
            let estimator = MoERouterWeightedHessian()
            let rows = (0 ..< groups).flatMap { _ in
                let basis = MLXRandom.normal([d, d / 4])
                return (0 ..< perGroup).map { _ in
                    let down = matmul(basis, MLXRandom.normal([d / 4, inter])) + MLXRandom.normal([d, inter]) * Float(0.02)
                    let hessian = estimator.hessian(
                        activations: MLXRandom.normal([48, inter]), gatingWeights: abs(MLXRandom.normal([48])) + Float(0.1))
                    return matmul(InputWhitenedSVD.computeWhiteningPair(hessian: hessian).s.T, down.T)
                }
            }
            eval(rows)
            let found = GroupLatentSpectrum.groupings(count: rows.count, dim: d, rank: d / 4, key: MLXRandom.key(6441)) { rows[$0] }
            let energies = GroupLatentSpectrum.retainedEnergy(found, dim: d, ranks: [d / 2]) { rows[$0] }
            let g4 = try #require(found.firstIndex { $0.groups == groups })

            #expect(Self.recovers(found[g4].labels, groups: groups, perGroup: perGroup))
            #expect(energies[g4][0] >= 0.97)
            #expect(energies[0][0] < energies[g4][0] - 0.05)
        }
    }

    @Test(
        "Real-model S0: moe-rw energy retained per MoE layer at l in {0.5d, 0.7d} for G in {1, 4, 8}, input and output side",
        .enabled(if: ProcessInfo.processInfo.environment["BASALT_E2E_644_S0"] != nil)
    )
    func realModelS0LayerSpectrumSweep() async throws {
        let env = ProcessInfo.processInfo.environment
        let path = try #require(env["BASALT_E2E_644_MODEL_PATH"])
        let maxLayers = env["BASALT_E2E_644_MAX_LAYERS"].flatMap { Int($0) } ?? .max
        let log = try await ModelRunLog("grouplatent-s0", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(.conversion)
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let config = MACaQConfig()
        let batches = await GSQRCOConverter.tokenizeCalibration(
            container, corpus: MultilingualMoECalibrationCorpus.defaultText,
            count: config.calibrationBatches, seqLen: config.calibrationSeqLen)
        try #require(!batches.isEmpty)
        log.record("GROUPLATENT", "layer | side | d | G | e(0.5d) | e(0.7d)")
        let rows = try await log.step("sweep") {
            try await container.perform { context in
                try GroupLatentCapture.sweep(context.model, batches: batches, maxLayers: maxLayers, log: log)
            }
        }
        GroupLatentCapture.summarize(rows, log: log)
        #expect(!rows.isEmpty)
        #expect(rows.allSatisfy { $0.energies.joined().allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1.0001 } })
    }
}
