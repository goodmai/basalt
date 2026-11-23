import Foundation
import MLX
import Testing
@testable import BasaltInference

struct GroupLatentBudget: Sendable {
    let model: String
    let hiddenDim: Int
    let numExperts: Int
    let intermediateDim: Int
    let latentDim: Int
    let groups: Int

    var denseExpertParams: Int {
        3 * numExperts * intermediateDim * hiddenDim
    }

    var compressedExpertParams: Int {
        3 * numExperts * intermediateDim * latentDim
    }

    var projectionOverheadParams: Int {
        groups * 2 * hiddenDim * latentDim
    }

    var totalCompressedParams: Int {
        compressedExpertParams + projectionOverheadParams
    }

    var compressionRatio: Double {
        Double(denseExpertParams) / Double(totalCompressedParams)
    }
}

private enum GroupLatentSpectrum {
    static func whitenedRows(_ experts: [MLXArray], hessian: MLXArray, epsilonFloor: Float = 1e-10) -> (
        whitened: [MLXArray], s: MLXArray, sInv: MLXArray
    ) {
        let (s, sInv) = InputWhitenedSVD.computeWhiteningPair(hessian: hessian, epsilonFloor: epsilonFloor)
        let whitened = experts.map { matmul($0.asType(.float32), s) }
        eval(whitened)
        return (whitened, s, sInv)
    }

    static func subspaceSignatures(whitenedExperts: [MLXArray], subspaceRank: Int) -> [[Float]] {
        whitenedExperts.map { mat in
            let (_, _, vt) = MLXLinalg.svd(mat, stream: .cpu)
            eval(vt)
            let r = min(max(1, subspaceRank), vt.dim(0))
            let vr = vt[0 ..< r, 0...]
            let proj = matmul(vr.T, vr)
            eval(proj)
            let raw = proj.asArray(Float.self)
            let norm = max(raw.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot(), 1e-12)
            return raw.map { $0 / norm }
        }
    }

    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var acc: Float = 0
        for i in 0 ..< a.count {
            acc += a[i] * b[i]
        }
        return acc
    }

    static func initializeCentroids(signatures: [[Float]], groups: Int) -> [[Float]] {
        var centroids: [[Float]] = [signatures[0]]
        while centroids.count < groups {
            var bestIdx = 0
            var lowestMaxSim = Float.greatestFiniteMagnitude
            for (idx, sig) in signatures.enumerated() {
                let maxSim = centroids.map { dot(sig, $0) }.max() ?? -1
                if maxSim < lowestMaxSim {
                    lowestMaxSim = maxSim
                    bestIdx = idx
                }
            }
            centroids.append(signatures[bestIdx])
        }
        return centroids
    }

    static func balancedAssign(signatures: [[Float]], centroids: [[Float]]) -> [Int] {
        let count = signatures.count
        let groups = centroids.count
        let baseCap = count / groups
        let extra = count % groups
        var remaining = (0 ..< groups).map { $0 < extra ? baseCap + 1 : baseCap }
        let scored: [(idx: Int, sims: [(g: Int, sim: Float)], margin: Float)] = signatures.enumerated().map { idx, sig in
            let ranked = centroids.enumerated().map { (g: $0.offset, sim: dot(sig, $0.element)) }
                .sorted { $0.sim > $1.sim }
            let margin = ranked.count > 1 ? ranked[0].sim - ranked[1].sim : ranked[0].sim
            return (idx, ranked, margin)
        }.sorted { $0.margin > $1.margin }

        var labels = [Int](repeating: 0, count: count)
        for item in scored {
            if let chosen = item.sims.first(where: { remaining[$0.g] > 0 }) {
                labels[item.idx] = chosen.g
                remaining[chosen.g] -= 1
            }
        }
        return labels
    }

    static func updateCentroids(signatures: [[Float]], labels: [Int], groups: Int, fallback: [[Float]]) -> [[Float]] {
        let dim = signatures[0].count
        var sums = [[Float]](repeating: [Float](repeating: 0, count: dim), count: groups)
        var counts = [Int](repeating: 0, count: groups)
        for (idx, sig) in signatures.enumerated() {
            let g = labels[idx]
            counts[g] += 1
            for j in 0 ..< dim {
                sums[g][j] += sig[j]
            }
        }
        return (0 ..< groups).map { g in
            guard counts[g] > 0 else { return fallback[g] }
            let norm = max(sums[g].reduce(Float(0)) { $0 + $1 * $1 }.squareRoot(), 1e-12)
            return sums[g].map { $0 / norm }
        }
    }

    static func equalSizeSubspaceKMeans(
        whitenedExperts: [MLXArray],
        groups: Int,
        subspaceRank: Int,
        iterations: Int = 8
    ) -> [Int] {
        let count = whitenedExperts.count
        guard groups > 1, count >= groups else {
            return [Int](repeating: 0, count: count)
        }
        let signatures = subspaceSignatures(whitenedExperts: whitenedExperts, subspaceRank: subspaceRank)
        var centroids = initializeCentroids(signatures: signatures, groups: groups)
        var labels = [Int](repeating: 0, count: count)
        for _ in 0 ..< iterations {
            labels = balancedAssign(signatures: signatures, centroids: centroids)
            centroids = updateCentroids(signatures: signatures, labels: labels, groups: groups, fallback: centroids)
        }
        return labels
    }

    static func groupedRetainedEnergy(
        whitenedExperts: [MLXArray],
        groupLabels: [Int],
        groups: Int,
        rank: Int
    ) -> Float {
        var kept: Float = 0
        var total: Float = 0
        for g in 0 ..< groups {
            let members = whitenedExperts.enumerated().compactMap { groupLabels[$0.offset] == g ? $0.element : nil }
            guard !members.isEmpty else { continue }
            let stacked = concatenated(members, axis: 0)
            let (_, sigma, _) = MLXLinalg.svd(stacked, stream: .cpu)
            eval(sigma)
            let sVals = sigma.asArray(Float.self)
            let r = min(max(1, rank), sVals.count)
            for (idx, val) in sVals.enumerated() {
                let e = val * val
                total += e
                if idx < r {
                    kept += e
                }
            }
        }
        return kept / max(total, 1e-12)
    }

    static func groupedMaxReconstructionError(
        experts: [MLXArray],
        x: MLXArray,
        hessian: MLXArray,
        groupLabels: [Int],
        groups: Int,
        rank: Int
    ) -> Float {
        let (whitened, _, sInv) = whitenedRows(experts, hessian: hessian)
        var maxErr: Float = 0
        for g in 0 ..< groups {
            let indices = experts.indices.filter { groupLabels[$0] == g }
            guard !indices.isEmpty else { continue }
            let stacked = concatenated(indices.map { whitened[$0] }, axis: 0)
            let (_, _, vt) = MLXLinalg.svd(stacked, stream: .cpu)
            eval(vt)
            let r = min(max(1, rank), vt.dim(0))
            let vtR = vt[0 ..< r, 0...]
            let vR = vtR.T
            let downG = matmul(vtR, sInv)
            let projectedX = matmul(x.asType(.float32), downG.T)
            for idx in indices {
                let coreE = matmul(whitened[idx], vR)
                let approx = matmul(projectedX, coreE.T)
                let exact = matmul(x.asType(.float32), experts[idx].asType(.float32).T)
                eval(approx, exact)
                maxErr = max(maxErr, abs(exact - approx).max().item(Float.self))
            }
        }
        return maxErr
    }
}

private enum GroupLatentSafetensorsProbe {
    static func readQuantizationSpec(modelDir: URL) -> (bits: Int, groupSize: Int) {
        let cfgURL = modelDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (4, 64)
        }
        let q = (json["quantization"] as? [String: Any]) ?? (json["quantization_config"] as? [String: Any])
        let bits = (q?["bits"] as? Int) ?? 4
        let groupSize = (q?["group_size"] as? Int) ?? 64
        return (bits, groupSize)
    }

    static func loadStackedExpertGateUp(
        modelDir: URL,
        layerIndex: Int
    ) throws -> (gateUp: [MLXArray], downTransposed: [MLXArray])? {
        let qSpec = readQuantizationSpec(modelDir: modelDir)
        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let prefixes = [
            "model.layers.\(layerIndex).mlp.switch_mlp.",
            "language_model.model.layers.\(layerIndex).mlp.switch_mlp.",
            "model.layers.\(layerIndex).experts.",
        ]
        for file in files {
            let tensors = try MLX.loadArrays(url: file)
            for prefix in prefixes {
                guard let gate = dequantizeIfNeeded(tensors: tensors, key: "\(prefix)gate_proj", bits: qSpec.bits, groupSize: qSpec.groupSize),
                      let up = dequantizeIfNeeded(tensors: tensors, key: "\(prefix)up_proj", bits: qSpec.bits, groupSize: qSpec.groupSize),
                      let down = dequantizeIfNeeded(tensors: tensors, key: "\(prefix)down_proj", bits: qSpec.bits, groupSize: qSpec.groupSize),
                      gate.ndim == 3, up.ndim == 3, down.ndim == 3 else {
                    continue
                }
                let eCount = gate.dim(0)
                let gateUp = (0 ..< eCount).map { concatenated([gate[$0], up[$0]], axis: 0) }
                let downT = (0 ..< eCount).map { down[$0].T }
                eval(gateUp)
                eval(downT)
                return (gateUp, downT)
            }
        }
        return nil
    }

    static func dequantizeIfNeeded(tensors: [String: MLXArray], key: String, bits: Int, groupSize: Int) -> MLXArray? {
        guard let weight = tensors["\(key).weight"] else { return nil }
        guard let scales = tensors["\(key).scales"] else { return weight.asType(.float32) }
        let biases = tensors["\(key).biases"]
        return MLX.dequantized(weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: .affine)
            .asType(.float32)
    }
}

struct GroupLatentBudgetCase: Sendable {
    let budget: GroupLatentBudget
    let expectedRatio: Double
}

@Suite("Group-Latent MoE S0 whitened spectrum, balanced subspace k-means, and byte budgets (#644)", .serialized)
struct GroupLatentMoESpikeTests {
    @Test(
        "Compression ratio arithmetic matches #644 table for Qwen3.6-35B-A3B and Gemma 4 26B-A4B",
        arguments: [
            GroupLatentBudgetCase(budget: .init(model: "Qwen3.6-35B-A3B", hiddenDim: 2048, numExperts: 256, intermediateDim: 512, latentDim: 1408, groups: 8), expectedRatio: 1.34),
            GroupLatentBudgetCase(budget: .init(model: "Qwen3.6-35B-A3B", hiddenDim: 2048, numExperts: 256, intermediateDim: 512, latentDim: 1024, groups: 8), expectedRatio: 1.85),
            GroupLatentBudgetCase(budget: .init(model: "Qwen3.6-35B-A3B", hiddenDim: 2048, numExperts: 256, intermediateDim: 512, latentDim: 1024, groups: 1), expectedRatio: 1.98),
            GroupLatentBudgetCase(budget: .init(model: "Gemma 4 26B-A4B", hiddenDim: 2816, numExperts: 128, intermediateDim: 704, latentDim: 1920, groups: 8), expectedRatio: 1.26),
            GroupLatentBudgetCase(budget: .init(model: "Gemma 4 26B-A4B", hiddenDim: 2816, numExperts: 128, intermediateDim: 704, latentDim: 1408, groups: 8), expectedRatio: 1.71),
            GroupLatentBudgetCase(budget: .init(model: "Gemma 4 26B-A4B", hiddenDim: 2816, numExperts: 128, intermediateDim: 704, latentDim: 1408, groups: 4), expectedRatio: 1.85),
        ]
    )
    func compressionBudgetMatchesSpecTable(_ item: GroupLatentBudgetCase) {
        let budget = item.budget
        #expect(abs(budget.compressionRatio - item.expectedRatio) < 0.01)
        let plainFourToThreeAffineRatio = (4.0 + 32.0 / 64.0) / (3.0 + 32.0 / 64.0)
        if budget.latentDim * 2 == budget.hiddenDim {
            #expect(budget.compressionRatio > plainFourToThreeAffineRatio)
        }
    }

    @Test("Balanced subspace k-means recovers latent expert groups and crosses the 97% S0 energy bar at l = 0.5d")
    func balancedSubspaceKMeansRecoversGroupedSpectrum() {
        MLXRandom.seed(6440)
        let (d, rowsPerExpert, numGroups, expertsPerGroup) = (32, 12, 4, 4)
        let halfRank = d / 2
        let sevenTenthsRank = Int((Double(d) * 0.7).rounded())
        let scales = MLXArray((0 ..< d).map { Float(exp(Double($0) * 0.05)) }).reshaped(1, d)
        let x = MLXRandom.normal([96, d], dtype: .float32) * scales
        let h = matmul(x.T, x)

        var experts: [MLXArray] = []
        for _ in 0 ..< numGroups {
            let basis = MLXRandom.normal([halfRank, d], dtype: .float32)
            for _ in 0 ..< expertsPerGroup {
                let coeffs = MLXRandom.normal([rowsPerExpert, halfRank], dtype: .float32)
                let noise = MLXRandom.normal([rowsPerExpert, d], dtype: .float32) * Float(0.02)
                experts.append(matmul(coeffs, basis) + noise)
            }
        }
        eval(experts)

        let (whitened, _, _) = GroupLatentSpectrum.whitenedRows(experts, hessian: h)
        let labelsG1 = GroupLatentSpectrum.equalSizeSubspaceKMeans(whitenedExperts: whitened, groups: 1, subspaceRank: 4)
        let labelsG4 = GroupLatentSpectrum.equalSizeSubspaceKMeans(whitenedExperts: whitened, groups: numGroups, subspaceRank: 4)

        for g in 0 ..< numGroups {
            #expect(labelsG4.filter { $0 == g }.count == expertsPerGroup)
        }

        let energyG1Half = GroupLatentSpectrum.groupedRetainedEnergy(whitenedExperts: whitened, groupLabels: labelsG1, groups: 1, rank: halfRank)
        let energyG4Half = GroupLatentSpectrum.groupedRetainedEnergy(whitenedExperts: whitened, groupLabels: labelsG4, groups: numGroups, rank: halfRank)
        let energyG4Seven = GroupLatentSpectrum.groupedRetainedEnergy(whitenedExperts: whitened, groupLabels: labelsG4, groups: numGroups, rank: sevenTenthsRank)

        #expect(energyG4Half >= 0.97)
        #expect(energyG4Half > energyG1Half + 0.05)
        #expect(energyG4Seven >= energyG4Half)
    }

    @Test("Output-side down_proj whitened spectrum and full-rank grouped D_g/U_g reconstruction are exact at l = d")
    func outputSideSpectrumAndFullRankExactness() {
        MLXRandom.seed(6441)
        let (d, inter, numExperts, groups) = (24, 10, 8, 4)
        let x = MLXRandom.normal([32, d], dtype: .float32)
        let h = matmul(x.T, x)
        let downTransposed = (0 ..< numExperts).map { _ in MLXRandom.normal([inter, d], dtype: .float32) }
        eval(downTransposed)

        let (whitened, _, _) = GroupLatentSpectrum.whitenedRows(downTransposed, hessian: h)
        let labels = GroupLatentSpectrum.equalSizeSubspaceKMeans(whitenedExperts: whitened, groups: groups, subspaceRank: 4)
        let fullEnergy = GroupLatentSpectrum.groupedRetainedEnergy(whitenedExperts: whitened, groupLabels: labels, groups: groups, rank: d)
        let fullErr = GroupLatentSpectrum.groupedMaxReconstructionError(
            experts: downTransposed,
            x: x,
            hessian: h,
            groupLabels: labels,
            groups: groups,
            rank: d
        )
        #expect(abs(fullEnergy - 1.0) < 1e-5)
        #expect(fullErr < 1e-3)
    }

    @Test(
        "Real-model S0 layer-by-layer safetensors spectrum sweep reports GO/NO-GO at l = 0.5d and G = 8",
        .enabled(if: ProcessInfo.processInfo.environment["BASALT_E2E_644_S0"] != nil)
    )
    func realModelS0LayerSpectrumSweep() throws {
        let path = try #require(ProcessInfo.processInfo.environment["BASALT_E2E_644_MODEL_PATH"])
        let modelDir = URL(fileURLWithPath: path)
        let maxLayers = Int(ProcessInfo.processInfo.environment["BASALT_E2E_644_MAX_LAYERS"] ?? "4") ?? 4
        var passingLayers = 0
        var evaluatedLayers = 0
        for layer in 0 ..< maxLayers {
            guard let pair = try GroupLatentSafetensorsProbe.loadStackedExpertGateUp(modelDir: modelDir, layerIndex: layer) else {
                continue
            }
            let d = pair.gateUp[0].dim(1)
            let identityH = MLXArray.eye(d, dtype: .float32)
            let (whitened, _, _) = GroupLatentSpectrum.whitenedRows(pair.gateUp, hessian: identityH)
            let labelsG8 = GroupLatentSpectrum.equalSizeSubspaceKMeans(whitenedExperts: whitened, groups: 8, subspaceRank: 4)
            let eHalfG8 = GroupLatentSpectrum.groupedRetainedEnergy(whitenedExperts: whitened, groupLabels: labelsG8, groups: 8, rank: d / 2)
            let eSevenG8 = GroupLatentSpectrum.groupedRetainedEnergy(whitenedExperts: whitened, groupLabels: labelsG8, groups: 8, rank: Int((Double(d) * 0.7).rounded()))
            print("[S0 #644] layer=\(layer) d=\(d) G=8 e(0.5d)=\(String(format: "%.4f", eHalfG8)) e(0.7d)=\(String(format: "%.4f", eSevenG8))")
            evaluatedLayers += 1
            if eHalfG8 >= 0.97 {
                passingLayers += 1
            }
        }
        #expect(evaluatedLayers > 0)
        print("[S0 #644] passingLayers=\(passingLayers)/\(evaluatedLayers)")
    }
}
