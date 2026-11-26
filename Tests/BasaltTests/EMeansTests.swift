import Foundation
import MLX
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

private enum EMeansTestFixtures {
    static func nextNormal(_ rng: inout SplitMix64) -> Float {
        let scale = 1.0 / Double(1 << 53)
        let u1 = max(1e-12, Double(rng.next() >> 11) * scale)
        let u2 = Double(rng.next() >> 11) * scale
        return Float(sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2))
    }

    static func bimodalSkewedClusters(
        macroClusters: Int,
        perLobe: Int,
        tailPockets: Int,
        perTail: Int,
        dim: Int,
        seed: UInt64
    ) -> MLXArray {
        var rng = SplitMix64(seed: seed)
        let total = macroClusters * 2 * perLobe + tailPockets * perTail
        var values: [Float] = []
        values.reserveCapacity(total * dim)
        var macroCenters: [[Float]] = []
        macroCenters.reserveCapacity(macroClusters)
        for _ in 0 ..< macroClusters {
            let center = (0 ..< dim).map { _ in nextNormal(&rng) * 2.5 }
            let lobe = (0 ..< dim).map { _ in nextNormal(&rng) * 0.5 }
            macroCenters.append(center)
            for sign in [Float(1), Float(-1)] {
                for _ in 0 ..< perLobe {
                    for d in 0 ..< dim {
                        values.append(center[d] + sign * lobe[d] + nextNormal(&rng) * 0.15)
                    }
                }
            }
        }
        for pocket in 0 ..< tailPockets {
            let base = macroCenters[pocket % macroClusters]
            let offset = (0 ..< dim).map { _ in nextNormal(&rng) * 0.275 }
            for _ in 0 ..< perTail {
                for d in 0 ..< dim {
                    values.append(base[d] + offset[d] + nextNormal(&rng) * 0.05)
                }
            }
        }
        let array = MLXArray(values, [total, dim])
        eval(array)
        return array
    }

    static func fitLloyd(points: MLXArray, clusters k: Int, epochs: Int, seed: UInt64) -> MLXArray {
        let x = points.asType(.float32)
        var rng = SplitMix64(seed: seed)
        let initIdx = EMeansCore.sampleUniqueIndices(count: x.dim(0), k: k, rng: &rng)
        var centers = x.take(MLXArray(initIdx), axis: 0)
        let clusterIds = MLXArray((0 ..< k).map(Int32.init)).reshaped(k, 1)
        for _ in 0 ..< epochs {
            let assign = EMeans.nearestIndices(points: x, centroids: centers, precision: .float32)
            let mask = (expandedDimensions(assign, axis: 0) .== clusterIds).asType(.float32)
            let counts = mask.sum(axis: -1, keepDims: true)
            let updated = matmul(mask, x) / maximum(counts, MLXArray(Float(1)))
            centers = which(counts .> MLXArray(Float(0)), updated, centers)
            eval(centers)
        }
        return centers
    }
}

@Suite("Meta FAIR e-means Table 2 scaling laws, cosine decay, and JSONL vector parsing (#836)")
struct EMeansScalingLawAndCoreTests {
    @Test("Table 2 kneeBatchSize follows power-of-two rounding, clamps to [1, N], and switches to the /8 regime above 2M points")
    func kneeBatchSizeFormulasAndLargeRegime() {
        #expect(EMeansScalingLaw.kneeBatchSize(points: 4096, clusters: 32, epochs: 2.5) == 128)
        #expect(EMeansScalingLaw.kneeBatchSize(points: 1_000_000, clusters: 1000, epochs: 2.5) == 32_768)
        #expect(EMeansScalingLaw.kneeBatchSize(points: 10_000_000, clusters: 1000, epochs: 2.5) == 32_768)

        let atBoundary = EMeansScalingLaw.kneeBatchSize(points: 2_000_000, clusters: 1024, epochs: 2.5)
        let aboveBoundary = EMeansScalingLaw.kneeBatchSize(points: 2_000_001, clusters: 1024, epochs: 2.5)
        #expect(atBoundary == 65_536)
        #expect(aboveBoundary == 8_192)
        #expect(atBoundary / aboveBoundary == 8)

        let clampedSmall = EMeansScalingLaw.kneeBatchSize(points: 3, clusters: 2, epochs: 10.0)
        #expect((1 ... 3).contains(clampedSmall))
    }

    @Test("Table 2 peakStep and Eq. 5 cosine stepAlpha satisfy exact analytical values and monotonic decay bounds")
    func peakStepAndCosineScheduleBoundaries() {
        let expectedPeak = Float(
            min(1.0, 0.824 * pow(32.0, -0.46) * pow(128.0, 0.90) * pow(4096.0, -0.45) * pow(2.5, -0.11)))
        let computedPeak = EMeansScalingLaw.peakStep(points: 4096, clusters: 32, batchSize: 128, epochs: 2.5)
        #expect(abs(computedPeak - expectedPeak) < 1e-6)
        #expect(computedPeak > 0 && computedPeak <= 1.0)

        let clampedPeak = EMeansScalingLaw.peakStep(points: 16, clusters: 2, batchSize: 16, epochs: 0.5)
        #expect(clampedPeak == 1.0)

        let m = 80
        let a0: Float = 0.3
        let eps: Float = 0.1
        let start = EMeansScalingLaw.stepAlpha(step: 0, totalSteps: m, peakAlpha: a0, floor: eps)
        let mid = EMeansScalingLaw.stepAlpha(step: m / 2, totalSteps: m, peakAlpha: a0, floor: eps)
        let last = EMeansScalingLaw.stepAlpha(step: m - 1, totalSteps: m, peakAlpha: a0, floor: eps)
        let end = EMeansScalingLaw.stepAlpha(step: m, totalSteps: m, peakAlpha: a0, floor: eps)
        #expect(abs(start - a0) < 1e-6)
        #expect(abs(mid - a0 * (eps + (1.0 - eps) * 0.5)) < 1e-6)
        #expect(last > eps * a0 && last < eps * a0 + 0.005)
        #expect(abs(end - eps * a0) < 1e-6)
        for step in 1 ..< m {
            let prev = EMeansScalingLaw.stepAlpha(step: step - 1, totalSteps: m, peakAlpha: a0, floor: eps)
            let curr = EMeansScalingLaw.stepAlpha(step: step, totalSteps: m, peakAlpha: a0, floor: eps)
            #expect(curr < prev)
        }
        #expect(EMeansScalingLaw.totalSteps(points: 4096, batchSize: 128, epochs: 2.5) == 80)
        #expect(EMeansScalingLaw.totalSteps(points: 10, batchSize: 10, epochs: 0.2) == 1)
    }

    @Test("JSONL vector parser decodes raw float arrays and embedding objects while rejecting malformed rows")
    func jsonlVectorParsingAndValidation() throws {
        let jsonl = """
        [1.0, 2.0, 3.0]
        {"id": "row-2", "embedding": [4.0, 5.0, 6.0]}

        [7.0, 8.0, 9.0]
        """
        let parsed = try EMeansCore.parseJSONLVectors(jsonl)
        #expect(parsed.rows == 3 && parsed.cols == 3)
        #expect(parsed.values == [1, 2, 3, 4, 5, 6, 7, 8, 9])

        #expect(throws: EMeansError.self) {
            _ = try EMeansCore.parseJSONLVectors("[1.0, 2.0]\n[3.0]")
        }
        #expect(throws: EMeansError.self) {
            _ = try EMeansCore.parseJSONLVectors("   \n  ")
        }
        #expect(throws: EMeansError.self) {
            _ = try EMeansCore.parseJSONLVectors("{\"wrong\": [1.0, 2.0]}")
        }
    }
}

@Suite("e-means GPU clustering beats 25-epoch Lloyd k-means in 2.5 epochs and revives dead clusters cleanly (#836)", .serialized)
struct EMeansClusteringConvergenceTests {
    @Test("Dead-cluster revival conserves total count sum, applies Eq. 8 alternating splits, and eliminates dead clusters")
    func deadClusterRevivalCountConservationAndSplitIdentity() {
        let k = 16
        let d = 32
        var rng = SplitMix64(seed: 17)
        let rawCentroids = (0 ..< k * d).map { _ in EMeansTestFixtures.nextNormal(&rng) * 3.0 + 1.0 }
        var centroids = MLXArray(rawCentroids, [k, d])
        let initialCounts: [Float] = [0, 0, 0, 0.01, 0.02, 0.03, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]
        var counts = MLXArray(initialCounts)
        var sums = centroids * expandedDimensions(counts, axis: -1)
        eval(centroids, counts, sums)

        let sumBefore = counts.sum().item(Float.self)
        let revived = EMeans.reviveDeadClusters(
            centroids: &centroids, counts: &counts, sums: &sums,
            threshold: 0.1, delta: EMeansConfig.defaultRevivalDelta, rng: &rng)

        let sumAfter = counts.sum().item(Float.self)
        let meanCount = sumAfter / Float(k)
        #expect(revived == 6)
        #expect(abs(sumAfter - sumBefore) < 1e-3)
        #expect(counts.min().item(Float.self) >= 0.1 * meanCount)
        let reconstructedSums = centroids * expandedDimensions(counts, axis: -1)
        #expect(abs(sums - reconstructedSums).max().item(Float.self) < 1e-3)
    }

    @Test("Dead-cluster revival eliminates starved clusters and lowers MSE on an imbalanced dataset")
    func endToEndRevivalEliminatesDeadClusters() throws {
        let points = EMeansTestFixtures.bimodalSkewedClusters(
            macroClusters: 8, perLobe: 44, tailPockets: 160, perTail: 2, dim: 32, seed: 31)
        var withRevival = EMeansConfig(clusters: 16, epochs: 2.5, seed: 11)
        withRevival.revivalInterval = 5
        withRevival.revivalThreshold = 0.1

        var withoutRevival = withRevival
        withoutRevival.revivalInterval = 0

        let revived = try EMeans.fit(points, config: withRevival)
        let unrevived = try EMeans.fit(points, config: withoutRevival)
        #expect(revived.metrics.revivals > 0)
        #expect(revived.metrics.deadClusters == 0)
        #expect(unrevived.metrics.revivals == 0)
        #expect(revived.metrics.trainMSE < unrevived.metrics.trainMSE)
    }

    @Test("e-means in 2.5 epochs beats 25-epoch full-batch Lloyd k-means MSE on N=4096 D=64 K=32 and is bit-deterministic")
    func eMeansBeats25EpochLloydAndIsDeterministic() throws {
        let points = EMeansTestFixtures.bimodalSkewedClusters(
            macroClusters: 16, perLobe: 100, tailPockets: 224, perTail: 4, dim: 64, seed: 101)
        #expect(points.shape == [4096, 64])

        let config = EMeansConfig(clusters: 32, epochs: 2.5, finalizer: .lastEpochMean, seed: 42)
        let first = try EMeans.fit(points, config: config)
        let second = try EMeans.fit(points, config: config)
        let lloydCentroids = EMeansTestFixtures.fitLloyd(points: points, clusters: 32, epochs: 25, seed: 42)
        let lloydMSE = EMeans.meanSquaredError(points: points, centroids: lloydCentroids)

        #expect(first.centroids.shape == [32, 64])
        #expect(first.counts.shape == [32])
        #expect(first.metrics.batchSize == 128)
        #expect(first.metrics.steps == 80)
        #expect(abs(first.counts.sum().item(Float.self) - 4096) < 1e-2)
        #expect(first.metrics.trainMSE < lloydMSE, "e-means MSE \(first.metrics.trainMSE) did not beat 25-epoch Lloyd MSE \(lloydMSE)")
        #expect(abs(first.centroids - second.centroids).max().item(Float.self) == 0)
        #expect(first.metrics == second.metrics)
    }
}

@Suite("e-means supports all three finalizers, mixed-precision assignment dtypes, and strict input validation (#836)", .serialized)
struct EMeansFinalizerAndPrecisionTests {
    @Test("All three finalizers converge cleanly with zero dead clusters and finite MSE",
          arguments: [EMeansFinalizer.none, .lastEpochMean, .slowEMA(speedMultiplier: 0.25)])
    func allFinalizersConverge(finalizer: EMeansFinalizer) throws {
        let points = EMeansTestFixtures.bimodalSkewedClusters(
            macroClusters: 4, perLobe: 48, tailPockets: 16, perTail: 2, dim: 32, seed: 73)
        let config = EMeansConfig(clusters: 8, epochs: 2.5, finalizer: finalizer, seed: 7)
        let result = try EMeans.fit(points, config: config)
        #expect(result.centroids.shape == [8, 32])
        #expect(result.centroids.dtype == .float32)
        #expect(result.metrics.trainMSE.isFinite && result.metrics.trainMSE > 0)
        #expect(result.metrics.deadClusters == 0)
        #expect(abs(result.counts.sum().item(Float.self) - Float(points.dim(0))) < 1e-2)
    }

    @Test("All three assignment GEMM dtypes produce consistent float32 centroids and finite MSE",
          arguments: EMeansPrecision.allCases)
    func allAssignmentPrecisionsConverge(precision: EMeansPrecision) throws {
        let points = EMeansTestFixtures.bimodalSkewedClusters(
            macroClusters: 4, perLobe: 48, tailPockets: 16, perTail: 2, dim: 32, seed: 79)
        let config = EMeansConfig(clusters: 8, epochs: 2.5, finalizer: .lastEpochMean, seed: 9, assignmentDType: precision)
        let result = try EMeans.fit(points, config: config)
        #expect(result.centroids.shape == [8, 32])
        #expect(result.centroids.dtype == .float32)
        #expect(result.metrics.trainMSE.isFinite && result.metrics.trainMSE > 0)
        let assign = EMeans.nearestIndices(points: points, centroids: result.centroids, precision: precision)
        #expect(assign.shape == [points.dim(0)] && assign.dtype == .int32)
    }

    @Test("EMeans.fit rejects invalid tensor rank, cluster counts, and hyperparameters with typed EMeansError")
    func fitRejectsInvalidInputs() {
        let matrix = MLXArray.zeros([16, 8], dtype: .float32)
        let vector = MLXArray.zeros([16], dtype: .float32)
        #expect(throws: EMeansError.invalidRank(expected: 2, actual: 1)) {
            _ = try EMeans.fit(vector, config: EMeansConfig(clusters: 4))
        }
        #expect(throws: EMeansError.invalidClusterCount(clusters: 0, points: 16)) {
            _ = try EMeans.fit(matrix, config: EMeansConfig(clusters: 0))
        }
        #expect(throws: EMeansError.invalidClusterCount(clusters: 32, points: 16)) {
            _ = try EMeans.fit(matrix, config: EMeansConfig(clusters: 32))
        }
        #expect(throws: EMeansError.invalidEpochs(0)) {
            _ = try EMeans.fit(matrix, config: EMeansConfig(clusters: 4, epochs: 0))
        }
        var badPeak = EMeansConfig(clusters: 4)
        badPeak.peakStep = 1.5
        #expect(throws: EMeansError.self) {
            _ = try EMeans.fit(matrix, config: badPeak)
        }
        #expect(throws: EMeansError.self) {
            _ = try EMeans.fit(matrix, config: EMeansConfig(clusters: 4, finalizer: .slowEMA(speedMultiplier: 0)))
        }
    }

    @Test("Cluster counts above 64 exercise the scatter-add accumulation path cleanly")
    func largeClusterCountScatterAddPath() throws {
        let points = EMeansTestFixtures.bimodalSkewedClusters(
            macroClusters: 12, perLobe: 12, tailPockets: 0, perTail: 0, dim: 8, seed: 83)
        let config = EMeansConfig(clusters: 72, epochs: 2.5, finalizer: .lastEpochMean, seed: 5)
        let result = try EMeans.fit(points, config: config)
        #expect(result.centroids.shape == [72, 8])
        #expect(result.counts.shape == [72])
        #expect(result.metrics.trainMSE.isFinite && result.metrics.trainMSE >= 0)
        #expect(abs(result.counts.sum().item(Float.self) - Float(points.dim(0))) < 1e-2)
    }
}

@Suite("RQ-KV e-means codebook integration and online dead-cluster revival (#836)", .serialized)
struct EMeansRQKVTests {
    @Test("CentroidCodebook.fitEMeans trains per-head RQ-KV codebooks and lowers 2-bit key MSE below direct affine",
          arguments: [4, 8, 16])
    func centroidCodebookFitEMeansAndRQKVRoundTrip(clusters: Int) throws {
        let (keys, _) = CentroidRQKVTests.anisotropicKeys(clusters: clusters, tokens: 256, heads: 2, dim: 64, seed: 53)
        let book = CentroidCodebook.fitEMeans(
            keys: keys, clusters: clusters, epochs: 2.5,
            finalizer: .slowEMA(speedMultiplier: 0.25), seed: 13)
        #expect(book.centroids.shape == [1, 2, clusters, 64])
        #expect(book.counts.shape == [1, 2, clusters, 1])

        var direct = ZonedKVCache.QuantizedRows(bits: 2, groupSize: 32)
        direct.append(keys)
        let directDecoded = try #require(direct.dequantizedRows())

        var rq = CentroidRQKVRows(clusters: clusters, bits: 2, groupSize: 32, codebook: book)
        rq.append(keys)
        let rqDecoded = try #require(rq.dequantizedRows(dtype: .float32))
        #expect(CentroidRQKVTests.mse(rqDecoded, keys) < CentroidRQKVTests.mse(directDecoded, keys))

        let grouped = MLXRandom.normal([1, 2, 2, 1, 64], key: MLXRandom.key(29))
        let expectedScores = matmul(grouped, expandedDimensions(rqDecoded, axis: 2).swappedAxes(-1, -2))
        #expect(abs(try #require(rq.scores(grouped)) - expectedScores).max().item(Float.self) < 1e-4)
    }

    @Test("CentroidCodebook.updateOnline revives dead centroids per head when reviveDeadThreshold is set")
    func centroidCodebookOnlineUpdateRevivesDeadClusters() {
        let (keys, _) = CentroidRQKVTests.anisotropicKeys(clusters: 8, tokens: 192, heads: 2, dim: 32, seed: 61)
        let base = CentroidCodebook.fitEMeans(keys: keys, clusters: 8, epochs: 2.5, finalizer: .lastEpochMean, seed: 5)
        let poisonedCentroids = base.centroids + MLXArray(Float(0))
        let poisonedCounts = MLXArray.ones([1, 2, 8, 1], dtype: .float32) * Float(24)
        let farOut = MLXArray.ones([1, 2, 2, 32], dtype: .float32) * Float(500)
        poisonedCentroids[0..., 0..., 6 ..< 8, 0...] = farOut
        poisonedCounts[0..., 0..., 6 ..< 8, 0...] = MLXArray.ones([1, 2, 2, 1], dtype: .float32) * Float(0.001)
        eval(poisonedCentroids, poisonedCounts)

        var book = CentroidCodebook(clusters: 8, centroids: poisonedCentroids, counts: poisonedCounts)
        book.updateOnline(keys, decay: 0.5, reviveDeadThreshold: 0.1)
        #expect(book.counts.min().item(Float.self) > 1.0)
        #expect(abs(book.centroids).max().item(Float.self) < 100.0)
    }
}

@Suite("CLI emeans command parsing, validation refusal, and safetensors/jsonl file round-trip (#836)", .serialized)
struct EMeansCLITests {
    @Test("EMeansCommand parses valid arguments and refuses invalid cluster, epoch, and override values")
    func cliParsingAndRefusal() throws {
        let parsed = try EMeansCommand.parse([
            "--input", "vectors.safetensors", "--clusters", "32", "--epochs", "2.5",
            "--output", "out.safetensors", "--finalizer", "slow-ema",
            "--slow-speed", "0.25", "--dtype", "bfloat16", "--seed", "7",
        ])
        let cfg = parsed.makeConfig()
        #expect(cfg.clusters == 32 && cfg.epochs == 2.5 && cfg.seed == 7)
        #expect(cfg.finalizer == .slowEMA(speedMultiplier: 0.25) && cfg.assignmentDType == .bfloat16)

        for badArgs in [
            ["--input", "a.safetensors", "--clusters", "0", "--output", "o.safetensors"],
            ["--input", "a.safetensors", "--clusters", "8", "--epochs", "0", "--output", "o.safetensors"],
            ["--input", "a.safetensors", "--clusters", "8", "--batch-size", "0", "--output", "o.safetensors"],
            ["--input", "a.safetensors", "--clusters", "8", "--peak-step", "2.0", "--output", "o.safetensors"],
            ["--input", "a.safetensors", "--clusters", "8", "--slow-speed", "0", "--output", "o.safetensors"],
            ["--input", "a.safetensors", "--clusters", "8", "--output", "o.json"],
        ] {
            #expect(throws: (any Error).self) {
                _ = try EMeansCommand.parse(badArgs)
            }
        }
    }

    @Test("EMeansCommand and EMeans.fitFile round-trip safetensors and jsonl vector files cleanly")
    func cliSafetensorsAndJsonlRoundTrip() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(component: "emeans-cli-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let points = EMeansTestFixtures.bimodalSkewedClusters(
            macroClusters: 2, perLobe: 16, tailPockets: 4, perTail: 2, dim: 8, seed: 19)
        let stIn = dir.appending(component: "input.safetensors")
        let stOut = dir.appending(component: "out_st.safetensors")
        try MLX.save(arrays: ["embeddings": points], url: stIn)
        var stCmd = try EMeansCommand.parse([
            "--input", stIn.path, "--key", "embeddings", "--clusters", "4",
            "--epochs", "2.5", "--output", stOut.path,
        ])
        try await stCmd.run()
        let loadedST = try MLX.loadArrays(url: stOut)
        #expect(loadedST["centroids"]?.shape == [4, 8])
        #expect(loadedST["counts"]?.shape == [4])

        let jsonlIn = dir.appending(component: "input.jsonl")
        let jsonlOut = dir.appending(component: "out_jsonl.safetensors")
        let host = points.asArray(Float.self)
        var lines: [String] = []
        for row in 0 ..< points.dim(0) {
            let slice = (0 ..< 8).map { String(host[row * 8 + $0]) }.joined(separator: ",")
            lines.append(row.isMultiple(of: 2) ? "[\(slice)]" : "{\"embedding\":[\(slice)]}")
        }
        try lines.joined(separator: "\n").write(to: jsonlIn, atomically: true, encoding: .utf8)
        let report = try EMeans.fitFile(
            inputURL: jsonlIn, outputURL: jsonlOut, config: EMeansConfig(clusters: 4, epochs: 2.5, seed: 3))
        #expect(report.points == points.dim(0) && report.dimensions == 8 && report.vectorsPerSecond > 0)
        let loadedJSONL = try MLX.loadArrays(url: jsonlOut)
        #expect(loadedJSONL["centroids"]?.shape == [4, 8])
    }
}
