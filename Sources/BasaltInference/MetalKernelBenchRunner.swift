import BasaltCore
import Foundation
import MLX

public enum MetalKernelBenchProfile: String, Sendable, CaseIterable {
    case production
    case smoke
}

public enum MetalKernelBenchRunner {
    struct TimingSpec: Sendable {
        let iterations: Int
        let warmup: Int
        let callsPerSample: Int
    }

    struct ParityResult: Sendable {
        let maxDiff: Double
        let tolerance: Double
        let passed: Bool
    }

    public static func runAll(
        profile: MetalKernelBenchProfile = .production,
        iterations: Int = 11,
        warmup: Int = 5,
        metadata: MetalKernelBenchMetadata
    ) -> MetalKernelBenchReport {
        if profile == .production {
            warmUpGPU()
        }
        let passes = profile == .production ? 3 : 1
        var passResults: [[MetalKernelBenchRow]] = []
        passResults.reserveCapacity(passes)
        for _ in 0 ..< passes {
            passResults.append(runSinglePass(profile: profile, iterations: iterations, warmup: warmup))
        }
        return MetalKernelBenchReport(metadata: metadata, rows: mergePasses(passResults))
    }

    private static func runSinglePass(
        profile: MetalKernelBenchProfile,
        iterations: Int,
        warmup: Int
    ) -> [MetalKernelBenchRow] {
        let repeats = profile == .production ? max(7, iterations) : max(3, iterations)
        let matmulSpec = TimingSpec(iterations: repeats, warmup: warmup, callsPerSample: profile == .production ? 16 : 2)
        let denseSpec = TimingSpec(iterations: repeats, warmup: warmup, callsPerSample: profile == .production ? 64 : 2)
        let routerSpec = TimingSpec(iterations: repeats, warmup: warmup, callsPerSample: profile == .production ? 128 : 2)
        let routerBatch = profile == .production ? 256 : 64
        let nf4K = profile == .production ? 5120 : 512
        let nf4N = profile == .production ? 17408 : 512
        let cacheDefeat = profile == .production ? (256 * 1024 * 1024) : 0
        let denseN = profile == .production ? 4096 : 512

        var rows: [MetalKernelBenchRow] = []
        for m in [1, 2, 4, 8, 16] {
            rows.append(benchNF4(m: m, k: nf4K, n: nf4N, cacheDefeatBytes: cacheDefeat, spec: matmulSpec))
            Memory.clearCache()
        }
        rows.append(benchNF4Gather(profile: profile, spec: matmulSpec))
        Memory.clearCache()
        for type in [GGMLType.q4K, .iq4XS, .iq3XXS] {
            rows.append(benchIQ(type: type, m: 1, k: nf4K, n: nf4N, spec: matmulSpec))
            Memory.clearCache()
        }
        for type in [GGMLType.q4K, .iq4XS] {
            rows.append(benchIQ(type: type, m: 4, k: nf4K, n: nf4N, spec: matmulSpec))
            Memory.clearCache()
        }
        rows.append(benchFusedMoERouter(batch: routerBatch, spec: routerSpec))
        rows.append(benchFlintRouter(batch: routerBatch, spec: routerSpec))
        rows.append(benchThreeBitQMV(k: denseN, n: denseN, spec: denseSpec))
        Memory.clearCache()
        rows.append(benchQTIP(k: denseN, n: denseN, spec: denseSpec))
        Memory.clearCache()
        return rows
    }

    private static func mergePasses(_ passes: [[MetalKernelBenchRow]]) -> [MetalKernelBenchRow] {
        guard passes.count > 1, let first = passes.first else { return passes.first ?? [] }
        return first.indices.map { idx in
            let group = passes.map { $0[idx] }
            let rawMedians = group.map(\.timing.medianMs)
            let sortedByMedian = group.sorted { $0.timing.medianMs < $1.timing.medianMs }
            let best = sortedByMedian[0]
            let second = sortedByMedian[1]
            let spreadMs = max(0.0, second.timing.medianMs - best.timing.medianMs)
            let worstDiff = group.map(\.parity.maxAbsDiff).max() ?? best.parity.maxAbsDiff
            let allPassed = group.allSatisfy(\.parity.passed)
            return MetalKernelBenchRow(
                name: best.name,
                family: best.family,
                shape: best.shape,
                timing: MetalKernelBenchTiming(
                    medianMs: best.timing.medianMs,
                    iqrMs: spreadMs,
                    effectiveGBps: best.timing.effectiveGBps,
                    passMediansMs: rawMedians
                ),
                parity: MetalKernelBenchParity(
                    maxAbsDiff: worstDiff,
                    tolerance: best.parity.tolerance,
                    passed: allPassed
                ),
                gate: best.gate
            )
        }
    }

    static func measureBatched(spec: TimingSpec, callsPerSample: Int? = nil, _ block: () -> Void) -> (median: Double, iqr: Double) {
        let clock = ContinuousClock()
        for _ in 0 ..< max(1, spec.warmup) { block() }
        var samples: [Double] = []
        samples.reserveCapacity(max(1, spec.iterations))
        let divisor = Double(max(1, callsPerSample ?? spec.callsPerSample))
        for _ in 0 ..< max(1, spec.iterations) {
            let start = clock.now
            block()
            let elapsed = start.duration(to: clock.now)
            let totalMs = Double(elapsed.components.seconds) * 1e3
                + Double(elapsed.components.attoseconds) / 1e15
            samples.append(totalMs / divisor)
        }
        return MetalKernelBenchMath.medianAndIQR(samples)
    }

    static func fullRandomUInt32(shape: [Int], seed: UInt64) -> MLXArray {
        let lo = MLXRandom.randInt(low: 0, high: 65536, shape, key: MLXRandom.key(seed)).asType(.uint32)
        let hi = MLXRandom.randInt(low: 0, high: 65536, shape, key: MLXRandom.key(seed &+ 1)).asType(.uint32)
        return bitwiseOr(lo, leftShift(hi, MLXArray(UInt32(16))))
    }

    static func warmUpGPU() {
        let a = MLXRandom.normal([2048, 2048], key: MLXRandom.key(42)).asType(.bfloat16)
        let b = MLXRandom.normal([2048, 2048], key: MLXRandom.key(43)).asType(.bfloat16)
        eval(a, b)
        let deadline = ContinuousClock.now + .seconds(2)
        while ContinuousClock.now < deadline {
            eval(matmul(a, b))
        }
        Memory.clearCache()
    }

    static func makeRow(
        name: String,
        family: String,
        shape: String,
        timing: MetalKernelBenchTiming,
        parity: ParityResult,
        gate: MetalKernelBenchGate = .gated
    ) -> MetalKernelBenchRow {
        let safeDiff = parity.maxDiff.isFinite ? parity.maxDiff : 1e9
        return MetalKernelBenchRow(
            name: name,
            family: family,
            shape: shape,
            timing: timing,
            parity: MetalKernelBenchParity(
                maxAbsDiff: safeDiff,
                tolerance: parity.tolerance,
                passed: parity.passed && parity.maxDiff.isFinite && parity.maxDiff <= parity.tolerance
            ),
            gate: gate
        )
    }

    static func timingStats(_ raw: (median: Double, iqr: Double), bytes: Double) -> MetalKernelBenchTiming {
        MetalKernelBenchTiming(
            medianMs: raw.median,
            iqrMs: raw.iqr,
            effectiveGBps: MetalKernelBenchMath.effectiveGBps(bytes: bytes, medianMs: raw.median)
        )
    }

    static func relMaxDiff(_ got: MLXArray, _ ref: MLXArray) -> Double {
        let num = abs(got - ref).max()
        let den = maximum(abs(ref).max(), MLXArray(Float(1e-6)))
        eval(num, den)
        return Double((num / den).item(Float.self))
    }

    static func makeIQPacked(type: GGMLType, rows: Int, k: Int, seed: UInt64) -> MLXArray {
        let (be, bb) = type.block
        let blocksPerRow = k / be
        let totalBlocks = rows * blocksPerRow
        var bytes = [UInt8](repeating: 0, count: totalBlocks * bb)
        var state: UInt64 = seed &* 6364136223846793005 &+ 1442695040888963407
        let dBits = Float16(0.01).bitPattern
        let dLo = UInt8(dBits & 0xFF)
        let dHi = UInt8(dBits >> 8)
        let dminBits = Float16(0.005).bitPattern
        let dmLo = UInt8(dminBits & 0xFF)
        let dmHi = UInt8(dminBits >> 8)
        for b in 0 ..< totalBlocks {
            let base = b * bb
            bytes[base] = dLo
            bytes[base + 1] = dHi
            let startOffset = type == .q4K ? 4 : 2
            if type == .q4K {
                bytes[base + 2] = dmLo
                bytes[base + 3] = dmHi
            }
            for i in startOffset ..< bb {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                bytes[base + i] = UInt8((state >> 33) & 0xFF)
            }
        }
        return MLXArray(bytes, [rows, blocksPerRow * bb])
    }
}

private extension MetalKernelBenchRunner {
    static func benchNF4(m: Int, k: Int, n: Int, cacheDefeatBytes: Int, spec: TimingSpec) -> MetalKernelBenchRow {
        let x = MLXRandom.normal([m, k], key: MLXRandom.key(UInt64(100 + m))).asType(.bfloat16)
        let codes = fullRandomUInt32(shape: [n, k / 8], seed: UInt64(200 + m))
        let scales = MLXRandom.uniform(0.01 ..< 0.05, [n, k / 32], key: MLXRandom.key(UInt64(300 + m))).asType(.bfloat16)
        let singleBytes = codes.nbytes + scales.nbytes
        let copyCount = cacheDefeatBytes > 0 ? max(1, cacheDefeatBytes / max(1, singleBytes)) : 1
        let copies: [(MLXArray, MLXArray)] = [(codes, scales)] + (1 ..< copyCount).map { _ in
            (codes + UInt32(0), scales + MLXArray(Float(0)).asType(.bfloat16))
        }
        eval([x] + copies.flatMap { [$0.0, $0.1] })

        let kernelEligible = Device.defaultDevice() == .gpu && m <= 16
        let ref = matmul(x.asType(.float32), NF4Metal.dequantized(codes, scales: scales, groupSize: 32).asType(.float32).T)
        let got = NF4Metal.matmul(x, codes: codes, scales: scales, groupSize: 32).asType(.float32)
        let diff = relMaxDiff(got, ref)

        let rounds = max(1, spec.callsPerSample / copies.count)
        let timing = measureBatched(spec: spec, callsPerSample: rounds * copies.count) {
            eval((0 ..< rounds).flatMap { _ in
                copies.map { NF4Metal.matmul(x, codes: $0.0, scales: $0.1, groupSize: 32) }
            })
        }
        return makeRow(
            name: "nf4_m\(m)", family: "NF4Metal", shape: "M=\(m) K=\(k) N=\(n)",
            timing: timingStats(timing, bytes: Double(singleBytes + x.nbytes)),
            parity: ParityResult(maxDiff: diff, tolerance: 1.5e-2, passed: kernelEligible),
            gate: m <= 8 ? .gated : .reportOnly("bimodal DVFS #889")
        )
    }

    static func benchNF4Gather(profile: MetalKernelBenchProfile, spec: TimingSpec) -> MetalKernelBenchRow {
        let experts = profile == .production ? 128 : 16
        let topK = profile == .production ? 8 : 4
        let k = profile == .production ? 2816 : 256
        let n = profile == .production ? 704 : 256
        let x = MLXRandom.normal([topK, 1, k], key: MLXRandom.key(410)).asType(.bfloat16)
        let codes = fullRandomUInt32(shape: [experts, n, k / 8], seed: 420)
        let scales = MLXRandom.uniform(0.01 ..< 0.05, [experts, n, k / 32], key: MLXRandom.key(430)).asType(.bfloat16)
        let picks = (0 ..< spec.callsPerSample).map { call in
            MLXArray((0 ..< topK).map { UInt32(($0 * 37 + call * 11) % experts) })
        }
        eval([x, codes, scales] + picks)

        let got = NF4Metal.gatherMatmul(x, codes: codes, scales: scales, rhsIndices: picks[0], groupSize: 32).asType(.float32)
        let pickedCodes = codes.take(picks[0].asType(.int32), axis: 0)
        let pickedScales = scales.take(picks[0].asType(.int32), axis: 0)
        let wRef = NF4Metal.dequantized(pickedCodes, scales: pickedScales, groupSize: 32).asType(.float32)
        let ref = matmul(x.asType(.float32), wRef.transposed(0, 2, 1))
        let diff = relMaxDiff(got, ref)

        let timing = measureBatched(spec: spec, callsPerSample: picks.count) {
            eval(picks.map { NF4Metal.gatherMatmul(x, codes: codes, scales: scales, rhsIndices: $0, groupSize: 32) })
        }
        let activeBytes = Double(topK * ((n * k / 2) + (n * k / 32) * 2) + x.nbytes)
        return makeRow(
            name: "nf4_gather_gemma4", family: "NF4Metal", shape: "E=\(experts) top\(topK) K=\(k) N=\(n)",
            timing: timingStats(timing, bytes: activeBytes),
            parity: ParityResult(maxDiff: diff, tolerance: 1.5e-2, passed: Device.defaultDevice() == .gpu)
        )
    }

    static func benchIQ(type: GGMLType, m: Int, k: Int, n: Int, spec: TimingSpec) -> MetalKernelBenchRow {
        let (_, bb) = type.block
        let packed = makeIQPacked(type: type, rows: n, k: k, seed: UInt64(500 + m * 17 + bb))
        let xs = (0 ..< spec.callsPerSample).map { idx in
            MLXRandom.normal([m, k], key: MLXRandom.key(UInt64(600 + idx))).asType(.bfloat16)
        }
        eval([packed] + xs)

        let supported = Device.defaultDevice() == .gpu && IQMetalKernels.supports(type)
        let bytes = packed.asArray(UInt8.self)
        let refW = MLXArray(GGUFDequant.floats(bytes.span.bytes, type: type, count: n * k), [n, k])
        let ref = matmul(xs[0].asType(.float32), refW.T)
        let got = (m == 1
            ? IQMetalKernels.matvec(xs[0], packed, type: type)
            : IQMetalKernels.smallM(xs[0], packed, type: type)
        ).asType(.float32)
        let diff = relMaxDiff(got, ref)

        let timing = measureBatched(spec: spec, callsPerSample: xs.count) {
            eval(xs.map {
                m == 1 ? IQMetalKernels.matvec($0, packed, type: type) : IQMetalKernels.smallM($0, packed, type: type)
            })
        }
        return makeRow(
            name: "iq_\(type.name)_m\(m)", family: "IQMetalKernels", shape: "M=\(m) K=\(k) N=\(n)",
            timing: timingStats(timing, bytes: Double(packed.nbytes + xs[0].nbytes)),
            parity: ParityResult(maxDiff: diff, tolerance: 1.5e-2, passed: supported),
            gate: m == 1 ? .gated : .reportOnly("smallM variance")
        )
    }
}

private extension MetalKernelBenchRunner {
    static func routerEdgeLogits(experts: Int, seed: UInt64) -> MLXArray {
        let randRows = MLXRandom.normal([8, experts], key: MLXRandom.key(seed)).asType(.float32)
        let equalRow = MLXArray(Array(repeating: Float(1.25), count: experts), [1, experts])
        var tieVals = Array(repeating: Float(-2.0), count: experts)
        for i in 0 ..< min(12, experts) { tieVals[i] = 3.5 }
        let tieRow = MLXArray(tieVals, [1, experts])
        var nanVals = (0 ..< experts).map { Float($0) * 0.01 }
        nanVals[0] = Float.nan
        nanVals[min(7, experts - 1)] = Float.nan
        let nanRow = MLXArray(nanVals, [1, experts])
        var infVals = (0 ..< experts).map { Float($0) * -0.01 }
        infVals[1] = Float.infinity
        infVals[2] = -Float.infinity
        let infRow = MLXArray(infVals, [1, experts])
        return concatenated([randRows, equalRow, tieRow, nanRow, infRow], axis: 0)
    }

    static func routerParity(got: FusedMoERouterOutput, ref: FusedMoERouterOutput) -> (indicesOK: Bool, maxDiff: Double) {
        let gw = got.weights.asType(.float32)
        let rw = ref.weights.asType(.float32)
        let bothNaN = isNaN(gw) .&& isNaN(rw)
        let diffArr = which(bothNaN, MLXArray(Float(0)), abs(gw - rw))
        eval(got.indices, ref.indices, diffArr)
        let idxOK = arrayEqual(got.indices, ref.indices).item(Bool.self)
        let maxDiff = Double(diffArr.max().item(Float.self))
        return (idxOK, maxDiff)
    }

    static func benchFusedMoERouter(batch: Int, spec: TimingSpec) -> MetalKernelBenchRow {
        let experts = 128
        let topK = 8
        let parityLogits = routerEdgeLogits(experts: experts, seed: 700)
        let benchLogits = MLXRandom.normal([batch, experts], key: MLXRandom.key(701)).asType(.bfloat16)
        eval(parityLogits, benchLogits)

        let supported = FusedMoERouterKernel.supports(benchLogits, k: topK)
            && FusedMoERouterKernel.supports(parityLogits, k: topK)
        let got = FusedMoERouterKernel.route(parityLogits, k: topK, normTopkProb: true, softmaxBeforeTopK: true)
        let ref = FusedMoERouterKernel.reference(parityLogits, k: topK, normTopkProb: true, softmaxBeforeTopK: true)
        let (idxOK, maxDiff) = routerParity(got: got, ref: ref)

        let timing = measureBatched(spec: spec) {
            eval((0 ..< spec.callsPerSample).flatMap { _ -> [MLXArray] in
                let out = FusedMoERouterKernel.route(benchLogits, k: topK, normTopkProb: true, softmaxBeforeTopK: true)
                return [out.indices, out.weights]
            })
        }
        return makeRow(
            name: "moe_router_qwen36", family: "FusedMoERouterKernel", shape: "B=\(batch) E=128 top8",
            timing: timingStats(timing, bytes: Double(benchLogits.nbytes)),
            parity: ParityResult(maxDiff: maxDiff, tolerance: 1e-3, passed: supported && idxOK),
            gate: .reportOnly("sub-10us dispatch")
        )
    }

    static func benchFlintRouter(batch: Int, spec: TimingSpec) -> MetalKernelBenchRow {
        let experts = 128
        let topK = 4
        let parityLogits = routerEdgeLogits(experts: experts, seed: 800)
        let benchLogits = MLXRandom.normal([batch, experts], key: MLXRandom.key(801)).asType(.float32)
        let bias = MLXRandom.normal([experts], key: MLXRandom.key(802)).asType(.float32) * 0.1
        eval(parityLogits, benchLogits, bias)

        let supported = FlintRouterKernel.supports(benchLogits, expertBias: bias, k: topK)
            && FlintRouterKernel.supports(parityLogits, expertBias: bias, k: topK)
        let got = FlintRouterKernel.route(parityLogits, expertBias: bias, k: topK)
        let ref = FlintRouterKernel.reference(parityLogits, expertBias: bias, k: topK)
        let (idxOK, maxDiff) = routerParity(got: got, ref: ref)

        let timing = measureBatched(spec: spec) {
            eval((0 ..< spec.callsPerSample).flatMap { _ -> [MLXArray] in
                let out = FlintRouterKernel.route(benchLogits, expertBias: bias, k: topK)
                return [out.indices, out.weights]
            })
        }
        return makeRow(
            name: "flint_router_40b", family: "FlintRouterKernel", shape: "B=\(batch) E=128 top4",
            timing: timingStats(timing, bytes: Double(benchLogits.nbytes + bias.nbytes)),
            parity: ParityResult(maxDiff: maxDiff, tolerance: 1e-3, passed: supported && idxOK),
            gate: .reportOnly("sub-10us dispatch")
        )
    }

    static func benchThreeBitQMV(k: Int, n: Int, spec: TimingSpec) -> MetalKernelBenchRow {
        let w = (MLXRandom.normal([n, k], key: MLXRandom.key(900)) * 0.05).asType(.bfloat16)
        let (wq, scales, optBiases) = quantized(w, groupSize: 64, bits: 3)
        let biases = optBiases ?? MLXArray.zeros(like: scales)
        let x = MLXRandom.normal([1, k], key: MLXRandom.key(901)).asType(.bfloat16)
        eval(wq, scales, biases, x)

        let ref = quantizedMM(x, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 3).asType(.float32)
        let got = ThreeBitQMV.matmul(x, indices: nil, weight: wq, scales: scales, biases: biases, groupSize: 64).asType(.float32)
        eval(ref, got)
        let diff = Double(abs(got - ref).max().item(Float.self))

        let timing = measureBatched(spec: spec) {
            eval((0 ..< spec.callsPerSample).map { _ in
                ThreeBitQMV.matmul(x, indices: nil, weight: wq, scales: scales, biases: biases, groupSize: 64)
            })
        }
        return makeRow(
            name: "threebit_qmv_m1", family: "ThreeBitQMV", shape: "M=1 K=\(k) N=\(n)",
            timing: timingStats(timing, bytes: Double(wq.nbytes + scales.nbytes + biases.nbytes)),
            parity: ParityResult(maxDiff: diff, tolerance: 2e-2, passed: Device.defaultDevice() == .gpu)
        )
    }

    static func benchQTIP(k: Int, n: Int, spec: TimingSpec) -> MetalKernelBenchRow {
        let rate: QTIPRate = .twoHalfBPW
        let tileRows = n / 16
        let tileCols = k / 16
        let packed = fullRandomUInt32(shape: [tileRows, tileCols * rate.wordsPerTile], seed: 1001)
        let rowScales = MLXRandom.uniform(0.05 ..< 0.15, [tileRows], key: MLXRandom.key(1002)).asType(.float16)
        let tlut = (MLXRandom.normal([1024], key: MLXRandom.key(1000)) * 0.1).asType(.float16)
        let x = MLXRandom.normal([1, k], key: MLXRandom.key(1003)).asType(.float16)
        eval(packed, rowScales, tlut, x)

        let got = QTIPKernels.qmv(x, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate).asType(.float32)
        let ref = QTIPDifferentiableDecode.linearRotated(
            x, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate, inFeatures: k
        ).asType(.float32)
        eval(got, ref)
        let diff = Double(abs(got - ref).max().item(Float.self))

        let timing = measureBatched(spec: spec) {
            eval((0 ..< spec.callsPerSample).map { _ in
                QTIPKernels.qmv(x, packed: packed, rowScales: rowScales, tlut: tlut, rate: rate)
            })
        }
        return makeRow(
            name: "qtip_2p5bpw_m1", family: "QTIPMetal", shape: "M=1 K=\(k) N=\(n)",
            timing: timingStats(timing, bytes: Double(packed.nbytes + tlut.nbytes + rowScales.nbytes)),
            parity: ParityResult(maxDiff: diff, tolerance: 2e-2, passed: Device.defaultDevice() == .gpu)
        )
    }
}
