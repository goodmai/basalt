import Darwin
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("MACaQ Metal kernel benchmarks: barrier-free single-simdgroup Hadamard vs legacy barrier Hadamard across WHT and rotated quantized projections (#535)", .serialized)
struct MACaQKernelBenchmarkTests {
    typealias Bench = SmallMQuantizedMatmulBenchmarkTests

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BASALT_BENCH_MACAQ_KERNEL"] == "1"
    }

    static func whtRun(x: MLXArray, pre: MLXArray, blockSize: Int, fusedSingleSimd: Bool) -> () -> Double {
        let copies = (0 ..< 64).map { _ in x + MLXArray(0, dtype: x.dtype) }
        eval([pre] + copies)
        let run = {
            eval(copies.map { inp in
                fusedSingleSimd
                    ? MACaQFusedKernels.blockHadamard(inp, scaleVector: pre, blockSize: blockSize, isInverse: false)
                    : FusedKroneckerRHT.blockHadamardFallback(inp, signs: pre, blockSize: blockSize, isInverse: false)
            })
        }
        run()
        return { Bench.microseconds(run) / Double(copies.count) }
    }

    static func rotatedDecodeRun(
        dense: MLXArray,
        x: MLXArray,
        pre: MLXArray,
        blockSize: Int,
        mode: MACaQTargetMode,
        useSingleSimdWHT: Bool
    ) -> () -> Double {
        let q = MLX.quantized(dense, groupSize: mode.groupSize, bits: mode.bits, mode: mode.quantizationMode)
        let bytes = q.wq.nbytes + q.scales.nbytes + (q.biases?.nbytes ?? 0)
        let copyCount = max(1, min(16, Bench.cacheDefeatBytes / max(1, bytes)))
        let copies = [(q.wq, q.scales, q.biases)] + (1 ..< copyCount).map { _ in
            (q.wq + 0, q.scales + 0, q.biases.map { $0 + 0 })
        }
        eval([x, pre] + copies.flatMap { [$0.0, $0.1] + ($0.2.map { [$0] } ?? []) })
        let rounds = max(1, 16 / copies.count)
        let run = {
            eval((0 ..< rounds).flatMap { _ in
                copies.map { item -> MLXArray in
                    let xRot = useSingleSimdWHT
                        ? MACaQFusedKernels.blockHadamard(x, scaleVector: pre, blockSize: blockSize, isInverse: false)
                        : FusedKroneckerRHT.blockHadamardFallback(x, signs: pre, blockSize: blockSize, isInverse: false)
                    return quantizedMM(
                        xRot, item.0, scales: item.1, biases: item.2,
                        groupSize: mode.groupSize, bits: mode.bits, mode: mode.quantizationMode)
                }
            })
        }
        run()
        return { Bench.microseconds(run) / Double(rounds * copies.count) }
    }

    @Test("Single-simdgroup barrier-free WHT µs/call vs legacy barrier WHT across M=1,4,8,64 and Qwen/Gemma quantized projections",
          .enabled(if: enabled, "set BASALT_BENCH_MACAQ_KERNEL=1 and run in release mode"))
    func benchmarkMACaQKernels() throws {
        guard case .taken = try ModelRunLog.claim(ModelRunLog.gpuLock, pid: getpid()) else {
            Issue.record("\(ModelRunLog.gpuLock) is held by another run; relaunch when free")
            return
        }
        var lock = GPULockGuard(path: ModelRunLog.gpuLock, pid: getpid())
        defer { _ = lock.release() }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let url = root.appendingPathComponent("logs/macaq-kernel-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        Bench.log("RUN \(Date().ISO8601Format()) commit \(ModelRunLog.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) machine \(SystemProfiler.detectChipModel()) load \(ModelRunLog.load())", to: url)
        Bench.log("WHT | width x rows | block | legacy_barrier_us | single_simd_us | speedup", to: url)
        for block in [64, 128] {
            for rows in [1, 4, 8, 64] {
                let x = MLXRandom.normal([rows, 5120], dtype: .bfloat16)
                let pre = MLXRandom.uniform(low: 0.5, high: 1.5, [5120]).asType(.bfloat16)
                let med = NF4KernelBenchmarkTests.interleavedMedians([
                    Self.whtRun(x: x, pre: pre, blockSize: block, fusedSingleSimd: false),
                    Self.whtRun(x: x, pre: pre, blockSize: block, fusedSingleSimd: true),
                ], rounds: 9)
                Bench.log(String(
                    format: "WHT | 5120 x %d | B=%d | %.2f | %.2f | %.2fx",
                    rows, block, med[0], med[1], med[0] / max(med[1], 1e-6)), to: url)
            }
        }
        Bench.log("PROJ | mode | K x N | M | B | legacy_wht_qmm_us | single_simd_wht_qmm_us | speedup", to: url)
        let projShapes: [(name: String, inputs: Int, outputs: Int)] = [
            ("gemma4-12b q_proj", 3840, 4096),
            ("gemma4-12b gate_proj", 3840, 15360),
            ("qwen3.8-27b q_proj", 5120, 6144),
        ]
        let modes: [MACaQTargetMode] = [.affine(bits: 3, groupSize: 64), .affine(bits: 4, groupSize: 64), .mxfp4]
        for shape in projShapes {
            let dense = MLXRandom.normal([shape.outputs, shape.inputs], dtype: .bfloat16, scale: 0.02)
            let pre = MLXRandom.uniform(low: 0.5, high: 1.5, [shape.inputs]).asType(.bfloat16)
            for mode in modes {
                for m in [1, 4] {
                    let x = MLXRandom.normal([m, shape.inputs], dtype: .bfloat16)
                    let med = NF4KernelBenchmarkTests.interleavedMedians([
                        Self.rotatedDecodeRun(dense: dense, x: x, pre: pre, blockSize: 64, mode: mode, useSingleSimdWHT: false),
                        Self.rotatedDecodeRun(dense: dense, x: x, pre: pre, blockSize: 64, mode: mode, useSingleSimdWHT: true),
                    ], rounds: 7)
                    Memory.clearCache()
                    Bench.log(String(
                        format: "%@ | %@ | %d x %d | M=%d | B=64 | %.1f | %.1f | %.2fx",
                        shape.name, mode.label, shape.inputs, shape.outputs, m, med[0], med[1], med[0] / max(med[1], 1e-6)), to: url)
                }
            }
        }
        Bench.log("END \(Date().ISO8601Format()) log \(url.path)", to: url)
    }
}

private final class LegacyBarrierMACaQLinear: QuantizedLinear {
    let hadamardBlock: Int
    let preScale: MLXArray

    init(from layer: MACaQQuantizedLinear) {
        self.hadamardBlock = layer.hadamardBlock
        self.preScale = layer.preScale
        if layer.mode == .affine {
            super.init(
                weight: layer.weight, bias: layer.bias, scales: layer.scales,
                biases: layer.biases, groupSize: layer.groupSize, bits: layer.bits)
        } else if layer.bias == nil {
            super.init(
                weight: layer.weight, scales: layer.scales, biases: nil,
                groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode)
        } else {
            let stub = MLXArray.zeros([1, layer.groupSize], dtype: .float16)
            super.init(weight: stub, bias: layer.bias, groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode)
            update(parameters: ModuleParameters.unflattened([("weight", layer.weight), ("scales", layer.scales)]))
        }
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rot = FusedKroneckerRHT.blockHadamardFallback(
            x, signs: preScale, blockSize: hadamardBlock, isInverse: false)
        return super.callAsFunction(rot)
    }
}

enum MACaQE2EHarness {
    static func corpusWindowPPL(_ model: any LanguageModel, tokens: [Int32]) throws -> Double {
        let cache = try model.newCache(parameters: nil)
        var (state, nll, counted) = (LMOutput.State?.none, 0.0, 0)
        for start in stride(from: 0, to: tokens.count - 1, by: 256) {
            let end = min(tokens.count - 1, start + 256)
            let out = model(LMInput.Text(tokens: MLXArray(tokens[start ..< end])[.newAxis]), cache: cache, state: state)
            state = out.state
            let logits = out.logits[0].asType(.float32)
            let lse = logSumExp(logits, axis: -1, keepDims: true)
            let targets = MLXArray(tokens[(start + 1) ... end])[0..., .newAxis]
            let picked = takeAlong(logits, targets, axis: -1) - lse
            let usable = abs(picked) .< Float.infinity
            let sumNLL = which(usable, picked, 0).sum()
            let countArr = usable.sum()
            eval(sumNLL, countArr, cache)
            nll -= Double(sumNLL.item(Float.self))
            counted += countArr.item(Int.self)
        }
        return exp(nll / Double(max(counted, 1)))
    }

    static func measureDecodeTokPerSec(
        _ model: any LanguageModel,
        prompt: [Int32],
        steps: Int = 64,
        temperature: Float = 0.7
    ) throws -> Double {
        var timings: [Double] = []
        for round in 0 ..< 4 {
            let elapsed = try withRandomState(MLXRandom.RandomState(seed: UInt64(5350 + round))) {
                let cache = try model.newCache(parameters: nil)
                var out = model(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: cache, state: nil)
                var state = out.state
                var tok = sampleToken(out.logits[0, -1], temperature: temperature)
                eval(tok, cache)
                let start = ContinuousClock.now
                for _ in 0 ..< steps {
                    out = model(LMInput.Text(tokens: tok.reshaped(1, 1)), cache: cache, state: state)
                    state = out.state
                    tok = sampleToken(out.logits[0, -1], temperature: temperature)
                    eval(tok, cache)
                }
                return ContinuousClock.now - start
            }
            if round >= 1 {
                let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
                timings.append(Double(steps) / max(secs, 1e-6))
            }
        }
        timings.sort()
        return timings[timings.count / 2]
    }

    private static func sampleToken(_ logits: MLXArray, temperature: Float) -> MLXArray {
        guard temperature > 0 else { return argMax(logits, axis: -1) }
        return categorical(logits.asType(.float32) * (Float(1.0) / temperature))
    }

    static func requantizeLinearsTo3BitRTN(_ root: Module) throws {
        var updates: [(String, Module)] = []
        for (path, mod) in root.namedModules() {
            guard PVTuner.textBlockIndex(path) != nil,
                  !path.contains("router"), !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a")
            else { continue }
            if let ql = mod as? QuantizedLinear, ql.globalScale == nil {
                let inF = (ql.weight.dim(1) * 32) / max(1, ql.bits)
                guard inF.isMultiple(of: 64) else { continue }
                let (dense, bias) = QTIPQuantizer.extractDenseWeight(ql)
                let q = MLX.quantized(dense.asType(.bfloat16), groupSize: 64, bits: 3, mode: .affine)
                updates.append((path, QuantizedLinear(
                    weight: q.wq, bias: bias, scales: q.scales, biases: q.biases, groupSize: 64, bits: 3)))
            } else if type(of: mod) == Linear.self, let lin = mod as? Linear, lin.shape.1.isMultiple(of: 64) {
                let q = MLX.quantized(lin.weight.asType(.bfloat16), groupSize: 64, bits: 3, mode: .affine)
                updates.append((path, QuantizedLinear(
                    weight: q.wq, bias: lin.bias, scales: q.scales, biases: q.biases, groupSize: 64, bits: 3)))
            }
        }
        if !updates.isEmpty {
            root.update(modules: ModuleChildren.unflattened(updates))
            eval(root)
        }
    }

    static func prepareTokens(
        container: ModelContainer,
        batches: Int,
        seqLen: Int
    ) async -> (cal: [[Int32]], evalToks: [Int32]) {
        let calText = PVEvalHarness.calibrationCorpus
        let evalText = AbliterationRealModelTests.corpus
        return await container.perform { ctx in
            let encCal = ctx.tokenizer.encode(text: calText, addSpecialTokens: false).map(Int32.init)
            let bCount = min(batches, encCal.count / seqLen)
            let calBatches = (0 ..< bCount).map { Array(encCal[($0 * seqLen) ..< (($0 + 1) * seqLen)]) }
            let evalToks = Array(ctx.tokenizer.encode(text: evalText, addSpecialTokens: false).prefix(1025)).map(Int32.init)
            return (calBatches, evalToks)
        }
    }
}

@Suite("MACaQ real-model quality, refusal guard, and decode speed across Qwen3-0.6B, Ornith-1.5-9B (qwen3_5), Gemma 4 12B, and BGE-small (BERT refusal) (#535)", .serialized)
struct MACaQRealModelTests {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BASALT_E2E_MACAQ"] == "1"
    }

    @Test("MACaQ refuses encoder-only BERT and improves 3-bit PPL + WHT decode on Qwen3-0.6B-8bit, Ornith-1.5-9B-MLX-4bit, and gemma-4-12B-it-qat-4bit",
          .enabled(if: enabled, "set BASALT_E2E_MACAQ=1 and run in release mode"))
    func evaluateRealModels() async throws {
        let bertPath = try #require(ModelCache.resolve(repoId: "mlx-community/bge-small-en-v1.5-bf16"))
        do {
            let log = try await ModelRunLog("macaq-e2e", modelPath: bertPath.path)
            let tmpOut = FileManager.default.temporaryDirectory.appending(component: "macaq-bert-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: tmpOut) }
            var refusedMessage = ""
            do {
                _ = try await MACaQRefiner.convertDirectory(source: bertPath, output: tmpOut, targetMode: .affine(bits: 3, groupSize: 64))
            } catch {
                refusedMessage = error.localizedDescription
            }
            #expect(refusedMessage.contains("bert"))
            log.record("REFUSAL", "mlx-community/bge-small-en-v1.5-bf16 | refused cleanly: \(refusedMessage)")
        }

        let causalRepos = [
            "mlx-community/Qwen3-0.6B-8bit",
            "ornith-ai/Ornith-1.5-9B-MLX-4bit",
            "mlx-community/gemma-4-12B-it-qat-4bit",
        ]
        for repo in causalRepos {
            let resolved = try #require(ModelCache.resolve(repoId: repo), "Missing cached snapshot for \(repo)")
            do {
                let log = try await ModelRunLog("macaq-e2e", modelPath: resolved.path)
                try await Self.evaluateCausalModel(repo: repo, path: resolved, log: log)
            }
            Memory.clearCache()
        }
    }

    private static func evaluateCausalModel(repo: String, path: URL, log: ModelRunLog) async throws {
        var calCfg = MACaQConfig()
        calCfg.calibrationBatches = 4
        calCfg.calibrationSeqLen = 256
        calCfg.hadamardBlock = 64
        calCfg.maxFlipIterations = 64
        let finalCfg = calCfg

        let passA = try await runPassA(repo: repo, path: path, config: finalCfg, log: log)
        Memory.clearCache()
        try await runPassB(repo: repo, path: path, rtnPPL: passA.rtnPPL, config: finalCfg, log: log)
    }

    private static func runPassA(
        repo: String,
        path: URL,
        config: MACaQConfig,
        log: ModelRunLog
    ) async throws -> (rtnPPL: Double, refPPL: Double) {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: path.path)
        let container = try #require(await engine.container)
        let toks = await MACaQE2EHarness.prepareTokens(container: container, batches: config.calibrationBatches, seqLen: config.calibrationSeqLen)

        let srcPPL = try await log.step("source_ppl \(repo)") {
            try await container.perform { ctx in try MACaQE2EHarness.corpusWindowPPL(ctx.model, tokens: toks.evalToks) }
        }
        let srcDecode = try await log.step("source_decode_t0.7 \(repo)") {
            try await container.perform { ctx in try MACaQE2EHarness.measureDecodeTokPerSec(ctx.model, prompt: Array(toks.evalToks.prefix(64))) }
        }
        log.record("SOURCE", String(format: "%@ | source | ppl=%.3f | decode_t0.7=%.2f tok/s", repo, srcPPL, srcDecode))

        try await container.perform { ctx in
            try MACaQE2EHarness.requantizeLinearsTo3BitRTN(ctx.model as Module)
        }
        let rtnPPL = try await log.step("rtn_3bit_ppl \(repo)") {
            try await container.perform { ctx in try MACaQE2EHarness.corpusWindowPPL(ctx.model, tokens: toks.evalToks) }
        }
        let rtnDecode = try await log.step("rtn_3bit_decode_t0.7 \(repo)") {
            try await container.perform { ctx in try MACaQE2EHarness.measureDecodeTokPerSec(ctx.model, prompt: Array(toks.evalToks.prefix(64))) }
        }
        let refSummary = try await log.step("macaq_r_3bit_refine \(repo)") {
            try await container.perform { ctx in
                let reader = try MACaQShardTeacherReader(directory: path)
                return try MACaQRefiner.refineModel(
                    ctx.model, calibrationTokenBatches: toks.cal, config: config,
                    teacherWeight: { try reader.denseWeight(for: $0) })
            }
        }
        let refPPL = try await log.step("macaq_r_3bit_ppl \(repo)") {
            try await container.perform { ctx in try MACaQE2EHarness.corpusWindowPPL(ctx.model, tokens: toks.evalToks) }
        }
        let refDecode = try await log.step("macaq_r_3bit_decode_t0.7 \(repo)") {
            try await container.perform { ctx in try MACaQE2EHarness.measureDecodeTokPerSec(ctx.model, prompt: Array(toks.evalToks.prefix(64))) }
        }
        log.record("MACAQ_R_3BIT", String(
            format: "%@ | affine:3:64 | rtn_ppl=%.3f (%.2f tok/s) -> macaq_r_ppl=%.3f (%.2f tok/s) | layers=%d | flips=%d | H_ratio=%.4f | wall=%.2fs",
            repo, rtnPPL, rtnDecode, refPPL, refDecode,
            refSummary.refinedLayers, refSummary.totalFlips, refSummary.meanLossRatio, refSummary.wallSeconds))
        #expect(refSummary.refinedLayers > 0)
        #expect(refSummary.meanLossRatio < 1.0)
        return (rtnPPL, refPPL)
    }

    private static func runPassB(
        repo: String,
        path: URL,
        rtnPPL: Double,
        config: MACaQConfig,
        log: ModelRunLog
    ) async throws {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: path.path)
        let container = try #require(await engine.container)
        let toks = await MACaQE2EHarness.prepareTokens(container: container, batches: config.calibrationBatches, seqLen: config.calibrationSeqLen)

        let qRes = try await log.step("macaq_3bit_quantize \(repo)") {
            try await container.perform { ctx in
                try MACaQRefiner.quantizeModel(
                    ctx.model, targetMode: .affine(bits: 3, groupSize: 64),
                    calibrationTokenBatches: toks.cal, config: config)
            }
        }
        let macaqPPL = try await log.step("macaq_3bit_ppl \(repo)") {
            try await container.perform { ctx in try MACaQE2EHarness.corpusWindowPPL(ctx.model, tokens: toks.evalToks) }
        }
        let promptSlice = Array(toks.evalToks.prefix(64))
        let singleSimdTokS = try await log.step("macaq_3bit_decode_single_simd \(repo)") {
            try await container.perform { ctx in try MACaQE2EHarness.measureDecodeTokPerSec(ctx.model, prompt: promptSlice) }
        }
        let legacyTokS = try await log.step("macaq_3bit_decode_legacy_wht \(repo)") {
            try await container.perform { ctx -> Double in
                let root = ctx.model as Module
                var saved: [(String, MACaQQuantizedLinear)] = []
                for (p, mod) in root.namedModules() {
                    if let m = mod as? MACaQQuantizedLinear {
                        saved.append((p, m))
                        try PVTuner.swapModule(p, with: LegacyBarrierMACaQLinear(from: m), in: root)
                    }
                }
                defer {
                    for (p, m) in saved { try? PVTuner.swapModule(p, with: m, in: root) }
                }
                return try MACaQE2EHarness.measureDecodeTokPerSec(ctx.model, prompt: promptSlice)
            }
        }
        log.record("MACAQ_3BIT", String(
            format: "%@ | macaq:affine:3:64 B=64 | rtn_ppl=%.3f -> macaq_ppl=%.3f | layers=%d | flips=%d | H_ratio=%.4f | legacy_wht=%.2f tok/s -> single_simd_wht=%.2f tok/s (%.2fx) | wall=%.2fs",
            repo, rtnPPL, macaqPPL, qRes.summary.refinedLayers, qRes.summary.totalFlips, qRes.summary.meanLossRatio,
            legacyTokS, singleSimdTokS, singleSimdTokS / max(legacyTokS, 1e-6), qRes.summary.wallSeconds))
        #expect(qRes.summary.refinedLayers > 0)
        #expect(macaqPPL < rtnPPL)
        #expect(singleSimdTokS > 0)
    }
}
