import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

enum QTIPEvalHarness {
    static let env = ProcessInfo.processInfo.environment

    static func calibrationBatches(_ container: ModelContainer, count: Int, seqLen: Int) async -> [[Int32]] {
        await container.perform { context in
            let encoded = context.tokenizer.encode(text: PVEvalHarness.calibrationCorpus, addSpecialTokens: false)
            let usable = min(count, encoded.count / max(1, seqLen))
            return (0 ..< usable).map { idx in
                let start = idx * seqLen
                return encoded[start ..< (start + seqLen)].map(Int32.init)
            }
        }
    }

    static func appendJSONL(_ metrics: PVEvalMetrics) {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let day = Date().ISO8601Format(.iso8601.year().month().day())
        let file = root.appendingPathComponent("logs/qtip-results-\(day).jsonl")
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        guard let data = try? enc.encode(metrics) else { return }
        let line = data + Data("\n".utf8)
        if FileManager.default.fileExists(atPath: file.path), let h = try? FileHandle(forWritingTo: file) {
            defer { try? h.close() }
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: line)
        } else {
            try? line.write(to: file)
        }
    }

    static func requantizeDenseAffine(
        _ root: Module, bits: Int, groupSize: Int = 64, maxSourceBits: Int = 8
    ) throws {
        var replaced = 0
        for (path, mod) in root.namedModules() {
            guard PVTuner.textBlockIndex(path) != nil,
                  !path.contains("router"), !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a") else { continue }
            if let ql = mod as? QuantizedLinear, ql.globalScale == nil, ql.bits <= maxSourceBits {
                let rows = ql.weight.dim(0)
                let cols = ql.weight.dim(1) * 32 / ql.bits
                guard rows >= 64, cols >= 64, cols.isMultiple(of: groupSize) else { continue }
                let dense = dequantized(
                    ql.weight, scales: ql.scales, biases: ql.biases,
                    groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
                let repacked = QuantizedLinear(weight: dense, bias: ql.bias, groupSize: groupSize, bits: bits, mode: .affine)
                eval(repacked)
                try PVTuner.swapModule(path, with: repacked, in: root)
                replaced += 1
                if replaced.isMultiple(of: 8) { Memory.clearCache() }
            }
        }
        MLXInferenceEngine.settle(root)
        Memory.clearCache()
    }

    static func parsedRates() -> [QTIPRate] {
        guard let raw = env["BASALT_QTIP_RATES"], !raw.isEmpty else {
            return QTIPRate.allCases
        }
        let wanted = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let matched = wanted.compactMap { label in QTIPRate.allCases.first { $0.label == label } }
        return matched.isEmpty ? QTIPRate.allCases : matched
    }

    static func evaluateQTIPArm(
        modelPath: String,
        rate: QTIPRate,
        armPrefix: String,
        maxSourceBits: Int,
        context: PVArmEvalContext
    ) async throws {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await context.log.step("load \(context.family) for qtip-\(rate.label)") {
            try await engine.load(modelPath: modelPath)
        }
        let container = try #require(await engine.container)
        let calibCount = Int(env["BASALT_QTIP_CALIB_BATCHES"] ?? "") ?? 8
        let calibSeqLen = Int(env["BASALT_QTIP_CALIB_SEQLEN"] ?? "") ?? 256
        let groupBlocks = Int(env["BASALT_QTIP_BLOCK_GROUP"] ?? "") ?? 8
        let calib = await calibrationBatches(container, count: calibCount, seqLen: calibSeqLen)
        var cfg = QTIPConfig(rate: rate)
        cfg.calibrationBatches = calibCount
        cfg.calibrationSeqLen = calibSeqLen
        cfg.blockGroupSize = groupBlocks
        cfg.maxSourceBits = maxSourceBits
        let finalCfg = cfg
        let qSummary = try await context.log.step("qtip-\(rate.label)bpw \(context.family)") {
            try await container.perform { ctx in
                try QTIPQuantizer.quantizeModel(ctx.model, calibrationTokenBatches: calib, config: finalCfg) {
                    context.log.record("qtip-progress", $0)
                }
            }
        }
        let tuneStub = PVTuneSummary(
            trainKLBefore: 0, trainKLAfter: 0, valKLBefore: 0, valKLAfter: 0,
            totalUpdatedCodes: qSummary.quantizedLayers, completedGroups: 0,
            wallSeconds: qSummary.wallSeconds, teacherTokens: qSummary.calibrationTokens)
        let armName = "\(armPrefix)QTIP \(rate.label) bpw (L=16, HYB Q=9)"
        let (metrics, _) = try await PVEvalHarness.evaluateArm(armName, engine: engine, context: context, tuneSummary: tuneStub)
        appendJSONL(metrics)
    }

    static func makeStub(layers: Int, wall: Double, tokens: Int) -> PVTuneSummary {
        PVTuneSummary(
            trainKLBefore: 0, trainKLAfter: 0, valKLBefore: 0, valKLAfter: 0,
            totalUpdatedCodes: layers, completedGroups: 0, wallSeconds: wall, teacherTokens: tokens)
    }

    static func evaluateQTIPMultiRate(
        modelPath: String,
        rates: [QTIPRate],
        armPrefix: String,
        maxSourceBits: Int,
        context: PVArmEvalContext
    ) async throws {
        guard let firstRate = rates.first else { return }
        if rates.count == 1 {
            return try await evaluateQTIPArm(
                modelPath: modelPath, rate: firstRate, armPrefix: armPrefix, maxSourceBits: maxSourceBits, context: context)
        }
        let stageDir = URL(fileURLWithPath: "/private/tmp/basalt-qtip-stage-\(ProcessInfo.processInfo.processIdentifier)")
        defer { try? FileManager.default.removeItem(at: stageDir) }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await context.log.step("load \(context.family) for qtip-multi") { try await engine.load(modelPath: modelPath) }
        let container = try #require(await engine.container)
        let calibCount = Int(env["BASALT_QTIP_CALIB_BATCHES"] ?? "") ?? 8
        let calibSeqLen = Int(env["BASALT_QTIP_CALIB_SEQLEN"] ?? "") ?? 256
        let calib = await calibrationBatches(container, count: calibCount, seqLen: calibSeqLen)
        var cfg = QTIPConfig(rate: firstRate)
        (cfg.calibrationBatches, cfg.calibrationSeqLen, cfg.maxSourceBits) = (calibCount, calibSeqLen, maxSourceBits)
        cfg.blockGroupSize = Int(env["BASALT_QTIP_BLOCK_GROUP"] ?? "") ?? 8
        let finalCfg = cfg
        let labels = rates.map(\.label).joined(separator: "+")
        let (pSummary, plan) = try await context.log.step("qtip-\(labels)bpw \(context.family)") {
            try await container.perform { ctx in
                try QTIPQuantizer.quantizeModelMultiRate(
                    ctx.model, calibrationTokenBatches: calib, rates: rates,
                    baseConfig: finalCfg, stageDirectory: stageDir
                ) { context.log.record("qtip-progress", $0) }
            }
        }
        let pStub = makeStub(layers: pSummary.quantizedLayers, wall: pSummary.wallSeconds, tokens: pSummary.calibrationTokens)
        let (pMetrics, _) = try await PVEvalHarness.evaluateArm(
            "\(armPrefix)QTIP \(firstRate.label) bpw (L=16, HYB Q=9)", engine: engine, context: context, tuneSummary: pStub)
        appendJSONL(pMetrics)
        for secRate in rates.dropFirst() {
            let applyStart = ContinuousClock.now
            let applied = try await context.log.step("apply-staged qtip-\(secRate.label)bpw \(context.family)") {
                try await container.perform { try QTIPQuantizer.applyStagedRate($0.model as Module, plan: plan, rate: secRate) }
            }
            let d = ContinuousClock.now - applyStart
            let sStub = makeStub(
                layers: applied, wall: pSummary.wallSeconds + Double(d.components.seconds), tokens: plan.calibrationTokens)
            let (sMetrics, _) = try await PVEvalHarness.evaluateArm(
                "\(armPrefix)QTIP \(secRate.label) bpw (L=16, HYB Q=9)", engine: engine, context: context, tuneSummary: sStub)
            appendJSONL(sMetrics)
        }
    }
}

@Suite("QTIP Phase 0 real-model evaluation on Qwen3.8-27B, Gemma 4 26B-A4B, and Qwen3.5-9B-8bit (#409)",
       .serialized, .timeLimit(.minutes(420)))
struct QTIPRealModelTests {
    static let enabled = QTIPEvalHarness.env["BASALT_E2E_QTIP"] == "1"
    static let familyFilter = QTIPEvalHarness.env["BASALT_QTIP_FAMILY"] ?? "all"

    @Test("Evaluate QTIP at 2.0, 2.5, and 3.0 bpw against MLX affine, GPTQ, nf4, GSQ-RCO, and PV-Tuning",
          .enabled(if: enabled, "set BASALT_E2E_QTIP=1"))
    func evaluateQTIPPhase0() async throws {
        let groups = CascadeDecisionRealModelTests.groups(SystemOneDecisionSet.generate(states: 100, seed: 234))
        let rates = QTIPEvalHarness.parsedRates()

        if Self.familyFilter == "all" || Self.familyFilter == "qwen" {
            try await runQwen27B(decisionGroups: groups, rates: rates)
        }
        if Self.familyFilter == "all" || Self.familyFilter == "gemma" {
            try await runGemma26B(decisionGroups: groups, rates: rates)
        }
        if Self.familyFilter == "qwen9b8" {
            try await runQwen9B8Bit(decisionGroups: groups, rates: rates)
        }
    }

    func runQwen27B(
        decisionGroups: [(request: SystemOneRequest, items: [LabelledDecision])],
        rates: [QTIPRate]
    ) async throws {
        let teacherPath = NF4Run.path("mlx-community/Qwen3.8-27B-GPTQ-4bit")
        let log = try await ModelRunLog("qtip", modelPath: teacherPath)
        defer { log.releaseGPU() }

        let teacherEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await log.step("load qwen-teacher") { try await teacherEngine.load(modelPath: teacherPath) }
        let teacherTraces = try await log.step("golden qwen-teacher") {
            try await GoldenRun.traces(teacherEngine, run: 0, log: log)
        }
        await teacherEngine.unload()
        Memory.clearCache()

        let ctx = PVArmEvalContext(
            family: "Qwen3.8-27B", teacherTraces: teacherTraces, decisionGroups: decisionGroups, log: log)
        try await QTIPEvalHarness.evaluateQTIPMultiRate(
            modelPath: teacherPath, rates: rates, armPrefix: "", maxSourceBits: 4, context: ctx)
    }

    func runGemma26B(
        decisionGroups: [(request: SystemOneRequest, items: [LabelledDecision])],
        rates: [QTIPRate]
    ) async throws {
        let gemmaPath = NF4Run.path("mlx-community/gemma-4-26B-A4B-it-qat-4bit")
        let log = try await ModelRunLog("qtip", modelPath: gemmaPath)
        defer { log.releaseGPU() }

        let teacherEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await log.step("load gemma-teacher") { try await teacherEngine.load(modelPath: gemmaPath) }
        let teacherTraces = try await log.step("golden gemma-teacher") {
            try await GoldenRun.traces(teacherEngine, run: 0, log: log)
        }
        await teacherEngine.unload()
        Memory.clearCache()

        let ctx = PVArmEvalContext(
            family: "Gemma-4-26B-A4B", teacherTraces: teacherTraces, decisionGroups: decisionGroups, log: log)
        if QTIPEvalHarness.env["BASALT_QTIP_SKIP_GEMMA_BASELINES"] != "1" {
            for bits in [2, 3] {
                let bEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
                try await log.step("load gemma-dense-\(bits)bit") { try await bEngine.load(modelPath: gemmaPath) }
                let bc = try #require(await bEngine.container)
                try await log.step("requant gemma-dense-\(bits)bit-rtn") {
                    try await bc.perform { c in try QTIPEvalHarness.requantizeDenseAffine(c.model as Module, bits: bits) }
                }
                let (m, _) = try await PVEvalHarness.evaluateArm("dense \(bits)-bit affine RTN", engine: bEngine, context: ctx)
                QTIPEvalHarness.appendJSONL(m)
                await bEngine.unload()
                Memory.clearCache()
            }
        }
        for rate in rates {
            try await QTIPEvalHarness.evaluateQTIPArm(
                modelPath: gemmaPath, rate: rate, armPrefix: "dense ", maxSourceBits: 8, context: ctx)
        }
    }

    func runQwen9B8Bit(
        decisionGroups: [(request: SystemOneRequest, items: [LabelledDecision])],
        rates: [QTIPRate]
    ) async throws {
        let modelPath = NF4Run.path("mlx-community/Qwen3.5-9B-8bit")
        let log = try await ModelRunLog("qtip", modelPath: modelPath)
        defer { log.releaseGPU() }

        let teacherEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await log.step("load qwen9b-8bit-teacher") { try await teacherEngine.load(modelPath: modelPath) }
        let baseCtx = PVArmEvalContext(
            family: "Qwen3.5-9B-8bit", teacherTraces: nil, decisionGroups: decisionGroups, log: log)
        let (tMetrics, teacherTraces) = try await PVEvalHarness.evaluateArm(
            "8-bit affine (teacher)", engine: teacherEngine, context: baseCtx)
        QTIPEvalHarness.appendJSONL(tMetrics)
        await teacherEngine.unload()
        Memory.clearCache()

        let ctx = PVArmEvalContext(
            family: "Qwen3.5-9B-8bit", teacherTraces: teacherTraces, decisionGroups: decisionGroups, log: log)
        for bits in [2, 3] {
            let bEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
            try await log.step("load qwen9b-\(bits)bit") { try await bEngine.load(modelPath: modelPath) }
            let bc = try #require(await bEngine.container)
            try await log.step("requant qwen9b-\(bits)bit-rtn") {
                try await bc.perform { c in try QTIPEvalHarness.requantizeDenseAffine(c.model as Module, bits: bits) }
            }
            let (m, _) = try await PVEvalHarness.evaluateArm("\(bits)-bit affine RTN", engine: bEngine, context: ctx)
            QTIPEvalHarness.appendJSONL(m)
            await bEngine.unload()
            Memory.clearCache()
        }
        for rate in rates {
            try await QTIPEvalHarness.evaluateQTIPArm(
                modelPath: modelPath, rate: rate, armPrefix: "", maxSourceBits: 8, context: ctx)
        }
    }
}
