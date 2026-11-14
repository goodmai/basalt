import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

struct PVEvalMetrics: Codable {
    var family: String
    var arm: String
    var modelBytesGB: Double
    var pplWindows: [Double]
    var pplMean16k: Double
    var goldenStep0KL: Double
    var goldenMaxKL: Double
    var systemOneCorrect: Int
    var systemOneTotal: Int
    var wallSeconds: Double
    var teacherTokens: Int
    var updatedCodes: Int
    var trainKLBefore: Double
    var trainKLAfter: Double
    var valKLBefore: Double
    var valKLAfter: Double
}

struct PVArmEvalContext {
    var family: String
    var teacherTraces: [String: GoldenTrace]?
    var decisionGroups: [(request: SystemOneRequest, items: [LabelledDecision])]
    var log: ModelRunLog
}

enum PVEvalHarness {
    static let env = ProcessInfo.processInfo.environment
    static let skipBaselines = env["BASALT_PV_SKIP_BASELINES"] == "1"
    static let skipQwen3Bit = env["BASALT_PV_SKIP_QWEN_3BIT"] == "1"

    static var calibrationCorpus: String {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let files = [
            "docs/commands.md", "docs/compose.md", "docs/ROADMAP.md", "docs/ROADMAP_EPIC_16_CLI_ENHANCEMENTS.md",
            "docs/ROADMAP_EPIC_16_DETAILED.md", "docs/EPIC_15_REVIEW_BA_DECOMPOSITION.md", "docs/rest-api.md",
            "docs/mac-app.md", "docs/integrations.md", "docs/architecture.md"
        ]
        return files.compactMap { try? String(contentsOf: root.appendingPathComponent($0), encoding: .utf8) }.joined(separator: "\n\n")
    }

    static func modelBytesGB(_ container: ModelContainer) async -> Double {
        await container.perform { ctx in
            Double(ctx.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }) / 1e9
        }
    }

    static func goldenStep0KL(reference: [String: GoldenTrace], candidate: [String: GoldenTrace]) -> (step0Mean: Double, maxKL: Double) {
        let fixture = GoldenFixture(revision: "teacher", tolerance: 1.0, prompts: reference)
        let verdict = fixture.compare(candidate)
        var (sum, count) = (0.0, 0)
        for (name, refTrace) in reference {
            guard let candTrace = candidate[name], let r0 = refTrace.steps.first, let c0 = candTrace.steps.first else { continue }
            sum += GoldenFixture.divergence(reference: r0, new: c0)
            count += 1
        }
        return (count > 0 ? sum / Double(count) : 0, verdict.maxKL)
    }

    static func fastPass16k(_ container: ModelContainer) async throws -> [Double] {
        try await container.perform { context in
            let tokens = Array(context.tokenizer.encode(text: AbliterationRealModelTests.corpus).prefix(16_385))
            let cache = try context.model.newCache(parameters: nil)
            var (windows, state, nll, counted) = ([Double](), LMOutput.State?.none, 0.0, 0)
            for start in stride(from: 0, to: tokens.count - 1, by: 256) {
                let end = min(tokens.count - 1, start + 256)
                let output = context.model(
                    LMInput.Text(tokens: MLXArray(tokens[start ..< end].map(Int32.init))[.newAxis]), cache: cache, state: state)
                state = output.state
                let logits = output.logits[0].asType(.float32)
                let lse = logSumExp(logits, axis: -1, keepDims: true)
                let targets = MLXArray(tokens[(start + 1) ... end].map(Int32.init))[0..., .newAxis]
                let picked = takeAlong(logits, targets, axis: -1) - lse
                let usable = abs(picked) .< Float.infinity
                let sumNLL = which(usable, picked, 0).sum()
                let countArr = usable.sum()
                eval(sumNLL, countArr, cache)
                nll -= Double(sumNLL.item(Float.self))
                counted += countArr.item(Int.self)
                if end % 4096 == 0 || end == tokens.count - 1 {
                    windows.append(exp(nll / Double(max(counted, 1))))
                    (nll, counted) = (0, 0)
                    Memory.clearCache()
                }
            }
            return windows
        }
    }

    static func evaluateArm(
        _ arm: String, engine: MLXInferenceEngine, context: PVArmEvalContext, tuneSummary: PVTuneSummary? = nil
    ) async throws -> (metrics: PVEvalMetrics, traces: [String: GoldenTrace]) {
        let container = try #require(await engine.container)
        let gb = await modelBytesGB(container)
        let pplWindows = try await context.log.step("ppl16k \(arm)") { try await fastPass16k(container) }
        let logMean = pplWindows.isEmpty ? 0 : exp(pplWindows.map(Foundation.log).reduce(0, +) / Double(pplWindows.count))
        let traces = try await context.log.step("golden \(arm)") { try await GoldenRun.traces(engine, run: 0, log: context.log) }
        let klPair = context.teacherTraces.map { goldenStep0KL(reference: $0, candidate: traces) } ?? (0.0, 0.0)
        let decider = ReadoutDecisionEngine(engine: engine, modelID: "self", readout: .uncalibrated)
        let sysArm = try await context.log.step("systemone \(arm)") {
            try await CascadeDecisionRealModelTests.run(decider, context.decisionGroups)
        }
        let m = PVEvalMetrics(
            family: context.family, arm: arm, modelBytesGB: gb, pplWindows: pplWindows, pplMean16k: logMean,
            goldenStep0KL: klPair.0, goldenMaxKL: klPair.1, systemOneCorrect: sysArm.correct, systemOneTotal: sysArm.total,
            wallSeconds: tuneSummary?.wallSeconds ?? 0, teacherTokens: tuneSummary?.teacherTokens ?? 0,
            updatedCodes: tuneSummary?.totalUpdatedCodes ?? 0,
            trainKLBefore: Double(tuneSummary?.trainKLBefore ?? 0), trainKLAfter: Double(tuneSummary?.trainKLAfter ?? 0),
            valKLBefore: Double(tuneSummary?.valKLBefore ?? 0), valKLAfter: Double(tuneSummary?.valKLAfter ?? 0))
        let winStr = pplWindows.map { String(format: "%.3f", $0) }.joined(separator: "/")
        context.log.record("PV_ARM", String(
            format: "%@ | %@ | %.2f GB | ppl16k %.3f [%@] | step0KL %.4f maxKL %.4f | sys1 %d/%d | codes %d | wall %.1fs | tok %d",
            context.family, arm, gb, logMean, winStr, klPair.0, klPair.1, sysArm.correct, sysArm.total,
            m.updatedCodes, m.wallSeconds, m.teacherTokens))
        appendJSONL(m)
        Memory.clearCache()
        return (m, traces)
    }

    static func appendJSONL(_ metrics: PVEvalMetrics) {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let day = Date().ISO8601Format(.iso8601.year().month().day())
        let file = root.appendingPathComponent("logs/pv-tuning-results-\(day).jsonl")
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
}

extension PVEvalHarness {
    static func runQwenBaselines(
        teacherPath: String, decisionGroups: [(request: SystemOneRequest, items: [LabelledDecision])], log: ModelRunLog
    ) async throws -> (batches: [PVTeacherBatch], traces: [String: GoldenTrace]) {
        let teacherEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await log.step("load qwen-gptq-4bit-teacher") { try await teacherEngine.load(modelPath: teacherPath) }
        let tc = try #require(await teacherEngine.container)
        let allBatches = await PVTuner.collectTeacherBatches(tc, texts: calibrationCorpus, count: 40, seqLen: 128, topK: 64)
        let baseCtx = PVArmEvalContext(family: "Qwen3.8-27B", teacherTraces: nil, decisionGroups: decisionGroups, log: log)
        let teacherTraces: [String: GoldenTrace]
        if skipBaselines {
            teacherTraces = try await log.step("golden teacher-ref") { try await GoldenRun.traces(teacherEngine, run: 0, log: log) }
        } else {
            teacherTraces = try await evaluateArm("4-bit GPTQ (teacher)", engine: teacherEngine, context: baseCtx).traces
        }
        await teacherEngine.unload()
        Memory.clearCache()
        if !skipBaselines {
            var armCtx = baseCtx
            armCtx.teacherTraces = teacherTraces
            let rtn4Engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
            try await log.step("load qwen-rtn-4bit") { try await rtn4Engine.load(modelPath: NF4Run.path("mlx-community/Qwen3.8-27B-4bit")) }
            _ = try await evaluateArm("4-bit affine RTN", engine: rtn4Engine, context: armCtx)
            await rtn4Engine.unload()
            Memory.clearCache()
            let gsqPath = "/Users/alexeiboklag/.cache/basalt/gguf/qwen38-27b-gsq-native"
            if FileManager.default.fileExists(atPath: gsqPath) {
                let gsqEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
                try await log.step("load qwen-gsq-rco-iq3s") { try await gsqEngine.load(modelPath: gsqPath) }
                _ = try await evaluateArm("GSQ-RCO IQ3_S native", engine: gsqEngine, context: armCtx)
                await gsqEngine.unload()
                Memory.clearCache()
            }
        }
        return (allBatches, teacherTraces)
    }

    static func runQwenTuned(
        teacherPath: String, batches: [PVTeacherBatch], context: PVArmEvalContext, config: PVTuneConfig
    ) async throws {
        let trainBatches = Array(batches.prefix(32))
        let valBatches = Array(batches.suffix(8))
        if !skipQwen3Bit {
            let s3Engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
            try await context.log.step("load qwen-3bit-rtn") {
                try await s3Engine.load(modelPath: NF4Run.path("lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"))
            }
            if !skipBaselines { _ = try await evaluateArm("3-bit affine RTN", engine: s3Engine, context: context) }
            let sc = try #require(await s3Engine.container)
            let summary = try await context.log.step("pv-tune qwen-3bit") {
                try await sc.perform { ctx in
                    try PVTuner.tune(ctx.model, train: trainBatches, val: valBatches, config: config) { context.log.record("pv-progress", $0) }
                }
            }
            _ = try await evaluateArm("3-bit affine + PV-Tuning", engine: s3Engine, context: context, tuneSummary: summary)
            await s3Engine.unload()
            Memory.clearCache()
        }
        let s2Engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await context.log.step("load qwen-4bit-for-2bit") { try await s2Engine.load(modelPath: teacherPath) }
        let sc2 = try #require(await s2Engine.container)
        try await context.log.step("requant qwen-2bit-rtn") {
            try await sc2.perform { ctx in try PVTuner.requantizeModel(ctx.model as Module, bits: 2, groupSize: 64, mode: .affine) }
        }
        if !skipBaselines { _ = try await evaluateArm("2-bit affine RTN", engine: s2Engine, context: context) }
        let summary2 = try await context.log.step("pv-tune qwen-2bit") {
            try await sc2.perform { ctx in
                try PVTuner.tune(ctx.model, train: trainBatches, val: valBatches, config: config) { context.log.record("pv-progress-2b", $0) }
            }
        }
        _ = try await evaluateArm("2-bit affine + PV-Tuning", engine: s2Engine, context: context, tuneSummary: summary2)
        await s2Engine.unload()
        Memory.clearCache()
    }
}

@Suite("PV-Tuning Phase 0 real-model evaluation on Qwen3.8-27B and Gemma 4 26B-A4B across 2/3-bit affine and nf4 (#410)", .serialized, .timeLimit(.minutes(180)))
struct PVTuningRealModelTests {
    static let enabled = PVEvalHarness.env["BASALT_E2E_PV_TUNING"] == "1"
    static let familyFilter = PVEvalHarness.env["BASALT_PV_FAMILY"] ?? "all"

    @Test("Qwen3.8-27B and Gemma 4 26B-A4B PV-Tuning Phase 0 vs RTN, GPTQ, nf4 and GSQ-RCO",
          .enabled(if: enabled, "set BASALT_E2E_PV_TUNING=1"))
    func evaluatePVTuning() async throws {
        let groups = CascadeDecisionRealModelTests.groups(SystemOneDecisionSet.generate(states: 100, seed: 234))
        if Self.familyFilter == "all" || Self.familyFilter == "qwen" {
            let teacherPath = NF4Run.path("mlx-community/Qwen3.8-27B-GPTQ-4bit")
            let log = try await ModelRunLog("pv-tuning", modelPath: teacherPath)
            defer { log.releaseGPU() }
            var cfg = PVTuneConfig()
            cfg.globalContinuousSteps = 32
            cfg.continuousStepsPerGroup = 10
            cfg.batchesPerDiscreteStep = Int(PVEvalHarness.env["BASALT_PV_DISC_BATCHES"] ?? "") ?? 6
            cfg.maxLayerGroups = Int(PVEvalHarness.env["BASALT_PV_QWEN_GROUPS"] ?? "") ?? 3
            cfg.maxActiveElements = 200_000_000
            let (batches, traces) = try await PVEvalHarness.runQwenBaselines(teacherPath: teacherPath, decisionGroups: groups, log: log)
            let ctx = PVArmEvalContext(family: "Qwen3.8-27B", teacherTraces: traces, decisionGroups: groups, log: log)
            try await PVEvalHarness.runQwenTuned(teacherPath: teacherPath, batches: batches, context: ctx, config: cfg)
        }
        if Self.familyFilter == "all" || Self.familyFilter == "gemma" {
            let gemmaPath = NF4Run.path("mlx-community/gemma-4-26B-A4B-it-qat-4bit")
            let log = try await ModelRunLog("pv-tuning", modelPath: gemmaPath)
            defer { log.releaseGPU() }
            let (batches, ctx, cfg) = try await runGemma3Bit(gemmaPath: gemmaPath, decisionGroups: groups, log: log)
            try await runGemmaNF4(gemmaPath: gemmaPath, batches: batches, context: ctx, baseConfig: cfg)
        }
    }

    func runGemma3Bit(
        gemmaPath: String, decisionGroups: [(request: SystemOneRequest, items: [LabelledDecision])], log: ModelRunLog
    ) async throws -> ([PVTeacherBatch], PVArmEvalContext, PVTuneConfig) {
        let gEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await log.step("load gemma-4bit-qat-teacher") { try await gEngine.load(modelPath: gemmaPath) }
        let gc = try #require(await gEngine.container)
        let allBatches = await PVTuner.collectTeacherBatches(gc, texts: PVEvalHarness.calibrationCorpus, count: 40, seqLen: 128, topK: 64)
        var ctx = PVArmEvalContext(family: "Gemma-4-26B-A4B", teacherTraces: nil, decisionGroups: decisionGroups, log: log)
        if PVEvalHarness.skipBaselines {
            ctx.teacherTraces = try await log.step("golden gemma-teacher-ref") { try await GoldenRun.traces(gEngine, run: 0, log: log) }
        } else {
            ctx.teacherTraces = try await PVEvalHarness.evaluateArm("4-bit QAT (teacher)", engine: gEngine, context: ctx).traces
        }
        try await log.step("requant gemma-3bit-rtn") {
            try await gc.perform { c in try PVTuner.requantizeModel(c.model as Module, bits: 3, groupSize: 64, mode: .affine) }
        }
        if !PVEvalHarness.skipBaselines { _ = try await PVEvalHarness.evaluateArm("3-bit affine RTN", engine: gEngine, context: ctx) }
        var cfg = PVTuneConfig()
        cfg.batchesPerDiscreteStep = Int(PVEvalHarness.env["BASALT_PV_DISC_BATCHES"] ?? "") ?? 6
        cfg.maxLayerGroups = Int(PVEvalHarness.env["BASALT_PV_GEMMA_GROUPS"] ?? "") ?? 3
        let gemmaCfg = cfg
        let trainBatches = Array(allBatches.prefix(32))
        let valBatches = Array(allBatches.suffix(8))
        let summary3 = try await log.step("pv-tune gemma-3bit") {
            try await gc.perform { c in
                try PVTuner.tune(c.model, train: trainBatches, val: valBatches, config: gemmaCfg) { log.record("pv-progress-gemma3b", $0) }
            }
        }
        _ = try await PVEvalHarness.evaluateArm("3-bit affine + PV-Tuning", engine: gEngine, context: ctx, tuneSummary: summary3)
        await gEngine.unload()
        Memory.clearCache()
        return (allBatches, ctx, gemmaCfg)
    }

    func runGemmaNF4(
        gemmaPath: String, batches: [PVTeacherBatch], context: PVArmEvalContext, baseConfig: PVTuneConfig
    ) async throws {
        let nf4Engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await context.log.step("reload gemma-for-nf4") { try await nf4Engine.load(modelPath: gemmaPath) }
        let nfc = try #require(await nf4Engine.container)
        try await context.log.step("requant gemma-nf4-rtn") {
            try await nfc.perform { c in
                let root = c.model as Module
                var replaced = 0
                for (path, mod) in root.namedModules() {
                    guard PVTuner.textBlockIndex(path) != nil, !path.contains("router"),
                          let ql = mod as? QuantizedLinear, ql.globalScale == nil, ql.bits <= 4,
                          (ql.weight.dim(-1) * 32 / ql.bits).isMultiple(of: 64) else { continue }
                    let dense = dequantized(ql.weight, scales: ql.scales, biases: ql.biases, groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
                    let repacked = QuantizedLinear(weight: dense, bias: ql.bias, groupSize: 64, bits: 4, mode: .nf4)
                    eval(repacked)
                    try PVTuner.swapModule(path, with: repacked, in: root)
                    replaced += 1
                    if replaced.isMultiple(of: 8) { Memory.clearCache() }
                }
                MLXInferenceEngine.settle(root)
                Memory.clearCache()
            }
        }
        _ = try await PVEvalHarness.evaluateArm("4-bit nf4 (dense) RTN", engine: nf4Engine, context: context)
        var nfCfg = baseConfig
        nfCfg.globalContinuousSteps = 0
        let finalNFCfg = nfCfg
        let trainBatches = Array(batches.prefix(32))
        let valBatches = Array(batches.suffix(8))
        let summaryNF4 = try await context.log.step("pv-tune gemma-nf4") {
            try await nfc.perform { c in
                try PVTuner.tune(c.model, train: trainBatches, val: valBatches, config: finalNFCfg) { context.log.record("pv-progress-gemmanf4", $0) }
            }
        }
        _ = try await PVEvalHarness.evaluateArm("4-bit nf4 (dense) + PV-Tuning", engine: nf4Engine, context: context, tuneSummary: summaryNF4)
        await nf4Engine.unload()
        Memory.clearCache()
    }
}
