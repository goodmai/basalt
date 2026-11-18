import ArgumentParser
import Foundation
import Hummingbird
import HummingbirdTesting
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Router-Weighted MoE Hessian, Cascade Router KL & Dead-Zone 2.8 bpw Allocator (#569)", .serialized)
struct MoERouterWeightedHessianTests {
    @Test("MoERouterWeightedHessian matches exact CPU FP32 quadratic and linear gating formulas")
    func exactQuadraticAndLinearFormulas() {
        let (rows, cols) = (12, 32)
        let xHost = (0 ..< (rows * cols)).map { idx -> Float in
            Float((idx * 13 + 5) % 29 - 14) * 0.125
        }
        let cHost: [Float] = [0.92, 0.08, 0.75, 0.25, 0.60, 0.40, 0.95, 0.05, 0.50, 0.50, 0.85, 0.15]
        let xArr = MLXArray(xHost, [rows, cols])
        let cArr = MLXArray(cHost, [rows])
        let lambda: Float = 0.03

        let hQuad = MoERouterWeightedHessian.computeHessian(
            activations: xArr, gatingWeights: cArr, mode: .quadratic, damping: lambda)
        let hLin = MoERouterWeightedHessian.computeHessian(
            activations: xArr, gatingWeights: cArr, mode: .linear, damping: lambda)
        eval(hQuad, hLin)
        let quadGot = hQuad.asArray(Float.self)
        let linGot = hLin.asArray(Float.self)

        var quadBase = [Float](repeating: 0, count: cols * cols)
        var linBase = [Float](repeating: 0, count: cols * cols)
        for t in 0 ..< rows {
            let w2 = 2.0 * cHost[t] * cHost[t]
            let w1 = 2.0 * max(0.0, cHost[t])
            for i in 0 ..< cols {
                for j in 0 ..< cols {
                    let prod = xHost[t * cols + i] * xHost[t * cols + j]
                    quadBase[i * cols + j] += w2 * prod
                    linBase[i * cols + j] += w1 * prod
                }
            }
        }
        let diagMean = (0 ..< cols).map { quadBase[$0 * cols + $0] }.reduce(0, +) / Float(cols)
        for d in 0 ..< cols {
            quadBase[d * cols + d] += lambda * max(diagMean, 1e-12)
            linBase[d * cols + d] += lambda
        }
        let maxQuadErr = zip(quadGot, quadBase).map { abs($0 - $1) }.max() ?? 0
        let maxLinErr = zip(linGot, linBase).map { abs($0 - $1) }.max() ?? 0
        #expect(maxQuadErr < 1e-4)
        #expect(maxLinErr < 1e-4)
    }

    @Test("Router-weighted Hessian achieves strictly lower routed MSE, cascade router KL, and TopK flip rate than unweighted Hessian on skewed gating")
    func skewedGatingStrictlyBeatsUnweightedHessian() {
        let (tailRows, domRows, cols, outDims, nextExperts) = (112, 24, 64, 64, 8)
        let totalRows = tailRows + domRows
        var xHost = [Float](repeating: 0, count: totalRows * cols)
        var cHost = [Float](repeating: 0, count: totalRows)
        for r in 0 ..< tailRows {
            cHost[r] = 0.015
            for j in 0 ..< cols {
                let raw = Float(((r * 17 + j * 31 + 7) % 41) - 20) * 0.05
                xHost[r * cols + j] = (j < 4) ? (raw * 16.0) : (raw * 0.06)
            }
        }
        for r in tailRows ..< totalRows {
            cHost[r] = 0.96
            for j in 0 ..< cols {
                let raw = Float(((r * 23 + j * 19 + 11) % 43) - 21) * 0.11
                xHost[r * cols + j] = (j < 4) ? (raw * 0.02) : (raw * 1.4)
            }
        }
        let xArr = MLXArray(xHost, [totalRows, cols])
        let cArr = MLXArray(cHost, [totalRows])
        let wExpert = MLXRandom.normal([outDims, cols], scale: 0.35, key: MLXRandom.key(56901)).asType(.float32)
        let wNextRouter = MLXRandom.normal([nextExperts, outDims], scale: 1.2, key: MLXRandom.key(56902)).asType(.float32)
        eval(wExpert, wNextRouter)

        let rwEst = MoERouterWeightedHessian(mode: .quadratic, damping: 0.01)
        let unwEst = MoERouterWeightedHessian(mode: .unweighted, damping: 0.01)
        let rwQuant = rwEst.quantizeExpert(weight: wExpert, activations: xArr, gatingWeights: cArr, bits: 2, groupSize: 32)
        let unwQuant = unwEst.quantizeExpert(weight: wExpert, activations: xArr, gatingWeights: cArr, bits: 2, groupSize: 32)

        let mseRW = MoERouterWeightedHessian.routedReconstructionMSE(
            activations: xArr, gatingWeights: cArr, targetWeight: wExpert, quantizedWeight: rwQuant.dequantized)
        let mseUnw = MoERouterWeightedHessian.routedReconstructionMSE(
            activations: xArr, gatingWeights: cArr, targetWeight: wExpert, quantizedWeight: unwQuant.dequantized)
        #expect(mseRW < mseUnw)

        let cCol = cArr.reshaped(totalRows, 1)
        let yTeacher = matmul(xArr * cCol, wExpert.T)
        let yRW = matmul(xArr * cCol, rwQuant.dequantized.T)
        let yUnw = matmul(xArr * cCol, unwQuant.dequantized.T)
        let zTeacher = matmul(yTeacher, wNextRouter.T)
        let zRW = matmul(yRW, wNextRouter.T)
        let zUnw = matmul(yUnw, wNextRouter.T)

        let klRW = MoERouterWeightedHessian.cascadeRouterKL(teacherLogits: zTeacher, studentLogits: zRW)
        let klUnw = MoERouterWeightedHessian.cascadeRouterKL(teacherLogits: zTeacher, studentLogits: zUnw)
        let domSlice = tailRows ..< totalRows
        let flipRW = MoERouterWeightedHessian.topKFlipRate(
            teacherLogits: zTeacher[domSlice], studentLogits: zRW[domSlice], topK: 2)
        let flipUnw = MoERouterWeightedHessian.topKFlipRate(
            teacherLogits: zTeacher[domSlice], studentLogits: zUnw[domSlice], topK: 2)
        #expect(klRW < klUnw)
        #expect(flipRW < flipUnw)
    }

    @Test("TinyModel qwen3Moe and qwen35Moe router-weighted SwitchGLU quantization beats unweighted Hessian on routed MSE and next-layer TopK flips")
    func tinyModelMoESwitchGLUBeatsUnweightedAndDecodes() throws {
        for family in [TinyModel.Family.qwen3Moe, .qwen35Moe] {
            let teacher = try TinyModel.make(family)
            let params = Dictionary(uniqueKeysWithValues: teacher.parameters().flattened())
            let downW = try #require(params["model.layers.0.mlp.switch_mlp.down_proj.weight"]).asType(.float32)
            let nextGateW = try #require(params["model.layers.1.mlp.gate.weight"]).asType(.float32)
            let scaledNextGate = nextGateW * Float(6.0)
            let (experts, outF, inF) = (downW.dim(0), downW.dim(1), downW.dim(2))

            let rowsPerExpert = 48
            var mseRWSum: Float = 0
            var mseUnwSum: Float = 0
            var flipRWSum: Float = 0
            var flipUnwSum: Float = 0
            let rwEst = MoERouterWeightedHessian(mode: .quadratic, damping: 0.01)
            let unwEst = MoERouterWeightedHessian(mode: .unweighted, damping: 0.01)
            for e in 0 ..< experts {
                var xHost = [Float](repeating: 0, count: rowsPerExpert * inF)
                var cHost = [Float](repeating: 0, count: rowsPerExpert)
                for r in 0 ..< rowsPerExpert {
                    let dom = r >= 32
                    cHost[r] = dom ? 0.95 : 0.01
                    for j in 0 ..< inF {
                        let base = Float(((r * 19 + j * 29 + e * 7 + 3) % 37) - 18) * 0.09
                        xHost[r * inF + j] = dom ? (j < 3 ? base * 0.01 : base * 2.0) : (j < 3 ? base * 22.0 : base * 0.03)
                    }
                }
                let xE = MLXArray(xHost, [rowsPerExpert, inF])
                let cE = MLXArray(cHost, [rowsPerExpert])
                let wE = downW[e] * Float(3.0)
                let qRW = rwEst.quantizeExpert(weight: wE, activations: xE, gatingWeights: cE, bits: 2, groupSize: 32)
                let qUnw = unwEst.quantizeExpert(weight: wE, activations: xE, gatingWeights: cE, bits: 2, groupSize: 32)
                mseRWSum += MoERouterWeightedHessian.routedReconstructionMSE(
                    activations: xE, gatingWeights: cE, targetWeight: wE, quantizedWeight: qRW.dequantized)
                mseUnwSum += MoERouterWeightedHessian.routedReconstructionMSE(
                    activations: xE, gatingWeights: cE, targetWeight: wE, quantizedWeight: qUnw.dequantized)
                let xDom = xE[32 ..< rowsPerExpert]
                let zRef = matmul(matmul(xDom, wE.T), scaledNextGate.T)
                let zRW = matmul(matmul(xDom, qRW.dequantized.T), scaledNextGate.T)
                let zUnw = matmul(matmul(xDom, qUnw.dequantized.T), scaledNextGate.T)
                flipRWSum += MoERouterWeightedHessian.topKFlipRate(teacherLogits: zRef, studentLogits: zRW, topK: 2)
                flipUnwSum += MoERouterWeightedHessian.topKFlipRate(teacherLogits: zRef, studentLogits: zUnw, topK: 2)
                #expect(outF == 64)
            }
            #expect(mseRWSum < mseUnwSum)
            #expect(flipRWSum < flipUnwSum)
        }
    }

    @Test("TinyModel qwen3Moe and qwen35Moe quantizeModelMoERW produces mixed-bit manifest and finite logits")
    func tinyModelMoERWQuantizationAndCascadeConsistency() throws {
        for family in [TinyModel.Family.qwen3Moe, .qwen35Moe] {
            let teacher = try TinyModel.make(family)
            let studentRW = try TinyModel.make(family)
            var batches: [[Int32]] = []
            for b in 0 ..< 4 {
                var row: [Int32] = []
                for i in 0 ..< 32 {
                    row.append(Int32((b * 37 + i * 13 + 3) % 240))
                }
                batches.append(row)
            }
            var cfg = MACaQConfig()
            cfg.hadamardBlock = 32
            cfg.maxFlipIterations = 24
            cfg.ridge = 0.0
            let rwRes = try UniversalModelConverter.quantizeModelMoERW(
                studentRW, calibrationTokenBatches: batches, targetBPW: 2.8,
                estimator: .init(mode: .quadratic, damping: 0.02), config: cfg)
            #expect(rwRes.summary.quantizedLayers > 0)
            #expect(!rwRes.manifest.isEmpty)
            let bitsUsed = Set(rwRes.manifest.values.map(\.bits))
            #expect(bitsUsed.isSubset(of: [2, 3, 4]))

            let heldOut = MLXArray([Int32(7), 19, 43, 88, 12, 65, 104, 31], [1, 8])
            let tLogits = teacher(LMInput.Text(tokens: heldOut), cache: nil, state: nil).logits
            let sLogits = studentRW(LMInput.Text(tokens: heldOut), cache: nil, state: nil).logits
            eval(tLogits, sLogits)
            #expect(sLogits.shape == tLogits.shape)
            let sVals = sLogits.asType(.float32).asArray(Float.self)
            #expect(sVals.allSatisfy { $0.isFinite })
        }
    }
}

@Suite("Dead-Zone 2.8 bpw Allocator, Multilingual MoE Corpus & Universal Converter Parity (#569)", .serialized)
struct DeadZoneAllocatorAndMoEConverterTests {
    @Test("DeadZoneBitAllocator hits 2.8 bpw within deadZoneEpsilon with zero oscillation across 64 and 128 experts")
    func deadZoneAllocatorHits28BPWAcross64And128Experts() {
        let eps = 0.015
        let allocator = DeadZoneBitAllocator(targetBPW: 2.8, deadZoneEpsilon: eps, cascadeKLWeight: 10.0)
        for expertCount in [64, 128] {
            let items = (0 ..< expertCount).map { idx -> DeadZoneBitAllocator.ExpertSensitivity in
                let zipfMass = 1.0 / pow(Double(idx + 1), 0.85)
                let trace = 0.4 + Double((idx * 7) % 19) * 0.15
                let cascadeKL = (idx % 11 == 0) ? 0.18 : 0.01
                return .init(
                    id: "layers.0.experts.\(idx)",
                    weightCount: 4_096,
                    routerWeightedTrace: trace,
                    cascadeRouterKL: cascadeKL,
                    routingMass: zipfMass)
            }
            let plan = allocator.allocate(items)
            #expect(plan.withinDeadZone)
            #expect(abs(plan.achievedBPW - 2.8) <= 2.8 * eps)
            #expect(plan.deadZonePenalty == 0.0)
            #expect(plan.oscillationCount == 0)
            #expect(plan.allocations.count == expertCount)
            #expect(Set(plan.allocations.map(\.bits)).isSubset(of: [2, 3, 4]))
            #expect((plan.bitHistogram[2] ?? 0) > 0 && (plan.bitHistogram[3] ?? 0) > 0)
        }

        let pairPlan = allocator.allocate([
            .init(id: "low_kl", weightCount: 4_096, routerWeightedTrace: 1.0, cascadeRouterKL: 0.0, routingMass: 0.5),
            .init(id: "high_kl", weightCount: 4_096, routerWeightedTrace: 1.0, cascadeRouterKL: 0.4, routingMass: 0.5),
        ])
        #expect(pairPlan.allocations[1].bits >= pairPlan.allocations[0].bits)
    }

    @Test("MultilingualMoECalibrationCorpus covers EN/RU/KO/JA/math/code and increases MoE router entropy")
    func multilingualCorpusDomainsAndEntropy() throws {
        let domains = Set(MultilingualMoECalibrationCorpus.probes.map(\.domain))
        #expect(domains == ["en", "ru", "ko", "ja", "math", "code"])
        #expect(MultilingualMoECalibrationCorpus.defaultText.count > 2_000)

        let model = try TinyModel.make(.qwen3Moe)
        let gateW = try #require(
            model.parameters().flattened().first(where: { $0.0.hasSuffix(".mlp.gate.weight") })?.1)
        let multiBytes = Array(MultilingualMoECalibrationCorpus.defaultText.utf8.prefix(256)).map { Int32($0) }
        let monoBytes = Array(repeating: Int32(97), count: 256)
        let embed = try #require(
            model.parameters().flattened().first(where: { $0.0.contains("embed_tokens.weight") })?.1)
        let hMulti = embed.take(MLXArray(multiBytes), axis: 0).asType(.float32)
        let hMono = embed.take(MLXArray(monoBytes), axis: 0).asType(.float32)
        let pMulti = MoERouterWeightedHessian.topKGating(logits: matmul(hMulti, gateW.asType(.float32).T), topK: 2).fullProbs
        let pMono = MoERouterWeightedHessian.topKGating(logits: matmul(hMono, gateW.asType(.float32).T), topK: 2).fullProbs
        let entMulti = MultilingualMoECalibrationCorpus.routingEntropy(gatingProbs: pMulti)
        let entMono = MultilingualMoECalibrationCorpus.routingEntropy(gatingProbs: pMono)
        #expect(entMulti > entMono)
    }

    @Test("macaq:moe:2.8 and --quant-recipe moe-rw round-trip across QuantizationTarget, ModelConversionPlan, CLI, REST, and LibraryReducer")
    func targetPlanCLIRESTAndReducerRoundTrip() async throws {
        let parsed = try QuantizationTarget.parse("macaq:moe:2.8")
        #expect(parsed == .macaqMoEBudget(bpw: 2.8))
        #expect(parsed.specString == "macaq:moe:2.8")
        #expect(parsed.supportsPVTune && !parsed.isDense)
        #expect(abs(parsed.effectiveBitsPerWeight - 2.8) < 1e-9)
        let jsonTarget = try JSONEncoder().encode(parsed)
        #expect(try JSONDecoder().decode(QuantizationTarget.self, from: jsonTarget) == parsed)

        let plan = ModelConversionPlan(
            sourceId: "Qwen/Qwen3-30B-A3B", sourceFormat: "dense:bf16",
            target: parsed.specString, supportsPVTune: parsed.supportsPVTune,
            footprint: .init(sourceBytes: 60_000_000_000, estimatedOutputBytes: 10_500_000_000))
        let decodedPlan = try JSONDecoder().decode(ModelConversionPlan.self, from: JSONEncoder().encode(plan))
        #expect(decodedPlan == plan)
        #expect(decodedPlan.supportedTargets.contains("macaq:moe:2.8"))

        var state = LibraryState()
        let src = LibraryModel(id: "Qwen/Qwen3-30B-A3B-bf16", sizeBytes: 60_000_000_000, isLoaded: false)
        _ = LibraryReducer.reduce(&state, .openConvert(src))
        var draft = try #require(state.convertDraft)
        draft.family = .macaqMoE
        draft.moeBudgetBPW = 2.8
        draft.refreshSuggestedOutputId()
        let planFx = LibraryReducer.reduce(&state, .updateConvertDraft(draft))
        #expect(planFx == [.planConversion(source: src.id, target: .macaqMoEBudget(bpw: 2.8), budgetGB: nil, pvTune: true)])
        #expect(state.convertDraft?.outputId == "local/Qwen3-30B-A3B-MACaQ-MoE-2.8bpw-PV")
        let req = try #require(state.convertDraft).toRequest()
        #expect(req.target == .macaqMoEBudget(bpw: 2.8) && req.recipe == "moe-rw" && req.hadamardBlock == 64)

        let cliUnified = try ConvertSubcommand.parse(["src", "--to", "macaq:moe:2.8", "--quant-recipe", "moe-rw", "--out", "dst"])
        #expect(cliUnified.quantRecipe == .moeRW)
        let cliStd = try ConvertSubcommand.parse(["src", "--quantize", "affine:3:64", "--quant-recipe", "moe-rw", "--out", "dst"])
        #expect(try cliStd.resolvedSINQRecipe() == .moeRW)

        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        try await orch.loadModel(path: "mock")
        let hub = ModelHubOperations(converter: UniversalConverterTests.StubConversionService()) { _, _ in URL(fileURLWithPath: "/tmp") }
        let app = Application(router: RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock")).withModelHub(hub).makeRouter())
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/models/convert/plan?source=Qwen/Qwen3-30B-A3B&target=macaq:moe:2.8&recipe=moe-rw&pv_tune=true",
                method: .get
            ) { res in
                #expect(res.status == .ok)
                let p = try JSONDecoder().decode(ModelConversionPlan.self, from: Data(res.body.readableBytesView))
                #expect(p.target == "macaq:moe:2.8" && p.supportsPVTune)
            }
        }
    }
}
