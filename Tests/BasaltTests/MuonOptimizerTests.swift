import ArgumentParser
import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import Synchronization
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI

private final class MixedRankModule: Module {
    var matrix: MLXArray
    var bias: MLXArray
    var scalar: MLXArray
    var tensor: MLXArray

    init(matrix: MLXArray, bias: MLXArray, scalar: MLXArray, tensor: MLXArray) {
        self.matrix = matrix
        self.bias = bias
        self.scalar = scalar
        self.tensor = tensor
        super.init()
    }
}

private final class MatrixOnlyModule: Module {
    var matrix: MLXArray

    init(matrix: MLXArray) {
        self.matrix = matrix
        super.init()
    }
}

private final class NonMatrixModule: Module {
    var bias: MLXArray
    var scalar: MLXArray
    var tensor: MLXArray

    init(bias: MLXArray, scalar: MLXArray, tensor: MLXArray) {
        self.bias = bias
        self.scalar = scalar
        self.tensor = tensor
        super.init()
    }
}

private final class BiasOnlyModule: Module {
    var bias: MLXArray

    init(bias: MLXArray) {
        self.bias = bias
        super.init()
    }
}

private enum MuonTestSupport {
    static let matrixValues: [Float] = (0 ..< 24).map { Float(sin(Double($0) * 1.7) + 0.1 * Double($0 % 5)) }
    static let matrixGradValues: [Float] = (0 ..< 24).map { Float(cos(Double($0) * 1.3) + 0.15 * Double($0 % 4)) }
    static let biasValues: [Float] = [0.5, -0.25, 0.75, -1.0, 0.125, 0.625]
    static let biasGradValues: [Float] = [0.2, -0.4, 0.6, -0.3, 0.5, -0.1]
    static let tensorValues: [Float] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    static let tensorGradValues: [Float] = [-0.3, 0.2, -0.1, 0.4, -0.5, 0.6, -0.2, 0.1]

    static func freshMatrix() -> MLXArray { MLXArray(matrixValues, [6, 4]) }
    static func freshMatrixGrad() -> MLXArray { MLXArray(matrixGradValues, [6, 4]) }
    static func freshBias() -> MLXArray { MLXArray(biasValues, [6]) }
    static func freshBiasGrad() -> MLXArray { MLXArray(biasGradValues, [6]) }
    static func freshScalar() -> MLXArray { MLXArray(Float(2.0)) }
    static func freshScalarGrad() -> MLXArray { MLXArray(Float(-0.35)) }
    static func freshTensor() -> MLXArray { MLXArray(tensorValues, [2, 2, 2]) }
    static func freshTensorGrad() -> MLXArray { MLXArray(tensorGradValues, [2, 2, 2]) }

    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let diff = MLX.abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }

    static func singularValues(_ array: MLXArray) -> MLXArray {
        MLX.sqrt(MLX.maximum(0, MLX.eigvalsh(matmul(array.T, array), stream: .cpu)))
    }

    static func scratch(_ prefix: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "\(prefix)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func writeCorpus(to directory: URL) throws -> URL {
        let file = directory.appending(component: "muon_corpus.jsonl")
        let lines = (0 ..< 8).map { idx in
            #"{"messages":[{"role":"user","content":"Question \#(idx): colour?"},{"role":"assistant","content":"The colour is blue \#(idx % 2)."}]}"#
        }.joined(separator: "\n")
        try Data(lines.utf8).write(to: file)
        return file
    }
}

@Suite("Muon optimizer: CLI/MCP parsing, 2-D Newton-Schulz partitioning, checkpoint resume, and trainer convergence (#823)", .serialized, .timeLimit(.minutes(2)))
struct MuonOptimizerTests {
    @Test("CLI models train, distill, and train-jepa default to adamw, accept muon and adamw, and reject invalid values")
    func cliOptionParsing() throws {
        let trainDefault = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl"])
        let trainAdamW = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--optimizer", "adamw"])
        let trainMuon = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--optimizer", "muon"])
        #expect(trainDefault.optimizer == .adamw && trainAdamW.optimizer == .adamw && trainMuon.optimizer == .muon)
        #expect(throws: (any Error).self) { _ = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--optimizer", "sgd"]) }

        let distillBase = ["org/model", "--prompts", "p.jsonl", "--teacher-url", "http://127.0.0.1:8080/v1", "--teacher-model", "t"]
        let distillDefault = try DistillSubcommand.parse(distillBase)
        let distillAdamW = try DistillSubcommand.parse(distillBase + ["--optimizer", "adamw"])
        let distillMuon = try DistillSubcommand.parse(distillBase + ["--optimizer", "muon"])
        #expect(distillDefault.optimizer == .adamw && distillAdamW.optimizer == .adamw && distillMuon.optimizer == .muon)
        #expect(throws: (any Error).self) { _ = try DistillSubcommand.parse(distillBase + ["--optimizer", "rmsprop"]) }

        let jepaBase = ["org/model", "--dataset", "scenes.jsonl"]
        let jepaDefault = try TrainJepaSubcommand.parse(jepaBase)
        let jepaAdamW = try TrainJepaSubcommand.parse(jepaBase + ["--optimizer", "adamw"])
        let jepaMuon = try TrainJepaSubcommand.parse(jepaBase + ["--optimizer", "muon"])
        #expect(jepaDefault.optimizer == .adamw && jepaAdamW.optimizer == .adamw && jepaMuon.optimizer == .muon)
        #expect(throws: (any Error).self) { _ = try TrainJepaSubcommand.parse(jepaBase + ["--optimizer", "invalid"]) }
    }

    @Test("Request structs default to adamw and MCP parsers accept muon and reject invalid optimizers")
    func requestDefaultsAndMCP() throws {
        #expect(CorpusTrainRequest(corpus: CorpusSourceSpec(source: "local.jsonl")).optimizer == .adamw)
        #expect(DistillRequest(prompts: "p.jsonl", teacher: TeacherSpec(url: "https://t.test/v1", model: "m"), gate: GateRequest(heldOut: ["a"])).optimizer == .adamw)
        #expect(LeJEPATrainRequest(datasetPath: "s.jsonl", iterations: 4).optimizer == .adamw)
        #expect(TrainRequest(tasks: .arithmetic(count: 4, seed: 1), gate: GateRequest(heldOut: ["a"])).optimizer == .adamw)

        let mcpDistill = try DistillRequest(mcp: [
            "prompts": "p.jsonl", "teacherUrl": "http://127.0.0.1:8080/v1", "teacherModel": "t",
            "heldOut": ["Explain gravity."], "optimizer": "muon",
        ])
        let mcpJepa = try LeJEPATrainRequest(mcp: ["datasetPath": "s.jsonl", "optimizer": "muon"])
        let mcpTrain = try TrainRequest(mcp: ["heldOut": ["Hi"], "optimizer": "muon"])
        #expect(mcpDistill.optimizer == .muon && mcpJepa.optimizer == .muon && mcpTrain.optimizer == .muon)
        #expect(ResearchTool.trainJepaProperties["optimizer"] != nil)
        #expect(ResearchTool.distillProperties["optimizer"] != nil)
        #expect(ResearchTool.trainProperties["optimizer"] != nil)

        #expect(throws: MCPFailure.self) { _ = try LeJEPATrainRequest(mcp: ["datasetPath": "s.jsonl", "optimizer": "sgd"]) }
        #expect(throws: MCPFailure.self) {
            _ = try DistillRequest(mcp: [
                "prompts": "p.jsonl", "teacherUrl": "http://127.0.0.1:8080/v1", "teacherModel": "t",
                "heldOut": ["Hi"], "optimizer": "sgd",
            ])
        }
        #expect(throws: MCPFailure.self) { _ = try TrainRequest(mcp: ["heldOut": ["Hi"], "optimizer": "sgd"]) }
    }

    @Test("Composite Muon routes 2-D matrices to Newton-Schulz orthogonalization and 0-D/1-D/3-D parameters to AdamW")
    func partitionsTwoDimToMuonAndOthersToAdamW() {
        let lr: Float = 0.05
        let wd: Float = 0.01
        let mixed = MixedRankModule(
            matrix: MuonTestSupport.freshMatrix(), bias: MuonTestSupport.freshBias(),
            scalar: MuonTestSupport.freshScalar(), tensor: MuonTestSupport.freshTensor()
        )
        let matrixRef = MatrixOnlyModule(matrix: MuonTestSupport.freshMatrix())
        let adamMatrixRef = MatrixOnlyModule(matrix: MuonTestSupport.freshMatrix())
        let nonMatrixRef = NonMatrixModule(
            bias: MuonTestSupport.freshBias(), scalar: MuonTestSupport.freshScalar(), tensor: MuonTestSupport.freshTensor()
        )

        let composite = TrainingOptimizerFactory.make(.muon, learningRate: lr, weightDecay: wd)
        let muonRef = Muon(learningRate: lr, weightDecay: wd)
        let adamRef = AdamW(learningRate: lr, weightDecay: wd)
        let adamFallbackRef = AdamW(learningRate: lr, weightDecay: wd)

        for _ in 0 ..< 2 {
            let mixedGrads = ModuleParameters.unflattened([
                ("matrix", MuonTestSupport.freshMatrixGrad()),
                ("bias", MuonTestSupport.freshBiasGrad()),
                ("scalar", MuonTestSupport.freshScalarGrad()),
                ("tensor", MuonTestSupport.freshTensorGrad()),
            ])
            let matrixGrads = ModuleParameters.unflattened([("matrix", MuonTestSupport.freshMatrixGrad())])
            let nonMatrixGrads = ModuleParameters.unflattened([
                ("bias", MuonTestSupport.freshBiasGrad()),
                ("scalar", MuonTestSupport.freshScalarGrad()),
                ("tensor", MuonTestSupport.freshTensorGrad()),
            ])
            composite.update(model: mixed, gradients: mixedGrads)
            muonRef.update(model: matrixRef, gradients: matrixGrads)
            adamRef.update(model: adamMatrixRef, gradients: matrixGrads)
            adamFallbackRef.update(model: nonMatrixRef, gradients: nonMatrixGrads)
            eval(mixed, composite, matrixRef, muonRef, adamMatrixRef, adamRef, nonMatrixRef, adamFallbackRef)
        }

        #expect(MuonTestSupport.maxAbsDiff(mixed.matrix, matrixRef.matrix) < 1e-5)
        #expect(MuonTestSupport.maxAbsDiff(mixed.matrix, adamMatrixRef.matrix) > 1e-3)
        #expect(MuonTestSupport.maxAbsDiff(mixed.bias, nonMatrixRef.bias) < 1e-5)
        #expect(MuonTestSupport.maxAbsDiff(mixed.scalar, nonMatrixRef.scalar) < 1e-5)
        #expect(MuonTestSupport.maxAbsDiff(mixed.tensor, nonMatrixRef.tensor) < 1e-5)
    }

    @Test("Step-1 Muon update conditions singular values via Newton-Schulz and learningRate setter propagates to both sub-optimizers")
    func newtonSchulzConditioningAndSchedulePropagation() {
        let model = MatrixOnlyModule(matrix: MuonTestSupport.freshMatrix())
        let before = MuonTestSupport.freshMatrix()
        let opt = TrainingOptimizerFactory.make(.muon, learningRate: 1.0, weightDecay: 0.0)
        opt.update(model: model, gradients: ModuleParameters.unflattened([("matrix", MuonTestSupport.freshMatrix())]))
        eval(model, opt)

        let scale = Float((6.0 / 4.0).squareRoot())
        let update = (before - model.matrix) / scale
        let inputSV = MuonTestSupport.singularValues(before)
        let updateSV = MuonTestSupport.singularValues(update)
        eval(inputSV, updateSV)
        let inputCond = inputSV.max().item(Float.self) / inputSV.min().item(Float.self)
        let updateCond = updateSV.max().item(Float.self) / updateSV.min().item(Float.self)
        #expect(inputCond > 10)
        #expect(updateCond < 2)
        #expect(updateSV.min().item(Float.self) > 0.5 && updateSV.max().item(Float.self) < 1.5)

        let zeroLRModel = MixedRankModule(
            matrix: MuonTestSupport.freshMatrix(), bias: MuonTestSupport.freshBias(),
            scalar: MuonTestSupport.freshScalar(), tensor: MuonTestSupport.freshTensor()
        )
        let scheduled = TrainingOptimizerFactory.make(.muon, learningRate: 0.1, weightDecay: 0.0)
        scheduled.learningRate = 0.0
        scheduled.update(
            model: zeroLRModel,
            gradients: ModuleParameters.unflattened([
                ("matrix", MuonTestSupport.freshMatrixGrad()),
                ("bias", MuonTestSupport.freshBiasGrad()),
                ("scalar", MuonTestSupport.freshScalarGrad()),
                ("tensor", MuonTestSupport.freshTensorGrad()),
            ])
        )
        eval(zeroLRModel, scheduled)
        #expect(MuonTestSupport.maxAbsDiff(zeroLRModel.matrix, MuonTestSupport.freshMatrix()) == 0)
        #expect(MuonTestSupport.maxAbsDiff(zeroLRModel.bias, MuonTestSupport.freshBias()) == 0)
    }
}

extension MuonOptimizerTests {
    @Test("All-2D, all-1D, and empty-gradient updates succeed without crashing and preserve state")
    func edgeCasesAndStateValidation() throws {
        let matrixOnly = MatrixOnlyModule(matrix: MuonTestSupport.freshMatrix())
        let muonMat = TrainingOptimizerFactory.make(.muon, learningRate: 0.05)
        muonMat.update(model: matrixOnly, gradients: ModuleParameters.unflattened([("matrix", MuonTestSupport.freshMatrixGrad())]))
        eval(matrixOnly, muonMat)
        let matState = Dictionary(uniqueKeysWithValues: muonMat.state().flattened().map { ($0.0, $0.1.shape) })
        #expect(matState == ["states.0.matrix": [6, 4]])
        #expect(AdapterCheckpoint.mismatch(weights: ["matrix": [6, 4]], optimizer: matState, trainable: ["matrix": [6, 4]], kind: .muon) == nil)
        #expect(AdapterCheckpoint.mismatch(weights: ["matrix": [6, 4]], optimizer: matState, trainable: ["matrix": [6, 4]], kind: .adamw) != nil)

        let snapMat = matrixOnly.matrix + MLXArray(Float(0))
        eval(snapMat)
        muonMat.update(model: matrixOnly, gradients: ModuleParameters())
        eval(matrixOnly, muonMat)
        #expect(MuonTestSupport.maxAbsDiff(matrixOnly.matrix, snapMat) == 0)
        #expect(muonMat.state().flattened().count == 1)

        let biasOnly = BiasOnlyModule(bias: MuonTestSupport.freshBias())
        let muonBias = TrainingOptimizerFactory.make(.muon, learningRate: 0.05)
        muonBias.update(model: biasOnly, gradients: ModuleParameters.unflattened([("bias", MuonTestSupport.freshBiasGrad())]))
        eval(biasOnly, muonBias)
        let biasState = Dictionary(uniqueKeysWithValues: muonBias.state().flattened().map { ($0.0, $0.1.shape) })
        #expect(Set(biasState.keys) == ["states.1.bias.m", "states.1.bias.v", "states.1.bias.step"])
        #expect(AdapterCheckpoint.mismatch(weights: ["bias": [6]], optimizer: biasState, trainable: ["bias": [6]], kind: .muon) == nil)

        let adamOpt = TrainingOptimizerFactory.make(.adamw, learningRate: 0.05)
        adamOpt.update(model: biasOnly, gradients: ModuleParameters.unflattened([("bias", MuonTestSupport.freshBiasGrad())]))
        eval(biasOnly, adamOpt)
        let snapBias = biasOnly.bias + MLXArray(Float(0))
        eval(snapBias)
        adamOpt.update(model: biasOnly, gradients: ModuleParameters())
        eval(biasOnly, adamOpt)
        #expect(MuonTestSupport.maxAbsDiff(biasOnly.bias, snapBias) == 0)
        #expect(adamOpt.state().flattened().count == 3)

        var muonReq = DistillRequest(prompts: "", teacher: TeacherSpec(url: "https://t.test/v1", model: "m"), gate: GateRequest(heldOut: ["a"]))
        muonReq.optimizer = .muon
        var adamReq = muonReq
        adamReq.optimizer = .adamw
        let muonProgress = AdapterProgress(muonReq, model: "m", data: "d", layerCount: 4).advanced(by: 1.0)
        let adamIdentity = AdapterProgress(adamReq, model: "m", data: "d", layerCount: 4)
        #expect(muonProgress.resumable(as: adamIdentity, iterations: 10)?.contains("same settings") == true)

        let legacyJSON = #"{"step":1,"losses":[1.0],"model":"m","data":"d","layers":[0,1,2,3],"layerCount":4,"rank":16,"batchSize":2,"learningRate":0.0001}"#
        let legacy = try JSONDecoder().decode(AdapterProgress.self, from: Data(legacyJSON.utf8))
        #expect(legacy.effectiveOptimizer == .adamw)
        #expect(legacy.resumable(as: adamIdentity, iterations: 10) == nil)
        #expect(legacy.resumable(as: AdapterProgress(muonReq, model: "m", data: "d", layerCount: 4), iterations: 10)?.contains("same settings") == true)
    }
}

extension MuonOptimizerTests {
    @Test("Corpus LoRA training with Muon converges on TinyModel, matches AdamW health, and supports PHM and OFT")
    func corpusTrainMuonConverges() async throws {
        let scratch = try MuonTestSupport.scratch("muon-corpus")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let jsonl = try MuonTestSupport.writeCorpus(to: scratch)
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let beforeWeights = await DistillTests.baseWeights(container)

        for kind in TrainingOptimizerKind.allCases {
            var req = CorpusTrainRequest(corpus: CorpusSourceSpec(source: jsonl.path, maxRows: 8, includeReasoning: false))
            req.method = .lora(rank: 4, scale: 8.0)
            req.optimizer = kind
            req.iterations = 10
            req.batchSize = 2
            req.learningRate = 5e-3
            req.maxSeqLen = 256
            req.checkpointEvery = 5
            req.output = scratch.appending(component: "lora-\(kind.rawValue)").path

            let report = try await engine.trainCorpus(req, lease: nil)
            #expect(!report.losses.isEmpty && report.losses.allSatisfy(\.isFinite))
            let nllBefore = try #require(report.nllBefore)
            let nllAfter = try #require(report.nllAfter)
            #expect(nllAfter < nllBefore, "\(kind.rawValue): \(nllBefore) -> \(nllAfter)")
            #expect(await DistillTests.baseWeights(container) == beforeWeights)
        }

        for (tag, method) in [
            ("phm", AdapterMethod.phm(hypercomplexDim: 4, factorRank: 1, sharedBasis: true, scale: 8.0)),
            ("oft", AdapterMethod.oft(blockSize: 8, parameterization: .cayley, blockShared: false)),
        ] {
            var req = CorpusTrainRequest(corpus: CorpusSourceSpec(source: jsonl.path, maxRows: 8, includeReasoning: false))
            req.method = method
            req.optimizer = .muon
            req.iterations = 8
            req.batchSize = 2
            req.learningRate = 5e-3
            req.maxSeqLen = 256
            req.checkpointEvery = 0
            req.output = scratch.appending(component: "muon-\(tag)").path
            let report = try await engine.trainCorpus(req, lease: nil)
            #expect(!report.losses.isEmpty && report.losses.allSatisfy(\.isFinite))
            #expect(try #require(report.nllAfter) < (try #require(report.nllBefore)))
            #expect(await DistillTests.baseWeights(container) == beforeWeights)
        }
    }

    @Test("Distillation with Muon lowers held-out NLL and resumes from its step-2 checkpoint identically")
    func distillMuonConvergesAndResumes() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        let scratch = try MuonTestSupport.scratch("muon-distill")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let (full, copy) = (scratch.appending(component: "full"), scratch.appending(component: "resumed"))
        let copied = Mutex(false)
        let snapshot = StepLease(enter: {
            let source = AdapterCheckpoint.directory(in: full)
            if PreemptibleTrainingTests.savedStep(full) == 2, !copied.withLock({ $0 }) {
                try? FileManager.default.createDirectory(at: copy, withIntermediateDirectories: true)
                try? FileManager.default.copyItem(at: source, to: AdapterCheckpoint.directory(in: copy))
                copied.withLock { $0 = true }
            }
            return 0
        }, leave: { _ in })

        var request = PreemptibleTrainingTests.request(output: full, iterations: 8)
        request.learningRate = 5e-3
        request.optimizer = .muon
        let reference = try await PreemptibleTrainingTests.run(engine, request, lease: snapshot)
        #expect(reference.losses.count == 8 && reference.losses.allSatisfy(\.isFinite))
        #expect(reference.nllAfter < reference.nllBefore, "\(reference.nllBefore) -> \(reference.nllAfter)")

        var resume = PreemptibleTrainingTests.request(output: copy, iterations: 8)
        resume.learningRate = 5e-3
        resume.optimizer = .muon
        resume.resume = true
        let resumed = try await PreemptibleTrainingTests.run(engine, resume)
        #expect(resumed.resumedAt == 2)
        #expect(resumed.losses.count == reference.losses.count)
        #expect(zip(resumed.losses, reference.losses).allSatisfy { Swift.abs($0 - $1) < 1e-4 })
        #expect(Swift.abs(resumed.nllAfter - reference.nllAfter) < 1e-4)
        #expect(await DistillTests.baseWeights(container) == before)
    }

    @Test("JEPA training with Muon lowers loss over mixed 2-D projector/LoRA and 1-D BatchNorm parameters on TinyModel")
    func trainJEPAMuonConvergesAgainstAdamW() async throws {
        let scratch = try MuonTestSupport.scratch("muon-jepa")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let dataset = try GoodViewDataset.parse(jsonl: LeJEPATrainTests.scenes)

        for kind in TrainingOptimizerKind.allCases {
            let container = try TinyModel.container(.qwen3)
            let before = await LeJEPATrainTests.weights(container)
            let directory = scratch.appending(component: "jepa-\(kind.rawValue)")
            var request = LeJEPATrainRequest(datasetPath: "-", iterations: 20)
            (request.rank, request.layers, request.batchSize, request.maxSeqLen, request.learningRate, request.warmupRatio) = (4, [1, 3], 4, 16, 5e-4, 0.1)
            request.optimizer = kind

            let report = try await container.performEvaluating(nonSendable: ()) { [request] context, _ in
                try withRandomState(MLXRandom.RandomState(seed: 7)) {
                    try MLXInferenceEngine.trainJEPA(request, dataset: dataset, directory: directory, context: context)
                }
            }
            #expect(report.losses.count == 20 && report.losses.allSatisfy(\.isFinite))
            let early = report.losses.prefix(5).reduce(0, +)
            let late = report.losses.suffix(5).reduce(0, +)
            #expect(late < early, "\(kind.rawValue): \(report.losses)")
            let rankAfter = try #require(report.embeddingRankAfter)
            #expect(rankAfter > 1)
            #expect(await LeJEPATrainTests.weights(container) == before)
            let adapter = try LoRAContainer.from(directory: directory)
            #expect(adapter.parameters.flattened().contains { MLX.abs($0.1).max().item(Float.self) > 0 && $0.0.hasSuffix("lora_b") })
            let projector = try MLX.loadArrays(url: URL(fileURLWithPath: try #require(report.projectorPath)))
            #expect(projector["fc4.weight"]?.shape == [MLXInferenceEngine.jepaProjection, MLXInferenceEngine.jepaHidden])
        }
    }
}
