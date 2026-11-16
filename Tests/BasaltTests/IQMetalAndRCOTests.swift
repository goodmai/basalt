import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("IQMetalKernels specialized K-quant + IQ matvec, smallM, and TinyModel #228 parity", .serialized)
struct IQMetalSpecializedKernelAdmissionTests {
    static let primaryFormats: [GGMLType] = IQMetalKernels.primarySub4Types
    static let allSpecializedFormats: [GGMLType] = IQMetalKernels.supportedTypes

    static func maxAbsError(_ actual: MLXArray, _ reference: MLXArray) -> Float {
        abs(actual.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
    }

    @Test("Every primary sub-4-bit and K-quant format in IQMetalKernels matches GGUFDequant within 1e-3 max-abs at M=1 and M=4",
          arguments: allSpecializedFormats, [DType.float32, .float16])
    func specializedMatvecAndSmallMMaxAbsParity(type: GGMLType, dtype: DType) throws {
        #expect(Set(Self.primaryFormats) == Set([.q3K, .q4K, .iq3S, .iq3XXS, .iq4XS, .iq2XS]))
        #expect(Self.primaryFormats.allSatisfy(IQMetalKernels.supports))
        let (outputs, inputs) = (29, 512)
        let blockCount = (outputs * inputs) / type.block.elements
        let bytes = GGUFKernelTests.blocks(type, count: blockCount, seed: UInt64(type.rawValue) &+ 408, scale: 0.001)
        let packed = GGUFKernelTests.packed(bytes, rows: outputs)
        let cpuDense = GGUFKernelTests.dense(bytes, type, rows: outputs, columns: inputs)
        let layer = GGUFLinear(type: type, outputs: outputs, inputs: inputs)
        try layer.update(parameters: ModuleParameters.unflattened(["weight": packed]), verify: .all)

        let tol: Float = dtype == .float32 ? 1e-3 : 2.5e-3
        let x1 = (MLXRandom.normal([1, inputs], key: MLXRandom.key(UInt64(type.rawValue) &+ 3)) * 0.1).asType(dtype)
        let ref1 = matmul(x1.asType(.float32), cpuDense.T)
        let mv1 = IQMetalKernels.matvec(x1, packed, type: type)
        let lin1 = layer(x1)
        eval(mv1, lin1, ref1)
        #expect(mv1.shape == [1, outputs] && mv1.dtype == dtype)
        #expect(Self.maxAbsError(mv1, ref1) < tol, "\(type.name) M=1 \(dtype)")
        #expect(Self.maxAbsError(lin1, ref1) < tol, "\(type.name) GGUFLinear M=1 \(dtype)")

        let x4 = (MLXRandom.normal([4, inputs], key: MLXRandom.key(UInt64(type.rawValue) &+ 19)) * 0.1).asType(dtype)
        let ref4 = matmul(x4.asType(.float32), cpuDense.T)
        let sm4 = IQMetalKernels.smallM(x4, packed, type: type)
        let lin4 = layer(x4.reshaped(1, 4, inputs))
        eval(sm4, lin4, ref4)
        #expect(sm4.shape == [4, outputs] && sm4.dtype == dtype)
        #expect(Self.maxAbsError(sm4, ref4) < tol, "\(type.name) M=4 \(dtype)")
        #expect(Self.maxAbsError(lin4.reshaped(4, outputs), ref4) < tol, "\(type.name) GGUFLinear M=4 \(dtype)")

        let x12 = (MLXRandom.normal([12, inputs], key: MLXRandom.key(UInt64(type.rawValue) &+ 31)) * 0.1).asType(dtype)
        let ref12 = matmul(x12.asType(.float32), cpuDense.T)
        let lin12 = layer(x12)
        eval(lin12, ref12)
        let prefillTol: Float = dtype == .float32 ? 1e-3 : 5e-3
        #expect(lin12.shape == [12, outputs] && Self.maxAbsError(lin12, ref12) < prefillTol)
    }

    @Test("TinyModel qwen35 and gemma4 greedy decode and prefill logits match CPU GGUFDequant reference (#228)",
          arguments: [TinyModel.Family.qwen35, TinyModel.Family.gemma4])
    func tinyModelQwen35AndGemma4Parity(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        var entries: [String: GGUFNativeLoader.Entry] = [:]
        var packedWeights: [(String, MLXArray)] = []
        var denseWeights: [(String, MLXArray)] = []
        let candidates = model.leafModules().flattened().compactMap { path, mod -> (String, Linear)? in
            guard type(of: mod) == Linear.self, let lin = mod as? Linear, lin.bias == nil else { return nil }
            guard lin.weight.dim(1).isMultiple(of: 32) else { return nil }
            return (path, lin)
        }
        #expect(candidates.count >= 16)
        for (idx, (path, lin)) in candidates.enumerated() {
            let (rows, cols) = (lin.weight.dim(0), lin.weight.dim(1))
            let qType: GGMLType = cols.isMultiple(of: 256)
                ? Self.primaryFormats[idx % Self.primaryFormats.count]
                : .iq4NL
            let blockCount = (rows * cols) / qType.block.elements
            let bytes = GGUFKernelTests.blocks(qType, count: blockCount, seed: UInt64(idx + 700), scale: 0.002)
            entries[path] = GGUFNativeLoader.Entry(type: qType.name, dtype: "\(lin.weight.dtype)")
            packedWeights.append((path + ".weight", GGUFKernelTests.packed(bytes, rows: rows)))
            denseWeights.append((path + ".weight", GGUFKernelTests.dense(bytes, qType, rows: rows, columns: cols).asType(lin.weight.dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(denseWeights))
        eval(model)

        let prompt = Array(TinyModel.tokens("specialized kquant and iquant admission").prefix(6))
        let promptTensor = MLXArray(prompt.map(Int32.init))[.newAxis]
        let refLogits = model(promptTensor, cache: nil).asType(.float32)
        let refGreedy = try CacheBlendQualityTests.greedy(prompt, count: 8, model: model)
        eval(refLogits)

        try GGUFNativeLoader.install(entries, in: model)
        try model.update(parameters: ModuleParameters.unflattened(packedWeights), verify: [.noUnusedKeys, .shapeMismatch])
        eval(model)
        let iqLogits = model(promptTensor, cache: nil).asType(.float32)
        let iqGreedy = try CacheBlendQualityTests.greedy(prompt, count: 8, model: model)
        eval(iqLogits)

        #expect(Self.maxAbsError(iqLogits, refLogits) < 1e-3, "family=\(family)")
        #expect(iqGreedy == refGreedy, "family=\(family)")
    }

    @Test("GSQRCOBlockEncoder round-trips all 10 encodable IQ/K-quant formats against GGUFDequant and IQMetalKernels",
          arguments: GSQRCOBlockEncoder.encodableTypes)
    func blockEncoderRoundTrip(type: GGMLType) {
        let (rows, cols) = (4, 256)
        let source = (0 ..< (rows * cols)).map { idx -> Float in
            let angle = Float(idx) * 0.173
            return Foundation.sin(angle) * 0.25 + Foundation.cos(angle * 0.37) * 0.1
        }
        let encoded = GSQRCOBlockEncoder.encodeMatrix(source, rows: rows, columns: cols, type: type)
        #expect(encoded.count == (rows * cols / type.block.elements) * type.block.bytes)
        let decoded = GGUFDequant.floats(encoded.span.bytes, type: type, count: rows * cols)
        let mse = zip(source, decoded).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) } / Double(source.count)
        #expect(mse < 0.02, "\(type.name) mse=\(mse)")

        if IQMetalKernels.supports(type) {
            let packed = GGUFKernelTests.packed(encoded, rows: rows)
            let x = MLXArray((0 ..< cols).map { Float($0 % 13) * 0.02 - 0.1 }, [1, cols])
            let mv = IQMetalKernels.matvec(x, packed, type: type)
            let ref = matmul(x, MLXArray(decoded, [rows, cols]).T)
            eval(mv, ref)
            #expect(Self.maxAbsError(mv, ref) < 1e-4)
        }
    }
}

@Suite("IQQuantFormat metadata and RCOSensitivityAllocator mixed-rate budget solver (#408, #409)", .serialized)
struct RCOMixedRateAllocatorTests {
    @Test("IQQuantFormat and IQBlockDescriptor expose exact block geometry, LUT flags, and fused kernel support")
    func quantFormatMetadata() {
        #expect(IQQuantFormat.allCases.count == 11)
        for fmt in IQQuantFormat.allCases {
            let desc = fmt.descriptor
            #expect(desc.quantFormat == fmt)
            #expect(desc.blockElements == fmt.ggmlType.block.elements)
            #expect(desc.superBlockBytes == fmt.ggmlType.block.bytes)
            #expect(abs(desc.effectiveBitsPerWeight - fmt.ggmlType.bitsPerWeight) < 1e-12)
            #expect(desc.usesCodebookLUT == desc.isIQ)
        }
        let fused = Set(IQBlockDescriptor.all.filter(\.supportsFusedKernel).map(\.ggmlType))
        #expect(fused == Set(IQMetalKernels.supportedTypes))
        #expect(RCOSensitivityAllocator.LayerRole.infer(from: "model.layers.0.linear_attn.in_proj_qkv.weight") == .ssm)
        #expect(RCOSensitivityAllocator.LayerRole.infer(from: "model.layers.1.self_attn.v_proj.weight") == .attentionValue)
        #expect(RCOSensitivityAllocator.LayerRole.infer(from: "model.layers.1.mlp.gate.weight") == .router)
        #expect(RCOSensitivityAllocator.LayerRole.infer(from: "model.layers.1.mlp.experts.2.gate_proj.weight") == .moeRouted)
    }

    @Test("allocate(layers:targetBitsPerWeight:candidates:) obeys QTIP 2.0/2.5/3.0/4.0 budget, monotonicity, and sensitive role protection")
    func qtipBudgetAllocationAndMonotonicity() {
        let layers: [RCOSensitivityAllocator.LayerSensitivity] = [
            .init(name: "layers.0.linear_attn.in_proj_qkv", weightCount: 65_536, sensitivity: 3.2, role: .ssm, columns: 256),
            .init(name: "layers.1.self_attn.v_proj", weightCount: 32_768, sensitivity: 4.1, role: .attentionValue, columns: 256),
            .init(name: "layers.0.mlp.router", weightCount: 16_384, sensitivity: 5.0, role: .router, columns: 256),
            .init(name: "layers.0.self_attn.q_proj", weightCount: 65_536, sensitivity: 1.9, role: .attention, columns: 256),
            .init(name: "layers.1.self_attn.q_proj", weightCount: 65_536, sensitivity: 0.8, role: .attention, columns: 256),
            .init(name: "layers.0.mlp.gate_proj", weightCount: 131_072, sensitivity: 1.5, role: .ffn, columns: 256),
            .init(name: "layers.1.mlp.gate_proj", weightCount: 131_072, sensitivity: 0.6, role: .ffn, columns: 256),
            .init(name: "layers.0.mlp.experts.0.up_proj", weightCount: 262_144, sensitivity: 0.9, role: .moeRouted, columns: 256),
            .init(name: "layers.1.mlp.experts.0.up_proj", weightCount: 262_144, sensitivity: 0.3, role: .moeRouted, columns: 256),
        ]
        let allocator = RCOSensitivityAllocator()
        for budget in [2.35, 2.75, 3.25] {
            let plan = allocator.allocate(
                layers: layers,
                targetBitsPerWeight: budget,
                candidates: RCOSensitivityAllocator.qtipBudgetCandidates)
            #expect(plan.achievedBPW <= budget + 1e-9)
            #expect(abs(plan.achievedBPW - budget) < 0.15)
            for alloc in plan.allocations where alloc.role.requiresSensitiveFloor {
                #expect(alloc.bitsPerWeight >= (budget >= 2.5 ? 3.0 : 2.5))
            }
            for i in layers.indices {
                for j in layers.indices where layers[i].role == layers[j].role && layers[i].sensitivity > layers[j].sensitivity {
                    #expect(plan.allocations[i].bitsPerWeight + 1e-12 >= plan.allocations[j].bitsPerWeight)
                }
            }
        }
    }
}

@Suite("GSQRCOConverter and --experimental-qtip-budget end-to-end TinyModel conversion (#408, #409)", .serialized)
struct GSQRCOAndQTIPBudgetEndToEndTests {
    @Test("GSQRCOConverter quantizes TinyModel qwen35 and prism, writes gguf_quantization checkpoint, and reloads bit-identically")
    func gsqRCOEndToEndRoundTrip() async throws {
        let qwen35 = try TinyModel.make(.qwen35)
        let calib: [[Int32]] = [(0 ..< 24).map { Int32(($0 * 7 + 5) % 200) }]
        var cfg = GSQRCOConfig(targetBPW: 5.5)
        cfg.minDimension = 32
        cfg.minFloorForSensitiveRoles = .iq4NL
        let qwenSummary = try GSQRCOConverter.quantizeModel(qwen35, calibrationTokenBatches: calib, config: cfg)
        #expect(qwenSummary.quantizedLayers >= 16)
        #expect(qwenSummary.plan.achievedBPW <= 5.5 + 1e-9)

        let model = try TinyModel.make(.prism)
        let summary = try GSQRCOConverter.quantizeModel(model, calibrationTokenBatches: calib, config: cfg)
        #expect(summary.quantizedLayers >= 16)
        #expect(summary.plan.achievedBPW <= 5.5 + 1e-9)

        let dir = FileManager.default.temporaryDirectory.appending(component: "gsq-rco-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let srcDir = dir.appending(component: "src")
        let outDir = dir.appending(component: "out")
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
        try Data(TinyModel.Family.prism.config.utf8).write(to: srcDir.appending(component: "config.json"))
        try Data("{}".utf8).write(to: srcDir.appending(component: "tokenizer_config.json"))

        let written = try GSQRCOConverter.writeGSQDirectory(model: model as Module, source: srcDir, output: outDir)
        #expect(written.shards == 1 && written.bytes > 0)
        let container = try #require(try await GGUFNativeLoader.loadContainer(from: outDir, tokenizerLoader: ByteTokenizerLoader()))
        let prompt: [Int32] = [3, 14, 29, 61]
        let directLogits = model(MLXArray(prompt)[.newAxis], cache: nil).asType(.float32)
        eval(directLogits)
        let directArr = directLogits.asArray(Float.self)
        let maxReloadDiff = await container.perform { ctx in
            let rLogits = ctx.model(MLXArray(prompt)[.newAxis], cache: nil).asType(.float32)
            eval(rLogits)
            return zip(directArr, rLogits.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1
        }
        #expect(maxReloadDiff < 1e-5)
    }

    @Test("quantizeModelQTIPBudget allocates mixed 2.0/2.5/3.0/4.0 bpw on qwen35, protects ssm/attn.v/router, and reloads via QTIPNativeLoader")
    func qtipBudgetMixedRateRoundTrip() async throws {
        let model = try TinyModel.make(.qwen35)
        let calib: [[Int32]] = [(0 ..< 24).map { Int32(($0 * 11 + 3) % 200) }]
        var qCfg = QTIPConfig(rate: .twoHalfBPW)
        qCfg.minDimension = 16
        qCfg.blockGroupSize = 2
        let summary = try GSQRCOConverter.quantizeModelQTIPBudget(
            model, calibrationTokenBatches: calib, targetBPW: 2.75, baseConfig: qCfg)
        #expect(summary.qtipLayers > 0)
        #expect(summary.plan.achievedBPW <= 2.75 + 1e-9)
        #expect(summary.skippedProtectedLayers > 0)
        for alloc in summary.plan.allocations where alloc.role == .ssm || alloc.role == .attentionValue {
            #expect(alloc.bitsPerWeight >= 3.0)
        }
        #expect(Set(summary.rateCounts.keys).count >= 2)

        let dir = FileManager.default.temporaryDirectory.appending(component: "qtip-budget-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let srcDir = dir.appending(component: "src")
        let outDir = dir.appending(component: "out")
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
        try Data(TinyModel.Family.prism.config.utf8).write(to: srcDir.appending(component: "config.json"))
        try Data("{}".utf8).write(to: srcDir.appending(component: "tokenizer_config.json"))

        let prismModel = try TinyModel.make(.prism)
        let prismSummary = try GSQRCOConverter.quantizeModelQTIPBudget(
            prismModel, calibrationTokenBatches: calib, targetBPW: 2.85, baseConfig: qCfg)
        let written = try QTIPNativeLoader.writeMixedDirectory(
            model: prismModel as Module, achievedBPW: prismSummary.plan.achievedBPW,
            seed: qCfg.seed, source: srcDir, output: outDir, affinePaths: prismSummary.affinePaths)
        #expect(written.shards == 1 && written.bytes > 0)
        let report = await ModelInspector().inspect(modelIdOrPath: outDir.path)
        #expect(report.intact && report.quantizationMode == "qtip")

        let reloaded = try #require(try await QTIPNativeLoader.loadContainer(from: outDir, tokenizerLoader: ByteTokenizerLoader()))
        let prompt: [Int32] = [5, 19, 44, 102]
        let direct = prismModel(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
        eval(direct)
        let directVals = direct.asArray(Float.self)
        let diff = await reloaded.perform { ctx in
            let r = ctx.model(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
            eval(r)
            return zip(directVals, r.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1
        }
        #expect(diff == 0)
    }

    @Test("ConvertSubcommand parses and validates --quantize gsq-rco --bits and --experimental-qtip-budget")
    func cliValidation() throws {
        let gsq = try ConvertSubcommand.parse(["src", "--out", "dst", "--quantize", "gsq-rco", "--bits", "3.25"])
        #expect(try gsq.gsqRCOBits() == 3.25)
        let gsqInline = try ConvertSubcommand.parse(["src", "--out", "dst", "--quantize", "gsq-rco:2.75"])
        #expect(try gsqInline.gsqRCOBits() == 2.75)
        let qtipB = try ConvertSubcommand.parse([
            "src", "--out", "dst", "--experimental-qtip-budget", "2.65", "--experimental-pv-tune", "--pv-calibration", "cal.txt"
        ])
        #expect(qtipB.experimentalQtipBudget == 2.65 && qtipB.experimentalPvTune && qtipB.pvCalibration == "cal.txt")

        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--quantize", "gsq-rco"])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--bits", "3.0"])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--experimental-qtip-budget", "1.5"])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse([
                "src", "--out", "dst", "--experimental-qtip", "2.5", "--experimental-qtip-budget", "2.5"
            ])
        }
    }

    @Test("PVQTIPLinear matches QTIPLinear forward output and two-stage PVTuner monotonically refines row_scales, tlut, and gated Viterbi trellis")
    func qtipTwoStagePVTuningParityAndMonotonicity() throws {
        let teacher = try TinyModel.make(.prism)
        let student = try TinyModel.make(.prism)
        let seqLen = 12
        let topK = 16
        let batches: [PVTeacherBatch] = (0 ..< 4).map { bIdx in
            let tokens = (0 ..< seqLen).map { Int32((bIdx * 17 + $0 * 7 + 5) % 200) }
            let logits = teacher(LMInput.Text(tokens: MLXArray(tokens)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
            let lse = logSumExp(logits, axis: -1, keepDims: true)
            let topIdx = argPartition(-logits, kth: topK - 1, axis: -1)[0..., 0 ..< topK]
            let topVal = takeAlong(logits, topIdx, axis: -1) - lse
            eval(topIdx, topVal)
            return PVTeacherBatch(
                tokens: tokens, topIDs: topIdx.asArray(Int32.self),
                topLogprobs: topVal.asArray(Float.self), seqLen: seqLen, topK: topK)
        }
        var qCfg = QTIPConfig(rate: .twoBPW)
        qCfg.minDimension = 16
        qCfg.blockGroupSize = 2
        _ = try GSQRCOConverter.quantizeModelQTIPBudget(
            student, calibrationTokenBatches: batches.map(\.tokens), targetBPW: 2.25, baseConfig: qCfg)
        let firstQTIP = try #require((student as Module).namedModules().compactMap { $0.1 as? QTIPLinear }.first)
        let pvWrapper = PVQTIPLinear(from: firstQTIP)
        let probeX = MLXRandom.normal([3, firstQTIP.inFeatures]).asType(.float16)
        let yNative = firstQTIP(probeX).asType(.float32)
        let yDiff = pvWrapper(probeX).asType(.float32)
        eval(yNative, yDiff)
        let maxForwardDiff = zip(yNative.asArray(Float.self), yDiff.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1
        #expect(maxForwardDiff < 5e-3)

        var pvCfg = PVTuneConfig()
        pvCfg.globalContinuousSteps = 4
        pvCfg.continuousStepsPerGroup = 3
        pvCfg.batchesPerDiscreteStep = 2
        pvCfg.continuousLR = 5e-4
        let summary = try PVTuner.tune(student, train: Array(batches.prefix(3)), val: [batches[3]], config: pvCfg)
        #expect(summary.completedGroups >= 1)
        #expect(summary.valKLAfter <= summary.valKLBefore + 1e-6)
        #expect(summary.trainKLAfter <= summary.trainKLBefore + 1e-6)
    }
}
