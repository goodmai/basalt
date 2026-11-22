import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("PV-Tuning continuous step trains quantized scales: QuantizedLinear.unfreeze no longer re-freezes them (#698)", .serialized)
struct PVContinuousUnfreezeTests {
    static let blocks: Set<Int> = [0, 1, 2, 3]

    static func affineQwen3() throws -> any LanguageModel {
        let model = try TinyModel.make(.qwen3)
        quantize(model: model, groupSize: 32, bits: 4) { path, _ in path.contains("layers.") }
        return model
    }

    static func macaqQwen35() throws -> any LanguageModel {
        let model = try TinyModel.make(.qwen35)
        var config = MACaQConfig()
        config.hadamardBlock = 32
        config.maxFlipIterations = 1
        _ = try MACaQRefiner.quantizeModel(
            model, targetMode: .affine(bits: 3, groupSize: 32), calibrationTokenBatches: [(0 ..< 24).map { Int32(($0 * 13 + 91) % 200) }], config: config)
        return model
    }

    static func gradientNorms(_ model: any LanguageModel, suffixes: [String]) -> [String: Float] {
        let root = model as Module
        let tokens = MLXArray((0 ..< 16).map { Int32(($0 * 7 + 3) % 200) })[.newAxis]
        let step = valueAndGrad(model: root) { _, arrays in [model(arrays[0], cache: nil).asType(.float32).square().mean()] }
        let (_, grads) = step(root, [tokens])
        return Dictionary(grads.flattened().filter { key, _ in suffixes.contains { key.hasSuffix($0) } }.map { ($0.0, abs($0.1).sum().item(Float.self)) },
                          uniquingKeysWith: { first, _ in first })
    }

    @Test("Affine QuantizedLinear: scales and biases are trainable, the packed weight is not, and their gradients are non-zero")
    func affineScalesTrain() throws {
        let model = try Self.affineQwen3()
        let swaps = try PVTuner.prepareContinuousUnfreeze(model as Module, blocks: Self.blocks, maxSwitchExperts: 64)
        #expect(!swaps.affine.isEmpty)
        let trainable = (model as Module).trainableParameters().flattened().map(\.0)
        #expect(trainable.contains { $0.hasSuffix("self_attn.q_proj.scales") })
        #expect(trainable.contains { $0.hasSuffix("self_attn.q_proj.biases") })
        #expect(!trainable.contains { $0.hasSuffix("self_attn.q_proj.weight") })
        let norms = Self.gradientNorms(model, suffixes: ["self_attn.q_proj.scales", "mlp.down_proj.biases"])
        #expect(!norms.isEmpty && norms.values.allSatisfy { $0 > 0 }, "\(norms)")
    }

    @Test("MACaQ dense layers: pre_scale, scales and biases are trainable with non-zero gradients; freezing clears them again")
    func macaqScalesTrain() throws {
        let model = try Self.macaqQwen35()
        _ = try PVTuner.prepareContinuousUnfreeze(model as Module, blocks: Self.blocks, maxSwitchExperts: 64)
        let trainable = (model as Module).trainableParameters().flattened().map(\.0)
        for suffix in ["pre_scale", "scales", "biases"] {
            #expect(trainable.contains { $0.contains("mlp.down_proj.") && $0.hasSuffix(suffix) }, "\(suffix)")
        }
        #expect(!trainable.contains { $0.hasSuffix("mlp.down_proj.weight") })
        let norms = Self.gradientNorms(model, suffixes: ["mlp.down_proj.pre_scale", "mlp.down_proj.scales"])
        #expect(!norms.isEmpty && norms.values.allSatisfy { $0 > 0 }, "\(norms)")
        (model as Module).freeze()
        #expect(!(model as Module).trainableParameters().flattened().contains { $0.0.hasSuffix(".scales") })
    }

    @Test("After a continuous group the affine wrappers are swapped back to plain QuantizedLinear and the model is frozen")
    func affineSwapsBack() throws {
        let teacher = try TinyModel.make(.qwen3)
        let student = try Self.affineQwen3()
        let data = MACaQPVTuneTests.makeCalibrationAndTeacher(teacher: teacher)
        var config = PVTuneConfig()
        config.continuousStepsPerGroup = 2
        config.continuousLR = 1e-3
        try PVTuner.runContinuousGroup(student, blocks: Self.blocks, train: data.train, val: data.val, stepSeed: 0, config: config)
        let linears = (student as Module).namedModules().compactMap { $0.1 as? QuantizedLinear }
        #expect(!linears.isEmpty && linears.allSatisfy { type(of: $0) == QuantizedLinear.self })
        #expect((student as Module).trainableParameters().flattened().isEmpty)
    }
}
