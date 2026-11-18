import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct SINQAbsorptionGroup: Sendable, Equatable {
    package let normWeightKey: String
    package let normBiasKey: String?
    package let quantizedBases: [String]
    package let denseCompensatedKeys: [String]
}

package enum SINQAbsorption {
    private static let allowedAttnQuantized: Set<String> = [
        "q_proj", "k_proj", "v_proj", "qkv_proj", "in_proj_qkv", "in_proj_z",
    ]
    private static let allowedAttnDense: Set<String> = ["in_proj_a", "in_proj_b"]
    private static let ignoredAttnDownstream: Set<String> = [
        "o_proj", "out_proj", "q_norm", "k_norm", "norm", "conv1d", "dt_bias", "A_log", "D", "sinks",
    ]

    private static let allowedMLPQuantizedSuffixes: [String] = [
        "gate_proj", "up_proj",
        "switch_mlp.gate_proj", "switch_mlp.up_proj",
        "shared_expert.gate_proj", "shared_expert.up_proj",
        "shared_experts.gate_proj", "shared_experts.up_proj",
        "w1", "w3",
    ]
    private static let allowedMLPDenseSuffixes: [String] = [
        "gate", "router", "shared_expert_gate",
    ]
    private static let ignoredMLPDownstreamSuffixes: [String] = [
        "down_proj", "switch_mlp.down_proj", "shared_expert.down_proj", "shared_experts.down_proj", "w2",
    ]

    package static func addsUnitOffset(config: [String: Any]) -> Bool {
        let top = (config["model_type"] as? String)?.lowercased() ?? ""
        let text = ((config["text_config"] as? [String: Any])?["model_type"] as? String)?.lowercased() ?? ""
        return top.hasPrefix("gemma") || text.hasPrefix("gemma")
    }

    package static func discoverGroups(keys: Set<String>) -> [SINQAbsorptionGroup] {
        let blockPrefixes = extractBlockPrefixes(from: keys)
        var groups: [SINQAbsorptionGroup] = []
        for block in blockPrefixes.sorted() {
            if let attn = buildAttentionGroup(block: block, keys: keys) {
                groups.append(attn)
            }
            if let mlp = buildMLPGroup(block: block, keys: keys) {
                groups.append(mlp)
            }
        }
        return groups
    }

    private static func extractBlockPrefixes(from keys: Set<String>) -> Set<String> {
        let normSuffixes = [
            ".input_layernorm.weight",
            ".operator_norm.weight",
            ".pre_feedforward_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".ffn_norm.weight",
        ]
        var prefixes: Set<String> = []
        for key in keys where !isNonLanguageKey(key) {
            for suffix in normSuffixes where key.hasSuffix(suffix) {
                prefixes.insert(String(key.dropLast(suffix.count)))
            }
        }
        return prefixes
    }

    private static func isNonLanguageKey(_ key: String) -> Bool {
        let parts = key.split(separator: ".")
        return parts.contains { part in
            ["vision", "visual", "audio", "mm_projector", "multi_modal_projector"].contains { part.contains($0) }
        }
    }

    private static func buildAttentionGroup(block: String, keys: Set<String>) -> SINQAbsorptionGroup? {
        let normBase: String
        if keys.contains("\(block).input_layernorm.weight") {
            normBase = "\(block).input_layernorm"
        } else if keys.contains("\(block).operator_norm.weight") {
            normBase = "\(block).operator_norm"
        } else {
            return nil
        }
        let subPrefixes = ["\(block).self_attn.", "\(block).linear_attn."]
        var quantized: [String] = []
        var dense: [String] = []
        for sub in subPrefixes {
            for key in keys where key.hasPrefix(sub) && key.hasSuffix(".weight") {
                let rel = String(key.dropFirst(sub.count).dropLast(".weight".count))
                if ignoredAttnDownstream.contains(rel) { continue }
                if allowedAttnQuantized.contains(rel) {
                    quantized.append(String(key.dropLast(".weight".count)))
                } else if allowedAttnDense.contains(rel) {
                    dense.append(key)
                } else {
                    return nil
                }
            }
        }
        guard !quantized.isEmpty else { return nil }
        let biasKey = keys.contains("\(normBase).bias") ? "\(normBase).bias" : nil
        return SINQAbsorptionGroup(
            normWeightKey: "\(normBase).weight",
            normBiasKey: biasKey,
            quantizedBases: quantized.sorted(),
            denseCompensatedKeys: dense.sorted())
    }

    private static func buildMLPGroup(block: String, keys: Set<String>) -> SINQAbsorptionGroup? {
        let normBase: String
        if keys.contains("\(block).pre_feedforward_layernorm.weight") {
            normBase = "\(block).pre_feedforward_layernorm"
        } else if keys.contains("\(block).post_attention_layernorm.weight") {
            normBase = "\(block).post_attention_layernorm"
        } else if keys.contains("\(block).ffn_norm.weight") {
            normBase = "\(block).ffn_norm"
        } else {
            return nil
        }
        let subPrefixes = ["\(block).mlp.", "\(block).feed_forward."]
        var quantized: [String] = []
        var dense: [String] = []
        for sub in subPrefixes {
            for key in keys where key.hasPrefix(sub) && key.hasSuffix(".weight") {
                let rel = String(key.dropFirst(sub.count).dropLast(".weight".count))
                if ignoredMLPDownstreamSuffixes.contains(rel) { continue }
                if allowedMLPQuantizedSuffixes.contains(rel) {
                    quantized.append(String(key.dropLast(".weight".count)))
                } else if allowedMLPDenseSuffixes.contains(rel) {
                    dense.append(key)
                } else {
                    return nil
                }
            }
        }
        guard !quantized.isEmpty else { return nil }
        let biasKey = keys.contains("\(normBase).bias") ? "\(normBase).bias" : nil
        return SINQAbsorptionGroup(
            normWeightKey: "\(normBase).weight",
            normBiasKey: biasKey,
            quantizedBases: quantized.sorted(),
            denseCompensatedKeys: dense.sorted())
    }
}

final class SINQActivationBox {
    let inFeatures: Int
    var sumSq: MLXArray
    var count = 0

    init(inFeatures: Int) {
        self.inFeatures = inFeatures
        self.sumSq = MLXArray.zeros([inFeatures], dtype: .float32)
        eval(self.sumSq)
    }

    func record(_ x: MLXArray) {
        let rows = inFeatures > 0 ? x.size / inFeatures : 0
        guard rows > 0 else { return }
        let flat = x.reshaped(rows, inFeatures).asType(.float32)
        sumSq = sumSq + sum(flat * flat, axis: 0)
        count += rows
        eval(sumSq)
    }

    func rms() -> MLXArray? {
        guard count > 0 else { return nil }
        let out = sqrt(sumSq * (Float(1.0) / Float(count)) + MLXArray(Float(1e-12)))
        eval(out)
        return out
    }
}

final class SINQLinearTap: Linear {
    let wrapped: Module
    let box: SINQActivationBox

    init(wrapping module: Module, box: SINQActivationBox) {
        self.wrapped = module
        self.box = box
        super.init(weight: MLXArray.zeros([1, 1], dtype: .float16), bias: nil)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        box.record(x)
        if let nf4 = wrapped as? NF4Linear { return nf4(x) }
        if let ql = wrapped as? QuantizedLinear { return ql(x) }
        if let lin = wrapped as? Linear { return lin(x) }
        preconditionFailure("Unsupported SINQLinearTap module: \(type(of: wrapped))")
    }
}

extension SINQAbsorption {
    package static func captureActivationRMS(
        source: URL,
        groups: [SINQAbsorptionGroup],
        calibrationText: String?
    ) async throws -> (rmsByNorm: [String: MLXArray], tokens: Int) {
        guard !groups.isEmpty else { return ([:], 0) }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0, disableExpertStreaming: true))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: source.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? PVTuner.defaultCalibrationCorpus
        let batches = await GSQRCOConverter.tokenizeCalibration(container, corpus: corpus, count: 8, seqLen: 128)
        let totalTokens = batches.reduce(0) { $0 + $1.count }
        guard !batches.isEmpty else { return ([:], 0) }

        let capturedHost: [String: [Float]] = try await container.perform { context in
            let root = context.model as Module
            root.train(false)
            let named = Dictionary(root.namedModules(), uniquingKeysWith: { first, _ in first })
            var originals: [(String, Module)] = []
            var boxes: [String: SINQActivationBox] = [:]

            for group in groups {
                let candidatePaths = group.quantizedBases + group.denseCompensatedKeys.map {
                    String($0.dropLast(".weight".count))
                }
                guard let tapPath = candidatePaths.first(where: { path in
                    guard let mod = named[path] else { return false }
                    return mod is Linear || mod is QuantizedLinear || mod is NF4Linear
                }), let mod = named[tapPath] else { continue }

                let width = inputWidth(of: mod)
                guard width > 0 else { continue }
                let box = SINQActivationBox(inFeatures: width)
                try PVTuner.swapModule(tapPath, with: SINQLinearTap(wrapping: mod, box: box), in: root)
                originals.append((tapPath, mod))
                boxes[group.normWeightKey] = box
            }

            defer {
                for (path, orig) in originals {
                    try? PVTuner.swapModule(path, with: orig, in: root)
                }
            }

            for seq in batches {
                let input = MLXArray(seq).reshaped(1, seq.count)
                let out = context.model(LMInput.Text(tokens: input), cache: nil, state: nil)
                eval(out.logits)
            }

            var out: [String: [Float]] = [:]
            for (normKey, box) in boxes {
                if let rms = box.rms() {
                    out[normKey] = rms.asType(.float32).asArray(Float.self)
                }
            }
            return out
        }
        let rmsByNorm = capturedHost.mapValues { MLXArray($0) }
        return (rmsByNorm, totalTokens)
    }

    private static func inputWidth(of module: Module) -> Int {
        if let nf4 = module as? NF4Linear { return nf4.shape.1 }
        if let ql = module as? QuantizedLinear { return (ql.weight.dim(1) * 32) / max(1, ql.bits) }
        if let lin = module as? Linear { return lin.shape.1 }
        return 0
    }
}
