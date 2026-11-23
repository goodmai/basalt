import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("YaRN RoPE: one formula for the text and multimodal Qwen 3.5 paths, Gemma 4 refuses it (#729 P0, #228)", .serialized)
struct YarnRoPETests {
    static let (dims, base, factor, original): (Int, Double, Double, Int) = (8, 1e7, 4, 32)

    static func referencePeriods(dims: Int, base: Double, factor: Double, original: Int) -> [Double] {
        func correction(_ rotations: Double) -> Double { Double(dims) * log(Double(original) / (rotations * 2 * .pi)) / (2 * log(base)) }
        let low = max(floor(correction(32)), 0)
        var high = min(ceil(correction(1)), Double(dims - 1))
        if low == high { high += 0.001 }
        return (0 ..< dims / 2).map { pair in
            let extrapolated = pow(base, Double(2 * pair) / Double(dims))
            let mask = 1 - min(max((Double(pair) - low) / (high - low), 0), 1)
            let interpolated = factor * extrapolated
            return interpolated * extrapolated / (interpolated * mask + extrapolated * (1 - mask))
        }
    }

    static func referenceRoPE(_ x: [Float], position: Int, periods: [Double], scale: Double) -> [Float] {
        let half = periods.count
        var out = x.map(Double.init)
        for pair in 0 ..< half {
            let angle = Double(position) / periods[pair]
            let (a, b) = (Double(x[pair]), Double(x[pair + half]))
            out[pair] = (a * cos(angle) - b * sin(angle)) * scale
            out[pair + half] = (b * cos(angle) + a * sin(angle)) * scale
        }
        return out.map(Float.init)
    }

    @Test("YarnRoPE matches a Double reference of Hugging Face's YaRN at offsets 0, 31, 100 and 4000, and factor 1 is stock RoPE")
    func matchesReference() {
        let periods = Self.referencePeriods(dims: Self.dims, base: Self.base, factor: Self.factor, original: Self.original)
        let computed = YarnScaling.periods(
            dimensions: Self.dims, base: Float(Self.base), scalingFactor: Float(Self.factor), originalMaxPositionEmbeddings: Self.original).asArray(Float.self)
        #expect(zip(computed, periods).allSatisfy { abs(Double($0) - $1) / $1 < 1e-5 })
        let scale = 0.1 * log(Self.factor) + 1
        #expect(abs(Double(YarnScaling.attentionFactor(scalingFactor: Float(Self.factor))) - scale) < 1e-6)
        let rope = YarnRoPE(dimensions: Self.dims, base: Float(Self.base), scalingFactor: Float(Self.factor), originalMaxPositionEmbeddings: Self.original)
        let x: [Float] = (0 ..< Self.dims).map { Float($0 % 3) - 0.7 + Float($0) * 0.1 }
        for offset in [0, 31, 100, 4000] {
            let out = rope(MLXArray(x).reshaped(1, 1, 1, Self.dims), offset: offset).reshaped(-1).asArray(Float.self)
            let expected = Self.referenceRoPE(x, position: offset, periods: periods, scale: scale)
            #expect(zip(out, expected).allSatisfy { abs($0 - $1) < 2e-3 }, "offset \(offset): \(out) vs \(expected)")
        }
        let unit = YarnRoPE(dimensions: Self.dims, base: Float(Self.base), scalingFactor: 1, originalMaxPositionEmbeddings: Self.original)
        let stock = RoPE(dimensions: Self.dims, traditional: false, base: Float(Self.base))
        let probe = MLXArray(x).reshaped(1, 1, 1, Self.dims)
        #expect(abs(unit(probe, offset: 100) - stock(probe, offset: 100)).max().item(Float.self) < 1e-5)
    }

    static func qwenConfig(yarn: Bool) -> String {
        let rope = #""rope_parameters": {"rope_type": "\#(yarn ? "yarn" : "default")", "factor": 4.0, "original_max_position_embeddings": 32, "rope_theta": 10000000.0, "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1], "mrope_interleaved": true}, "#
        let text = TinyModel.Family.qwen35.config.replacingOccurrences(of: #""model_type": "qwen3_5_text", "#, with: #""model_type": "qwen3_5_text", "# + rope)
        return #"{"model_type": "qwen3_5", "text_config": \#(text), "vision_config": {"model_type": "qwen3_vl", "depth": 1, "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 16}}"#
    }

    static func logits(yarn: Bool) throws -> (text: MLXArray, multimodal: MLXArray) {
        let data = Data(Self.qwenConfig(yarn: yarn).utf8)
        let decoder = JSONDecoder()
        let text = try withRandomState(MLXRandom.RandomState(seed: 729)) { () throws -> any LanguageModel in
            MLXLLM.Qwen35Model(try decoder.decode(MLXLLM.Qwen35Configuration.self, from: data))
        }
        let multimodal = MLXVLM.Qwen35(try decoder.decode(MLXVLM.Qwen35Configuration.self, from: data))
        try multimodal.update(parameters: text.parameters(), verify: [])
        let tokens = MLXArray((0 ..< 64).map { Int32(($0 * 29 + 7) % 250) })[.newAxis]
        let textLogits = text(tokens, cache: nil)
        let multimodalLogits = multimodal(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        return (textLogits, multimodalLogits)
    }

    @Test("With a yarn rope_parameters entry, the multimodal Qwen 3.5 path gives the text path's logits over 64 tokens past a 32-token original context; both differ from stock RoPE")
    func multimodalHonorsYarn() throws {
        let yarn = try Self.logits(yarn: true)
        let stock = try Self.logits(yarn: false)
        #expect(abs(stock.text - stock.multimodal).max().item(Float.self) < 1e-4)
        #expect(abs(yarn.text - yarn.multimodal).max().item(Float.self) < 1e-4)
        #expect(argMax(yarn.text, axis: -1).asArray(Int32.self) == argMax(yarn.multimodal, axis: -1).asArray(Int32.self))
        #expect(abs(yarn.text - stock.text).max().item(Float.self) > 1e-3)
    }

    @Test("A yarn rope_type in a Gemma 4 config is refused on load, on the text and the multimodal path, naming the field")
    func gemma4Refuses() throws {
        let entry = #""rope_parameters": {"full_attention": {"rope_type": "yarn", "factor": 4.0, "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}, "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}, "#
        let text = "{" + entry + TinyModel.Family.gemma4.config.drop { $0 != "{" }.dropFirst()
        let unified = TinyModel.Family.gemma4Unified.config.replacingOccurrences(of: #""model_type": "gemma4_unified_text", "#, with: #""model_type": "gemma4_unified_text", "# + entry)
        for (family, config) in [(TinyModel.Family.gemma4, text), (.gemma4Unified, unified)] {
            #expect(config.contains(#""rope_type": "yarn""#))
            #expect {
                _ = try family.model(Data(config.utf8))
            } throws: { error in
                "\(error)".contains("rope_type 'yarn' is not supported")
            }
        }
        _ = try TinyModel.make(.gemma4KEqV)
    }
}
