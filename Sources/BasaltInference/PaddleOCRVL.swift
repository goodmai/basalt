import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

struct PaddleOCRVLPositions: Equatable {
    var rows: [[Int32]] = [[], [], []]
    var delta = 0

    init(tokens: [Int], frames: [THW], imageToken: Int, mergeSize: Int) {
        var next: Int32 = 0
        var start = 0
        func text(_ count: Int) {
            for row in 0 ..< 3 { rows[row] += (0 ..< Int32(count)).map { next + $0 } }
            next += Int32(count)
        }
        for frame in frames {
            guard let index = tokens[start...].firstIndex(of: imageToken) else { break }
            text(index - start)
            let (t, h, w) = (frame.t, frame.h / mergeSize, frame.w / mergeSize)
            for ti in 0 ..< t {
                for hi in 0 ..< h {
                    rows[0] += Array(repeating: next + Int32(ti), count: w)
                    rows[1] += Array(repeating: next + Int32(hi), count: w)
                    rows[2] += (0 ..< Int32(w)).map { next + $0 }
                }
            }
            next += Int32(max(t, h, w))
            start = index + t * h * w
        }
        text(tokens.count - start)
        delta = Int(next) - tokens.count
    }

    var array: MLXArray { MLXArray(rows.flatMap { $0 }).reshaped(3, 1, -1) }
}

final class PaddleOCRVL: Module, VLMModel, KVCacheDimensionProvider {
    static let modelType = "paddleocr_vl"
    static let ropeDelta = LMOutput.Key<MLXArray>("paddleocr_vl.ropeDelta")

    let config: PaddleOCRVLConfiguration
    let kvHeads: [Int]
    @ModuleInfo var visual: PaddleOCRVLVision.Model
    @ModuleInfo(key: "language_model") var languageModel: PaddleOCRVLText.LanguageModel

    var loraLayers: [Module] { languageModel.model.layers }

    init(_ config: PaddleOCRVLConfiguration) {
        self.config = config
        kvHeads = Array(repeating: config.kvHeads, count: config.layers)
        _visual.wrappedValue = PaddleOCRVLVision.Model(config.vision, textHidden: config.hiddenSize)
        _languageModel.wrappedValue = PaddleOCRVLText.LanguageModel(config)
    }

    func hidden(_ embeddings: MLXArray, cache: [KVCache]?, positions: MLXArray) -> MLXArray {
        let rotary = PaddleOCRVLText.rotary(
            positions: positions, dims: config.headDim, base: config.ropeTheta,
            sections: config.ropeScaling.mropeSection, dtype: embeddings.dtype)
        return languageModel.model(embeddings, cache: cache, rotary: rotary)
    }

    func logits(_ tokens: MLXArray, cache: [KVCache]?, delta: Int) -> MLXArray {
        let ids = tokens.ndim == 1 ? tokens[.newAxis] : tokens
        let positions = PaddleOCRVLText.positions(after: cache?.first, delta: delta, batch: ids.dim(0), length: ids.dim(1))
        return languageModel.lmHead(hidden(languageModel.model.embedTokens(ids), cache: cache, positions: positions))
    }

    func merged(_ ids: MLXArray, pixels: MLXArray, frames: [THW]) throws(BasaltError) -> MLXArray {
        let embeddings = languageModel.model.embedTokens(ids)
        let features = visual(pixels.asType(visual.embeddings.patchEmbedding.weight.dtype), frames: frames)
        let mask = ids .== config.imageTokenId
        let count = mask.asType(.int32).sum().item(Int32.self)
        guard Int(count) == features.dim(0) else {
            throw .invalidRequestStructure(details: "PaddleOCR-VL: \(count) image tokens for \(features.dim(0)) image features")
        }
        let index = maximum(cumsum(mask.asType(.int32), axis: -1) - 1, 0)
        return which(mask[.ellipsis, .newAxis], features.take(index, axis: 0), embeddings)
    }

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        guard let image = input.image, let frames = image.frames, !frames.isEmpty else { return .tokens(input.text) }
        guard (cache.first?.offset ?? 0) == 0 else {
            throw BasaltError.invalidRequestStructure(details: "PaddleOCR-VL reads an image only at the start of a prompt, not after a cached prefix")
        }
        let ids = input.text.tokens.ndim == 1 ? input.text.tokens[.newAxis] : input.text.tokens
        let layout = PaddleOCRVLPositions(
            tokens: ids[0].asArray(Int.self), frames: frames, imageToken: config.imageTokenId, mergeSize: config.vision.mergeSize)
        let h = hidden(try merged(ids, pixels: image.pixels, frames: frames), cache: cache, positions: layout.array)
        var state = LMOutput.State()
        state[Self.ropeDelta] = MLXArray(Int32(layout.delta))
        prefill.progress?(ids.dim(1), ids.dim(1))
        return .logits(LMOutput(logits: languageModel.lmHead(h[0..., (h.dim(1) - 1)...]), state: state))
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        let delta = state?[Self.ropeDelta].map { Int($0.item(Int32.self)) } ?? 0
        return LMOutput(logits: logits(input.tokens, cache: cache, delta: delta), state: state)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs, cache: cache, delta: 0)
    }
}
