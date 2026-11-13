import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

struct DeepSeekOCRImageLayout: Equatable {
    static let pixelsPerQuery = 64

    let side: Int
    let columns: Int
    let rows: Int
    let tile: Int
    var newlines = true

    var tiled: Bool { columns * rows > 1 }
    var frames: [THW] { [THW(1, side, side), THW(rows, columns, tile)] }
    var globalValues: Int { side * side * 3 }
    var values: Int { globalValues + (tiled ? columns * rows * tile * tile * 3 : 0) }

    var tokens: Int {
        let (global, local) = (side / Self.pixelsPerQuery, tile / Self.pixelsPerQuery)
        guard newlines else { return global * global + 1 + (tiled ? local * local * columns * rows : 0) }
        return (global + 1) * global + 1 + (tiled ? (local * columns + 1) * (local * rows) : 0)
    }

    static func parse(_ frames: [THW], newlines: Bool = true) throws(BasaltError) -> [Self] {
        guard frames.count.isMultiple(of: 2) else {
            throw .invalidRequestStructure(details: "DeepSeek-OCR: an image needs a global view and a tile grid, got \(frames.count) frames")
        }
        return try stride(from: 0, to: frames.count, by: 2).map { index throws(BasaltError) in
            let (view, grid) = (frames[index], frames[index + 1])
            let layout = DeepSeekOCRImageLayout(side: view.h, columns: grid.h, rows: grid.t, tile: grid.w, newlines: newlines)
            let sides = layout.tiled ? [layout.side, layout.tile] : [layout.side]
            guard view.t == 1, view.h == view.w, layout.columns > 0, layout.rows > 0,
                  sides.allSatisfy({ $0 > 0 && $0.isMultiple(of: pixelsPerQuery) }) else {
                throw .invalidRequestStructure(details: "DeepSeek-OCR: image \(index / 2 + 1) has views that are not squares of a multiple of \(pixelsPerQuery) pixels")
            }
            return layout
        }
    }
}

protocol DeepSeekOCRDecoding: VLMModel {
    var languageModel: DeepSeekOCRText.LanguageModel { get }
    var imageToken: Int { get }
    var newlines: Bool { get }

    func features(_ pixels: MLXArray, _ layout: DeepSeekOCRImageLayout) throws(BasaltError) -> MLXArray
}

extension DeepSeekOCRDecoding {
    func merged(_ ids: MLXArray, pixels: MLXArray, frames: [THW]) throws(BasaltError) -> MLXArray {
        let layouts = try DeepSeekOCRImageLayout.parse(frames, newlines: newlines)
        let placeholders = ids[0].asArray(Int.self).count { $0 == imageToken }
        let (tokens, values) = (layouts.map(\.tokens).reduce(0, +), layouts.map(\.values).reduce(0, +))
        guard placeholders == tokens, pixels.size == values else {
            throw .invalidRequestStructure(
                details: "DeepSeek-OCR: \(layouts.count) images need \(tokens) placeholder tokens and \(values) pixel values, got \(placeholders) and \(pixels.size)")
        }
        var offset = 0
        var parts = [MLXArray]()
        for layout in layouts {
            parts.append(try features(pixels.reshaped(-1)[offset ..< offset + layout.values], layout))
            offset += layout.values
        }
        let features = concatenated(parts, axis: 0)
        let embeddings = languageModel.model.embedTokens(ids)
        let mask = ids .== imageToken
        let index = maximum(cumsum(mask.asType(.int32), axis: -1) - 1, 0)
        return which(mask[.ellipsis, .newAxis], features.take(index, axis: 0).asType(embeddings.dtype), embeddings)
    }

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        guard let image = input.image, let frames = image.frames, !frames.isEmpty else { return .tokens(input.text) }
        guard (cache.first?.offset ?? 0) == 0 else {
            throw BasaltError.invalidRequestStructure(details: "DeepSeek-OCR reads an image only at the start of a prompt, not after a cached prefix")
        }
        let ids = input.text.tokens.ndim == 1 ? input.text.tokens[.newAxis] : input.text.tokens
        let h = languageModel.model(try merged(ids, pixels: image.pixels, frames: frames), cache: cache)
        prefill.progress?(ids.dim(1), ids.dim(1))
        return .logits(LMOutput(logits: languageModel.lmHead(h[0..., (h.dim(1) - 1)...])))
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        LMOutput(logits: callAsFunction(input.tokens, cache: cache), state: state)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let ids = inputs.ndim == 1 ? inputs[.newAxis] : inputs
        return languageModel(languageModel.model.embedTokens(ids), cache: cache)
    }
}
