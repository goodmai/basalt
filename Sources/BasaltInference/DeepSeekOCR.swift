import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

struct DeepSeekOCRImageLayout: Equatable {
    static let pixelsPerQuery = 64

    let side: Int
    let columns: Int
    let rows: Int
    let tile: Int

    var tiled: Bool { columns * rows > 1 }
    var frames: [THW] { [THW(1, side, side), THW(rows, columns, tile)] }
    var globalValues: Int { side * side * 3 }
    var values: Int { globalValues + (tiled ? columns * rows * tile * tile * 3 : 0) }

    var tokens: Int {
        let (global, local) = (side / Self.pixelsPerQuery, tile / Self.pixelsPerQuery)
        return (global + 1) * global + 1 + (tiled ? (local * columns + 1) * (local * rows) : 0)
    }

    static func parse(_ frames: [THW]) throws(BasaltError) -> [Self] {
        guard frames.count.isMultiple(of: 2) else {
            throw .invalidRequestStructure(details: "DeepSeek-OCR: an image needs a global view and a tile grid, got \(frames.count) frames")
        }
        return try stride(from: 0, to: frames.count, by: 2).map { index throws(BasaltError) in
            let (view, grid) = (frames[index], frames[index + 1])
            let layout = DeepSeekOCRImageLayout(side: view.h, columns: grid.h, rows: grid.t, tile: grid.w)
            let sides = layout.tiled ? [layout.side, layout.tile] : [layout.side]
            guard view.t == 1, view.h == view.w, layout.columns > 0, layout.rows > 0,
                  sides.allSatisfy({ $0 > 0 && $0.isMultiple(of: pixelsPerQuery) }) else {
                throw .invalidRequestStructure(details: "DeepSeek-OCR: image \(index / 2 + 1) has views that are not squares of a multiple of \(pixelsPerQuery) pixels")
            }
            return layout
        }
    }
}

final class DeepSeekOCR: Module, VLMModel, KVCacheDimensionProvider {
    static let modelType = "deepseekocr"

    final class Projector: Module {
        @ModuleInfo var layers: Linear

        init(_ config: DeepSeekOCRConfiguration.Projector) {
            _layers.wrappedValue = Linear(config.inputDim, config.embedDim)
        }
    }

    let config: DeepSeekOCRConfiguration
    let kvHeads: [Int]
    @ModuleInfo(key: "sam_model") var sam: DeepSeekOCRSAM.Encoder
    @ModuleInfo(key: "vision_model") var clip: DeepSeekOCRCLIP.Model
    @ModuleInfo var projector: Projector
    @ParameterInfo(key: "image_newline") var imageNewline: MLXArray
    @ParameterInfo(key: "view_separator") var viewSeparator: MLXArray
    @ModuleInfo(key: "language_model") var languageModel: DeepSeekOCRText.LanguageModel

    var loraLayers: [Module] { languageModel.model.layers }

    init(_ config: DeepSeekOCRConfiguration) {
        self.config = config
        kvHeads = Array(repeating: config.text.kvHeads, count: config.text.layers)
        _sam.wrappedValue = DeepSeekOCRSAM.Encoder(config.sam, imageSize: config.samImageSize)
        _clip.wrappedValue = DeepSeekOCRCLIP.Model(config.clip)
        _projector.wrappedValue = Projector(config.projector)
        _imageNewline.wrappedValue = zeros([config.projector.embedDim])
        _viewSeparator.wrappedValue = zeros([config.projector.embedDim])
        _languageModel.wrappedValue = DeepSeekOCRText.LanguageModel(config.text)
    }

    func encoded(_ views: MLXArray) -> MLXArray {
        let samFeatures = sam(views.asType(sam.patchEmbed.proj.weight.dtype))
        let clipFeatures = clip(samFeatures)
        let flat = samFeatures.reshaped(views.dim(0), -1, samFeatures.dim(-1))
        return projector.layers(concatenated([clipFeatures[0..., 1...], flat], axis: -1))
    }

    func rowsWithNewlines(_ grid: MLXArray) -> MLXArray {
        let (h, n) = (grid.dim(0), grid.dim(-1))
        return concatenated([grid, broadcast(imageNewline.reshaped(1, 1, n), to: [h, 1, n])], axis: 1).reshaped(-1, n)
    }

    func features(_ pixels: MLXArray, _ layout: DeepSeekOCRImageLayout) -> MLXArray {
        func square(_ x: MLXArray) -> MLXArray {
            let side = Int(Double(x.dim(-2)).squareRoot().rounded())
            return x.reshaped(x.dim(0), side, side, -1)
        }
        let global = square(encoded(pixels[..<layout.globalValues].reshaped(1, layout.side, layout.side, 3)))[0]
        var parts = [rowsWithNewlines(global), viewSeparator.reshaped(1, -1)]
        if layout.tiled {
            let views = pixels[layout.globalValues ..< layout.values].reshaped(layout.rows * layout.columns, layout.tile, layout.tile, 3)
            let tiles = square(encoded(views))
            let (q, n) = (tiles.dim(1), tiles.dim(-1))
            let grid = tiles.reshaped(layout.rows, layout.columns, q, q, n).transposed(0, 2, 1, 3, 4)
            parts.insert(rowsWithNewlines(grid.reshaped(layout.rows * q, layout.columns * q, n)), at: 0)
        }
        return concatenated(parts, axis: 0)
    }

    func merged(_ ids: MLXArray, pixels: MLXArray, frames: [THW]) throws(BasaltError) -> MLXArray {
        let layouts = try DeepSeekOCRImageLayout.parse(frames)
        let placeholders = ids[0].asArray(Int.self).count { $0 == config.imageToken }
        let (tokens, values) = (layouts.map(\.tokens).reduce(0, +), layouts.map(\.values).reduce(0, +))
        guard placeholders == tokens, pixels.size == values else {
            throw .invalidRequestStructure(
                details: "DeepSeek-OCR: \(layouts.count) images need \(tokens) placeholder tokens and \(values) pixel values, got \(placeholders) and \(pixels.size)")
        }
        var offset = 0
        let features = concatenated(layouts.map { layout in
            defer { offset += layout.values }
            return features(pixels.reshaped(-1)[offset ..< offset + layout.values], layout)
        }, axis: 0)
        let embeddings = languageModel.model.embedTokens(ids)
        let mask = ids .== config.imageToken
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
