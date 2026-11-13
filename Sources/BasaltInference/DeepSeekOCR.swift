import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

final class DeepSeekOCR: Module, DeepSeekOCRDecoding, KVCacheDimensionProvider {
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
    var imageToken: Int { config.imageToken }
    var newlines: Bool { true }

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
}
