import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

struct DeepSeekOCR2Configuration: Codable, Sendable {
    struct Encoder: Codable, Sendable {
        let dim: Int
        let layerCount: Int?
        let heads: Int?
        let kvHeads: Int?
        let intermediateSize: Int?

        var layers: Int { layerCount ?? 24 }
        var attentionHeads: Int { heads ?? 14 }
        var keyValueHeads: Int { kvHeads ?? 2 }
        var hiddenSize: Int { intermediateSize ?? 4864 }
        var eps: Float { 1e-6 }
        var ropeTheta: Float { 1_000_000 }

        enum CodingKeys: String, CodingKey {
            case dim, heads
            case layerCount = "layers"
            case kvHeads = "kv_heads"
            case intermediateSize = "intermediate_size"
        }
    }

    struct Vision: Codable, Sendable {
        struct Towers: Codable, Sendable {
            let sam: DeepSeekOCRConfiguration.SAM
            let encoder: Encoder

            enum CodingKeys: String, CodingKey {
                case sam = "sam_vit_b"
                case encoder = "qwen2-0-5b"
            }
        }

        let imageSize: Int
        let width: Towers

        enum CodingKeys: String, CodingKey {
            case imageSize = "image_size"
            case width
        }
    }

    let text: DeepSeekOCRConfiguration.Text
    let vision: Vision
    let projector: DeepSeekOCRConfiguration.Projector
    let imageTokenId: Int?

    var imageToken: Int { imageTokenId ?? 128_815 }
    var encoder: Encoder { vision.width.encoder }

    enum CodingKeys: String, CodingKey {
        case text = "language_config"
        case vision = "vision_config"
        case projector = "projector_config"
        case imageTokenId = "image_token_index"
    }

    func validated() throws(BasaltError) -> Self {
        let reasons = text.unsupported + [
            projector.type != "linear" ? "projector \(projector.type)" : nil,
            projector.inputDim != encoder.dim ? "a projector input of \(projector.inputDim) after a \(encoder.dim)-wide encoder" : nil,
            encoder.dim % encoder.attentionHeads != 0 || encoder.attentionHeads % encoder.keyValueHeads != 0
                ? "an encoder of width \(encoder.dim) with \(encoder.attentionHeads) heads and \(encoder.keyValueHeads) key-value heads" : nil,
        ].compactMap { $0 }
        guard reasons.isEmpty else {
            throw .modelConfigurationInvalid(path: "config.json", reason: "DeepSeek-OCR-2: this port does not run \(reasons.joined(separator: ", "))")
        }
        return self
    }
}

final class DeepSeekOCR2: Module, DeepSeekOCRDecoding, KVCacheDimensionProvider {
    static let modelType = "deepseekocr_2"

    let config: DeepSeekOCR2Configuration
    let kvHeads: [Int]
    @ModuleInfo(key: "sam_model") var sam: DeepSeekOCRSAM.Encoder
    @ModuleInfo(key: "vision_model") var vision: DeepSeekOCR2Encoder.Queries
    @ModuleInfo var projector: DeepSeekOCR.Projector
    @ParameterInfo(key: "view_separator") var viewSeparator: MLXArray
    @ModuleInfo(key: "language_model") var languageModel: DeepSeekOCRText.LanguageModel

    var loraLayers: [Module] { languageModel.model.layers }
    var imageToken: Int { config.imageToken }
    var newlines: Bool { false }

    init(_ config: DeepSeekOCR2Configuration) {
        self.config = config
        kvHeads = Array(repeating: config.text.kvHeads, count: config.text.layers)
        let samConfig = config.vision.width.sam
        _sam.wrappedValue = DeepSeekOCRSAM.Encoder(samConfig, imageSize: samConfig.imageSize ?? config.vision.imageSize, outputChannels: config.encoder.dim)
        _vision.wrappedValue = DeepSeekOCR2Encoder.Queries(config.encoder)
        _projector.wrappedValue = DeepSeekOCR.Projector(config.projector)
        _viewSeparator.wrappedValue = zeros([config.projector.embedDim])
        _languageModel.wrappedValue = DeepSeekOCRText.LanguageModel(config.text)
    }

    func encoded(_ views: MLXArray) throws(BasaltError) -> MLXArray {
        let encoded = try vision.encoder(sam(views.asType(sam.patchEmbed.proj.weight.dtype)))
        return projector.layers(encoded).reshaped(-1, config.projector.embedDim)
    }

    func features(_ pixels: MLXArray, _ layout: DeepSeekOCRImageLayout) throws(BasaltError) -> MLXArray {
        var parts = [try encoded(pixels[..<layout.globalValues].reshaped(1, layout.side, layout.side, 3)), viewSeparator.reshaped(1, -1)]
        if layout.tiled {
            let views = pixels[layout.globalValues ..< layout.values].reshaped(layout.rows * layout.columns, layout.tile, layout.tile, 3)
            parts.insert(try encoded(views), at: 0)
        }
        return concatenated(parts, axis: 0)
    }
}
