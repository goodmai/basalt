import BasaltCore
import Foundation

struct PaddleOCRVLConfiguration: Codable, Sendable {
    struct Vision: Codable, Sendable {
        let hiddenSize: Int
        let intermediateSize: Int
        let layers: Int
        let heads: Int
        let channels: Int
        let imageSize: Int
        let patchSize: Int
        let layerNormEps: Float
        let mergeSize: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case layers = "num_hidden_layers"
            case heads = "num_attention_heads"
            case channels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case layerNormEps = "layer_norm_eps"
            case mergeSize = "spatial_merge_size"
        }
    }

    struct RopeScaling: Codable, Sendable {
        let mropeSection: [Int]

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
        }
    }

    let hiddenSize: Int
    let layers: Int
    let intermediateSize: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let vocabularySize: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let ropeScaling: RopeScaling
    let imageTokenId: Int
    let vision: Vision

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case layers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case heads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case imageTokenId = "image_token_id"
        case vision = "vision_config"
    }
}

struct PaddleOCRVLProcessorConfiguration: Codable, Sendable {
    let minPixels: Int
    let maxPixels: Int
    let patchSize: Int
    let mergeSize: Int
    let imageMean: [Float]
    let imageStd: [Float]

    var factor: Int { patchSize * mergeSize }

    enum CodingKeys: String, CodingKey {
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case patchSize = "patch_size"
        case mergeSize = "merge_size"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }

    func resized(width: Int, height: Int) throws(BasaltError) -> (width: Int, height: Int) {
        guard width > 0, height > 0 else {
            throw .invalidRequestStructure(details: "PaddleOCR-VL cannot read a \(width)x\(height) image")
        }
        var (width, height) = (Double(width), Double(height))
        let factor = Double(factor)
        if height < factor { (width, height) = ((width * factor / height).rounded(.toNearestOrEven), factor) }
        if width < factor { (width, height) = (factor, (height * factor / width).rounded(.toNearestOrEven)) }
        guard max(height, width) / min(height, width) <= 200 else {
            throw .invalidRequestStructure(details: "PaddleOCR-VL takes images with an aspect ratio up to 200, got \(Int(width))x\(Int(height))")
        }
        var h = (height / factor).rounded(.toNearestOrEven) * factor
        var w = (width / factor).rounded(.toNearestOrEven) * factor
        if h * w > Double(maxPixels) {
            let beta = (height * width / Double(maxPixels)).squareRoot()
            h = (height / beta / factor).rounded(.down) * factor
            w = (width / beta / factor).rounded(.down) * factor
        } else if h * w < Double(minPixels) {
            let beta = (Double(minPixels) / (height * width)).squareRoot()
            h = (height * beta / factor).rounded(.up) * factor
            w = (width * beta / factor).rounded(.up) * factor
        }
        return (Int(w), Int(h))
    }
}
