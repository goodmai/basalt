import BasaltCore
import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

struct DeepSeekOCRProcessorConfiguration: Codable, Sendable {
    let imageMean: [Float]
    let imageStd: [Float]

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }
}

struct DeepSeekOCRProcessor: UserInputProcessor {
    struct Geometry: Sendable {
        static let v1 = Geometry(name: "DeepSeek-OCR", tile: OCRResolution.tileSide, maxTiles: 9, newlines: true, resolutions: OCRResolution.allCases)
        static let v2 = Geometry(name: "DeepSeek-OCR-2", tile: 768, maxTiles: 6, newlines: false, resolutions: [.base, .gundam])

        let name: String
        let tile: Int
        let maxTiles: Int
        let newlines: Bool
        let resolutions: [OCRResolution]
    }

    static let processorType = "DeepseekVLV2Processor"
    static let ocr2ProcessorType = "DeepseekOCR2Processor"
    static let imageToken = "<image>"
    static let groundingToken = "<|grounding|>"
    static let padding: CGFloat = 127 / 255

    let config: DeepSeekOCRProcessorConfiguration
    let tokenizer: any Tokenizer
    var geometry = Geometry.v1

    init(_ config: DeepSeekOCRProcessorConfiguration, tokenizer: any Tokenizer, geometry: Geometry = .v1) {
        (self.config, self.tokenizer, self.geometry) = (config, tokenizer, geometry)
    }

    static func mode(_ context: [String: any Sendable]?) -> OCRMode {
        let resolution = (context?[OCRMode.resolutionKey] as? String).flatMap(OCRResolution.init(rawValue:)) ?? .gundam
        return OCRMode(resolution: resolution, grounding: context?[OCRMode.groundingKey] as? Bool ?? false)
    }

    static func prompt(_ input: UserInput, grounding: Bool) -> String {
        let turns: [(text: String, images: Int)] = switch input.prompt {
        case .text(let text): [(text, input.images.count)]
        case .chat(let messages): messages.map { ($0.content, $0.images.count) }
        case .messages(let messages): messages.enumerated().map { (($1["content"] as? String) ?? "", $0 == 0 ? input.images.count : 0) }
        }
        return turns.enumerated().map { index, turn in
            String(repeating: imageToken + "\n", count: turn.images) + (grounding && index == turns.count - 1 ? groundingToken : "") + turn.text + " "
        }.joined()
    }

    func tokens(_ prompt: String, layouts: [DeepSeekOCRImageLayout]) throws(BasaltError) -> [Int] {
        let pieces = prompt.components(separatedBy: Self.imageToken)
        guard pieces.count == layouts.count + 1 else {
            throw .invalidRequestStructure(details: "DeepSeek-OCR: the prompt has \(pieces.count - 1) image places for \(layouts.count) images")
        }
        guard let bos = tokenizer.bosToken.flatMap(tokenizer.convertTokenToId), let image = tokenizer.convertTokenToId(Self.imageToken) else {
            throw .inferenceError("DeepSeek-OCR: the tokenizer has no beginning-of-sentence or \(Self.imageToken) token")
        }
        let ids = pieces.enumerated().reduce(into: [bos]) { ids, entry in
            ids += tokenizer.encode(text: entry.element, addSpecialTokens: false)
            if entry.offset < layouts.count { ids += Array(repeating: image, count: layouts[entry.offset].tokens) }
        }
        return Array(ids.dropLast())
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let mode = Self.mode(input.additionalContext)
        let views = try input.images.map { try view(MediaProcessing.apply(try $0.asCIImage(), processing: input.processing), mode.resolution) }
        let ids = MLXArray(try tokens(Self.prompt(input, grounding: mode.grounding), layouts: views.map(\.layout)))[.newAxis]
        guard !views.isEmpty else { return LMInput(text: .init(tokens: ids)) }
        return LMInput(
            text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
            image: .init(pixels: concatenated(views.map(\.pixels)), frames: views.flatMap(\.layout.frames)))
    }

    static func contained(width: Int, height: Int, side: Int) -> CGRect {
        let (w, h, s) = (Double(width), Double(height), Double(side))
        let size = w == h ? (s, s) : w > h ? (s, max(1, (h / w * s).rounded(.toNearestOrEven))) : (max(1, (w / h * s).rounded(.toNearestOrEven)), s)
        let top = ((s - size.1) * 0.5).rounded(.toNearestOrEven)
        let left = ((s - size.0) * 0.5).rounded(.toNearestOrEven)
        return CGRect(x: left, y: s - top - size.1, width: size.0, height: size.1)
    }

    func view(_ image: CIImage, _ resolution: OCRResolution) throws -> (pixels: MLXArray, layout: DeepSeekOCRImageLayout) {
        guard !image.extent.isInfinite, image.extent.width.isFinite, image.extent.height.isFinite,
              image.extent.width * image.extent.height <= CGFloat(ImageInput.maxPixels) else {
            throw BasaltError.invalidRequestStructure(details: "DeepSeek-OCR cannot read an image with an unbounded or oversized extent (\(image.extent.size))")
        }
        let extent = image.extent.integral
        let (width, height, side) = (Int(extent.width), Int(extent.height), resolution.side)
        guard width > 0, height > 0 else { throw BasaltError.invalidRequestStructure(details: "DeepSeek-OCR cannot read a \(width)x\(height) image") }
        guard geometry.resolutions.contains(resolution) else {
            throw BasaltError.invalidRequestStructure(
                details: "\(geometry.name) has no \(resolution.rawValue) mode; use \(geometry.resolutions.map(\.rawValue).joined(separator: " or "))")
        }
        let (columns, rows) = resolution.grid(width: width, height: height, tile: geometry.tile, maxTiles: geometry.maxTiles)
        let layout = DeepSeekOCRImageLayout(
            side: side, columns: columns, rows: rows, tile: resolution == .gundam ? geometry.tile : 0, newlines: geometry.newlines)
        let stretched = resolution != .gundam && side <= OCRResolution.tileSide
        var views = [try ImageCanvas.rgb(
            image, width: side, height: side, placing: stretched ? nil : Self.contained(width: width, height: height, side: side), fill: Self.padding)]
        if layout.tiled {
            let tile = layout.tile
            let page = try ImageCanvas.rgb(image, width: tile * columns, height: tile * rows)
            views.append(page.reshaped(rows, tile, columns, tile, 3).transposed(0, 2, 1, 3, 4))
        }
        let pixels = concatenated(views.map { (($0.asType(.float32) / 255 - MLXArray(config.imageMean)) / MLXArray(config.imageStd)).reshaped(-1) })
        return (pixels, layout)
    }
}
