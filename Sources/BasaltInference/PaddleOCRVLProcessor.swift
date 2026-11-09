import BasaltCore
import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

struct PaddleOCRVLProcessor: UserInputProcessor {
    static let processorType = "PaddleOCRVLProcessor"
    static let placeholder = "<|IMAGE_PLACEHOLDER|>"

    let config: PaddleOCRVLProcessorConfiguration
    let tokenizer: any Tokenizer

    init(_ config: PaddleOCRVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    static func messages(_ input: UserInput) -> [Message] {
        func message(_ role: String, _ text: String, images: Int) -> Message {
            ["role": role, "content": Array(repeating: ["type": "image"], count: images) + [["type": "text", "text": text]]]
        }
        return switch input.prompt {
        case .text(let text): [message("user", text, images: input.images.count)]
        case .messages(let messages): messages
        case .chat(let messages): messages.map { message($0.role.rawValue, $0.content, images: $0.images.count) }
        }
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let tokens = try tokenizer.applyChatTemplate(
            messages: Self.messages(input), tools: input.tools, additionalContext: input.additionalContext)
        guard !input.images.isEmpty else { return LMInput(tokens: MLXArray(tokens)) }
        let images = try input.images.map { try patches(MediaProcessing.apply(try $0.asCIImage(), processing: input.processing)) }
        guard let token = tokenizer.convertTokenToId(Self.placeholder) else {
            throw BasaltError.inferenceError("PaddleOCR-VL: the tokenizer has no \(Self.placeholder) token")
        }
        let ids = MLXArray(try Self.expanded(tokens, token: token, frames: images.map(\.frame), mergeSize: config.mergeSize))[.newAxis]
        return LMInput(
            text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
            image: .init(pixels: concatenated(images.map(\.patches)), frames: images.map(\.frame)))
    }

    static func expanded(_ tokens: [Int], token: Int, frames: [THW], mergeSize: Int) throws(BasaltError) -> [Int] {
        guard tokens.count(where: { $0 == token }) == frames.count else {
            throw .invalidRequestStructure(details: "PaddleOCR-VL: the prompt has a different number of image placeholders than images")
        }
        var pending = frames[...]
        return tokens.flatMap { id -> [Int] in
            guard id == token, let frame = pending.popFirst() else { return [id] }
            return Array(repeating: token, count: frame.product / (mergeSize * mergeSize))
        }
    }

    func patches(_ image: CIImage) throws -> (patches: MLXArray, frame: THW) {
        guard !image.extent.isInfinite, image.extent.width.isFinite, image.extent.height.isFinite else {
            throw BasaltError.invalidRequestStructure(details: "PaddleOCR-VL cannot read an image with an unbounded extent (\(image.extent.size))")
        }
        let extent = image.extent.integral
        let size = try config.resized(width: Int(extent.width), height: Int(extent.height))
        let rgb = try Self.rgb(image, width: size.width, height: size.height).asType(.float32) / 255
        let normalized = (rgb - MLXArray(config.imageMean)) / MLXArray(config.imageStd)
        let (p, rows, columns) = (config.patchSize, size.height / config.patchSize, size.width / config.patchSize)
        let patches = normalized.reshaped(rows, p, columns, p, 3).transposed(0, 2, 1, 3, 4).reshaped(rows * columns, p, p, 3)
        return (patches, THW(1, rows, columns))
    }

    static func rgb(_ image: CIImage, width: Int, height: Int) throws(BasaltError) -> MLXArray {
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let source = context.createCGImage(image, from: image.extent, format: .RGBA8, colorSpace: space),
              let canvas = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw .inferenceError("PaddleOCR-VL cannot render the image")
        }
        canvas.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        canvas.fill(rect)
        canvas.interpolationQuality = .high
        canvas.draw(source, in: rect)
        guard let data = canvas.data else { throw .inferenceError("PaddleOCR-VL cannot read the rendered image") }
        return MLXArray(Data(bytes: data, count: width * height * 4), [height, width, 4], type: UInt8.self)[0..., 0..., ..<3]
    }
}
