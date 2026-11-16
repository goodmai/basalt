import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import MLX
import BasaltCore

public struct DecodedImageBuffer: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let channels: Int
    public let pixels: [Float]
    public let hasTransparency: Bool

    public init(width: Int, height: Int, channels: Int, pixels: [Float], hasTransparency: Bool) {
        self.width = width
        self.height = height
        self.channels = channels
        self.pixels = pixels
        self.hasTransparency = hasTransparency
    }

    public func rgba(x: Int, y: Int) -> (r: Float, g: Float, b: Float, a: Float) {
        let clampedX = min(max(0, x), max(0, width - 1))
        let clampedY = min(max(0, y), max(0, height - 1))
        let base = (clampedY * width + clampedX) * channels
        guard base + 2 < pixels.count else { return (0, 0, 0, 1) }
        let alpha = channels >= 4 && base + 3 < pixels.count ? pixels[base + 3] : 1.0
        return (pixels[base], pixels[base + 1], pixels[base + 2], alpha)
    }
}

public struct EncodedImageBuffer: Sendable, Equatable {
    public let data: Data
    public let format: ImageOutputFormat
    public let hasAlpha: Bool

    public init(data: Data, format: ImageOutputFormat, hasAlpha: Bool) {
        self.data = data
        self.format = format
        self.hasAlpha = hasAlpha
    }
}

public enum ImagePNGCodec {
    private static var srgbSpace: CGColorSpace {
        CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    }

    private static var premultipliedRGBAInfo: UInt32 {
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    }

    private static var opaqueRGBXInfo: UInt32 {
        CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    }

    public static func encode(
        _ tensor: MLXArray,
        format: ImageOutputFormat = .png,
        forceOpaque: Bool = false
    ) throws(BasaltError) -> EncodedImageBuffer {
        let hwc = try normalizedHWCTensor(tensor)
        eval(hwc)
        let dims = ImageDimensions(width: hwc.dim(1), height: hwc.dim(0))
        let channels = hwc.dim(2)
        let floats = hwc.asType(.float32).asArray(Float.self)
        return try encodePixels(
            floats,
            dimensions: dims,
            channels: channels,
            format: format,
            forceOpaque: forceOpaque
        )
    }

    public static func encodePixels(
        _ pixels: [Float],
        dimensions: ImageDimensions,
        channels: Int,
        format: ImageOutputFormat = .png,
        forceOpaque: Bool = false
    ) throws(BasaltError) -> EncodedImageBuffer {
        guard dimensions.width > 0, dimensions.height > 0, channels == 3 || channels == 4 else {
            throw .invalidRequestStructure(details: "image codec: expected positive dimensions and 3 or 4 channels")
        }
        let expectedCount = dimensions.width * dimensions.height * channels
        guard pixels.count == expectedCount else {
            throw .invalidRequestStructure(details: "image codec: pixel count \(pixels.count) != \(expectedCount)")
        }
        let emitAlpha = channels == 4 && !forceOpaque && format != .jpeg
        let packed = packRGBABytes(pixels, pixelCount: dimensions.width * dimensions.height, channels: channels, emitAlpha: emitAlpha)
        let bitmapInfo = emitAlpha ? premultipliedRGBAInfo : opaqueRGBXInfo
        guard let provider = CGDataProvider(data: Data(packed) as CFData),
              let cgImage = CGImage(
                  width: dimensions.width,
                  height: dimensions.height,
                  bitsPerComponent: 8,
                  bitsPerPixel: 32,
                  bytesPerRow: dimensions.width * 4,
                  space: srgbSpace,
                  bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              ) else {
            throw .inferenceError("image codec: failed to construct CGImage")
        }
        let (encoded, resolvedFormat) = try writeDestination(cgImage, requested: format)
        let effectiveAlpha = emitAlpha && resolvedFormat != .jpeg
        return EncodedImageBuffer(data: encoded, format: resolvedFormat, hasAlpha: effectiveAlpha)
    }

    public static func decode(
        _ data: Data,
        targetWidth: Int? = nil,
        targetHeight: Int? = nil
    ) throws(BasaltError) -> MLXArray {
        let decoded = try decodePixels(data, targetWidth: targetWidth, targetHeight: targetHeight)
        return MLXArray(decoded.pixels, [decoded.height, decoded.width, 4])
    }

    public static func decodePixels(
        _ data: Data,
        targetWidth: Int? = nil,
        targetHeight: Int? = nil
    ) throws(BasaltError) -> DecodedImageBuffer {
        guard !data.isEmpty,
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw .invalidRequestStructure(details: "image codec: input bytes are not a valid decodable image")
        }
        let outW = targetWidth ?? image.width
        let outH = targetHeight ?? image.height
        guard outW > 0, outH > 0 else {
            throw .invalidRequestStructure(details: "image codec: decoded target dimensions must be positive")
        }
        return try renderBilinearRGBA(image, width: outW, height: outH)
    }
}

extension ImagePNGCodec {
    private static func normalizedHWCTensor(_ tensor: MLXArray) throws(BasaltError) -> MLXArray {
        let hwc: MLXArray
        if tensor.ndim == 4 {
            guard tensor.dim(0) == 1 else {
                throw .invalidRequestStructure(details: "image codec: 4D tensor batch size must be 1, got \(tensor.dim(0))")
            }
            hwc = tensor[0]
        } else if tensor.ndim == 3 {
            hwc = tensor
        } else {
            throw .invalidRequestStructure(details: "image codec: expected 3D [H,W,C] or 4D [1,H,W,C] tensor, got \(tensor.shape)")
        }
        let c = hwc.dim(2)
        guard hwc.dim(0) > 0, hwc.dim(1) > 0, c == 3 || c == 4 else {
            throw .invalidRequestStructure(details: "image codec: expected H>0, W>0 and C in {3,4}, got \(hwc.shape)")
        }
        return hwc
    }

    private static func clampUnit(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return min(1.0, max(0.0, value))
    }

    private static func packRGBABytes(
        _ pixels: [Float],
        pixelCount: Int,
        channels: Int,
        emitAlpha: Bool
    ) -> [UInt8] {
        var bytes = [UInt8](repeating: 255, count: pixelCount * 4)
        for idx in 0 ..< pixelCount {
            let src = idx * channels
            let dst = idx * 4
            let r = clampUnit(pixels[src])
            let g = clampUnit(pixels[src + 1])
            let b = clampUnit(pixels[src + 2])
            if emitAlpha && channels == 4 {
                let a = clampUnit(pixels[src + 3])
                bytes[dst] = UInt8((r * a * 255.0).rounded())
                bytes[dst + 1] = UInt8((g * a * 255.0).rounded())
                bytes[dst + 2] = UInt8((b * a * 255.0).rounded())
                bytes[dst + 3] = UInt8((a * 255.0).rounded())
            } else {
                bytes[dst] = UInt8((r * 255.0).rounded())
                bytes[dst + 1] = UInt8((g * 255.0).rounded())
                bytes[dst + 2] = UInt8((b * 255.0).rounded())
                bytes[dst + 3] = 255
            }
        }
        return bytes
    }

    private static func writeDestination(
        _ image: CGImage,
        requested: ImageOutputFormat
    ) throws(BasaltError) -> (Data, ImageOutputFormat) {
        let candidates: [(ImageOutputFormat, UTType)] = switch requested {
        case .png: [(.png, .png)]
        case .jpeg: [(.jpeg, .jpeg)]
        case .webp: [(.webp, .webP), (.png, .png)]
        }
        for (fmt, utType) in candidates {
            let buffer = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(buffer as CFMutableData, utType.identifier as CFString, 1, nil) else {
                continue
            }
            let props: CFDictionary? = fmt == .jpeg
                ? ([kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
                : nil
            CGImageDestinationAddImage(dest, image, props)
            if CGImageDestinationFinalize(dest), buffer.length > 0 {
                return (buffer as Data, fmt)
            }
        }
        throw .inferenceError("image codec: failed to finalize ImageIO destination for \(requested.rawValue)")
    }

    private static func renderBilinearRGBA(
        _ source: CGImage,
        width: Int,
        height: Int
    ) throws(BasaltError) -> DecodedImageBuffer {
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: srgbSpace,
            bitmapInfo: premultipliedRGBAInfo
        ) else {
            throw .inferenceError("image codec: failed to allocate sRGB CGContext")
        }
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        context.clear(rect)
        context.interpolationQuality = .medium
        context.draw(source, in: rect)
        guard let rendered = context.makeImage(),
              let rawCF = rendered.dataProvider?.data else {
            throw .inferenceError("image codec: failed to read rendered CGContext image")
        }
        let raw = rawCF as Data
        return unpackPremultipliedRGBA(raw, width: width, height: height, bytesPerRow: rendered.bytesPerRow)
    }

    private static func unpackPremultipliedRGBA(
        _ raw: Data,
        width: Int,
        height: Int,
        bytesPerRow: Int
    ) -> DecodedImageBuffer {
        var floats = [Float](repeating: 0, count: width * height * 4)
        var sawTransparency = false
        for y in 0 ..< height {
            let rowStart = y * bytesPerRow
            for x in 0 ..< width {
                let src = rowStart + x * 4
                let dst = (y * width + x) * 4
                guard src + 3 < raw.count else { continue }
                let rP = raw[src]
                let gP = raw[src + 1]
                let bP = raw[src + 2]
                let aP = raw[src + 3]
                if aP < 255 { sawTransparency = true }
                let alpha = Float(aP) / 255.0
                if aP > 0 {
                    let inv = 1.0 / Float(aP)
                    floats[dst] = min(1.0, Float(rP) * inv)
                    floats[dst + 1] = min(1.0, Float(gP) * inv)
                    floats[dst + 2] = min(1.0, Float(bP) * inv)
                }
                floats[dst + 3] = alpha
            }
        }
        return DecodedImageBuffer(
            width: width,
            height: height,
            channels: 4,
            pixels: floats,
            hasTransparency: sawTransparency
        )
    }
}
