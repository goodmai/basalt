import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct RenderedBitmap: Sendable {
    public let width: Int
    public let height: Int
    public let rgba: Data
    public let png: Data
    public let cgImage: CGImage

    public init(width: Int, height: Int, rgba: Data, png: Data, cgImage: CGImage) {
        self.width = width
        self.height = height
        self.rgba = rgba
        self.png = png
        self.cgImage = cgImage
    }
}

public enum BitmapCodec {
    private static var srgbSpace: CGColorSpace {
        CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    }

    private static var bitmapInfo: UInt32 {
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    }

    public static func fromCGImage(_ source: CGImage) throws(VisualTestError) -> RenderedBitmap {
        let width = source.width
        let height = source.height
        guard width > 0, height > 0 else {
            throw .invalidImageData("empty CGImage")
        }
        guard let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: srgbSpace, bitmapInfo: bitmapInfo
        ) else {
            throw .invalidImageData("failed to allocate sRGB CGContext")
        }
        context.interpolationQuality = .high
        context.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let normalized = context.makeImage(),
              let cfData = normalized.dataProvider?.data else {
            throw .invalidImageData("failed to extract normalized CGImage data")
        }
        let raw = cfData as Data
        let rowBytes = width * 4
        let stride = normalized.bytesPerRow
        let rgba: Data
        if stride == rowBytes, raw.count == rowBytes * height {
            rgba = raw
        } else {
            var packed = Data(capacity: rowBytes * height)
            for row in 0 ..< height {
                let start = row * stride
                guard start + rowBytes <= raw.count else {
                    throw .invalidImageData("truncated CGImage row buffer")
                }
                packed.append(raw.subdata(in: start ..< (start + rowBytes)))
            }
            rgba = packed
        }
        let png = try encodePNG(normalized)
        return RenderedBitmap(width: width, height: height, rgba: rgba, png: png, cgImage: normalized)
    }

    public static func fromRGBA(_ rgba: Data, width: Int, height: Int) throws(VisualTestError) -> RenderedBitmap {
        guard width > 0, height > 0, rgba.count == width * height * 4,
              let provider = CGDataProvider(data: rgba as CFData),
              let image = CGImage(
                  width: width, height: height,
                  bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                  space: srgbSpace, bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else {
            throw .invalidImageData("invalid RGBA buffer for \(width)x\(height)")
        }
        let png = try encodePNG(image)
        return RenderedBitmap(width: width, height: height, rgba: rgba, png: png, cgImage: image)
    }

    public static func decodePNG(_ data: Data) throws(VisualTestError) -> RenderedBitmap {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw .invalidImageData("cannot decode PNG data")
        }
        return try fromCGImage(image)
    }

    public static func encodePNG(_ image: CGImage) throws(VisualTestError) -> Data {
        let mutable = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            mutable as CFMutableData, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw .invalidImageData("cannot create PNG destination")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw .invalidImageData("cannot finalize PNG destination")
        }
        return mutable as Data
    }
}

public enum BaselineStore {
    public static var defaultDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/VisualGoldens")
    }

    public static func fileURL(
        spec: VisualSceneSpec,
        variant: TraitVariant,
        directory: URL = defaultDirectory
    ) -> URL {
        directory.appendingPathComponent(spec.goldenFilename(for: variant))
    }

    public static func read(
        spec: VisualSceneSpec,
        variant: TraitVariant,
        directory: URL = defaultDirectory
    ) throws(VisualTestError) -> RenderedBitmap {
        let url = fileURL(spec: spec, variant: variant, directory: directory)
        guard let data = try? Data(contentsOf: url) else {
            throw .baselineMissing(path: url.path)
        }
        return try BitmapCodec.decodePNG(data)
    }

    public static func write(
        _ png: Data,
        spec: VisualSceneSpec,
        variant: TraitVariant,
        directory: URL = defaultDirectory
    ) throws(VisualTestError) {
        let url = fileURL(spec: spec, variant: variant, directory: directory)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: url, options: .atomic)
        } catch {
            throw .ioFailed(error.localizedDescription)
        }
    }
}

public enum BeforeAfterCompositor {
    public static func makeComparison(
        before: RenderedBitmap,
        after: RenderedBitmap,
        mask: [UInt8]
    ) throws(VisualTestError) -> RenderedBitmap {
        let gap = 12
        let panelWidth = max(before.width, after.width)
        let panelHeight = max(before.height, after.height)
        let totalWidth = panelWidth * 3 + gap * 2
        let heatmapImage: CGImage
        if before.width == after.width, before.height == after.height, mask.count == after.width * after.height {
            let heatRGBA = PixelDiffEngine.makeHeatmapRGBA(
                after: after.rgba.bytes, mask: mask, width: after.width, height: after.height
            )
            heatmapImage = try BitmapCodec.fromRGBA(heatRGBA, width: after.width, height: after.height).cgImage
        } else {
            let fullMask = [UInt8](repeating: 220, count: after.width * after.height)
            let heatRGBA = PixelDiffEngine.makeHeatmapRGBA(
                after: after.rgba.bytes, mask: fullMask, width: after.width, height: after.height
            )
            heatmapImage = try BitmapCodec.fromRGBA(heatRGBA, width: after.width, height: after.height).cgImage
        }
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(
            data: nil, width: totalWidth, height: panelHeight,
            bitsPerComponent: 8, bytesPerRow: totalWidth * 4,
            space: space, bitmapInfo: info
        ) else {
            throw .invalidImageData("cannot create comparison CGContext")
        }
        context.setFillColor(CGColor(srgbRed: 0.11, green: 0.11, blue: 0.12, alpha: 1.0))
        context.fill(CGRect(x: 0, y: 0, width: totalWidth, height: panelHeight))
        context.draw(
            before.cgImage,
            in: CGRect(x: 0, y: panelHeight - before.height, width: before.width, height: before.height)
        )
        context.draw(
            after.cgImage,
            in: CGRect(x: panelWidth + gap, y: panelHeight - after.height, width: after.width, height: after.height)
        )
        context.draw(
            heatmapImage,
            in: CGRect(x: (panelWidth + gap) * 2, y: panelHeight - after.height, width: after.width, height: after.height)
        )
        guard let composite = context.makeImage() else {
            throw .invalidImageData("cannot finalize comparison CGImage")
        }
        return try BitmapCodec.fromCGImage(composite)
    }
}
