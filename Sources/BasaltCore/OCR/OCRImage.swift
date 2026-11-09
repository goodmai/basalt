import CoreGraphics
import Foundation
import ImageIO
import PDFKit
import UniformTypeIdentifiers

enum OCRImage {
    static let maxSide = 4096.0

    static func render(pdf data: Data, pages selection: OCRPageSelection?, dpi: Double) throws(BasaltError) -> [(number: Int, image: ImageInput)] {
        guard let document = PDFDocument(data: data), document.pageCount > 0 else {
            throw .invalidRequestStructure(details: "the PDF cannot be read or has no pages")
        }
        guard !document.isLocked else { throw .invalidRequestStructure(details: "the PDF is encrypted; send an unlocked copy") }
        let numbers = try OCRPageSelection.resolve(selection, pageCount: document.pageCount)
        var pages: [(number: Int, image: ImageInput)] = []
        for number in numbers {
            guard let page = document.page(at: number - 1), let image = render(page, dpi: dpi) else {
                throw .invalidRequestStructure(details: "page \(number) of the PDF cannot be rendered")
            }
            pages.append((number, try ImageInput(data: try png(image), mime: "image/png")))
        }
        return pages
    }

    static func render(_ page: PDFPage, dpi: Double) -> CGImage? {
        let bounds = page.bounds(for: .cropBox)
        let size = page.rotation % 180 == 0 ? bounds.size : CGSize(width: bounds.height, height: bounds.width)
        guard size.width > 0, size.height > 0 else { return nil }
        let scale = min(dpi / 72, maxSide / max(size.width, size.height))
        let width = Int((size.width * scale).rounded()), height = Int((size.height * scale).rounded())
        guard let context = canvas(width: width, height: height) else { return nil }
        context.scaleBy(x: CGFloat(width) / size.width, y: CGFloat(height) / size.height)
        page.draw(with: .cropBox, to: context)
        return context.makeImage()
    }

    static func canvas(width: Int, height: Int) -> CGContext? {
        guard width > 0, height > 0, let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .high
        return context
    }

    static func decode(_ data: Data) throws(BasaltError) -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary) else {
            throw .invalidRequestStructure(details: "the page image cannot be decoded")
        }
        return image
    }

    static func png(_ image: CGImage) throws(BasaltError) -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw .inferenceError("cannot create a PNG encoder")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw .inferenceError("cannot encode the page as PNG") }
        return data as Data
    }

    static func pngInput(_ image: CGImage) throws(BasaltError) -> ImageInput {
        try ImageInput(data: try png(image), mime: "image/png")
    }

    static func resized(_ image: CGImage, width: Int, height: Int) -> CGImage? {
        guard let context = canvas(width: width, height: height) else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }

    static func crop(_ image: CGImage, to box: OCRBox) -> CGImage? {
        let (width, height) = (Double(image.width), Double(image.height))
        let x1 = (box.x1 * width).rounded(.toNearestOrEven), y1 = (box.y1 * height).rounded(.toNearestOrEven)
        let x2 = (box.x2 * width).rounded(.toNearestOrEven), y2 = (box.y2 * height).rounded(.toNearestOrEven)
        guard x2 - x1 >= 1, y2 - y1 >= 1 else { return nil }
        return image.cropping(to: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1))
    }

    static func rotated(_ image: CGImage, counterclockwise degrees: Int) -> CGImage? {
        let (width, height) = (CGFloat(image.width), CGFloat(image.height))
        let quarter = ((degrees / 90) % 4 + 4) % 4
        guard quarter != 0 else { return image }
        let sideways = quarter % 2 == 1
        guard let context = canvas(width: Int(sideways ? height : width), height: Int(sideways ? width : height)) else { return nil }
        let offsets = [CGPoint.zero, CGPoint(x: height, y: 0), CGPoint(x: width, y: height), CGPoint(x: 0, y: width)]
        context.translateBy(x: offsets[quarter].x, y: offsets[quarter].y)
        context.rotate(by: CGFloat(quarter) * .pi / 2)
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }

    static func fitted(_ image: CGImage, minEdge: Int, maxRatio: Double) -> CGImage? {
        var image = image
        let (width, height) = (image.width, image.height)
        if Double(max(width, height)) / Double(min(width, height)) > maxRatio {
            let padded = width > height
                ? (width, Int((Double(width) / maxRatio).rounded(.up))) : (Int((Double(height) / maxRatio).rounded(.up)), height)
            guard let context = canvas(width: padded.0, height: padded.1) else { return nil }
            let top = (padded.1 - height) / 2
            context.draw(image, in: CGRect(x: (padded.0 - width) / 2, y: padded.1 - height - top, width: width, height: height))
            guard let pasted = context.makeImage() else { return nil }
            image = pasted
        }
        guard min(image.width, image.height) < minEdge else { return image }
        let scale = Double(minEdge) / Double(min(image.width, image.height))
        return resized(image, width: Int((Double(image.width) * scale).rounded(.up)), height: Int((Double(image.height) * scale).rounded(.up)))
    }
}
