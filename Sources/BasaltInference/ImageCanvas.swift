import BasaltCore
import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXVLM

enum ImageCanvas {
    static func rgb(_ image: CIImage, width: Int, height: Int, placing rect: CGRect? = nil, fill: CGFloat = 1) throws(BasaltError) -> MLXArray {
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        let canvasRect = CGRect(x: 0, y: 0, width: width, height: height)
        let target = rect ?? canvasRect
        let origin = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        let scaled = origin.extent.size == target.size ? origin : MediaProcessing.resampleBicubic(origin, to: target.size)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let source = context.createCGImage(scaled, from: CGRect(origin: .zero, size: target.size), format: .RGBA8, colorSpace: space),
              let canvas = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw .inferenceError("cannot render the image")
        }
        canvas.setFillColor(CGColor(colorSpace: space, components: [fill, fill, fill, 1]) ?? CGColor(gray: fill, alpha: 1))
        canvas.fill(canvasRect)
        canvas.draw(source, in: target)
        guard let data = canvas.data else { throw .inferenceError("cannot read the rendered image") }
        return MLXArray(Data(bytes: data, count: width * height * 4), [height, width, 4], type: UInt8.self)[0..., 0..., ..<3]
    }
}
