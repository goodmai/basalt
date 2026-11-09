import Foundation
import CryptoKit
import ImageIO

public struct ImageInput: Sendable, Codable, Hashable {
    public static let maxCount = 8
    public static let maxBytes = 20 * 1024 * 1024
    public static let maxTotalBytes = 32 * 1024 * 1024
    public static let maxPixels = 50_000_000
    public static let mimeTypes: Set<String> = ["image/png", "image/jpeg", "image/webp", "image/heic"]
    public static let remoteRefusal = "remote image URLs are not fetched; send the image as a data: URI"

    public let data: Data
    public let mime: String
    public let digest: String

    private init(data: Data, mime: String, digest: String) {
        self.data = data
        self.mime = mime
        self.digest = digest
    }

    public init(data: Data, mime: String) throws(BasaltError) {
        let mime = mime.lowercased()
        guard Self.mimeTypes.contains(mime) else {
            throw .invalidRequestStructure(details: "image type '\(mime)' is not supported; send image/png, image/jpeg, image/webp or image/heic")
        }
        guard data.count <= Self.maxBytes else {
            throw .invalidRequestStructure(details: "an image of \(data.count) bytes is over the limit of 20 MB per image")
        }
        try Self.checkPixels(data)
        self.init(data: data, mime: mime, digest: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    }

    public static func checkPixels(_ data: Data) throws(BasaltError) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int else { return }
        guard width <= maxPixels, height <= maxPixels, width * height <= maxPixels else {
            throw .invalidRequestStructure(details: "an image of \(width)x\(height) pixels is over the limit of \(maxPixels / 1_000_000) megapixels")
        }
    }

    public init(base64: some StringProtocol, mime: String) throws(BasaltError) {
        guard let data = Data(base64Encoded: String(base64)), !data.isEmpty else {
            throw .invalidRequestStructure(details: "the image data is empty or not valid base64")
        }
        try self.init(data: data, mime: mime)
    }

    public init(url: String) throws(BasaltError) {
        switch url.prefix(while: { $0 != ":" }).lowercased() {
        case "http", "https":
            throw .invalidRequestStructure(details: Self.remoteRefusal)
        case "data":
            guard let comma = url.firstIndex(of: ","), url[..<comma].hasSuffix(";base64") else {
                throw .invalidRequestStructure(details: "an image data: URI must have the form data:<mime>;base64,<data>")
            }
            let mime = url[url.index(url.startIndex, offsetBy: 5) ..< comma].dropLast(";base64".count)
            try self.init(base64: url[url.index(after: comma)...], mime: String(mime))
        default:
            throw .invalidRequestStructure(details: "an image URL must be a data:<mime>;base64,<data> URI")
        }
    }

    public static func checkLimits(_ images: [ImageInput]) throws(BasaltError) {
        try checkCount(images.count)
        let total = images.reduce(0) { $0 + $1.data.count }
        guard total <= maxTotalBytes else {
            throw .invalidRequestStructure(details: "the images of a request add up to \(total) bytes, over the limit of 32 MB per request")
        }
    }

    public static func checkCount(_ count: Int) throws(BasaltError) {
        guard count <= maxCount else {
            throw .invalidRequestStructure(details: "a request carries at most \(maxCount) images; this one has \(count)")
        }
    }

    enum CodingKeys: String, CodingKey {
        case data, mime, sha256, bytes
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let mime = try container.decode(String.self, forKey: .mime)
        if let data = try container.decodeIfPresent(Data.self, forKey: .data) {
            try self.init(data: data, mime: mime)
        } else {
            self.init(data: Data(), mime: mime, digest: try container.decode(String.self, forKey: .sha256))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(mime, forKey: .mime)
        try container.encode(digest, forKey: .sha256)
        try container.encode(data.count, forKey: .bytes)
    }
}
