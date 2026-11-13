import Foundation

public enum OCRResolution: String, Sendable, CaseIterable, Codable {
    case tiny, small, base, large, gundam

    public static let tileSide = 640
    public static let pixelsPerQuery = 64
    public static let tileQueries = tileSide / pixelsPerQuery

    public var side: Int {
        switch self {
        case .tiny: 512
        case .small: 640
        case .base, .gundam: 1024
        case .large: 1280
        }
    }

    public var queries: Int { side / Self.pixelsPerQuery }

    public func grid(width: Int, height: Int, tile: Int = OCRResolution.tileSide, maxTiles: Int = 9) -> (columns: Int, rows: Int) {
        guard self == .gundam, width > tile || height > tile, width > 0, height > 0, maxTiles >= 2 else { return (1, 1) }
        let aspect = Double(width) / Double(height)
        let candidates = (1 ... maxTiles).flatMap { i in (1 ... maxTiles).map { (i, $0) } }
            .filter { (2 ... maxTiles).contains($0.0 * $0.1) }
            .sorted { ($0.0 * $0.1, $0.0) < ($1.0 * $1.1, $1.0) }
        var best = (columns: 1, rows: 1)
        var bestDifference = Double.infinity
        for (columns, rows) in candidates {
            let difference = abs(aspect - Double(columns) / Double(rows))
            if difference < bestDifference {
                (best, bestDifference) = ((columns, rows), difference)
            } else if difference == bestDifference, Double(width * height) > 0.5 * Double(tile * tile * columns * rows) {
                best = (columns, rows)
            }
        }
        return best
    }

    public func visionTokens(width: Int, height: Int) -> Int {
        let global = (queries + 1) * queries + 1
        let (columns, rows) = grid(width: width, height: height)
        guard columns * rows > 1 else { return global }
        return global + (Self.tileQueries * columns + 1) * (Self.tileQueries * rows)
    }
}

public struct OCRMode: Sendable, Equatable {
    public static let resolutionKey = "ocr_resolution"
    public static let groundingKey = "ocr_grounding"

    public let resolution: OCRResolution
    public let grounding: Bool

    public init(resolution: OCRResolution, grounding: Bool) {
        (self.resolution, self.grounding) = (resolution, grounding)
    }

    public var templateContext: [String: any Sendable] {
        [Self.resolutionKey: resolution.rawValue, Self.groundingKey: grounding]
    }
}
