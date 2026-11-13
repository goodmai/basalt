import Foundation

public struct PixelFrameDimensions: Equatable, Sendable {
    public var expectedWidth: Int
    public var expectedHeight: Int
    public var actualWidth: Int
    public var actualHeight: Int

    public init(width: Int, height: Int) {
        self.expectedWidth = width
        self.expectedHeight = height
        self.actualWidth = width
        self.actualHeight = height
    }

    public init(expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int) {
        self.expectedWidth = expectedWidth
        self.expectedHeight = expectedHeight
        self.actualWidth = actualWidth
        self.actualHeight = actualHeight
    }
}

public enum PixelDiffEngine {
    public static func compare(
        expected: RawSpan,
        actual: RawSpan,
        dimensions: PixelFrameDimensions,
        tolerance: DiffTolerance
    ) -> (result: DiffResult, mask: [UInt8]) {
        let total = dimensions.expectedWidth * dimensions.expectedHeight
        let sameDims = dimensions.expectedWidth == dimensions.actualWidth
            && dimensions.expectedHeight == dimensions.actualHeight
        guard sameDims, expected.byteCount == total * 4, actual.byteCount == total * 4 else {
            return (mismatchResult(dimensions: dimensions, actualByteCount: actual.byteCount), [])
        }
        var changed = 0
        var maxDelta = 0.0
        var bounds: PixelBounds?
        var mask = [UInt8](repeating: 0, count: total)
        let width = dimensions.expectedWidth
        for index in 0 ..< total {
            let offset = index * 4
            let delta = yiqDelta(expected: expected, actual: actual, offset: offset)
            if delta > maxDelta { maxDelta = delta }
            if delta > tolerance.pixelDeltaThreshold {
                changed += 1
                let scaled = Int((delta * 255.0 * 4.0).rounded())
                mask[index] = UInt8(min(255, max(64, scaled)))
                let x = width > 0 ? index % width : 0
                let y = width > 0 ? index / width : 0
                bounds = expand(bounds, x: x, y: y)
            }
        }
        let ratio = total > 0 ? Double(changed) / Double(total) : 0.0
        let verdict: DiffVerdict =
            if changed == 0 {
                .identical
            } else if ratio <= tolerance.maxChangedRatio {
                .withinTolerance
            } else {
                .mismatched
            }
        return (
            DiffResult(
                totalPixels: total, changedPixels: changed, changedRatio: ratio,
                maxPixelDelta: maxDelta, bounds: bounds, verdict: verdict
            ),
            mask
        )
    }

    public static func makeHeatmapRGBA(after: RawSpan, mask: [UInt8], width: Int, height: Int) -> Data {
        let total = width * height
        guard after.byteCount == total * 4, mask.count == total else {
            return Data(repeating: 0, count: max(0, total * 4))
        }
        var output = [UInt8](repeating: 255, count: total * 4)
        for index in 0 ..< total {
            let offset = index * 4
            let r = Double(after[offset])
            let g = Double(after[offset + 1])
            let b = Double(after[offset + 2])
            let strength = Double(mask[index]) / 255.0
            if strength > 0 {
                let blend = max(0.55, strength)
                output[offset] = UInt8(min(255.0, r * (1.0 - blend) + 255.0 * blend))
                output[offset + 1] = UInt8(min(255.0, g * (1.0 - blend) + 45.0 * blend))
                output[offset + 2] = UInt8(min(255.0, b * (1.0 - blend) + 85.0 * blend))
                output[offset + 3] = 255
            } else {
                let luma = UInt8(min(255.0, (0.299 * r + 0.587 * g + 0.114 * b) * 0.25 + 18.0))
                output[offset] = luma
                output[offset + 1] = luma
                output[offset + 2] = luma
                output[offset + 3] = 255
            }
        }
        return Data(output)
    }

    private static func yiqDelta(expected: RawSpan, actual: RawSpan, offset: Int) -> Double {
        let r1 = Double(expected[offset]) / 255.0
        let g1 = Double(expected[offset + 1]) / 255.0
        let b1 = Double(expected[offset + 2]) / 255.0
        let a1 = Double(expected[offset + 3]) / 255.0
        let r2 = Double(actual[offset]) / 255.0
        let g2 = Double(actual[offset + 1]) / 255.0
        let b2 = Double(actual[offset + 2]) / 255.0
        let a2 = Double(actual[offset + 3]) / 255.0
        let dr = (r1 * a1) - (r2 * a2)
        let dg = (g1 * a1) - (g2 * a2)
        let db = (b1 * a1) - (b2 * a2)
        let y = dr * 0.29889531 + dg * 0.58662247 + db * 0.11448223
        let i = dr * 0.59597799 - dg * 0.27417610 - db * 0.32180189
        let q = dr * 0.21147017 - dg * 0.52261711 + db * 0.31114694
        return 0.5053 * y * y + 0.299 * i * i + 0.1957 * q * q
    }

    private static func mismatchResult(dimensions: PixelFrameDimensions, actualByteCount: Int) -> DiffResult {
        let total = dimensions.expectedWidth * dimensions.expectedHeight
        let inferredHeight = dimensions.actualWidth > 0 ? (actualByteCount / 4) / dimensions.actualWidth : 0
        let actualH = dimensions.actualHeight > 0 ? dimensions.actualHeight : inferredHeight
        return DiffResult(
            totalPixels: total, changedPixels: max(total, 1), changedRatio: 1.0,
            maxPixelDelta: 1.0, bounds: nil,
            verdict: .dimensionMismatch(
                expectedWidth: dimensions.expectedWidth,
                expectedHeight: dimensions.expectedHeight,
                actualWidth: dimensions.actualWidth,
                actualHeight: actualH
            )
        )
    }

    private static func expand(_ current: PixelBounds?, x: Int, y: Int) -> PixelBounds {
        guard var current else { return PixelBounds(minX: x, minY: y, maxX: x, maxY: y) }
        current.minX = min(current.minX, x)
        current.minY = min(current.minY, y)
        current.maxX = max(current.maxX, x)
        current.maxY = max(current.maxY, y)
        return current
    }
}
