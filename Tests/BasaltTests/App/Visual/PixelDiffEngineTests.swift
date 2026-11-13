import Foundation
import Testing

@Suite("Visual diff value core: perceptual YIQ RawSpan comparison, heatmap generation, and manifest serialization (#474)")
struct PixelDiffEngineTests {
    static func solidRGBA(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) -> Data {
        var bytes = [UInt8]()
        bytes.reserveCapacity(width * height * 4)
        for _ in 0 ..< (width * height) {
            bytes += [r, g, b, 255]
        }
        return Data(bytes)
    }

    @Test("Identical RGBA8 buffers produce zero changed pixels, zero delta, and an identical verdict")
    func identicalBuffers() {
        let frame = Self.solidRGBA(width: 8, height: 6, r: 32, g: 64, b: 128)
        let dims = PixelFrameDimensions(width: 8, height: 6)
        let (result, mask) = PixelDiffEngine.compare(expected: frame.bytes, actual: frame.bytes, dimensions: dims, tolerance: .crispUI)
        #expect(result.verdict == .identical)
        #expect(result.changedPixels == 0)
        #expect(result.maxPixelDelta == 0)
        #expect(result.bounds == nil)
        #expect(mask.allSatisfy { $0 == 0 })
    }

    @Test("Single-LSB anti-aliasing jitter stays below crispUI pixel threshold while a localized patch reports exact PixelBounds")
    func toleranceAndBounds() {
        let base = Self.solidRGBA(width: 10, height: 10, r: 100, g: 100, b: 100)
        var jittered = [UInt8](base)
        jittered[0] = 101
        let dims = PixelFrameDimensions(width: 10, height: 10)
        let jitterDiff = PixelDiffEngine.compare(expected: base.bytes, actual: Data(jittered).bytes, dimensions: dims, tolerance: .crispUI)
        #expect(jitterDiff.result.verdict == .identical)

        var patched = [UInt8](base)
        for y in 2 ... 4 {
            for x in 3 ... 6 {
                let offset = (y * 10 + x) * 4
                patched[offset] = 250
                patched[offset + 1] = 20
                patched[offset + 2] = 40
            }
        }
        let patchData = Data(patched)
        let (diff, mask) = PixelDiffEngine.compare(expected: base.bytes, actual: patchData.bytes, dimensions: dims, tolerance: .crispUI)
        #expect(diff.verdict == .mismatched)
        #expect(diff.changedPixels == 12)
        #expect(diff.bounds == PixelBounds(minX: 3, minY: 2, maxX: 6, maxY: 4))

        let heatmap = PixelDiffEngine.makeHeatmapRGBA(after: patchData.bytes, mask: mask, width: 10, height: 10)
        let changedOffset = (2 * 10 + 3) * 4
        let unchangedOffset = 0
        #expect(heatmap[changedOffset] > 200 && heatmap[changedOffset + 1] < 80)
        #expect(heatmap[unchangedOffset] == heatmap[unchangedOffset + 1] && heatmap[unchangedOffset] < 60)
    }

    @Test("Dimension mismatches and manifest JSON/Markdown round-trips report exact geometry and slugs")
    func dimensionMismatchAndManifest() throws {
        let small = Self.solidRGBA(width: 4, height: 4, r: 10, g: 10, b: 10)
        let tall = Self.solidRGBA(width: 4, height: 8, r: 10, g: 10, b: 10)
        let dims = PixelFrameDimensions(expectedWidth: 4, expectedHeight: 4, actualWidth: 4, actualHeight: 8)
        let (mismatch, _) = PixelDiffEngine.compare(expected: small.bytes, actual: tall.bytes, dimensions: dims, tolerance: .exact)
        #expect(mismatch.verdict == .dimensionMismatch(expectedWidth: 4, expectedHeight: 4, actualWidth: 4, actualHeight: 8))
        #expect(TraitVariant.darkReducedTransparency.slug == "dark-opaque-hc-nomotion")
        #expect(TraitVariant.darkRussian.slug == "dark-ru_RU")

        let record = SceneComparisonRecord(
            sceneID: .quickAskCompact, variant: .darkStandard,
            beforePath: "before/q.png", afterPath: "after/q.png", comparisonPath: "comparison/q.png", diff: mismatch
        )
        let video = RecordedVideoRecord(
            scenarioID: "quickask-morph", path: "videos/quickask-morph.mp4",
            durationSeconds: 3.2, byteSize: 49_152, width: 1360, height: 880
        )
        let manifest = BeforeAfterManifest(
            commitBase: "abc1234", commitHead: "def5678", createdAt: "2026-11-13T16:00:00Z",
            comparisons: [record], videos: [video]
        )
        let data = try manifest.encodedJSON()
        let decoded = try JSONDecoder().decode(BeforeAfterManifest.self, from: data)
        #expect(decoded == manifest)
        let markdown = manifest.markdownSummary()
        #expect(markdown.contains("`quickask.compact`") && markdown.contains("`quickask-morph`"))
    }
}
