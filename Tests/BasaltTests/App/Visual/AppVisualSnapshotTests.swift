import Foundation
import Testing

@Suite("Deterministic visual snapshots across all 10 Mac app scenes and appearance/accessibility traits (#474)", .serialized)
@MainActor
struct AppVisualSnapshotTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }

    static var shouldUpdateGoldens: Bool {
        env["BASALT_UPDATE_VISUAL_GOLDENS"] == "1"
    }

    static var exportDirectory: URL? {
        guard let raw = env["BASALT_VISUAL_EXPORT_DIR"], !raw.isEmpty else { return nil }
        return URL(fileURLWithPath: raw)
    }

    @Test("Every canonical scene renders deterministically at 2x sRGB and matches its golden PNG within perceptual YIQ tolerance",
          arguments: VisualSceneCatalog.entries)
    func snapshotMatchesGolden(entry: VisualSceneMatrixEntry) async throws {
        let actual = try await VisualSceneCatalog.render(spec: entry.spec, variant: entry.variant)
        #expect(actual.width == entry.spec.width * entry.spec.scale)
        #expect(actual.height == entry.spec.height * entry.spec.scale)

        if let exportDir = Self.exportDirectory {
            try BaselineStore.write(actual.png, spec: entry.spec, variant: entry.variant, directory: exportDir)
        }

        if Self.shouldUpdateGoldens {
            try BaselineStore.write(actual.png, spec: entry.spec, variant: entry.variant)
            return
        }

        let expected = try BaselineStore.read(spec: entry.spec, variant: entry.variant)
        let dims = PixelFrameDimensions(
            expectedWidth: expected.width, expectedHeight: expected.height,
            actualWidth: actual.width, actualHeight: actual.height
        )
        let (diff, mask) = PixelDiffEngine.compare(
            expected: expected.rgba.bytes,
            actual: actual.rgba.bytes,
            dimensions: dims,
            tolerance: entry.spec.tolerance
        )

        if diff.verdict == .mismatched || diff.verdict != .identical && diff.verdict != .withinTolerance {
            let comparison = try BeforeAfterCompositor.makeComparison(before: expected, after: actual, mask: mask)
            let slug = "\(entry.spec.id.rawValue)__\(entry.variant.slug)"
            Attachment.record(expected.cgImage, named: "\(slug)__before.png", as: .png)
            Attachment.record(actual.cgImage, named: "\(slug)__after.png", as: .png)
            Attachment.record(comparison.cgImage, named: "\(slug)__comparison.png", as: .png)
        }

        #expect(
            diff.verdict == .identical || diff.verdict == .withinTolerance,
            "\(entry): changedRatio=\(diff.changedRatio) maxDelta=\(diff.maxPixelDelta) verdict=\(diff.verdict)"
        )
    }
}
