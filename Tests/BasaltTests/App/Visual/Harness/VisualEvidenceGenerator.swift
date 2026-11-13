import Foundation

@MainActor
public enum VisualEvidenceGenerator {
    public static func generatePackage(
        baselineDirectory: URL,
        outputDirectory: URL,
        commitBase: String,
        commitHead: String,
        includeVideos: Bool
    ) async throws {
        let fm = FileManager.default
        let beforeDir = outputDirectory.appendingPathComponent("before")
        let afterDir = outputDirectory.appendingPathComponent("after")
        let comparisonDir = outputDirectory.appendingPathComponent("comparison")
        let videosDir = outputDirectory.appendingPathComponent("videos")
        for dir in [beforeDir, afterDir, comparisonDir, videosDir] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        var comparisons: [SceneComparisonRecord] = []
        for entry in VisualSceneCatalog.entries {
            let record = try await compareEntry(
                entry,
                baselineDirectory: baselineDirectory,
                beforeDir: beforeDir,
                afterDir: afterDir,
                comparisonDir: comparisonDir
            )
            comparisons.append(record)
        }

        let videos = includeVideos ? try await renderVideos(into: videosDir) : []
        let createdAt = ISO8601DateFormatter().string(from: Date())
        let manifest = BeforeAfterManifest(
            commitBase: commitBase,
            commitHead: commitHead,
            createdAt: createdAt,
            comparisons: comparisons,
            videos: videos
        )
        try manifest.encodedJSON().write(to: outputDirectory.appendingPathComponent("manifest.json"), options: .atomic)
        let summary = Data(manifest.markdownSummary().utf8)
        try summary.write(to: outputDirectory.appendingPathComponent("report.md"), options: .atomic)
    }

    private static func compareEntry(
        _ entry: VisualSceneMatrixEntry,
        baselineDirectory: URL,
        beforeDir: URL,
        afterDir: URL,
        comparisonDir: URL
    ) async throws -> SceneComparisonRecord {
        let filename = entry.spec.goldenFilename(for: entry.variant)
        let after = try await VisualSceneCatalog.render(spec: entry.spec, variant: entry.variant)
        let before = (try? BaselineStore.read(spec: entry.spec, variant: entry.variant, directory: baselineDirectory)) ?? after
        try BaselineStore.write(before.png, spec: entry.spec, variant: entry.variant, directory: beforeDir)
        try BaselineStore.write(after.png, spec: entry.spec, variant: entry.variant, directory: afterDir)

        let dims = PixelFrameDimensions(
            expectedWidth: before.width, expectedHeight: before.height,
            actualWidth: after.width, actualHeight: after.height
        )
        let beforeRGBA = before.rgba
        let afterRGBA = after.rgba
        let (diff, mask) = PixelDiffEngine.compare(
            expected: beforeRGBA.bytes,
            actual: afterRGBA.bytes,
            dimensions: dims,
            tolerance: entry.spec.tolerance
        )
        let composite = try BeforeAfterCompositor.makeComparison(before: before, after: after, mask: mask)
        try composite.png.write(to: comparisonDir.appendingPathComponent(filename), options: .atomic)
        return SceneComparisonRecord(
            sceneID: entry.spec.id,
            variant: entry.variant,
            beforePath: "before/\(filename)",
            afterPath: "after/\(filename)",
            comparisonPath: "comparison/\(filename)",
            diff: diff
        )
    }

    private static func renderVideos(into videosDir: URL) async throws -> [RecordedVideoRecord] {
        let scenarios: [(String, [RenderedBitmap])] = [
            ("quickask-morph", try VideoScenarioCatalog.renderQuickAskMorphFrames()),
            ("chat-stream-pin-and-jump", try VideoScenarioCatalog.renderChatStreamFrames()),
            ("engine-recovery-banner", try await VideoScenarioCatalog.renderEngineRecoveryFrames()),
        ]
        var records: [RecordedVideoRecord] = []
        for (name, frames) in scenarios {
            let url = videosDir.appendingPathComponent("\(name).mp4")
            let recorded = try await ChoreographedMP4Writer.writeMP4(frames: frames, fps: 60, to: url)
            records.append(RecordedVideoRecord(
                scenarioID: name,
                path: "videos/\(name).mp4",
                durationSeconds: recorded.durationSeconds,
                byteSize: recorded.byteSize,
                width: recorded.width,
                height: recorded.height
            ))
        }
        return records
    }
}
