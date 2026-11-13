import Foundation

public struct SceneComparisonRecord: Equatable, Codable, Sendable {
    public var sceneID: VisualSceneID
    public var variant: TraitVariant
    public var beforePath: String
    public var afterPath: String
    public var comparisonPath: String
    public var diff: DiffResult

    public init(
        sceneID: VisualSceneID,
        variant: TraitVariant,
        beforePath: String,
        afterPath: String,
        comparisonPath: String,
        diff: DiffResult
    ) {
        self.sceneID = sceneID
        self.variant = variant
        self.beforePath = beforePath
        self.afterPath = afterPath
        self.comparisonPath = comparisonPath
        self.diff = diff
    }
}

public struct RecordedVideoRecord: Equatable, Codable, Sendable {
    public var scenarioID: String
    public var path: String
    public var durationSeconds: Double
    public var byteSize: Int64
    public var width: Int
    public var height: Int

    public init(
        scenarioID: String,
        path: String,
        durationSeconds: Double,
        byteSize: Int64,
        width: Int,
        height: Int
    ) {
        self.scenarioID = scenarioID
        self.path = path
        self.durationSeconds = durationSeconds
        self.byteSize = byteSize
        self.width = width
        self.height = height
    }
}

public struct BeforeAfterManifest: Equatable, Codable, Sendable {
    public var commitBase: String
    public var commitHead: String
    public var createdAt: String
    public var comparisons: [SceneComparisonRecord]
    public var videos: [RecordedVideoRecord]

    public init(
        commitBase: String,
        commitHead: String,
        createdAt: String,
        comparisons: [SceneComparisonRecord],
        videos: [RecordedVideoRecord] = []
    ) {
        self.commitBase = commitBase
        self.commitHead = commitHead
        self.createdAt = createdAt
        self.comparisons = comparisons
        self.videos = videos
    }

    public func encodedJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    public func markdownSummary() -> String {
        var lines = [
            "# Basalt UI Visual & Video Evidence",
            "",
            "- Base: `\(commitBase)`",
            "- Head: `\(commitHead)`",
            "- Generated: `\(createdAt)`",
            "",
            "## Before / After Visual Comparisons",
            "",
            "| Scene | Variant | Changed % | Max ΔE | Verdict | Comparison |",
            "|---|---|---:|---:|---|---|",
        ]
        for entry in comparisons {
            let pct = String(format: "%.2f%%", entry.diff.changedRatio * 100.0)
            let maxDelta = String(format: "%.4f", entry.diff.maxPixelDelta)
            let label = verdictLabel(entry.diff.verdict)
            lines.append("| `\(entry.sceneID.rawValue)` | `\(entry.variant.slug)` | \(pct) | \(maxDelta) | \(label) | `\(entry.comparisonPath)` |")
        }
        if !videos.isEmpty {
            lines += [
                "",
                "## Recorded Interaction Videos (MP4)",
                "",
                "| Scenario | Resolution | Duration | Size | File |",
                "|---|---|---:|---:|---|",
            ]
            for video in videos {
                let dur = String(format: "%.2f s", video.durationSeconds)
                let kb = String(format: "%.1f KB", Double(video.byteSize) / 1024.0)
                lines.append("| `\(video.scenarioID)` | `\(video.width)x\(video.height)` | \(dur) | \(kb) | `\(video.path)` |")
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private func verdictLabel(_ verdict: DiffVerdict) -> String {
        switch verdict {
        case .identical: "identical"
        case .withinTolerance: "within-tolerance"
        case .mismatched: "changed"
        case .dimensionMismatch(let ew, let eh, let aw, let ah): "resized (\(ew)x\(eh) -> \(aw)x\(ah))"
        }
    }
}
