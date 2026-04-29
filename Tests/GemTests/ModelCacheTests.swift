import Testing
import Foundation
@testable import GemCore

@Suite("ModelCache path generation")
struct ModelCacheTests {

    @Test("cacheDir converts slash to double-dash")
    func cacheDirPath() {
        let url = ModelCache.cacheDir(for: "mlx-community/gemma-4-e2b-it-4bit")
        let name = url.pathComponents
        // Should contain "models--mlx-community--gemma-4-e2b-it-4bit"
        #expect(name.contains("models--mlx-community--gemma-4-e2b-it-4bit"))
    }

    @Test("cacheDir is under ~/.cache/huggingface/hub")
    func cacheDirUnderRoot() {
        let root = ModelCache.root.path
        let dir  = ModelCache.cacheDir(for: "mlx-community/test-model").path
        #expect(dir.hasPrefix(root))
    }

    @Test("cacheDir ends with snapshots/main")
    func cacheDirEndsWithSnapshotsMain() {
        let url = ModelCache.cacheDir(for: "org/model")
        #expect(url.lastPathComponent == "main")
        #expect(url.deletingLastPathComponent().lastPathComponent == "snapshots")
    }

    @Test("isDownloaded returns false for non-existent model")
    func notDownloaded() {
        // Use a nonsense ID that definitely isn't cached
        let fake = "nonexistent-org/definitely-not-cached-xyzzy-\(Int.random(in: 0..<999999))"
        #expect(ModelCache.isDownloaded(repoId: fake) == false)
    }

    @Test("HFModelInfo derives correct quantization from name")
    func quantizationParsing() {
        let m4bit  = HFModelInfo(id: "mlx-community/gemma-4-e2b-it-4bit",  downloads: nil, likes: nil, tags: nil)
        let m8bit  = HFModelInfo(id: "mlx-community/gemma-4-31b-8bit",     downloads: nil, likes: nil, tags: nil)
        let mbf16  = HFModelInfo(id: "mlx-community/gemma-4-31b-bf16",     downloads: nil, likes: nil, tags: nil)

        #expect(m4bit.quantization == "4bit")
        #expect(m8bit.quantization == "8bit")
        #expect(mbf16.quantization == "bf16")
    }

    @Test("HFModelInfo formats large downloads as k")
    func downloadFormatting() {
        let m = HFModelInfo(id: "mlx-community/x", downloads: 227_825, likes: nil, tags: nil)
        #expect(m.formattedDownloads.hasSuffix("k"))
    }

    @Test("HFModelInfo formats small downloads as raw number")
    func smallDownloadFormatting() {
        let m = HFModelInfo(id: "mlx-community/x", downloads: 42, likes: nil, tags: nil)
        #expect(m.formattedDownloads == "42")
    }
}
