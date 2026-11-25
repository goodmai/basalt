import Foundation
import MLX
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("ModelOpt auto-import: a cached ModelOpt snapshot is imported once into <repo>-mlx and every cache lookup then resolves to the pack (#677)", .serialized)
struct ModelOptAutoImportTests {
    static func cachedSnapshot(_ repo: String, root: URL) throws -> URL {
        let model = root.appending(component: "models--" + repo.replacingOccurrences(of: "/", with: "--"))
        try FileManager.default.createDirectory(at: model.appending(component: "refs"), withIntermediateDirectories: true)
        try Data("rev\n".utf8).write(to: model.appending(component: "refs/main"))
        return model.appending(component: "snapshots/rev")
    }

    @Test("Imports once: resolve returns the snapshot before and the nvfp4 pack after, no staging is left, and a second call reuses the pack")
    func importsOnceAndRedirects() async throws {
        let root = ModelOptImporterTests.temporary("auto")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try Self.cachedSnapshot("org/tiny", root: root)
        _ = try ModelOptImporterTests.write(.gemma4Moe, at: source, algorithm: "NVFP4") { _ in false }
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root) == source)

        let pack = try #require(try await ModelOptAutoImport.ensureImported(repoId: "org/tiny", cacheRoot: root) { _ in })
        #expect(pack.standardizedFileURL == root.appending(component: "models--org--tiny-mlx/snapshots/local").standardizedFileURL)
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root)?.standardizedFileURL == pack.standardizedFileURL)
        let config = try #require(ModelCache.config(at: pack))
        #expect((config["quantization"] as? [String: Any])?["mode"] as? String == "nvfp4")
        #expect(!ModelCache.isModelOpt(pack))
        let siblings = try FileManager.default.contentsOfDirectory(atPath: pack.deletingLastPathComponent().path)
        #expect(siblings == ["local"])

        let shard = try #require(try FileManager.default.contentsOfDirectory(at: pack, includingPropertiesForKeys: nil).first { $0.pathExtension == "safetensors" })
        let written = try shard.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        let again = try await ModelOptAutoImport.ensureImported(repoId: "org/tiny", cacheRoot: root) { _ in Issue.record("re-imported an existing pack") }
        #expect(again?.standardizedFileURL == pack.standardizedFileURL)
        #expect(try shard.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate == written)
    }

    @Test("A plain MLX checkpoint is left alone and resolves to its own snapshot")
    func plainCheckpointIsNotImported() async throws {
        let root = ModelOptImporterTests.temporary("plain")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try Self.cachedSnapshot("org/plain", root: root)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data(TinyModel.Family.gemma4Moe.config.utf8).write(to: source.appending(component: "config.json"))
        #expect(try await ModelOptAutoImport.ensureImported(repoId: "org/plain", cacheRoot: root) { _ in } == nil)
        #expect(ModelCache.resolve(repoId: "org/plain", root: root) == source)
        #expect(!FileManager.default.fileExists(atPath: root.appending(component: "models--org--plain-mlx").path))
    }

    @Test("A pack without refs/main, as an interrupted import leaves it, is not served")
    func packWithoutRefsIsIgnored() throws {
        let root = ModelOptImporterTests.temporary("partial")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try Self.cachedSnapshot("org/tiny", root: root)
        _ = try ModelOptImporterTests.write(.gemma4Moe, at: source, algorithm: "NVFP4") { _ in false }
        let partial = root.appending(component: "models--org--tiny-mlx/snapshots/local")
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
        try Data(TinyModel.Family.gemma4Moe.config.utf8).write(to: partial.appending(component: "config.json"))
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root) == source)
    }
}
