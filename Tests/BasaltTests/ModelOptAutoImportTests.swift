import Foundation
import MLX
import Synchronization
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("ModelOpt auto-import: a cached ModelOpt snapshot is imported once into <repo>-mlx and every cache lookup then resolves to the pack (#677)", .serialized)
struct ModelOptAutoImportTests {
    static func cachedSnapshot(_ repo: String, root: URL, revision: String = "rev") throws -> URL {
        let model = root.appending(component: "models--" + repo.replacingOccurrences(of: "/", with: "--"))
        try FileManager.default.createDirectory(at: model.appending(component: "refs"), withIntermediateDirectories: true)
        try Data("\(revision)\n".utf8).write(to: model.appending(component: "refs/main"))
        return model.appending(component: "snapshots/\(revision)")
    }

    @Test("Imports once: resolve returns the snapshot before and the nvfp4 pack after, no staging is left, and a second call reuses the pack")
    func importsOnceAndRedirects() async throws {
        let root = ModelOptImporterTests.temporary("auto")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try Self.cachedSnapshot("org/tiny", root: root)
        _ = try ModelOptImporterTests.write(.gemma4Moe, at: source, algorithm: "NVFP4") { _ in false }
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root)?.path == source.path)

        let pack = try #require(try await ModelOptAutoImport.ensureImported(repoId: "org/tiny", cacheRoot: root) { _ in })
        #expect(pack.path == root.appending(component: "models--org--tiny-mlx/snapshots/local-rev").path)
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root)?.path == pack.path)
        let config = try #require(ModelCache.config(at: pack))
        #expect((config["quantization"] as? [String: Any])?["mode"] as? String == "nvfp4")
        #expect(!ModelCache.isModelOpt(pack))
        let siblings = try FileManager.default.contentsOfDirectory(atPath: pack.deletingLastPathComponent().path)
        #expect(siblings == ["local-rev"])

        let shard = try #require(try FileManager.default.contentsOfDirectory(at: pack, includingPropertiesForKeys: nil).first { $0.pathExtension == "safetensors" })
        let written = try shard.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        let again = try await ModelOptAutoImport.ensureImported(repoId: "org/tiny", cacheRoot: root) { _ in Issue.record("re-imported an existing pack") }
        #expect(again?.path == pack.path)
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
        #expect(ModelCache.resolve(repoId: "org/plain", root: root)?.path == source.path)
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
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root)?.path == source.path)
    }

    @Test("A ModelOpt snapshot whose quant_algo the importer does not take is left alone, with the reason reported, instead of re-quantized to RTN nvfp4")
    func unsupportedAlgorithmIsNotImported() async throws {
        let root = ModelOptImporterTests.temporary("awq")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try Self.cachedSnapshot("org/awq", root: root)
        _ = try ModelOptImporterTests.write(.gemma4Moe, at: source, algorithm: "NVFP4") { _ in false }
        let quant: [String: Any] = ["quantization": ["quant_algo": "W4A8_AWQ", "group_size": 128]]
        try JSONSerialization.data(withJSONObject: quant).write(to: source.appending(component: "hf_quant_config.json"))
        let reasons = Mutex<[String]>([])
        #expect(try await ModelOptAutoImport.ensureImported(repoId: "org/awq", cacheRoot: root) { line in reasons.withLock { $0.append(line) } } == nil)
        #expect(reasons.withLock { $0.contains { $0.contains("W4A8_AWQ") } })
        #expect(!FileManager.default.fileExists(atPath: root.appending(component: "models--org--awq-mlx").path))
    }

    @Test("A new source revision re-imports into its own pack and drops the old one; a pack made from another revision is never served")
    func newRevisionReimports() async throws {
        let root = ModelOptImporterTests.temporary("revision")
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try Self.cachedSnapshot("org/tiny", root: root, revision: "rev1")
        _ = try ModelOptImporterTests.write(.gemma4Moe, at: first, algorithm: "NVFP4") { _ in false }
        let old = try #require(try await ModelOptAutoImport.ensureImported(repoId: "org/tiny", cacheRoot: root) { _ in })
        let second = try Self.cachedSnapshot("org/tiny", root: root, revision: "rev2")
        _ = try ModelOptImporterTests.write(.gemma4Moe, at: second, algorithm: "NVFP4") { _ in false }
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root)?.path == second.path)
        let new = try #require(try await ModelOptAutoImport.ensureImported(repoId: "org/tiny", cacheRoot: root) { _ in })
        #expect(new.lastPathComponent == "local-rev2" && ModelCache.importSource(of: new) == "rev2")
        #expect(!FileManager.default.fileExists(atPath: old.path))
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root)?.path == new.path)
    }

    @Test("A downloaded <repo>-mlx repository is never overwritten or served as the import of <repo>")
    func downloadedMLXRepoIsLeftAlone() async throws {
        let root = ModelOptImporterTests.temporary("foreign")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try Self.cachedSnapshot("org/tiny", root: root)
        _ = try ModelOptImporterTests.write(.gemma4Moe, at: source, algorithm: "NVFP4") { _ in false }
        let foreign = try Self.cachedSnapshot("org/tiny-mlx", root: root, revision: "abc123")
        try FileManager.default.createDirectory(at: foreign, withIntermediateDirectories: true)
        try Data(TinyModel.Family.gemma4Moe.config.utf8).write(to: foreign.appending(component: "config.json"))
        #expect(try await ModelOptAutoImport.ensureImported(repoId: "org/tiny", cacheRoot: root) { _ in } == nil)
        #expect(try String(contentsOf: root.appending(component: "models--org--tiny-mlx/refs/main"), encoding: .utf8) == "abc123\n")
        #expect(ModelCache.resolve(repoId: "org/tiny", root: root)?.path == source.path)
    }
}
