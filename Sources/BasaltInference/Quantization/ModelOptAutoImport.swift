import Foundation
import BasaltCore

package enum ModelOptAutoImport {
    package static func ensureImported(
        repoId: String, cacheRoot: URL = ModelCache.root, onProgress: @escaping @Sendable (String) -> Void
    ) async throws -> URL? {
        guard !repoId.hasPrefix("/"), !repoId.hasPrefix("."), repoId.contains("/") else { return nil }
        let source = ModelCache.cacheDir(for: repoId, root: cacheRoot)
        guard ModelCache.hasUsableConfig(at: source), ModelCache.isModelOpt(source) else { return nil }
        if let pack = ModelCache.importedPack(for: repoId, source: source, root: cacheRoot) { return pack }
        let outputId = ModelCache.importedRepoId(for: repoId)
        let snapshots = try UniversalModelConverter.cacheSnapshotDirectory(for: outputId, cacheRoot: cacheRoot).deletingLastPathComponent()
        let final = snapshots.appending(component: "local")
        let staging = snapshots.appending(component: ".importing-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: snapshots, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }
        let request = ModelConversionRequest(source: source.path, outputId: outputId, target: .nvfp4, recipe: SINQRecipe.rtn.rawValue, pvTune: false, stream: false)
        _ = try await UniversalModelConverter.convertToDirectory(request: request, sourceURL: source, outputURL: staging, cacheRoot: cacheRoot) { onProgress($0.message) }
        try? FileManager.default.removeItem(at: final)
        try FileManager.default.moveItem(at: staging, to: final)
        let refs = snapshots.deletingLastPathComponent().appending(component: "refs")
        try FileManager.default.createDirectory(at: refs, withIntermediateDirectories: true)
        try Data("local\n".utf8).write(to: refs.appending(component: "main"))
        return final
    }
}
