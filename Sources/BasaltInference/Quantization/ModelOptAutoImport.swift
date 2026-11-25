import Darwin
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
        do {
            _ = try ModelOptImporter(source: source)
        } catch {
            onProgress("not importing \(repoId): \(error.description)")
            return nil
        }
        let outputId = ModelCache.importedRepoId(for: repoId)
        guard !ModelCache.holdsDownloadedSnapshot(outputId, root: cacheRoot) else {
            onProgress("not importing \(repoId): \(outputId) is a downloaded repository, not a basalt import")
            return nil
        }
        let snapshots = try UniversalModelConverter.cacheSnapshotDirectory(for: outputId, cacheRoot: cacheRoot).deletingLastPathComponent()
        try FileManager.default.createDirectory(at: snapshots, withIntermediateDirectories: true)
        let lock = try await exclusiveLock(snapshots.deletingLastPathComponent().appending(component: ".import.lock"))
        defer { _ = Darwin.close(lock) }
        if let pack = ModelCache.importedPack(for: repoId, source: source, root: cacheRoot) { return pack }
        return try await importPack(source: source, outputId: outputId, snapshots: snapshots, cacheRoot: cacheRoot, onProgress: onProgress)
    }

    static func importPack(
        source: URL, outputId: String, snapshots: URL, cacheRoot: URL, onProgress: @escaping @Sendable (String) -> Void
    ) async throws -> URL {
        let name = "local-\(source.lastPathComponent)"
        let final = snapshots.appending(component: name)
        let staging = snapshots.appending(component: ".importing-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: staging) }
        let request = ModelConversionRequest(source: source.path, outputId: outputId, target: .nvfp4, recipe: SINQRecipe.rtn.rawValue, pvTune: false, stream: false)
        _ = try await UniversalModelConverter.convertToDirectory(request: request, sourceURL: source, outputURL: staging, cacheRoot: cacheRoot) { onProgress($0.message) }
        guard ModelCache.importSource(of: staging) == source.lastPathComponent else {
            throw BasaltError.modelConfigurationInvalid(path: source.path, reason: "the ModelOpt import did not take the lossless importer path")
        }
        try? FileManager.default.removeItem(at: final)
        try FileManager.default.moveItem(at: staging, to: final)
        let refs = snapshots.deletingLastPathComponent().appending(component: "refs")
        try FileManager.default.createDirectory(at: refs, withIntermediateDirectories: true)
        try Data("\(name)\n".utf8).write(to: refs.appending(component: "main"), options: .atomic)
        for old in (try? FileManager.default.contentsOfDirectory(atPath: snapshots.path)) ?? [] where old.hasPrefix("local") && old != name {
            try? FileManager.default.removeItem(at: snapshots.appending(component: old))
        }
        return final
    }

    static func exclusiveLock(_ path: URL) async throws -> Int32 {
        let descriptor = Darwin.open(path.path, O_RDONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else { throw BasaltError.modelConfigurationInvalid(path: path.path, reason: String(cString: strerror(errno))) }
        while flock(descriptor, LOCK_EX | LOCK_NB) != 0 {
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                _ = Darwin.close(descriptor)
                throw error
            }
        }
        return descriptor
    }
}
