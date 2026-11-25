import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

extension ServeCommand {
    func importedModelOpt(_ modelId: String) async throws -> URL? {
        let source = ModelCache.cacheDir(for: modelId)
        if requireVerified, ModelCache.hasUsableConfig(at: source), ModelCache.isModelOpt(source) { try await refuseUnverified(source.path) }
        do {
            return try await ModelOptAutoImport.ensureImported(repoId: modelId) { fputs("[serve] ModelOpt import: \($0)\n", stderr) }
        } catch {
            log("\(red("[error]")) ModelOpt import of \(modelId) failed: \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func verificationTarget(_ path: String) -> String {
        guard modelPath == nil else { return path }
        let source = ModelCache.cacheDir(for: model ?? Self.defaultModel)
        return ModelCache.importSource(of: URL(fileURLWithPath: path)) == source.lastPathComponent ? source.path : path
    }
}
