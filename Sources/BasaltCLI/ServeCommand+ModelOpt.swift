import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

extension ServeCommand {
    func importedModelOpt(_ modelId: String) async throws -> URL? {
        do {
            return try await ModelOptAutoImport.ensureImported(repoId: modelId) { fputs("[serve] ModelOpt import: \($0)\n", stderr) }
        } catch {
            log("\(red("[error]")) ModelOpt import of \(modelId) failed: \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }
}
