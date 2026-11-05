import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

extension ServeCommand {
    func load(_ orchestrator: ModelOrchestratorActor, from path: String) async throws {
        log("Loading model from \(bold(path))…")
        do {
            try await orchestrator.loadModel(path: path)
            await orchestrator.setModelId(targetModel)
            log("Model ready.")
        } catch {
            switch error {
            case .modelConfigurationInvalid, .modelProbeFailed:
                log("\(red("[error]")) \(error.localizedDescription)")
                throw ExitCode.failure
            case .modelNotFound, .weightsCorrupted, .outOfMemory, .inferenceHardwareFailure, .invalidRequestStructure,
                 .contextLengthExceeded, .rateLimitExceeded, .authenticationFailed, .modelInferenceError, .nonFiniteLogits,
                 .inferenceError, .modelNotCached, .modelNotLoaded, .modelSwitching, .serverBusy, .memoryPressure:
                log("\(yellow("[warn]")) \(error.localizedDescription) — running in stub mode.")
            }
        }
    }

    func loadEmbeddings() async throws -> EmbeddingService? {
        guard embeddingModel != nil || rerankModel != nil else { return nil }
        func source(_ id: String?) async throws -> (id: String, directory: URL)? {
            guard let id else { return nil }
            return (id, URL(fileURLWithPath: try await resolvedPath(id, quant: nil)))
        }
        do {
            let service = try await EmbeddingService(embeddingModel: try await source(embeddingModel), rerankModel: try await source(rerankModel))
            log("Serving \(service.models.map { "\($0.id) (\($0.task.rawValue))" }.joined(separator: ", "))")
            return service
        } catch let error as BasaltError {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func resolveModelPath() async throws -> String {
        let path = try await requestedModelPath()
        if requireVerified { try await refuseUnverified(path) }
        return path
    }

    func refuseUnverified(_ path: String) async throws {
        log("Verifying every file of \(bold(path)) (--require-verified)…")
        do throws(HFError) {
            try await HubSnapshot(directory: URL(fileURLWithPath: path)).requireVerified(hub: HuggingFaceHub())
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    private func requestedModelPath() async throws -> String {
        if let explicit = modelPath { return explicit }

        let modelId = model ?? Self.defaultModel
        if model == nil { log("No model specified — using default: \(dim(modelId))") }
        return try await resolvedPath(modelId, quant: quant)
    }

    func resolvedPath(_ modelId: String, quant: String?) async throws -> String {
        guard modelId.contains("/") else { return modelId }

        let snapshot = ModelCache.cacheDir(for: modelId)

        if let resolved = ModelCache.resolve(repoId: modelId, quant: quant) {
            if quant == nil, resolved != snapshot {
                log("Auto-detected quantization: \(dim(resolved.lastPathComponent))")
            }
            return resolved.path
        }

        log("Model not in local cache: \(bold(modelId))\(quant.map { " [\($0)]" } ?? ""). Downloading…")
        let hub = HuggingFaceHub(requireVerified: requireVerified)
        do {
            let dest = try await hub.download(repoId: modelId, subfolder: quant, token: nil,
                                              onFile: DownloadProgressReporter().callAsFunction)
            print("\n")
            return dest.path
        } catch {
            log("\(red("Download failed:")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func log(_ msg: String) {
        fputs("[serve] \(msg)\n", stderr)
    }
}
