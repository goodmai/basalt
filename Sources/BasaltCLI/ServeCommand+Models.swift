import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

extension ServeCommand {
    func load(_ orchestrator: some ModelOrchestrating, from path: String) async throws {
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
        try await loadAdapters(into: orchestrator)
    }

    func loadAdapters(into orchestrator: some ModelOrchestrating) async throws {
        guard !adapters.isEmpty else { return }
        do {
            try await orchestrator.loadAdapters(try await adapterDirectories())
            let names = await orchestrator.adapterNames
            log("Adapters: \(names.joined(separator: ", "))")
        } catch let error as BasaltError {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    static func adapterSources(_ specs: [String]) throws -> [String: ServedAdapterSource] {
        do throws(ServedAdapterSource.Invalid) {
            return try ServedAdapterSource.parse(specs)
        } catch {
            throw ValidationError(error.description)
        }
    }

    func adapterDirectories() async throws -> [String: URL] {
        var directories: [String: URL] = [:]
        for (name, source) in try Self.adapterSources(adapters) {
            switch source {
            case .directory(let url): directories[name] = url
            case .repository(let repo): directories[name] = try await adapterDirectory(repo, name: name)
            }
        }
        return directories
    }

    func adapterDirectory(_ repo: String, name: String) async throws -> URL {
        let cached = ModelCache.cacheDir(for: repo)
        if ServedAdapterSource.isAdapterDirectory(cached) { return cached }
        log("Adapter \(bold(name)): \(repo) is not in the local cache. Downloading…")
        let hub = HuggingFaceHub(requireVerified: requireVerified)
        do {
            let files = try await hub.modelDetails(repoId: repo).siblings?.map(\.rfilename) ?? []
            guard ServedAdapterSource.hasAdapterFiles(files) else {
                log("\(red("Adapter refused:")) \(repo) has no adapter_config.json beside adapter_model.safetensors or adapters.safetensors at its top level; nothing was downloaded")
                throw ExitCode.failure
            }
            let directory = try await hub.download(
                repoId: repo, subfolder: nil, token: nil, onFile: DownloadProgressReporter(useStderr: true).callAsFunction)
            fputs("\n", stderr)
            return directory
        } catch let exit as ExitCode {
            throw exit
        } catch {
            log("\(red("Adapter download failed:")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func loadEmbeddings() async throws -> (any EmbeddingEngine)? {
        guard embeddingModel != nil || rerankModel != nil else { return nil }
        func source(_ id: String?) async throws -> (id: String, directory: URL)? {
            guard let id else { return nil }
            return (id, URL(fileURLWithPath: try await resolvedPath(id, quant: nil)))
        }
        let adapterURL = embeddingAdapter.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        do {
            if let embedding = try await source(embeddingModel), CoreAIEmbeddingService.isBundle(embedding.directory) {
                guard rerankModel == nil, adapterURL == nil else {
                    throw BasaltError.invalidRequestStructure(details: "a Core AI embedding bundle serves /v1/embeddings alone; --rerank-model and --embedding-adapter need an MLX embedding model")
                }
                log("Specializing the Core AI embedding bundle for the Neural Engine (first start only; cached afterwards)…")
                let service = try await CoreAIEmbeddingService(directory: embedding.directory)
                log("Serving \(service.models.map(\.id).joined(separator: ", ")) (embedding, Core AI on the Neural Engine)")
                return service
            }
            let service = try await EmbeddingService(
                embeddingModel: try await source(embeddingModel),
                rerankModel: try await source(rerankModel),
                embeddingAdapter: adapterURL
            )
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
        if quant == nil, let pack = try await importedModelOpt(modelId) { return pack.path }

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
                                              onFile: DownloadProgressReporter(useStderr: true).callAsFunction)
            fputs("\n", stderr)
            fflush(stderr)
            if quant == nil, let pack = try await importedModelOpt(modelId) { return pack.path }
            return dest.path
        } catch let exit as ExitCode {
            throw exit
        } catch {
            log("\(red("Download failed:")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func importedModelOpt(_ modelId: String) async throws -> URL? {
        do {
            return try await ModelOptAutoImport.ensureImported(repoId: modelId) { fputs("[serve] ModelOpt import: \($0)\n", stderr) }
        } catch {
            log("\(red("[error]")) ModelOpt import of \(modelId) failed: \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func log(_ msg: String) {
        fputs("[serve] \(msg)\n", stderr)
    }
}
