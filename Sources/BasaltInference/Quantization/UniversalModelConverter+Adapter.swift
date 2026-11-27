import BasaltCore
import Foundation
import MLX

extension UniversalModelConverter {
    package static func resolveAdapterDirectory(
        _ spec: String,
        cacheRoot: URL = ModelCache.root,
        requireVerified: Bool = false
    ) async throws -> URL {
        let parsed: ServedAdapterSource
        do throws(ServedAdapterSource.Invalid) {
            parsed = try ServedAdapterSource.parseSingle(spec)
        } catch {
            throw BasaltError.invalidRequestStructure(details: "convert: \(error.description)")
        }
        switch parsed {
        case .directory(let url):
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw BasaltError.modelNotFound(identifier: spec)
            }
            return url
        case .repository(let repo):
            if let cached = ServedAdapterSource.cachedDirectory(for: repo, cacheRoot: cacheRoot) {
                return cached
            }
            let hub = HuggingFaceHub(requireVerified: requireVerified)
            let files = try await hub.modelDetails(repoId: repo).siblings?.map(\.rfilename) ?? []
            guard ServedAdapterSource.hasAdapterFiles(files) else {
                throw BasaltError.invalidRequestStructure(
                    details: "convert: adapter repo '\(repo)' has no adapter_config.json beside adapter_model.safetensors or adapters.safetensors")
            }
            return try await hub.download(repoId: repo, subfolder: nil, token: nil) { _, _, _ in }
        }
    }

    static func needsAdapterPreFuse(target: QuantizationTarget) -> Bool {
        switch target {
        case .denseBF16, .denseFP16, .affine, .mixed, .mxfp4, .nvfp4, .nf4, .gaugeRoPE:
            return false
        case .macaqAffine, .macaqMXFP4, .macaqMoEBudget, .flatQuant, .lirQuant, .qtip, .qtipBudget, .gsqRCO:
            return true
        }
    }

    static func fuseAdapterIntoDense(
        sourceURL: URL,
        adapterURL: URL,
        outputURL: URL,
        dtype: DType,
        detected: DetectedSourceCheckpoint,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> WeightConverter.Summary {
        var canonStaging: URL?
        defer {
            if let canonStaging {
                try? FileManager.default.removeItem(at: canonStaging)
            }
        }
        let mergeSource: URL
        if detected.requiresCanonicalDecode {
            let tmp = FileManager.default.temporaryDirectory.appending(path: "basalt-adapter-canon-\(UUID().uuidString)")
            canonStaging = tmp
            _ = try await UniversalSourceReader.exportCanonicalDense(from: sourceURL, to: tmp, dtype: dtype)
            mergeSource = tmp
        } else {
            mergeSource = sourceURL
        }
        onProgress(.init(phase: "fusing", completed: 1, total: 2, message: "Merging LoRA adapter \(adapterURL.lastPathComponent)"))
        let converter = WeightConverter(precision: .dense(dtype), adapter: adapterURL)
        return try converter.run(source: mergeSource, output: outputURL)
    }
}
