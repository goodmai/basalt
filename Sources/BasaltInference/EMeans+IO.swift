import BasaltCore
import Foundation
import MLX

public extension EMeans {
    static func fitFile(
        inputURL: URL,
        outputURL: URL,
        tensorKey: String? = nil,
        config: EMeansConfig
    ) throws -> EMeansFileReport {
        let points = try loadVectors(from: inputURL, tensorKey: tensorKey)
        let started = ContinuousClock.now
        let result = try fit(points, config: config)
        let elapsed = (ContinuousClock.now - started) / .seconds(1)
        let parent = outputURL.deletingLastPathComponent()
        if !parent.path.isEmpty {
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        }
        try MLX.save(arrays: ["centroids": result.centroids, "counts": result.counts], url: outputURL)
        return EMeansFileReport(
            points: points.dim(0),
            dimensions: points.dim(1),
            clusters: config.clusters,
            metrics: result.metrics,
            elapsedSeconds: elapsed,
            outputPath: outputURL.path
        )
    }

    static func loadVectors(from url: URL, tensorKey: String? = nil) throws -> MLXArray {
        switch url.pathExtension.lowercased() {
        case "safetensors":
            return try loadSafetensorsVectors(from: url, tensorKey: tensorKey)
        case "jsonl":
            let text = try String(contentsOf: url, encoding: .utf8)
            let parsed = try EMeansCore.parseJSONLVectors(text)
            return MLXArray(parsed.values, [parsed.rows, parsed.cols])
        default:
            throw EMeansError.invalidInput(
                "Unsupported input extension '.\(url.pathExtension)'; expected .safetensors or .jsonl")
        }
    }

    private static func loadSafetensorsVectors(from url: URL, tensorKey: String?) throws -> MLXArray {
        let tensors = try MLX.loadArrays(url: url)
        if let tensorKey {
            guard let tensor = tensors[tensorKey] else {
                throw EMeansError.invalidInput("Tensor key '\(tensorKey)' not found in \(url.lastPathComponent)")
            }
            guard isValidVectorTensor(tensor) else {
                throw EMeansError.invalidInput("Tensor '\(tensorKey)' must be a 2-D floating-point tensor")
            }
            return tensor
        }
        for key in tensors.keys.sorted() {
            if let tensor = tensors[key], isValidVectorTensor(tensor) {
                return tensor
            }
        }
        throw EMeansError.invalidInput("No 2-D floating-point tensor found in \(url.lastPathComponent)")
    }

    private static func isValidVectorTensor(_ tensor: MLXArray) -> Bool {
        tensor.ndim == 2 && tensor.dtype.isFloatingPoint && !tensor.dtype.isComplex
    }
}
