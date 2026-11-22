import Foundation
import CoreAI
import MLX
import MLXLMCommon
import BasaltCore

public struct CoreAIVisionTower: Sendable {
    static let maxInFlight = 2
    let bundle: CoreAIVisionTowerBundle
    public let cacheKey: String
    public let loadedFromCache: Bool
    let admission = InFlightLimit(maxInFlight)
    private let functions: [String: InferenceFunction]

    public static func isBundle(_ directory: URL) -> Bool { CoreAIVisionTowerBundle.isBundle(directory) }

    public init(directory: URL, cacheDirectory: URL = CoreAIVisionTowerCache.defaultDirectory) async throws(BasaltError) {
        let parsedBundle = try CoreAIVisionTowerBundle(directory: directory)
        let osBuild = CoreAIVisionTowerCache.currentOSBuild()
        let digest = try CoreAIVisionTowerCache.bundleDigest(bundleDirectory: directory, asset: parsedBundle.asset)
        let key = CoreAIVisionTowerCache.cacheKey(bundleSHA256: digest, osBuild: osBuild)
        bundle = parsedBundle
        cacheKey = key
        let path = directory.path
        let modelURL = directory.appending(component: "\(parsedBundle.asset).aimodel")
        let options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        if let cached = Self.resolveCached(
            key: key, modelURL: modelURL, options: options, bundle: parsedBundle, path: path, cacheDirectory: cacheDirectory
        ) {
            functions = cached.functions
            loadedFromCache = true
            Self.persistEntry(
                key: key, digest: digest, osBuild: osBuild, bundle: parsedBundle,
                bookmark: cached.bookmark, in: cacheDirectory
            )
            return
        }
        do {
            let model = try await AIModel.specialize(
                contentsOf: modelURL, options: options, cache: .default, cachePolicy: .persistent)
            functions = try Self.loadFunctions(from: model, bundle: parsedBundle, path: path)
            loadedFromCache = false
            Self.persistEntry(
                key: key, digest: digest, osBuild: osBuild, bundle: parsedBundle,
                bookmark: model.bookmarkData, in: cacheDirectory
            )
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelConfigurationInvalid(path: path, reason: error.localizedDescription)
        }
    }

    private static func loadFunctions(
        from model: AIModel, bundle: CoreAIVisionTowerBundle, path: String
    ) throws -> [String: InferenceFunction] {
        var loadedFunctions: [String: InferenceFunction] = [:]
        for function in bundle.functions {
            guard let loaded = try model.loadFunction(named: function.name) else {
                throw BasaltError.modelConfigurationInvalid(path: path, reason: "the vision tower has no function \(function.name)")
            }
            loadedFunctions[function.name] = loaded
        }
        return loadedFunctions
    }

    private static func resolveCached(
        key: String, modelURL: URL, options: SpecializationOptions, bundle: CoreAIVisionTowerBundle,
        path: String, cacheDirectory: URL
    ) -> (functions: [String: InferenceFunction], bookmark: Data)? {
        if let entry = CoreAIVisionTowerCache.loadEntry(forKey: key, in: cacheDirectory),
           let encoded = entry.bookmarkBase64, let data = Data(base64Encoded: encoded),
           let model = try? AIModel(resolvingBookmark: data),
           let loaded = try? loadFunctions(from: model, bundle: bundle, path: path) {
            return (loaded, model.bookmarkData)
        }
        if let model = try? AIModelCache.default.model(for: modelURL, options: options),
           let loaded = try? loadFunctions(from: model, bundle: bundle, path: path) {
            return (loaded, model.bookmarkData)
        }
        return nil
    }

    private static func persistEntry(
        key: String, digest: String, osBuild: String, bundle: CoreAIVisionTowerBundle, bookmark: Data, in cacheDirectory: URL
    ) {
        let entry = CoreAIVisionTowerCache.Entry(
            cacheKey: key, bundleSHA256: digest, osBuild: osBuild, asset: bundle.asset,
            functions: bundle.functions.map(\.name), bookmarkBase64: bookmark.base64EncodedString()
        )
        try? CoreAIVisionTowerCache.saveEntry(entry, in: cacheDirectory)
    }

    func fits(_ visual: PaddleOCRVLVision.Model, frames: [THW]) -> Bool { bundle.fits(visual, frames: frames) }

    struct Encoded: Sendable {
        let function: String
        let inputs: [String: NDArray]
        let rows: Int
    }

    func inputs(_ image: LMInput.ProcessedImage, visual: PaddleOCRVLVision.Model) throws -> [Encoded]? { try bundle.inputs(image, visual: visual) }

    func run(_ encoded: [Encoded]) async throws -> [NDArray] {
        var results: [NDArray] = []
        for item in encoded {
            guard let inference = functions[item.function] else { throw BasaltError.inferenceError("the vision tower has no function \(item.function)") }
            var outputs = try await inference.run(inputs: item.inputs)
            guard let features = outputs.remove("features")?.ndArray else { throw BasaltError.inferenceError("the vision tower returned no features") }
            results.append(features)
        }
        return results
    }

    static func features(_ outputs: [NDArray], encoded: [Encoded], dtype: DType) throws -> MLXArray {
        concatenated(try zip(outputs, encoded).map { try rows($0, count: $1.rows) }, axis: 0).asType(dtype)
    }

    func features(_ image: LMInput.ProcessedImage, visual: PaddleOCRVLVision.Model) async throws -> MLXArray? {
        guard let encoded = try inputs(image, visual: visual) else { return nil }
        return try Self.features(try await run(encoded), encoded: encoded, dtype: visual.embeddings.patchEmbedding.weight.dtype)
    }

    static func planar(_ rows: MLXArray, length: Int, perHead: Bool = false) throws -> NDArray {
        let (count, channels) = (rows.dim(0), rows.dim(1))
        let total = channels * length
        let padded = count == length ? rows : concatenated([rows, MLXArray.zeros([length - count, channels], dtype: rows.dtype)], axis: 0)
        let shape = perHead ? [1, 1, channels, length] : [1, channels, 1, length]
        return try UnifiedMemoryBridge.withContiguousSpan(of: padded.transposed(), as: Float16.self) { source in
            var array = NDArray(shape: shape, scalarType: .float16)
            do {
                guard var target = array.mutableView(as: Float16.self).contiguousElements,
                      target.count == total, source.count == total
                else {
                    throw BasaltError.inferenceError("a Core AI input of \(channels)x\(length) is not row-major")
                }
                UnifiedMemoryBridge.copyScalars(from: source, into: &target)
            }
            return array
        }
    }

    static func rows(_ planar: NDArray, count: Int) throws -> MLXArray {
        guard planar.shape.count == 4, planar.shape[0] == 1, planar.shape[2] == 1 else {
            throw BasaltError.inferenceError("the vision tower output is not a row-major (1, C, 1, N) array")
        }
        let (channels, length) = (planar.shape[1], planar.shape[3])
        let total = channels * length
        guard count <= length, planar.view(as: Float16.self).contiguousElements?.count == total else {
            throw BasaltError.inferenceError("the vision tower output is not a row-major (1, C, 1, N) array")
        }
        guard total > 0, count > 0 else { return MLXArray.zeros([count, channels], dtype: .float16) }
        let values = try UnifiedMemoryBridge.makeMLXArray(shape: [channels, length], as: Float16.self) { dst throws(BasaltError) in
            guard let elements = planar.view(as: Float16.self).contiguousElements, elements.count == total else {
                throw BasaltError.inferenceError("the vision tower output is not a row-major (1, C, 1, N) array")
            }
            UnifiedMemoryBridge.copyScalars(from: elements, into: &dst)
        }
        return values[0..., ..<count].transposed()
    }
}

extension CoreAIVisionTowerBundle {
    func fits(_ visual: PaddleOCRVLVision.Model, frames: [THW]) -> Bool {
        visual.headDim == headDim && visual.projector.merge == merge
            && visual.projector.linear2.weight.dim(0) == textHidden && frames.allSatisfy { function(for: $0) != nil }
    }

    func inputs(_ image: LMInput.ProcessedImage, visual: PaddleOCRVLVision.Model) throws -> [CoreAIVisionTower.Encoded]? {
        guard let frames = image.frames, !frames.isEmpty, fits(visual, frames: frames) else { return nil }
        var start = 0
        return try frames.map { frame in
            defer { start += frame.h * frame.w }
            return try encoded(image.pixels[start ..< start + frame.h * frame.w], frame: frame, visual: visual)
        }
    }

    func mask(real count: Int, length: Int) -> NDArray {
        var array = NDArray(shape: [1, length, 1, 1], scalarType: .float16)
        let closed = Float16(masked)
        var filled = false
        do {
            if var span = array.mutableView(as: Float16.self).contiguousElements, span.count == length {
                for index in 0 ..< length {
                    span[index] = index < count ? 0 : closed
                }
                filled = true
            }
        }
        if !filled {
            var view = array.mutableView(as: Float16.self)
            view.copyElements(from: (0 ..< length).lazy.map { $0 < count ? Float16(0) : closed })
        }
        return array
    }

    func encoded(_ patches: MLXArray, frame: THW, visual: PaddleOCRVLVision.Model) throws -> CoreAIVisionTower.Encoded {
        guard let bucket = function(for: frame) else {
            throw BasaltError.inferenceError("the vision tower has no bucket for a \(frame.h)x\(frame.w) grid")
        }
        let (count, length) = (frame.h * frame.w, bucket.tokens)
        let gather = MLXArray(order(frame))
        let (cos, sin) = visual.rotary(frame)
        let inputs = [
            "patches": try CoreAIVisionTower.planar(patches.reshaped(count, -1).take(gather, axis: 0), length: length),
            "pos": try CoreAIVisionTower.planar(visual.embeddings.positions(frame).take(gather, axis: 0), length: length),
            "cos": try CoreAIVisionTower.planar(cos.reshaped(count, -1).take(gather, axis: 0), length: length, perHead: true),
            "sin": try CoreAIVisionTower.planar(sin.reshaped(count, -1).take(gather, axis: 0), length: length, perHead: true),
            "bias": mask(real: count, length: length),
        ]
        return CoreAIVisionTower.Encoded(function: bucket.name, inputs: inputs, rows: count / (merge * merge))
    }
}
