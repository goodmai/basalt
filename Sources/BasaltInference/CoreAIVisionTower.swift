import Foundation
import CoreAI
import MLX
import MLXLMCommon
import BasaltCore

public struct CoreAIVisionTower: Sendable {
    static let maxInFlight = 2
    let bundle: CoreAIVisionTowerBundle
    let admission = InFlightLimit(maxInFlight)
    private let functions: [String: InferenceFunction]

    public static func isBundle(_ directory: URL) -> Bool { CoreAIVisionTowerBundle.isBundle(directory) }

    public init(directory: URL) async throws(BasaltError) {
        bundle = try CoreAIVisionTowerBundle(directory: directory)
        let path = directory.path
        do {
            let model = try await AIModel.specialize(
                contentsOf: directory.appending(component: "\(bundle.asset).aimodel"), options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            var functions: [String: InferenceFunction] = [:]
            for function in bundle.functions {
                guard let loaded = try model.loadFunction(named: function.name) else {
                    throw BasaltError.modelConfigurationInvalid(path: path, reason: "the vision tower has no function \(function.name)")
                }
                functions[function.name] = loaded
            }
            self.functions = functions
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelConfigurationInvalid(path: path, reason: error.localizedDescription)
        }
    }

    func fits(_ visual: PaddleOCRVLVision.Model, frames: [THW]) -> Bool {
        visual.headDim == bundle.headDim && visual.projector.merge == bundle.merge
            && visual.projector.linear2.weight.dim(0) == bundle.textHidden && frames.allSatisfy { bundle.function(for: $0) != nil }
    }

    struct Encoded: Sendable {
        let function: String
        let inputs: [String: NDArray]
        let rows: Int
    }

    func inputs(_ image: LMInput.ProcessedImage, visual: PaddleOCRVLVision.Model) throws -> [Encoded]? {
        guard let frames = image.frames, !frames.isEmpty, fits(visual, frames: frames) else { return nil }
        var start = 0
        return try frames.map { frame in
            defer { start += frame.h * frame.w }
            return try encoded(image.pixels[start ..< start + frame.h * frame.w], frame: frame, visual: visual)
        }
    }

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

    func encoded(_ patches: MLXArray, frame: THW, visual: PaddleOCRVLVision.Model) throws -> Encoded {
        guard let function = bundle.function(for: frame) else {
            throw BasaltError.inferenceError("the vision tower has no bucket for a \(frame.h)x\(frame.w) grid")
        }
        let (count, length) = (frame.h * frame.w, function.tokens)
        let order = MLXArray(bundle.order(frame))
        let (cos, sin) = visual.rotary(frame)
        var bias = NDArray(shape: [1, length, 1, 1], scalarType: .float16)
        var view = bias.mutableView(as: Float16.self)
        view.copyElements(from: bundle.bias(real: count, length: length).map(Float16.init))
        let inputs = [
            "patches": try Self.planar(patches.reshaped(count, -1).take(order, axis: 0), length: length),
            "pos": try Self.planar(visual.embeddings.positions(frame).take(order, axis: 0), length: length),
            "cos": try Self.planar(cos.reshaped(count, -1).take(order, axis: 0), length: length, perHead: true),
            "sin": try Self.planar(sin.reshaped(count, -1).take(order, axis: 0), length: length, perHead: true),
            "bias": bias,
        ]
        return Encoded(function: function.name, inputs: inputs, rows: count / (bundle.merge * bundle.merge))
    }

    static func strides(of shape: [Int]) -> [Int] {
        shape.indices.map { shape[($0 + 1)...].reduce(1, *) }
    }

    static func planar(_ rows: MLXArray, length: Int, perHead: Bool = false) throws -> NDArray {
        let (count, channels) = (rows.dim(0), rows.dim(1))
        let padded = count == length ? rows : concatenated([rows, MLXArray.zeros([length - count, channels], dtype: rows.dtype)], axis: 0)
        let planar = contiguous(padded.transposed().asType(.float16))
        eval(planar)
        let shape = perHead ? [1, 1, channels, length] : [1, channels, 1, length]
        var array = NDArray(shape: shape, scalarType: .float16)
        guard Array(array.strides) == strides(of: shape) else {
            throw BasaltError.inferenceError("a Core AI input of \(channels)x\(length) is not row-major")
        }
        let bytes = planar.asData(access: .noCopyIfContiguous).data
        let view = array.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { pointer, _, _ in
            bytes.withUnsafeBytes { source in _ = memcpy(pointer, source.baseAddress!, min(source.count, channels * length * 2)) }
        }
        withExtendedLifetime(planar) {}
        return array
    }

    static func rows(_ planar: NDArray, count: Int) throws -> MLXArray {
        let (channels, length) = (planar.shape[1], planar.shape[3])
        guard Array(planar.strides) == strides(of: planar.shape), count <= length else {
            throw BasaltError.inferenceError("the vision tower output is not a row-major (1, C, 1, N) array")
        }
        let values = planar.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
            MLXArray(UnsafeBufferPointer(start: pointer, count: channels * length), [channels, length])
        }
        return values[0..., ..<count].transposed()
    }
}
