import Foundation
import CoreAI
import Tokenizers
import BasaltCore

public struct CoreAIEmbeddingService: EmbeddingEngine {
    public let models: [EmbeddingModelInfo]
    let bundle: CoreAIEmbeddingBundle
    private let tokenizer: any Tokenizers.Tokenizer
    private let table: InferenceFunction
    private let encoders: [String: InferenceFunction]

    public static func isBundle(_ directory: URL) -> Bool { CoreAIEmbeddingBundle.isBundle(directory) }

    public init(directory: URL) async throws(BasaltError) {
        bundle = try CoreAIEmbeddingBundle(directory: directory)
        let path = directory.path
        func failure(_ error: any Error) -> BasaltError { .modelConfigurationInvalid(path: path, reason: error.localizedDescription) }
        do {
            tokenizer = try await MLXInferenceEngine.upstreamTokenizer(at: directory.appending(component: "tokenizer"))
            let tableModel = try await AIModel.specialize(
                contentsOf: directory.appending(component: "\(bundle.tokenTable).aimodel"), options: SpecializationOptions(preferredComputeUnitKind: .cpu))
            let encoderModel = try await AIModel.specialize(
                contentsOf: directory.appending(component: "\(bundle.encoder).aimodel"), options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            guard let table = try tableModel.loadFunction(named: "main") else { throw BasaltError.modelConfigurationInvalid(path: path, reason: "token_table has no main function") }
            self.table = table
            var encoders: [String: InferenceFunction] = [:]
            for function in bundle.functions {
                guard let loaded = try encoderModel.loadFunction(named: function.name) else {
                    throw BasaltError.modelConfigurationInvalid(path: path, reason: "the encoder has no function \(function.name)")
                }
                encoders[function.name] = loaded
            }
            self.encoders = encoders
        } catch let error as BasaltError {
            throw error
        } catch {
            throw failure(error)
        }
        models = [EmbeddingModelInfo(id: bundle.id, task: .embedding, truncatesDimensions: bundle.matryoshka.count > 1)]
    }

    public func embed(_ inputs: [EmbeddingInput], dimensions: Int?, yieldingTo _: GenerationGate?) async throws(BasaltError) -> Embeddings {
        let tokens = inputs.map { input in
            switch input {
            case .text(let text): bundle.tokens(tokenizer.encode(text: bundle.documentPrompt + text, addSpecialTokens: false))
            case .tokens(let ids): bundle.tokens(ids)
            }
        }
        let total = tokens.reduce(0) { $0 + $1.count }
        guard total <= EmbeddingLimits.maxTokens else { throw .contextLengthExceeded(maxTokens: EmbeddingLimits.maxTokens, requested: total) }
        var vectors: [[Float]] = []
        for row in tokens {
            do {
                try Task.checkCancellation()
                vectors.append(try bundle.truncated(try await vector(row), to: dimensions))
            } catch let error as BasaltError {
                throw error
            } catch is CancellationError {
                throw .inferenceError("embedding was cancelled")
            } catch {
                throw .inferenceError("Core AI embedding failed: \(error.localizedDescription)")
            }
        }
        return Embeddings(vectors: vectors, promptTokens: total)
    }

    public func rerank(query _: String, documents _: [String], yieldingTo _: GenerationGate?) async throws(BasaltError) -> [Double] {
        throw .invalidRequestStructure(details: "the Core AI embedding backend serves /v1/embeddings only; rerank needs --rerank-model on the MLX backend")
    }

    func vector(_ tokens: [Int]) async throws -> [Float] {
        let function = bundle.function(for: tokens.count)
        let (length, real) = (function.length, min(tokens.count, function.length))
        guard let encoder = encoders[function.name] else { throw BasaltError.inferenceError("no encoder function \(function.name)") }
        var ids = NDArray(shape: [1, length], scalarType: .int32)
        var idView = ids.mutableView(as: Int32.self)
        idView.copyElements(from: bundle.padded(tokens, to: length))
        var tableOutputs = try await table.run(inputs: ["input_ids": ids])
        guard let embeds = tableOutputs.remove("inputs_embeds")?.ndArray else { throw BasaltError.inferenceError("token_table returned no inputs_embeds") }
        var bias = NDArray(shape: [1, length, 1, length], scalarType: .float16)
        var biasView = bias.mutableView(as: Float16.self)
        biasView.copyElements(from: bundle.bias(real: real, length: length).map(Float16.init))
        var pool = NDArray(shape: [1, length], scalarType: .float16)
        var poolView = pool.mutableView(as: Float16.self)
        poolView.copyElements(from: bundle.poolWeights(real: real, length: length).map(Float16.init))
        var outputs = try await encoder.run(inputs: ["inputs_embeds": embeds, "bias": bias, "pool_weights": pool])
        guard let embedding = outputs.remove("embedding")?.ndArray, let values = embedding.view(as: Float16.self).contiguousElements else {
            throw BasaltError.inferenceError("the encoder returned no contiguous embedding")
        }
        return (0 ..< values.count).map { Float(values[$0]) }
    }
}
