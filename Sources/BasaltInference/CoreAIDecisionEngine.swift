import Foundation
import CoreAI
import Tokenizers
import BasaltCore

public struct CoreAIDecisionEngine: DecisionEngine {
    public let modelID: String
    public let runsBesideDecode = true
    let bundle: CoreAIDecisionBundle
    let tokenizer: any Tokenizers.Tokenizer
    private let table: InferenceFunction
    private let encoders: [String: InferenceFunction]

    public static func isBundle(_ directory: URL) -> Bool { CoreAIDecisionBundle.isBundle(directory) }

    public init(directory: URL) async throws(BasaltError) {
        bundle = try CoreAIDecisionBundle(directory: directory)
        modelID = bundle.modelID
        let path = directory.path
        do {
            tokenizer = try await MLXInferenceEngine.upstreamTokenizer(at: directory.appending(component: "tokenizer"))
            let tableModel = try await AIModel.specialize(
                contentsOf: directory.appending(component: "token_table.aimodel"), options: SpecializationOptions(preferredComputeUnitKind: .cpu))
            let encoderModel = try await AIModel.specialize(
                contentsOf: directory.appending(component: "decision.aimodel"), options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            guard let table = try tableModel.loadFunction(named: "main") else { throw BasaltError.modelConfigurationInvalid(path: path, reason: "token_table has no main function") }
            self.table = table
            var encoders: [String: InferenceFunction] = [:]
            for function in bundle.functions {
                guard let loaded = try encoderModel.loadFunction(named: function.name) else {
                    throw BasaltError.modelConfigurationInvalid(path: path, reason: "decision.aimodel has no function \(function.name)")
                }
                encoders[function.name] = loaded
            }
            self.encoders = encoders
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelConfigurationInvalid(path: path, reason: error.localizedDescription)
        }
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        let names = request.questions.keys.sorted()
        let questions = names.map { request.questions[$0]! }
        let tokenize = { (text: String) in tokenizer.encode(text: text, addSpecialTokens: false) }
        let state = tokenize(CoreAIDecisionBundle.clean(request.state))
        var encoded: [CoreAIDecisionBundle.Encoded] = []
        for question in questions {
            encoded.append(try bundle.encode(question, state: state, tokenize: tokenize))
        }
        let calls = bundle.plan(encoded.map(\.ids.count))
        var scores: [[Float]] = []
        do {
            for call in calls {
                try Task.checkCancellation()
                scores.append(try await run(call.function, rows: call.rows.map { encoded[$0].ids }, kinds: call.rows.map { questions[$0].type }))
            }
        } catch let error as BasaltError {
            throw error
        } catch is CancellationError {
            throw .inferenceError("systemone was cancelled")
        } catch {
            throw .inferenceError("systemone: Core AI decision failed: \(error.localizedDescription)")
        }
        return bundle.readout(request, encoded: encoded, calls: calls, scores: scores)
    }

    func run(_ function: CoreAIDecisionBundle.Function, rows: [[Int]], kinds: [SystemOneKind]) async throws -> [Float] {
        guard let encoder = encoders[function.name] else { throw BasaltError.inferenceError("no decision function \(function.name)") }
        var ids = NDArray(shape: [1, function.batch * function.length], scalarType: .int32)
        var idView = ids.mutableView(as: Int32.self)
        idView.copyElements(from: bundle.padded(rows, to: function))
        var tableOutputs = try await table.run(inputs: ["input_ids": ids])
        guard let embeds = tableOutputs.remove("inputs_embeds")?.ndArray else { throw BasaltError.inferenceError("token_table returned no inputs_embeds") }
        var bias = NDArray(shape: [function.batch, function.length, 1, 1], scalarType: .float16)
        var biasView = bias.mutableView(as: Float16.self)
        biasView.copyElements(from: bundle.bias(rows.map(\.count), to: function).map(Float16.init))
        var types = NDArray(shape: [function.batch, 3], scalarType: .float16)
        var typeView = types.mutableView(as: Float16.self)
        typeView.copyElements(from: bundle.types(kinds, to: function).map(Float16.init))
        var outputs = try await encoder.run(inputs: ["inputs_embeds": embeds, "bias": bias, "qtype": types])
        guard let scores = outputs.remove("scores")?.ndArray, let values = scores.view(as: Float16.self).contiguousElements else {
            throw BasaltError.inferenceError("the decision encoder returned no contiguous scores")
        }
        return (0 ..< values.count).map { Float(values[$0]) }
    }
}
