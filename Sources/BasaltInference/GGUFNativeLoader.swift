import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum GGUFNativeLoader {
    struct Entry: Codable, Sendable, Equatable {
        let type: String
        var inputOrder: Bool? = nil
        var dtype: String? = nil

        enum CodingKeys: String, CodingKey {
            case type, dtype
            case inputOrder = "input_order"
        }

        var activationType: DType { DType.allCases.first { "\($0)" == dtype } ?? .bfloat16 }
    }

    struct Configuration: Decodable {
        let ggufQuantization: [String: Entry]?

        enum CodingKeys: String, CodingKey { case ggufQuantization = "gguf_quantization" }
    }

    static func entries(in directory: URL) -> [String: Entry]? {
        guard let data = try? Data(contentsOf: directory.appending(component: "config.json")) else { return nil }
        return (try? JSONDecoder().decode(Configuration.self, from: data))?.ggufQuantization
    }

    static func loadContainer(from directory: URL, tokenizerLoader: any TokenizerLoader) async throws -> ModelContainer? {
        guard let entries = entries(in: directory) else { return nil }
        let configuration = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configuration)
        var failure: (any Error)?
        for family in ExpertStreamingLoader.families where await family.types.contains(base.modelType) {
            do {
                let model = try await family.types.createModel(configuration: configuration, modelType: base.modelType)
                try install(entries, in: model)
                let registry = ModelTypeRegistry<any LanguageModel>(creators: [base.modelType: { _ in model }])
                return try await family.factory(registry).loadContainer(from: directory, using: tokenizerLoader)
            } catch {
                failure = error
            }
        }
        throw failure ?? BasaltError.inferenceError("GGUF native: no model factory knows model_type \(base.modelType)")
    }

    static func install(_ entries: [String: Entry], in model: Module) throws(BasaltError) {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        var updates: [(String, Module)] = []
        for (path, entry) in entries.sorted(by: { $0.key < $1.key }) {
            guard let type = GGMLType(name: entry.type), type.block.elements > 1 else {
                throw .inferenceError("GGUF native: \(path) has type \(entry.type), which has no basalt kernel")
            }
            switch modules[path] {
            case let linear as Linear:
                let (outputs, inputs) = linear.shape
                updates.append((path, GGUFLinear(type: type, outputs: outputs, inputs: inputs, permutedInput: entry.inputOrder ?? false)))
            case let embedding as Embedding:
                updates.append((path, GGUFEmbedding(type: type, count: embedding.weight.dim(0), dimensions: embedding.weight.dim(1), dtype: entry.activationType)))
            default:
                throw .inferenceError("GGUF native: \(path) is not a linear or embedding layer of this model")
            }
        }
        model.update(modules: ModuleChildren.unflattened(updates))
    }
}
