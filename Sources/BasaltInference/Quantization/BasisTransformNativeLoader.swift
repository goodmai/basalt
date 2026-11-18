import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct FallbackCalibrationTokenizer: MLXLMCommon.Tokenizer {
    let vocabSize: Int

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let cap = max(8, vocabSize)
        return text.utf8.map { Int($0) % cap }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(clamping: $0 & 0x7F) }, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { encode(text: $0["content"] as? String ?? "", addSpecialTokens: false) }
    }
}

struct FallbackCalibrationTokenizerLoader: TokenizerLoader {
    let vocabSize: Int

    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        if let real = try? await DefaultTokenizerLoader().load(from: directory) {
            return real
        }
        return FallbackCalibrationTokenizer(vocabSize: vocabSize)
    }
}

package enum BasisTransformNativeLoader {
    package static let flatQuantConfigKey = "flatquant_quantization"
    package static let lirQuantConfigKey = "lirquant_quantization"

    package struct FlatQuantManifest: Codable, Sendable, Equatable {
        package var version: Int = 1
        package var bits: Int
        package var groupSize: Int
        package var alpha: Float
        package var modules: [String: FlatQuantLayerSpec]

        enum CodingKeys: String, CodingKey {
            case version, bits, alpha, modules
            case groupSize = "group_size"
        }

        package init(version: Int = 1, bits: Int, groupSize: Int, alpha: Float, modules: [String: FlatQuantLayerSpec]) {
            self.version = version
            self.bits = bits
            self.groupSize = groupSize
            self.alpha = alpha
            self.modules = modules
        }
    }

    package struct LIRQuantManifest: Codable, Sendable, Equatable {
        package var version: Int = 1
        package var bits: Int
        package var groupSize: Int
        package var subspaceDim: Int
        package var modules: [String: LIRQuantLayerSpec]

        enum CodingKeys: String, CodingKey {
            case version, bits, modules
            case groupSize = "group_size"
            case subspaceDim = "subspace_dim"
        }

        package init(version: Int = 1, bits: Int, groupSize: Int, subspaceDim: Int, modules: [String: LIRQuantLayerSpec]) {
            self.version = version
            self.bits = bits
            self.groupSize = groupSize
            self.subspaceDim = subspaceDim
            self.modules = modules
        }
    }

    private struct WrapperConfig: Decodable {
        let flatquantQuantization: FlatQuantManifest?
        let lirquantQuantization: LIRQuantManifest?

        enum CodingKeys: String, CodingKey {
            case flatquantQuantization = "flatquant_quantization"
            case lirquantQuantization = "lirquant_quantization"
        }
    }

    package static func flatQuantManifest(in directory: URL) throws(BasaltError) -> FlatQuantManifest? {
        let configURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              json[flatQuantConfigKey] != nil else {
            return nil
        }
        guard let manifest = (try? JSONDecoder().decode(WrapperConfig.self, from: data))?.flatquantQuantization,
              manifest.version == 1 else {
            throw .modelConfigurationInvalid(path: configURL.path, reason: "invalid flatquant_quantization manifest or unsupported version")
        }
        return manifest
    }

    package static func lirQuantManifest(in directory: URL) throws(BasaltError) -> LIRQuantManifest? {
        let configURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              json[lirQuantConfigKey] != nil else {
            return nil
        }
        guard let manifest = (try? JSONDecoder().decode(WrapperConfig.self, from: data))?.lirquantQuantization,
              manifest.version == 1 else {
            throw .modelConfigurationInvalid(path: configURL.path, reason: "invalid lirquant_quantization manifest or unsupported version")
        }
        return manifest
    }
}

extension BasisTransformNativeLoader {
    package static func loadContainer(
        from directory: URL, tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContainer? {
        let fqManifest = try flatQuantManifest(in: directory)
        let lirManifest = try lirQuantManifest(in: directory)
        guard fqManifest != nil || lirManifest != nil else { return nil }
        let rawConfig = try Data(contentsOf: directory.appending(component: "config.json"))
        guard let json = try JSONSerialization.jsonObject(with: rawConfig) as? [String: Any],
              let modelType = (json["model_type"] as? String)
                  ?? ((json["text_config"] as? [String: Any])?["model_type"] as? String)
        else {
            throw BasaltError.inferenceError("BasisTransform native: missing model_type in config.json")
        }
        var failure: (any Error)?
        await MLXInferenceEngine.registerAliases()
        let scaleKeys = MACaQNativeLoader.safetensorsScaleKeys(in: directory)
        let perLayer = (try? JSONDecoder.json5().decode(BaseConfiguration.self, from: rawConfig))?.perLayerQuantization
        for family in ExpertStreamingLoader.families where await family.types.contains(modelType) {
            do {
                let model = try await family.types.createModel(configuration: rawConfig, modelType: modelType)
                if let fqManifest { try installFlatQuant(fqManifest.modules, in: model) }
                if let lirManifest { try installLIRQuant(lirManifest.modules, in: model) }
                if let perLayer {
                    try MACaQNativeLoader.installUntouchedQuantized(perLayer: perLayer, scaleKeys: scaleKeys, in: model)
                }
                let registry = ModelTypeRegistry<any LanguageModel>(creators: [modelType: { _ in model }])
                return try await family.factory(registry).loadContainer(from: directory, using: tokenizerLoader)
            } catch {
                failure = error
            }
        }
        throw failure ?? BasaltError.inferenceError("BasisTransform native: no model factory knows model_type \(modelType)")
    }

    package static func installFlatQuant(_ modules: [String: FlatQuantLayerSpec], in model: Module) throws(BasaltError) {
        let named = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        for (path, spec) in modules.sorted(by: { $0.key < $1.key }) {
            guard spec.d1 > 0, spec.d2 > 0,
                  [2, 3, 4, 8].contains(spec.bits), [32, 64, 128].contains(spec.groupSize) else {
                throw .modelConfigurationInvalid(path: path, reason: "invalid FlatQuant spec")
            }
            guard let linear = named[path] as? Linear, !(linear is QuantizedLinear),
                  linear.shape.1 == spec.d1 * spec.d2, linear.shape.1.isMultiple(of: spec.groupSize)
            else {
                throw .modelConfigurationInvalid(path: path, reason: "not a compatible Linear for FlatQuant")
            }
            let replacement = FlatQuantLinear.placeholder(replacing: linear, spec: spec)
            try MACaQNativeLoader.swapChild(path: path, replacement: replacement, named: named, root: model)
        }
        model.invalidateCompiledTraces()
    }

    package static func installLIRQuant(_ modules: [String: LIRQuantLayerSpec], in model: Module) throws(BasaltError) {
        let named = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        for (path, spec) in modules.sorted(by: { $0.key < $1.key }) {
            guard spec.subspaceDim >= 32, spec.numLayers >= 0, spec.pairsPerLayer >= 0,
                  [2, 3, 4, 8].contains(spec.bits), [32, 64, 128].contains(spec.groupSize) else {
                throw .modelConfigurationInvalid(path: path, reason: "invalid LIRQuant spec")
            }
            guard let linear = named[path] as? Linear, !(linear is QuantizedLinear),
                  linear.shape.1.isMultiple(of: 32), linear.shape.1.isMultiple(of: spec.groupSize)
            else {
                throw .modelConfigurationInvalid(path: path, reason: "not a compatible Linear for LIRQuant")
            }
            let replacement = LIRQuantLinear.placeholder(replacing: linear, spec: spec)
            try MACaQNativeLoader.swapChild(path: path, replacement: replacement, named: named, root: model)
        }
        model.invalidateCompiledTraces()
    }
}
