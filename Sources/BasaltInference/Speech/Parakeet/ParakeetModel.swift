import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import BasaltCore

final class ParakeetModel: Module, LLMModel, SpeechRecognitionModel {
    static let modelType = "parakeet_tdt"
    static let windowSeconds = 120

    let configuration: ParakeetConfiguration
    let tokenizer: ParakeetTokenizer
    @ModuleInfo(key: "encoder") var encoder: ParakeetEncoder
    @ModuleInfo(key: "decoder") var decoder: ParakeetDecoder
    @ModuleInfo(key: "joint") var joint: ParakeetJoint

    init(_ configuration: ParakeetConfiguration) {
        self.configuration = configuration
        tokenizer = ParakeetTokenizer(vocabulary: configuration.joint.vocabulary)
        _encoder.wrappedValue = ParakeetEncoder(configuration.encoder)
        _decoder.wrappedValue = ParakeetDecoder(configuration.decoder)
        _joint.wrappedValue = ParakeetJoint(configuration)
        super.init()
        train(false)
    }

    convenience init(configurationData data: Data) throws {
        self.init(try ParakeetConfiguration.decode(data))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        joint.output(joint.predictionProjection(decoder.prediction(inputs, state: nil).0))
    }

    var loraLayers: [Module] { [] }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] { [] }

    func transcribe(audio: [Float], tokenizer _: any MLXLMCommon.Tokenizer, temperature _: Float, language _: String?) throws -> String {
        let preprocessor = configuration.preprocessor
        let filters = WhisperAudio.melFilters(count: preprocessor.features, fftSize: preprocessor.fftSize)
        var texts: [String] = []
        for start in stride(from: 0, to: max(audio.count, 1), by: configuration.windowSamples) {
            try Task.checkCancellation()
            let mel = ParakeetAudio.logMel(audio[start ..< min(start + configuration.windowSamples, audio.count)], preprocessor: preprocessor, filters: filters)
            let features = encoder(mel.expandedDimensions(axis: 0))[0]
            texts.append(tokenizer.decode(tokenIds: try greedyTokens(features), skipSpecialTokens: true))
        }
        return texts.filter { !$0.isEmpty }.joined(separator: " ")
    }
}

struct ParakeetTokenizer: MLXLMCommon.Tokenizer {
    let vocabulary: [String]

    static func isSpecial(_ piece: String) -> Bool { piece.hasPrefix("<") && piece.hasSuffix(">") }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let ids = Dictionary(vocabulary.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        let unknown = ids["<unk>"] ?? 0
        var rest = Substring("▁" + text.replacingOccurrences(of: " ", with: "▁"))
        var tokens: [Int] = []
        while !rest.isEmpty {
            let match = (1 ... rest.count).reversed().lazy.compactMap { length in ids[String(rest.prefix(length))].map { (length, $0) } }.first
            tokens.append(match?.1 ?? unknown)
            rest = rest.dropFirst(match?.0 ?? 1)
        }
        return tokens
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let pieces = tokenIds.compactMap(convertIdToToken).filter { !(skipSpecialTokens && Self.isSpecial($0)) }
        return pieces.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }

    func convertTokenToId(_ token: String) -> Int? { vocabulary.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? { vocabulary.indices.contains(id) ? vocabulary[id] : nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw MLXLMCommon.TokenizerError.missingChatTemplate
    }
}

struct ParakeetInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        throw BasaltError.invalidRequestStructure(details: "Parakeet only transcribes speech; send audio to /v1/audio/transcriptions")
    }
}

enum ParakeetLoader {
    static func loadContainer(from directory: URL) async throws -> ModelContainer? {
        guard let config = ModelCache.config(at: directory), ParakeetConfiguration.describes(config) else { return nil }
        let model = try ParakeetModel(configurationData: Data(contentsOf: directory.appending(component: "config.json")))
        try await loadWeights(modelDirectory: directory, model: model)
        return ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(directory: directory), model: model, processor: ParakeetInputProcessor(), tokenizer: model.tokenizer))
    }
}
