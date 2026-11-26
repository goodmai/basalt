import Foundation
import MLXLMCommon

struct WhisperConfiguration: Decodable, Sendable {
    let melCount: Int
    let audioContext: Int
    let audioState: Int
    let audioHeads: Int
    let audioLayers: Int
    let vocabularySize: Int
    let textContext: Int
    let textState: Int
    let textHeads: Int
    let textLayers: Int

    enum CodingKeys: String, CodingKey {
        case melCount = "n_mels", audioContext = "n_audio_ctx", audioState = "n_audio_state"
        case audioHeads = "n_audio_head", audioLayers = "n_audio_layer", vocabularySize = "n_vocab"
        case textContext = "n_text_ctx", textState = "n_text_state", textHeads = "n_text_head", textLayers = "n_text_layer"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        melCount = try c.decode(Int.self, forKey: .melCount)
        audioContext = try c.decode(Int.self, forKey: .audioContext)
        audioState = try c.decode(Int.self, forKey: .audioState)
        audioHeads = try c.decode(Int.self, forKey: .audioHeads)
        audioLayers = try c.decode(Int.self, forKey: .audioLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        textContext = try c.decode(Int.self, forKey: .textContext)
        textState = try c.decode(Int.self, forKey: .textState)
        textHeads = try c.decode(Int.self, forKey: .textHeads)
        textLayers = try c.decode(Int.self, forKey: .textLayers)
        guard isValid else {
            throw DecodingError.dataCorruptedError(
                forKey: .audioState, in: c,
                debugDescription: "whisper needs positive sizes, states divisible by their head counts at least 4 audio state channels and a text context of at least 4 tokens")
        }
    }

    var isValid: Bool {
        let sizes = [melCount, audioContext, audioState, audioHeads, audioLayers, vocabularySize, textContext, textState, textHeads, textLayers]
        return sizes.allSatisfy { $0 > 0 } && audioState % audioHeads == 0 && textState % textHeads == 0 && audioState >= 4 && textContext >= 4
    }

    var melFrames: Int { 2 * audioContext }
    var windowSamples: Int { melFrames * WhisperAudio.hopLength }
}

struct WhisperSpecialTokens: Equatable, Sendable {
    static let languageCodes = """
        en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk \
        te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo \
        ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue
        """.split(separator: " ").map(String.init)

    let multilingual: Bool
    let languageCount: Int
    var endOfText: Int
    var startOfTranscript: Int
    var transcribe: Int
    var noTimestamps: Int

    init(vocabularySize: Int) {
        multilingual = vocabularySize >= 51_865
        languageCount = vocabularySize - 51_765 - (multilingual ? 1 : 0)
        endOfText = multilingual ? 50_257 : 50_256
        startOfTranscript = endOfText + 1
        transcribe = startOfTranscript + languageCount + 2
        noTimestamps = startOfTranscript + languageCount + 6
    }

    var languages: Range<Int> { (startOfTranscript + 1) ..< (startOfTranscript + 1 + max(languageCount, 0)) }

    func resolved(by tokenizer: any MLXLMCommon.Tokenizer) -> WhisperSpecialTokens {
        var tokens = self
        tokens.endOfText = Self.id(of: "<|endoftext|>", in: tokenizer) ?? endOfText
        tokens.startOfTranscript = Self.id(of: "<|startoftranscript|>", in: tokenizer) ?? startOfTranscript
        tokens.transcribe = Self.id(of: "<|transcribe|>", in: tokenizer) ?? transcribe
        tokens.noTimestamps = Self.id(of: "<|notimestamps|>", in: tokenizer) ?? noTimestamps
        return tokens
    }

    func language(_ code: String?, tokenizer: any MLXLMCommon.Tokenizer) -> Int? {
        guard multilingual, let code = code?.split(separator: "-").first.map(String.init) else { return nil }
        if let id = Self.id(of: "<|\(code)|>", in: tokenizer) { return id }
        guard let index = Self.languageCodes.firstIndex(of: code), index < languages.count else { return nil }
        return languages.lowerBound + index
    }

    func prompt(language: Int?) -> [Int] {
        guard multilingual, let language else { return [startOfTranscript, noTimestamps] }
        return [startOfTranscript, language, transcribe, noTimestamps]
    }

    static func id(of token: String, in tokenizer: any MLXLMCommon.Tokenizer) -> Int? {
        guard let id = tokenizer.convertTokenToId(token), tokenizer.convertIdToToken(id) == token else { return nil }
        return id
    }
}
