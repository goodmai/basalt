import Foundation
import MLXLMCommon
import MLXVLM

enum TinyGemma4Unified {
    static func model(_ config: Data) throws -> any LanguageModel {
        Gemma4Unified(try JSONDecoder.json5().decode(Gemma4UnifiedConfiguration.self, from: config))
    }
}
