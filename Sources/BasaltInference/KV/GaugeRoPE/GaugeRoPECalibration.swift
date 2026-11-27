import BasaltCore
import Foundation
import MLX
import MLXLMCommon

enum GaugeRoPECalibration {
    static let storedCalibrationTokens = 8192
    static let chunk = 512

    static func codecs(model: any LanguageModel, tokens: [Int], layout: GaugeRoPELayout) throws -> [Int: GaugeRoPECodec] {
        guard tokens.count >= layout.calibrationTokens else {
            throw BasaltError.invalidRequestStructure(
                details: "gaugerope: the calibration text is \(tokens.count) tokens and the codec is calibrated on the first \(layout.calibrationTokens); give a longer calibration text (#647)")
        }
        let count = min(tokens.count, max(layout.calibrationTokens, storedCalibrationTokens))
        let offline = GaugeRoPELayout(
            rotaryDims: layout.rotaryDims, ropeBase: layout.ropeBase, calibrationTokens: count, carrierRank: layout.carrierRank, codeRank: layout.codeRank)
        let caches = GaugeRoPEKVCache.replacingFullAttention(try model.newCache(parameters: nil), layout: offline)
        for start in stride(from: 0, to: count, by: chunk) {
            _ = model(MLXArray(Array(tokens[start ..< min(count, start + chunk)]))[.newAxis], cache: caches)
            eval(caches)
        }
        return Dictionary(uniqueKeysWithValues: caches.enumerated().compactMap { index, cache in
            (cache as? GaugeRoPEKVCache)?.codec.map { (index, $0) }
        })
    }

    static func codecs(_ container: ModelContainer, layout: GaugeRoPELayout, calibrationText: String?) async throws -> [Int: GaugeRoPECodec] {
        let corpus = (calibrationText?.isEmpty == false ? calibrationText : nil) ?? MultilingualMoECalibrationCorpus.defaultText
        return try await container.perform { context in
            try Self.codecs(model: context.model, tokens: context.tokenizer.encode(text: corpus, addSpecialTokens: false), layout: layout)
        }
    }
}
