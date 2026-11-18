import Foundation

public struct VoiceEnhancementOptions: Sendable, Codable, Equatable {
    public var noiseSuppressionEnabled: Bool
    public var semanticTurnTakingEnabled: Bool
    public var duplexSpeechToSpeechEnabled: Bool

    public init(
        noiseSuppressionEnabled: Bool = true,
        semanticTurnTakingEnabled: Bool = true,
        duplexSpeechToSpeechEnabled: Bool = false
    ) {
        self.noiseSuppressionEnabled = noiseSuppressionEnabled
        self.semanticTurnTakingEnabled = semanticTurnTakingEnabled
        self.duplexSpeechToSpeechEnabled = duplexSpeechToSpeechEnabled
    }
}

extension AppSettings {
    public static let noiseSuppressionKey = "voice.noiseSuppression"
    public static let semanticTurnTakingKey = "voice.semanticTurnTaking"
    public static let duplexSpeechToSpeechKey = "voice.duplexSpeechToSpeech"

    public static func voiceEnhancementOptions(in defaults: UserDefaults = .standard) -> VoiceEnhancementOptions {
        let noise = (defaults.object(forKey: noiseSuppressionKey) as? Bool) ?? true
        let turnTaking = (defaults.object(forKey: semanticTurnTakingKey) as? Bool) ?? true
        let duplex = (defaults.object(forKey: duplexSpeechToSpeechKey) as? Bool) ?? false
        return VoiceEnhancementOptions(
            noiseSuppressionEnabled: noise,
            semanticTurnTakingEnabled: turnTaking,
            duplexSpeechToSpeechEnabled: duplex
        )
    }

    public static func setVoiceEnhancementOptions(_ options: VoiceEnhancementOptions, in defaults: UserDefaults = .standard) {
        defaults.set(options.noiseSuppressionEnabled, forKey: noiseSuppressionKey)
        defaults.set(options.semanticTurnTakingEnabled, forKey: semanticTurnTakingKey)
        defaults.set(options.duplexSpeechToSpeechEnabled, forKey: duplexSpeechToSpeechKey)
    }
}
