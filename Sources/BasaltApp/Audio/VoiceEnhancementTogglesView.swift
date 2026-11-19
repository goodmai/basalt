import BasaltAppCore
import SwiftUI

public struct VoiceEnhancementTogglesView: View {
    let voice: AudioSessionController

    public init(voice: AudioSessionController) {
        self.voice = voice
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(
                "Noise Suppression",
                isOn: Binding(
                    get: { voice.store.state.enhancement.noiseSuppressionEnabled },
                    set: { enabled in
                        var next = voice.store.state.enhancement
                        next.noiseSuppressionEnabled = enabled
                        voice.store.send(.setEnhancement(next))
                    }
                )
            )
            Toggle(
                "Semantic Turn-Taking",
                isOn: Binding(
                    get: { voice.store.state.enhancement.semanticTurnTakingEnabled },
                    set: { enabled in
                        var next = voice.store.state.enhancement
                        next.semanticTurnTakingEnabled = enabled
                        voice.store.send(.setEnhancement(next))
                    }
                )
            )
            Toggle(
                "Duplex Speech-to-Speech",
                isOn: Binding(
                    get: { voice.store.state.enhancement.duplexSpeechToSpeechEnabled },
                    set: { enabled in
                        var next = voice.store.state.enhancement
                        next.duplexSpeechToSpeechEnabled = enabled
                        voice.store.send(.setEnhancement(next))
                    }
                )
            )
        }
    }
}
