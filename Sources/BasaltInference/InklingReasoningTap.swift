import Foundation
import MLXLMCommon

struct InklingReasoningStateMachine: ReasoningTokenParser {
    struct Controls: Equatable, Sendable {
        let messageModel: Int
        let thinking: Int
        let endMessage: Int
        let endSampling: Int
        let contentKinds: Set<Int>
    }

    private enum Phase {
        case betweenFrames
        case header(tokens: Int)
        case reasoning
        case nonReasoning
    }

    private let tokenizer: any MLXLMCommon.Tokenizer
    private let controls: Controls
    private var detokenizer: NaiveStreamingDetokenizer
    private var phase: Phase = .betweenFrames
    private let maxHeaderTokens = 64

    init?(tokenizer: any MLXLMCommon.Tokenizer) {
        let id = { (token: String) in tokenizer.convertTokenToId(token) }
        guard let messageModel = id("<|message_model|>"), let thinking = id("<|content_thinking|>"),
              let endMessage = id("<|end_message|>"), let endSampling = id("<|content_model_end_sampling|>")
        else { return nil }
        let kinds = ["<|content_text|>", "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>", "<|content_xml|>"].compactMap(id)
        self.tokenizer = tokenizer
        self.controls = Controls(messageModel: messageModel, thinking: thinking, endMessage: endMessage, endSampling: endSampling, contentKinds: Set(kinds))
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    }

    mutating func push(_ token: Int) -> OnyxReasoningStateMachine.Step {
        if token == controls.messageModel {
            phase = .header(tokens: 0)
            return .none
        }
        if token == controls.endMessage || token == controls.endSampling {
            phase = .betweenFrames
            return .none
        }
        switch phase {
        case .header(let count):
            if token == controls.thinking {
                detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                phase = .reasoning
                return .none
            }
            if controls.contentKinds.contains(token) {
                phase = .nonReasoning
                return .enteredNonReasoning
            }
            phase = count + 1 >= maxHeaderTokens ? .betweenFrames : .header(tokens: count + 1)
            return .none
        case .reasoning:
            detokenizer.append(token: token)
            guard let text = detokenizer.next(), !text.isEmpty else { return .none }
            return .reasoningChunk(text)
        case .betweenFrames, .nonReasoning:
            return .none
        }
    }
}
