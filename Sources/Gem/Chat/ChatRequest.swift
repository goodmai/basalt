import Foundation

// MARK: — Canonical chat request
//
// Every wire protocol (OpenAI, Anthropic, native) decodes into this one type.
// It is deliberately model-agnostic: nothing here knows about `<think>` markers,
// `<|im_start|>`, or any other template syntax. Turning a ChatRequest into tokens
// is the job of the tokenizer's own chat template, reached via ModelProfile.
//
// The previous design let the OpenAI adapter hand-assemble a prompt string with
// Gemma-3 markers, which (a) produced the wrong prompt for every non-Gemma-3
// model and (b) put the stream parser into thinking mode for the whole
// generation, so `content` came back empty while `completion_tokens` was
// non-zero. Keeping prompt syntax out of the transport layer makes that class
// of bug unrepresentable.

public struct ChatMessageInput: Sendable, Equatable {
    public enum Role: String, Sendable {
        case system, user, assistant, tool
    }

    public let role: Role
    public let text: String

    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }
}

/// How much the model should think before answering.
///
/// Expressed as intent, not as a vendor knob: each ModelProfile maps these cases
/// onto whatever its chat template actually understands. Qwen reads
/// `reasoning_effort` plus `enable_thinking`; Gemma has no knob at all and
/// ignores the policy entirely.
public enum ReasoningPolicy: Sendable, Equatable {
    /// Let the template pick. For Qwen3.x that means `xhigh`, which can spend the
    /// entire token budget reasoning — fine for hard problems, ruinous for
    /// "what is 127 + 458".
    case modelDefault
    /// Emit an already-closed thinking block so the model answers directly.
    case disabled
    /// A named budget the template understands (`xhigh` | `medium` | `low`).
    case effort(String)

    public init?(rawValue: String?) {
        switch rawValue?.lowercased() {
        case nil:                     return nil
        case "none", "off", "false":  self = .disabled
        case "default", "auto":       self = .modelDefault
        case .some(let value):        self = .effort(value)
        }
    }
}

public struct ChatRequest: Sendable {
    public let messages: [ChatMessageInput]
    public let maxTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let reasoning: ReasoningPolicy
    public let stream: Bool
    /// Model the caller asked for, if it named one. Used for hot-swap.
    public let requestedModel: String?

    public init(
        messages: [ChatMessageInput],
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        reasoning: ReasoningPolicy = .modelDefault,
        stream: Bool = false,
        requestedModel: String? = nil
    ) {
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.reasoning = reasoning
        self.stream = stream
        self.requestedModel = requestedModel
    }
}

// MARK: — Canonical result

public struct ChatResult: Sendable {
    public enum Stop: String, Sendable {
        /// Model emitted an end-of-turn token.
        case endTurn
        /// Generation was cut off at the token ceiling — the answer is incomplete.
        case tokenLimit
        /// A degenerate repetition loop was detected and generation was abandoned.
        case repetitionLoop

        public var openAIFinishReason: String {
            switch self {
            case .endTurn:        return "stop"
            case .tokenLimit:     return "length"
            case .repetitionLoop: return "length"
            }
        }
    }

    public let content: String
    public let reasoning: String?
    public let promptTokens: Int
    public let completionTokens: Int
    public let stop: Stop
    public let tokensPerSecond: Double
    public let timeToFirstToken: Double

    public init(
        content: String,
        reasoning: String?,
        promptTokens: Int,
        completionTokens: Int,
        stop: Stop,
        tokensPerSecond: Double,
        timeToFirstToken: Double
    ) {
        self.content = content
        self.reasoning = reasoning
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.stop = stop
        self.tokensPerSecond = tokensPerSecond
        self.timeToFirstToken = timeToFirstToken
    }
}
