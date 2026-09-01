import Foundation

// MARK: — Per-family model behaviour
//
// Everything that differs between model families lives behind this protocol, so
// the transport adapters and the engine stay family-agnostic. Adding a family is
// one new conformance plus one registry line — no edits to request handling.

public protocol ModelProfile: Sendable {
    /// Human-readable family name, for logs and benchmark reports.
    var name: String { get }

    /// Extra variables handed to the tokenizer's chat template. This is the ONLY
    /// sanctioned way to influence prompt construction: no adapter may build
    /// prompt syntax by hand.
    func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable]

    /// Markers that delimit a thinking span *in the generated stream*.
    ///
    /// Note these describe generation, not the prompt. Qwen's template opens
    /// `<think>` in the prompt itself and only ever emits the closing tag, which
    /// is why a parser that keys off the raw prompt gets it backwards.
    var thinkingMarkers: ThinkingMarkers { get }

    /// Ceiling used when the caller does not name one.
    var defaultMaxTokens: Int { get }
}

public struct ThinkingMarkers: Sendable {
    /// Tokens that open a thinking span mid-stream.
    public let open: [String]
    /// Tokens that close it.
    public let close: [String]
    /// True when the chat template leaves a thinking span open at the end of the
    /// prompt, so generation starts *inside* it and only a closing marker appears.
    public let openedByTemplate: Bool

    public init(open: [String], close: [String], openedByTemplate: Bool) {
        self.open = open
        self.close = close
        self.openedByTemplate = openedByTemplate
    }

    public static let none = ThinkingMarkers(open: [], close: [], openedByTemplate: false)
}

// MARK: — Qwen 3.x (dense and MoE)

/// Covers Qwen3.5 / 3.6 / 3.8, dense and MoE, including the abliterated and
/// heretic derivatives — they all ship the same chat template.
public struct QwenProfile: ModelProfile {
    public let name = "qwen3.x"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        switch reasoning {
        case .modelDefault:
            // Template defaults reasoning_effort to `xhigh`.
            return [:]
        case .disabled:
            // Makes the template emit `<think>\n\n</think>` — an already-closed
            // span — so the model answers immediately. Verified: "127 + 458"
            // drops from 800 tokens of repetition to 3 tokens ("585").
            return ["enable_thinking": false]
        case .effort(let level):
            // The template raises on anything outside this set, which would surface
            // as an opaque 500. Fail closed to the default instead.
            guard ["xhigh", "medium", "low"].contains(level) else { return [:] }
            return ["reasoning_effort": level]
        }
    }

    public var thinkingMarkers: ThinkingMarkers {
        ThinkingMarkers(open: ["<think>"], close: ["</think>"], openedByTemplate: true)
    }

    public var defaultMaxTokens: Int { 8192 }
}

// MARK: — Gemma 3 / 4

public struct GemmaProfile: ModelProfile {
    public let name = "gemma"

    public init() {}

    /// Gemma templates expose no reasoning knob; the policy is silently inapplicable.
    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] { [:] }

    public var thinkingMarkers: ThinkingMarkers { .none }

    public var defaultMaxTokens: Int { 4096 }
}

// MARK: — Fallback

public struct GenericProfile: ModelProfile {
    public let name = "generic"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] { [:] }
    public var thinkingMarkers: ThinkingMarkers { .none }
    public var defaultMaxTokens: Int { 4096 }
}

// MARK: — Selection

public enum ModelProfileRegistry {
    /// Picks a profile from the `model_type` in the model's own config.json.
    ///
    /// Matching is on prefix because the family covers several ids: `qwen3_5`,
    /// `qwen3_5_moe`, `qwen3_6`, `qwen3_8`… all share one template. Note that
    /// Qwen3.6 and 3.8 checkpoints both declare `model_type: qwen3_5` upstream —
    /// the id names the architecture family, not the release.
    public static func profile(forModelType modelType: String?) -> any ModelProfile {
        guard let t = modelType?.lowercased() else { return GenericProfile() }
        if t.hasPrefix("qwen3") { return QwenProfile() }
        if t.hasPrefix("gemma") { return GemmaProfile() }
        return GenericProfile()
    }

    /// Reads `model_type` out of a model directory, tolerating a missing or
    /// unreadable config rather than failing the load.
    public static func profile(forModelAt path: String) -> any ModelProfile {
        let url = URL(fileURLWithPath: path).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return GenericProfile() }
        return profile(forModelType: json["model_type"] as? String)
    }
}
