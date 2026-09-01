import Testing
@testable import GemCore

// Each test below pins one of the defects that made the previous adapter
// misbehave, so a regression fails here instead of six hours into a benchmark.

@Suite("Reasoning policy")
struct ReasoningPolicyTests {

    @Test("`none` disables thinking rather than being passed through as an effort level")
    func noneDisablesThinking() {
        #expect(ReasoningPolicy(rawValue: "none") == .disabled)
        #expect(ReasoningPolicy(rawValue: "off") == .disabled)
        #expect(ReasoningPolicy(rawValue: nil) == nil)
        #expect(ReasoningPolicy(rawValue: "medium") == .effort("medium"))
    }

    @Test("Qwen maps disabled onto enable_thinking, not reasoning_effort")
    func qwenDisabled() {
        let ctx = QwenProfile().templateContext(for: .disabled)
        #expect(ctx["enable_thinking"] as? Bool == false)
        #expect(ctx["reasoning_effort"] == nil)
    }

    @Test("Qwen forwards known effort levels verbatim")
    func qwenEffort() {
        let ctx = QwenProfile().templateContext(for: .effort("low"))
        #expect(ctx["reasoning_effort"] as? String == "low")
    }

    @Test("An unknown effort level falls back instead of raising in the template")
    func qwenUnknownEffort() {
        // The Jinja template calls raise_exception on anything outside the set,
        // which reaches the caller as an opaque 500.
        #expect(QwenProfile().templateContext(for: .effort("ultra")).isEmpty)
    }

    @Test("Gemma has no reasoning knob and stays empty for every policy")
    func gemmaIgnoresPolicy() {
        #expect(GemmaProfile().templateContext(for: .disabled).isEmpty)
        #expect(GemmaProfile().templateContext(for: .effort("low")).isEmpty)
    }
}

@Suite("Profile selection")
struct ProfileRegistryTests {

    @Test("Qwen ids across releases and topologies all resolve to one profile")
    func qwenVariants() {
        // Qwen3.6 and 3.8 checkpoints both declare qwen3_5 upstream: the id names
        // the architecture family, not the release.
        for id in ["qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_8", "qwen3_8_moe"] {
            #expect(ModelProfileRegistry.profile(forModelType: id).name == "qwen3.x")
        }
    }

    @Test("Gemma and unknown ids resolve correctly")
    func otherVariants() {
        #expect(ModelProfileRegistry.profile(forModelType: "gemma4").name == "gemma")
        #expect(ModelProfileRegistry.profile(forModelType: "llama").name == "generic")
        #expect(ModelProfileRegistry.profile(forModelType: nil).name == "generic")
    }
}

@Suite("Repetition guard")
struct RepetitionGuardTests {

    @Test("A verbatim loop is caught")
    func catchesLoop() {
        var guard_ = RepetitionGuard()
        let cycle = "    # Let's use a small step size for Integration. However, we can use the property that the sum of the series is:\n"
        var stopped = false
        for _ in 0..<40 where !stopped {
            stopped = guard_.shouldStop(after: cycle)
        }
        #expect(stopped, "459 lines of this is what Gemma4-12B actually emitted")
    }

    @Test("Ordinary prose is not flagged")
    func allowsProse() {
        var guard_ = RepetitionGuard()
        var stopped = false
        for i in 0..<80 where !stopped {
            stopped = guard_.shouldStop(after: "Line \(i): computing coefficient number \(i) using Simpson's rule over the interval.\n")
        }
        #expect(!stopped)
    }

    @Test("Short repetitive answers are left alone")
    func allowsShortRepeats() {
        var guard_ = RepetitionGuard()
        // Legitimately repetitive output below the minimum length, e.g. a grid.
        var stopped = false
        for _ in 0..<5 where !stopped {
            stopped = guard_.shouldStop(after: "0 0 0\n")
        }
        #expect(!stopped)
    }
}

@Suite("Stop reason")
struct StopReasonTests {

    @Test("Truncation is reported as length, not as a clean stop")
    func truncationIsVisible() {
        // The old engine hardcoded .stop, so a run cut off at the ceiling looked
        // identical to one that finished — truncated code read as a model failure.
        #expect(ChatResult.Stop.tokenLimit.openAIFinishReason == "length")
        #expect(ChatResult.Stop.repetitionLoop.openAIFinishReason == "length")
        #expect(ChatResult.Stop.endTurn.openAIFinishReason == "stop")
    }
}
