import ArgumentParser

struct ThinkingOptions: ParsableArguments {
    @Flag(
        name: .customLong("experimental-adaptive-thinking-budget"),
        help: "Give a request that sets neither thinking_budget nor reasoning_effort a thinking budget from a difficulty estimate of its last user message (keywords, length, tools): easy 512 tokens, medium 2048, hard uncapped. Unmeasured heuristic (#41); off by default"
    )
    var experimentalAdaptiveThinkingBudget = false
}
