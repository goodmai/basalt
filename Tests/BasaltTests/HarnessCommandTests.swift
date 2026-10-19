import Testing
@testable import BasaltCLI
@testable import BasaltHarness

@Suite("harness optimize CLI")
struct HarnessCommandTests {
    @Test("--proposer-max-tokens is optional and limited to the allowed range")
    func proposerMaxTokens() throws {
        let base = ["--model", "m", "--task", "t"]
        #expect(try HarnessOptimizeCommand.parse(base).proposerMaxTokens == nil)
        #expect(try HarnessOptimizeCommand.parse(base + ["--proposer-max-tokens", "4096"]).proposerMaxTokens == 4096)
        for bad in ["10", "1000000"] {
            #expect(throws: (any Error).self) { try HarnessOptimizeCommand.parse(base + ["--proposer-max-tokens", bad]) }
        }
    }
}
