import Foundation
import Testing
import Yams
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltServer

@Suite("Round 9 audit concurrency, security & crash regression tests (#708, #709)", .serialized)
struct Round9AuditRegressionTests {
    @Test("OnDemandCore.leave(epoch:) ignores stale release from superseded loadEpoch (#708)")
    func onDemandCoreStaleEpochReleaseIgnored() {
        var core = OnDemandCore()
        _ = core.enter()
        core.loaded(true)
        let oldEpoch = core.loadEpoch
        _ = core.enter()
        _ = core.beginUnload()
        core.unloaded()
        _ = core.enter()
        core.loaded(true)
        _ = core.enter()
        #expect(core.users == 1)
        #expect(core.leave(epoch: oldEpoch) == nil)
        #expect(core.users == 1)
    }

    @Test("ComposeFile.Walk.seconds returns nil on overflowing duration instead of trapping (#709)")
    func composeDurationOverflowSafe() {
        let node = Node.scalar(Node.Scalar("9999999999999999h"))
        #expect(ComposeFile.Walk.seconds(node) == nil)
    }

    @Test("MCPArguments rejects Float-overflowing Double and JSON booleans in number arrays (#709)")
    func mcpArgumentsFloatOverflowAndBoolGuard() {
        #expect(throws: MCPFailure.self) {
            _ = try MCPArguments.number("x", in: ["x": 1e40])
        }
        #expect(throws: MCPFailure.self) {
            _ = try MCPArguments.numbers("v", in: ["v": [true, false]])
        }
    }

    @Test("SubmissionFormatter handles duplicate sample IDs and invalid JSON objects safely (#709)")
    func submissionFormatterDuplicateIDsAndInvalidJSON() {
        let samples = [
            BenchmarkSampleRecord(id: "dup", adapter: "lcb", model: "m", prompt: "p", answer: "a", score: 1.0),
            BenchmarkSampleRecord(id: "dup", adapter: "lcb", model: "m", prompt: "p", answer: "b", score: 0.0)
        ]
        let artifact = BenchmarkRunArtifact(
            runID: "r1", suiteID: "s", model: "m",
            scorecard: BenchmarkScorecard(metrics: ["accuracy": 0.5]), samples: samples
        )
        let plan = SubmissionFormatter.build(target: .liveCodeBench, artifact: artifact, submit: nil)
        #expect(!plan.files.isEmpty)
        #expect(SubmissionFormatter.jsonData("not-a-json-container").isEmpty)
    }

    @Test("ComposeCommand.Bench.validateRunID rejects path traversal (#709)")
    func composeBenchRejectsTraversalRunID() {
        #expect(throws: (any Error).self) {
            try ComposeCommand.Bench.validateRunID("../escape")
        }
        #expect(throws: (any Error).self) {
            try ComposeCommand.Bench.validateRunID("a/b")
        }
        #expect(throws: Never.self) {
            try ComposeCommand.Bench.validateRunID("bench-smoke-123")
        }
    }

    @Test("MCPRoute rejects deeply nested JSON without stack overflow (#709)")
    func mcpRouteRejectsDeepNesting() {
        let deep = String(repeating: "[", count: 200) + String(repeating: "]", count: 200)
        let route = MCPRoute(line: deep)
        if case .failed = route {
            #expect(Bool(true))
        } else {
            Issue.record("Expected .failed for deeply nested JSON")
        }
    }

    @Test("HuggingFaceHub.listModels throws invalidURL instead of force-unwrapping malformed endpoint (#709)")
    func hfHubMalformedEndpointThrows() async {
        let hub = HuggingFaceHub(endpoint: "http://[::1")
        await #expect(throws: HFError.self) {
            _ = try await hub.listModels()
        }
    }
}
