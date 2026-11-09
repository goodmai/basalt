import Foundation
import ArgumentParser
import Testing
@testable import BasaltCore
@testable import BasaltCLI

@Suite("serve --decision-model cloud:<model> with --decision-escalate and serve.decisions.* (#407)")
struct DecisionOptionsTests {
    @Test("A cloud decision model needs a local escalation model; escalation needs a cloud model; the threshold lies in 0...1")
    func validation() throws {
        let cascade = try ServeCommand.parse(["--decision-model", "cloud:~typesafe/jev-latest", "--decision-escalate", "self"])
        #expect(cascade.decision.cloudModel == "~typesafe/jev-latest")
        #expect(cascade.decision.localModel == "self")
        #expect(cascade.decision.escalateBelow == 0.6)
        let local = try ServeCommand.parse(["--decision-model", "self"])
        #expect(local.decision.cloudModel == nil && local.decision.localModel == "self")
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--decision-model", "cloud:~typesafe/jev-latest"]) }
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--decision-model", "cloud:", "--decision-escalate", "self"]) }
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--decision-model", "cloud:a", "--decision-escalate", "cloud:b"]) }
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--decision-model", "self", "--decision-escalate", "self"]) }
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--decision-model", "cloud:a", "--decision-escalate", "self", "--decision-escalate-below", "1.2"]) }
    }

    @Test("serve.decisions keys give the same options as the flags")
    func configMatchesFlags() throws {
        let read: (ServeCommand) -> String = { "\($0.decision.model ?? "") \($0.decision.escalate ?? "") \($0.decision.escalateBelow)" }
        let yaml = "defaults:\n  serve:\n    decisions: { model: \"cloud:~typesafe/jev-latest\", escalate: self, escalate_below: 0.75 }\n"
        let flags = try ServeCommand.parse(["--decision-model", "cloud:~typesafe/jev-latest", "--decision-escalate", "self", "--decision-escalate-below", "0.75"])
        #expect(read(try ServeConfigFileTests.serve(yaml)) == read(flags))
        #expect(read(flags) == "cloud:~typesafe/jev-latest self 0.75")
    }

    @Test("Without the key a cloud decision model refuses to start; with it, serve builds a cascade whose local stage is gated")
    func servedCascade() throws {
        let command = try ServeCommand.parse(["--decision-model", "cloud:~typesafe/jev-latest", "--decision-escalate", "self"])
        let gate = GenerationGate()
        let local = StubLocalEngine()
        let served = try command.servedDecisions(local, gate: gate, besideDecode: true, apiKey: "sk-or-test")
        #expect((served as? CascadeDecisionEngine)?.local is GatedDecisionEngine)
        #expect(throws: (any Error).self) { try command.servedDecisions(local, gate: gate, besideDecode: true, apiKey: "") }
        #expect(try ServeCommand.parse(["--decision-model", "self"]).servedDecisions(local, gate: gate, besideDecode: true, apiKey: "") is GatedDecisionEngine)
    }
}

struct StubLocalEngine: DecisionEngine {
    let modelID = "self"

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        SystemOneResponse(model: modelID, answers: [:], usage: SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0), calibrated: false)
    }
}
