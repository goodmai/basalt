import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltHarness

@Suite("Harness2 PoC")
struct Harness2Tests {

    @Test("HarnessCacheReusePlan separates shared prefix for PromptCacheTree and chunks for CacheBlend")
    func cacheReusePlanComputesSavings() {
        let base = HarnessTree(files: [
            "CLAUDE.md": String(repeating: "System invariant instructions. ", count: 80),
            "prompts/guardrails.md": "Base guardrails."
        ])
        let candidate = base.applying(delta: [
            "prompts/guardrails.md": "Updated guardrails with stricter checks.",
            "skills/checker/SKILL.md": "---\nname: checker\ndescription: Check output\n---\nVerify all numbers."
        ])

        let plan = HarnessCacheReusePlan.build(
            taskSlug: "issue-90",
            taskPrompt: "Implement recursive agent harness.",
            baseTree: base,
            candidateTree: candidate,
            spec: .claude,
            kBudget: 2
        )

        #expect(plan.promptCacheKey == "harness:issue-90")
        #expect(plan.invariantPrefixText.contains("System invariant instructions."))
        #expect(plan.invariantPrefixText.contains("Implement recursive agent harness."))
        #expect(plan.blendableChunks.count == 2)
        #expect(plan.composedPrompt.contains("\n\n"))
        #expect(plan.prefillSavingsRatio > 0.5)
    }

    @Test("HarnessDiff marks multi-component edits as confounded and single-component edits as helped or hurt")
    func componentAttributionDetectsConfoundedEdits() {
        let base = HarnessTree(files: [
            "CLAUDE.md": "Base prompt",
            "prompts/guardrails.md": "Base guardrails"
        ])
        let singleEdit = base.applying(delta: [
            "prompts/guardrails.md": "Improved guardrails"
        ])
        let isolatedAttrs = HarnessDiff.attribute(
            candidateId: "cand_inv_0",
            base: base,
            candidate: singleEdit,
            improvedOverBase: true,
            regressedFromBase: false
        )
        #expect(isolatedAttrs.count == 1)
        #expect(isolatedAttrs[0].component == .guardrails)
        #expect(isolatedAttrs[0].verdict == .helped)

        let multiEdit = base.applying(delta: [
            "prompts/guardrails.md": "Improved guardrails",
            "scripts/check.py": "print('ok')"
        ])
        let confoundedAttrs = HarnessDiff.attribute(
            candidateId: "cand_comp_0",
            base: base,
            candidate: multiEdit,
            improvedOverBase: true,
            regressedFromBase: false
        )
        #expect(confoundedAttrs.count == 2)
        #expect(confoundedAttrs.allSatisfy { $0.verdict == .confounded })
    }

    @Test("HarnessLedger enforces R=4k and P=3k and HarnessRecursionRunner executes full pipeline")
    func recursionRunnerAndLedger() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let library = HarnessExperienceLibrary(rootDirectory: tempDir)
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        try await orchestrator.loadModel(path: "mock-model")

        let runner = HarnessRecursionRunner(
            orchestrator: orchestrator,
            library: library,
            spec: .claude
        )

        let baseTree = HarnessTree(files: [
            "CLAUDE.md": "Follow instructions and use skill domain-helper.",
            "prompts/guardrails.md": "Check files."
        ])

        let result = try await runner.optimize(
            taskSlug: "audit-repo",
            taskPrompt: "Audit the repository structure.",
            domain: "swift",
            baseTree: baseTree,
            invariantDeltas: [["prompts/guardrails.md": "Run self-check before returning."]],
            domainDeltas: [["skills/domain-helper/SKILL.md": "---\nname: domain-helper\ndescription: Swift helper\n---\nCheck Package.swift"]],
            k: 1
        )

        #expect(result.ledger.maxRollouts == 4)
        #expect(result.ledger.maxProposerCalls == 3)
        #expect(result.ledger.remainingRollouts == 0)
        #expect(result.ledger.remainingProposerCalls == 0)
        #expect(result.winningCandidateId == "compose_0")
        #expect(result.cachePlans.count == 4)
        #expect(result.cachePlans.allSatisfy { $0.promptCacheKey == "harness:audit-repo" })

        let stored = await library.query(topicOrCue: "audit-repo", domain: "swift")
        #expect(stored.count == 1)
        #expect(!stored[0].attributions.isEmpty)
    }

    @Test("MCPServer gemma_add_knowledge persists to ExperienceLibrary and harness_library_query retrieves it")
    func mcpKnowledgePersistenceAndValidation() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let library = HarnessExperienceLibrary(rootDirectory: tempDir)
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())

        actor OutputCapturer {
            var lines: [String] = []
            func capture(_ line: String) { lines.append(line) }
        }
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orchestrator, library: library) { line in
            Task { await capturer.capture(line) }
        }

        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "gemma_add_knowledge", "arguments": {"topic": "kv-cache-sharing", "content": "Pin invariant prefix in PromptCacheTree and chunk skills for CacheBlend.", "domain": "mlx", "confidence": "high"}}}
        """)

        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "harness_library_query", "arguments": {"query": "CacheBlend", "domain": "mlx"}}}
        """)

        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "harness_validate", "arguments": {"files": {"CLAUDE.md": "Use skill s1", "skills/s1/SKILL.md": "---\\nname: s1\\ndescription: test\\n---\\nBody"}}}}
        """)

        try? await Task.sleep(for: .milliseconds(50))
        let outputs = await capturer.lines
        #expect(outputs.count == 3)
        #expect(outputs[0].contains("Successfully added knowledge about 'kv-cache-sharing'"))
        #expect(outputs[1].contains("PromptCacheTree"))
        #expect(outputs[2].contains("\\\"isValid\\\":true"))

        let indexFile = tempDir.appendingPathComponent("mlx/INDEX.md")
        #expect(FileManager.default.fileExists(atPath: indexFile.path))
    }
}
