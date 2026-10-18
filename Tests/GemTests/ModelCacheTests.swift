import Testing
import Foundation
@testable import GemCore

@Suite("ModelCache path generation")
struct ModelCacheTests {

    @Test("cacheDir converts slash to double-dash")
    func cacheDirPath() {
        let url = ModelCache.cacheDir(for: "mlx-community/gemma-4-e2b-it-4bit")
        let name = url.pathComponents
        // Should contain "models--mlx-community--gemma-4-e2b-it-4bit"
        #expect(name.contains("models--mlx-community--gemma-4-e2b-it-4bit"))
    }

    @Test("cacheDir is under ~/.cache/huggingface/hub")
    func cacheDirUnderRoot() {
        let root = ModelCache.root.path
        let dir  = ModelCache.cacheDir(for: "mlx-community/test-model").path
        #expect(dir.hasPrefix(root))
    }

    @Test("cacheDir ends with snapshots/main")
    func cacheDirEndsWithSnapshotsMain() {
        let url = ModelCache.cacheDir(for: "org/model")
        #expect(url.lastPathComponent == "main")
        #expect(url.deletingLastPathComponent().lastPathComponent == "snapshots")
    }

    @Test("isDownloaded returns false for non-existent model")
    func notDownloaded() {
        // Use a nonsense ID that definitely isn't cached
        let fake = "nonexistent-org/definitely-not-cached-xyzzy-\(Int.random(in: 0..<999999))"
        #expect(ModelCache.isDownloaded(repoId: fake) == false)
    }

    @Test("HFModelInfo derives correct quantization from name")
    func quantizationParsing() {
        let m4bit  = HFModelInfo(id: "mlx-community/gemma-4-e2b-it-4bit",  downloads: nil, likes: nil, tags: nil)
        let m8bit  = HFModelInfo(id: "mlx-community/gemma-4-31b-8bit",     downloads: nil, likes: nil, tags: nil)
        let mbf16  = HFModelInfo(id: "mlx-community/gemma-4-31b-bf16",     downloads: nil, likes: nil, tags: nil)

        #expect(m4bit.quantization == "4bit")
        #expect(m8bit.quantization == "8bit")
        #expect(mbf16.quantization == "bf16")
    }

    @Test("HFModelInfo formats large downloads as k")
    func downloadFormatting() {
        let m = HFModelInfo(id: "mlx-community/x", downloads: 227_825, likes: nil, tags: nil)
        #expect(m.formattedDownloads.hasSuffix("k"))
    }

    @Test("HFModelInfo formats small downloads as raw number")
    func smallDownloadFormatting() {
        let m = HFModelInfo(id: "mlx-community/x", downloads: 42, likes: nil, tags: nil)
        #expect(m.formattedDownloads == "42")
    }
}

@Suite("Multi-Slot Conversation Cache & Parking")
struct MultiSlotConversationCacheTests {

    @Test("Parking budget is exactly 1/8 of max KV buffer")
    func parkingBudgetOneEighth() async {
        let cache = MultiSlotConversationCache(maxKVBufferTokens: 8192)
        #expect(await cache.parkingBudgetTokens == 1024)

        let cacheLarge = MultiSlotConversationCache(maxKVBufferTokens: 32768)
        #expect(await cacheLarge.parkingBudgetTokens == 4096)
    }

    @Test("Parking inactive conversation holds at most 1/8 buffer and preserves tool prefix")
    func parkSlotPreservesToolPrefixWithinBudget() async {
        let cache = MultiSlotConversationCache(maxKVBufferTokens: 1024)
        // 1024 / 8 = 128 tokens budget (~512 chars)
        let toolPrefix = "System: You have tool `run_command(cmd)` and `view_file(path)` available."
        let longHistory = String(repeating: "Intermediate chat turn between user and agent. ", count: 50)
        let fullPrompt = "\(toolPrefix)\n\n\(longHistory)"

        await cache.recordTurn(slotId: "subagent-1", fullPrompt: fullPrompt, toolPrefix: toolPrefix)
        
        let beforeSlot = await cache.slot(for: "subagent-1")
        #expect(beforeSlot?.isParked == false)
        #expect((beforeSlot?.prompt.count ?? 0) > 1000)

        // Park slot
        await cache.parkSlot(id: "subagent-1")
        let parkedSlot = await cache.slot(for: "subagent-1")
        let budget = await cache.parkingBudgetTokens
        #expect(parkedSlot?.isParked == true)
        #expect((parkedSlot?.estimatedTokens ?? 0) <= budget)
        #expect(parkedSlot?.prompt.contains("System: You have tool `run_command") == true)
    }

    @Test("Intermediate tool call does not reset prefix cache (prefill only on delta)")
    func intermediateToolCallIncrementalDelta() async {
        let cache = MultiSlotConversationCache(maxKVBufferTokens: 4096)
        let slotId = "session-coding-1"
        let toolPrefix = "[Tools]\nname: run_command\ndescription: Execute shell commands"
        let turn1 = "\(toolPrefix)\n\n[User]: Run test\n[Assistant]: {\"tool\": \"run_command\", \"args\": {\"cmd\": \"swift test\"}}"
        
        await cache.recordTurn(slotId: slotId, fullPrompt: turn1, toolPrefix: toolPrefix)

        // Intermediate tool result arrives:
        let toolResult = "\n[Tool Result]: 290 tests passed\n[Assistant]: All tests passed successfully."
        let turn2 = "\(turn1)\(toolResult)"

        let delta = await cache.computePrefillDelta(slotId: slotId, newPrompt: turn2)
        #expect(delta.isIncremental == true)
        #expect(delta.cachedPrefixLength == turn1.count)
        #expect(delta.deltaPrompt == toolResult)
        #expect(delta.cachedTokensEstimate > 0)
    }

    @Test("Switching between parallel subagents preserves independent slots")
    func parallelSubagentsSwitching() async {
        let cache = MultiSlotConversationCache(maxKVBufferTokens: 2048)
        let agentA = "subagent-researcher"
        let agentB = "subagent-coder"

        await cache.recordTurn(slotId: agentA, fullPrompt: "Researcher prompt with tool definitions", toolPrefix: "Researcher tools")
        await cache.recordTurn(slotId: agentB, fullPrompt: "Coder prompt with compiler tools", toolPrefix: "Coder tools")

        // Park all except coder
        await cache.parkInactiveSlots(except: agentB)
        
        let slotA = await cache.slot(for: agentA)
        let slotB = await cache.slot(for: agentB)

        #expect(slotA?.isParked == true)
        #expect(slotB?.isParked == false)
        #expect(slotA?.toolPrefix == "Researcher tools")
        #expect(slotB?.toolPrefix == "Coder tools")
    }
}

@Suite("Tool Call JSON Repair")
struct ToolCallRepairerTests {

    @Test("Repairs JSON wrapped in markdown codeblocks")
    func repairsMarkdownFences() {
        let input = """
        ```json
        {
            "name": "run_command",
            "arguments": {
                "CommandLine": "swift test"
            }
        }
        ```
        """
        let repaired = ToolCallRepairer.repairJSON(input)
        #expect(repaired != nil)
        let call = ToolCallRepairer.extractAndRepairToolCall(from: input)
        #expect(call?.name == "run_command")
        #expect(call?.arguments["CommandLine"] as? String == "swift test")
    }

    @Test("Repairs single-quoted dictionaries and Python literals")
    func repairsSingleQuotesAndLiterals() {
        let input = "{'name': 'view_file', 'arguments': {'AbsolutePath': '/path/file.swift', 'verbose': True, 'unused': None}}"
        let repaired = ToolCallRepairer.repairJSON(input)
        #expect(repaired != nil)
        let call = ToolCallRepairer.extractAndRepairToolCall(from: input)
        #expect(call?.name == "view_file")
        #expect(call?.arguments["AbsolutePath"] as? String == "/path/file.swift")
        #expect(call?.arguments["verbose"] as? Bool == true)
    }

    @Test("Repairs unquoted keys and trailing commas")
    func repairsUnquotedKeysAndTrailingCommas() {
        let input = "{tool: \"run_command\", args: {command: \"ls -la\", count: 5, }, }"
        let repaired = ToolCallRepairer.repairJSON(input)
        #expect(repaired != nil)
        let call = ToolCallRepairer.extractAndRepairToolCall(from: input)
        #expect(call?.name == "run_command")
        #expect(call?.arguments["command"] as? String == "ls -la")
        #expect(call?.arguments["count"] as? Int == 5)
    }

    @Test("Repairs unclosed strings and missing closing braces")
    func repairsUnclosedBracesAndStrings() {
        // Quantized model cut off at context boundary
        let broken = "{\"name\": \"run_command\", \"arguments\": {\"CommandLine\": \"swift build"
        let repaired = ToolCallRepairer.repairJSON(broken)
        #expect(repaired != nil)
        #expect(repaired?.contains("\"CommandLine\"") == true)
        #expect(repaired?.hasSuffix("}}") == true)
    }

    @Test("Extracts tool calls when preceded by chatter")
    func extractsFromChatter() {
        let response = """
        I will run the build now.
        {
            "tool": "run_command",
            "parameters": {
                "cmd": "make test"
            }
        }
        """
        let call = ToolCallRepairer.extractAndRepairToolCall(from: response)
        #expect(call?.name == "run_command")
        #expect(call?.arguments["cmd"] as? String == "make test")
    }
}

