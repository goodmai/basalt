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

    @Test("Slot ledger: an intermediate tool call needs only the delta (engine-side reuse is #56)")
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

    @Test("A rejected, malformed tool call is rebuilt with the declared name and repaired arguments")
    func repairRejectedToolCall() throws {
        let call = try #require(ToolCallRepairer.repairToolCall(
            name: "shell", callID: "c7",
            rawText: "{'name': 'shell', 'arguments': {'command': ['ls', '-la'], 'timeout_ms': 1000,}}",
            truncated: false))
        #expect(call.id == "c7")
        #expect(call.name == "shell")
        #expect(call.arguments == #"{"command":["ls","-la"],"timeout_ms":1000}"#)

        // Arguments carried as a JSON string are decoded too; an object without them is the arguments.
        let nested = try #require(ToolCallRepairer.repairToolCall(
            name: "read", callID: nil, rawText: #"{"name": "read", "arguments": "{path: 'a.txt'}"}"#, truncated: false))
        #expect(nested.arguments == #"{"path":"a.txt"}"#)
        #expect(nested.id.hasPrefix("call_"))
        let flat = try #require(ToolCallRepairer.repairToolCall(
            name: "read", callID: "c1", rawText: "{path: 'b.txt'", truncated: false))
        #expect(flat.arguments == #"{"path":"b.txt"}"#)
    }

    @Test("Valid JSON is returned untouched, apostrophes and literals inside strings included")
    func validJSONIsNotRewritten() {
        let valid = #"{"msg": "it's True that {x} is None"}"#
        #expect(ToolCallRepairer.repairJSON(valid) == valid)
    }

    @Test("Nothing is invented: no name, a truncated payload or no JSON at all gives nil")
    func repairRejectedToolCallRefuses() {
        #expect(ToolCallRepairer.repairToolCall(name: nil, callID: nil, rawText: "{a: 1}", truncated: false) == nil)
        #expect(ToolCallRepairer.repairToolCall(name: "x", callID: nil, rawText: "{a: 1}", truncated: true) == nil)
        #expect(ToolCallRepairer.repairToolCall(name: "x", callID: nil, rawText: "no json here", truncated: false) == nil)
    }

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

    @Test("ToolCallRepairer on chatter with JSON strips preamble (demonstrating need for hasTools gating)")
    func demonstratesPreambleStrippingOnChatter() {
        let prose = "Here is an example config:\n{\"env\": \"production\"}"
        let repaired = ToolCallRepairer.repairJSON(prose)
        // Without hasTools gating, the preamble is stripped:
        #expect(repaired?.starts(with: "Here is") == false)
        #expect(repaired == "{\"env\": \"production\"}")
    }

    @Test("Parking preserves clean boundary without synthetic separators")
    func cleanParkingBoundary() async {
        let cache = MultiSlotConversationCache(maxKVBufferTokens: 1024) // budget = 128 tokens = 512 chars
        let longPrompt = String(repeating: "Hello world and greetings to all agents. ", count: 20)
        let slotId = "slot-long"
        await cache.recordTurn(slotId: slotId, fullPrompt: longPrompt)

        await cache.parkSlot(id: slotId)
        let slot = await cache.slot(for: slotId)
        #expect(slot?.isParked == true)
        #expect(slot?.prompt.contains("\n...\n") == false)
        #expect(slot!.prompt.count <= 512)
        // Must end cleanly on a full word boundary, not mid-word
        let words = ["Hello", "world", "and", "greetings", "to", "all", "agents."]
        let lastWord = String(slot!.prompt.split(separator: " ").last ?? "")
        #expect(words.contains(lastWord))
    }

    @Test("Eviction prioritizes parked slots over active unparked slots")
    func evictionPrioritizesParkedSlots() async {
        let cache = MultiSlotConversationCache(maxKVBufferTokens: 4096, maxSlots: 3)
        await cache.recordTurn(slotId: "slot-active", fullPrompt: "Active prompt")
        await cache.recordTurn(slotId: "slot-idle1", fullPrompt: "Idle prompt 1")
        await cache.recordTurn(slotId: "slot-idle2", fullPrompt: "Idle prompt 2")

        // Park inactive slots except slot-active
        await cache.parkInactiveSlots(except: "slot-active")

        // Adding a 4th slot should evict a parked idle slot, not slot-active
        _ = await cache.getOrCreateSlot(id: "slot-new")
        let active = await cache.slot(for: "slot-active")
        let newSlot = await cache.slot(for: "slot-new")
        #expect(active != nil)
        #expect(newSlot != nil)
    }

    @Test("Mid-token prefill delta rolls back to clean word boundary")
    func midTokenRollback() async {
        let cache = MultiSlotConversationCache(maxKVBufferTokens: 4096)
        let p1 = "The quick brown fox jumps over the lazy dog and runs to the forest."
        let p2 = "The quick brown fox jumps over the lazy dog and runs to the foreign land."
        await cache.recordTurn(slotId: "fox-slot", fullPrompt: p1)

        let delta = await cache.computePrefillDelta(slotId: "fox-slot", newPrompt: p2)
        #expect(delta.isIncremental == true)
        // Common prefix ended at "The quick brown fox jumps over the lazy dog and runs to the fore"
        // But "fore" is a mid-word split of "forest" vs "foreign", so it must roll back to "runs to the "
        #expect(delta.cachedPrefix.hasSuffix("runs to the "))
    }
}

