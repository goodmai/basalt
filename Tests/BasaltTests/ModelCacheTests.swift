import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

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

    /// What the engine does with a rejected call — the shape mlx-swift-lm actually produces: the
    /// whole protocol span as raw text and, for malformed syntax, no parsed name or call id.
    private func resolve(_ raw: String, reason: String = "malformed_syntax", name: String? = nil,
                         truncated: Bool = false, declared: Set<String> = ["shell", "read"]) -> StreamChunk {
        MLXInferenceEngine.resolveRejectedToolCall(reason: reason, toolName: name, callID: nil,
                                                   rawText: raw, truncated: truncated, declaredTools: declared)
    }

    @Test("A malformed call inside <tool_call> tags is repaired into a real call")
    func repairTaggedMalformedCall() throws {
        let chunk = resolve("<tool_call>{'name': 'shell', 'arguments': {'command': ['ls', '-la'], 'timeout_ms': 1000,}}</tool_call>")
        guard case .toolCall(let call) = chunk else { Issue.record("expected a tool call, got \(chunk)"); return }
        #expect(call.name == "shell")
        #expect(call.arguments == #"{"command":["ls","-la"],"timeout_ms":1000}"#)
        #expect(call.id.hasPrefix("call_"))
    }

    @Test("A brace inside an unclosed argument does not cut the rest of the argument off")
    func repairKeepsArgumentTail() {
        let chunk = resolve(#"<tool_call>{"name":"shell","arguments":{"command":"git log --format={h} -n 5</tool_call>"#)
        guard case .toolCall(let call) = chunk else { Issue.record("expected a tool call, got \(chunk)"); return }
        #expect(call.arguments == #"{"command":"git log --format={h} -n 5"}"#)
    }

    @Test("Mistral [TOOL_CALLS] payloads are repaired too")
    func repairMistralPayload() {
        guard case .toolCall(let call) = resolve("[TOOL_CALLS]read[ARGS]{path: 'c.txt'", name: "read") else {
            Issue.record("mistral"); return
        }
        #expect(call.arguments == #"{"path":"c.txt"}"#)
    }

    @Test("Function-style and string-encoded arguments are read like the repairer's other schemas")
    func repairOtherSchemas() throws {
        guard case .toolCall(let fn) = resolve(#"{"function": {"name": "read", "arguments": "{path: 'a.txt'}"}}"#) else {
            Issue.record("function schema"); return
        }
        #expect(fn.name == "read")
        #expect(fn.arguments == #"{"path":"a.txt"}"#)

        // The parser knew the name and the payload is the bare arguments object.
        guard case .toolCall(let bare) = resolve("{path: 'b.txt'", name: "read") else { Issue.record("bare args"); return }
        #expect(bare.arguments == #"{"path":"b.txt"}"#)
    }

    @Test("Nothing is invented: undeclared names, truncated payloads, non-objects and other reasons give a visible note")
    func rejectedCallsBecomeNotes() {
        let cases: [StreamChunk] = [
            resolve("<tool_call>{'name': 'rm_rf', 'arguments': {}}</tool_call>"),            // undeclared tool
            resolve("<tool_call>{'name': 'shell', 'arguments': {'a': 1}}", truncated: true), // truncated preview
            resolve("<tool_call>{'name': 'shell', 'arguments': [1, 2]}</tool_call>"),        // arguments not an object
            resolve("no json here"),
            resolve(#"{"name": "shell", "arguments": {}}"#, reason: "invalid_arguments", name: "shell")
        ]
        for chunk in cases {
            guard case .text(let note) = chunk else { Issue.record("expected a note, got \(chunk)"); continue }
            #expect(note.contains("rejected"))
        }
    }

    @Test("Valid JSON is returned untouched, apostrophes and literals inside strings included")
    func validJSONIsNotRewritten() {
        let valid = #"{"msg": "it's True that {x} is None"}"#
        #expect(ToolCallRepairer.repairJSON(valid) == valid)
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

