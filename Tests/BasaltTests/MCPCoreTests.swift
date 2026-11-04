import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltServer
@testable import BasaltHarness

@Suite("MCP value core")
struct MCPCoreTests {

    private func object(_ json: String) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    private func failure(_ body: () throws -> Void) -> MCPFailure? {
        do {
            try body()
            return nil
        } catch {
            return error as? MCPFailure
        }
    }

    @Test("route: bad JSON is a parse error without id; an unknown method keeps the id")
    func routeFailures() {
        guard case .failed(nil, let parse) = MCPRoute(line: "not json") else { Issue.record("not a failure"); return }
        #expect(parse == .parseError)
        guard case .failed(.int(3)?, let unknown) = MCPRoute(line: #"{"jsonrpc":"2.0","id":3,"method":"nope"}"#) else {
            Issue.record("not a failure"); return
        }
        #expect(unknown == MCPFailure(code: -32601, message: "Method not found: nope"))
    }

    @Test("route: responses and initialized notifications are ignored", arguments: [
        #"{"jsonrpc":"2.0","id":1,"result":{}}"#,
        #"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        #"{"jsonrpc":"2.0","method":"initialized"}"#,
    ])
    func routeIgnored(line: String) {
        guard case .ignored = MCPRoute(line: line) else { Issue.record("routed \(line)"); return }
    }

    @Test("route: tools/call carries its id and params")
    func routeToolsCall() throws {
        let route = MCPRoute(line: #"{"jsonrpc":"2.0","id":"a","method":"tools/call","params":{"name":"status"}}"#)
        guard case .toolsCall(.string("a")?, let params) = route else { Issue.record("routed \(route)"); return }
        #expect(try MCPArguments.toolCall(params).name == "status")
    }

    @Test("tool names: gemma_* aliases resolve to the same tool, unknown names to nil")
    func toolAliases() {
        #expect(MCPTool(name: "gemma_generate") == .generate && MCPTool(name: "basalt_generate") == .generate)
        #expect(MCPTool(name: "gemma_status") == .health && MCPTool(name: "basalt_add_knowledge") == .addKnowledge)
        #expect(MCPTool(name: "playwright_screenshot") == nil)
    }

    @Test("tool arguments: object, null and repairable string pass; other shapes are -32602")
    func toolArguments() throws {
        #expect(try MCPArguments.arguments(["a": 1])?["a"] as? Int == 1)
        #expect(try MCPArguments.arguments(NSNull()) == nil)
        #expect(try MCPArguments.arguments(nil) == nil)
        #expect(try MCPArguments.arguments(#"{"prompt":"hi""#)?["prompt"] as? String == "hi")
        #expect(failure { _ = try MCPArguments.arguments(7) } == .invalidParams("Tool arguments must be a JSON object"))
        #expect(failure { _ = try MCPArguments.arguments("][") }
            == .invalidParams("Tool arguments are not a JSON object and could not be repaired"))
        #expect(failure { _ = try MCPArguments.toolCall(AnyCodable(["arguments": [String: Any]()])) } == .invalidParams("Missing tool name"))
    }

    @Test("generation: prompt is required, lookup flags must be booleans, values reach the request")
    func generation() throws {
        #expect(failure { _ = try MCPArguments.generation([:]) } == .invalidParams("Missing required argument: prompt"))
        let numeric = try object(#"{"prompt":"p","speculativeLookup":1}"#)
        #expect(failure { _ = try MCPArguments.generation(numeric) } == .invalidParams("speculativeLookup must be a boolean"))
        let request = try MCPArguments.generation(try object(#"{"prompt":"p","maxTokens":8,"grammarJumpForward":true,"speculativeLookup":null}"#))
        #expect(request.prompt == "p" && request.maxTokens == 8)
        #expect(request.grammarJumpForward == true && request.speculativeLookup == nil)
    }

    @Test("knowledge: missing topic wins over a missing library, which wins over a bad domain")
    func knowledgeOrder() {
        #expect(failure { _ = try KnowledgeArguments(["topic": "t"], libraryConfigured: false) }
            == .invalidParams("Missing required arguments: topic, content"))
        #expect(failure { _ = try KnowledgeArguments(["topic": "t", "content": "c", "domain": 5], libraryConfigured: false) }
            == .libraryMissing)
        #expect(failure { _ = try KnowledgeArguments(["topic": "t", "content": "c", "domain": 5], libraryConfigured: true) }
            == .invalidParams("domain must be a string"))
    }

    @Test("knowledge: defaults fill domain, cue and confidence")
    func knowledgeDefaults() throws {
        let knowledge = try KnowledgeArguments(["topic": "t", "content": "c"], libraryConfigured: true)
        #expect(knowledge.domain == "general" && knowledge.cue == "" && knowledge.confidence == "medium")
    }

    @Test("harness input: surface, alwaysOnMaxChars and source are checked in order")
    func harnessInput() throws {
        #expect(failure { _ = try HarnessInputArguments(["surface": "vim"]) }?.message.hasPrefix("Unknown surface 'vim'") == true)
        #expect(failure { _ = try HarnessInputArguments(["alwaysOnMaxChars": 0]) }
            == .invalidParams("alwaysOnMaxChars must be a positive integer"))
        #expect(failure { _ = try HarnessInputArguments([:]) } == .invalidParams("Missing required argument: directory or files"))
        let input = try HarnessInputArguments(["directory": "", "files": ["CLAUDE.md": "a"], "alwaysOnMaxChars": 5])
        guard case .files(let files) = input.source else { Issue.record("not files"); return }
        #expect(files == ["CLAUDE.md": "a"] && input.spec.alwaysOnMaxChars == 5)
    }

    @Test("harness optimize: booleans are not integers, the slug derives from the task")
    func harnessOptimize() throws {
        let booleanK = try object(#"{"task":"t","k":true}"#)
        let booleanTokens = try object(#"{"task":"t","proposerMaxTokens":true}"#)
        #expect(failure { _ = try HarnessOptimizeArguments(booleanK) } == .invalidParams("k must be an integer in \(HarnessLedger.kRange)"))
        #expect(failure { _ = try HarnessOptimizeArguments(booleanTokens) }
            == .invalidParams("proposerMaxTokens must be an integer in \(HarnessOptimizer.TokenBudgets.proposerRange)"))
        #expect(failure { _ = try HarnessOptimizeArguments(["task": "t", "requiredTerms": "x"]) }
            == .invalidParams("requiredTerms must be an array of strings"))
        let arguments = try HarnessOptimizeArguments(try object(#"{"task":"Audit the targets","k":2,"expectsJSON":true}"#))
        #expect(arguments.k == 2 && arguments.proposerMaxTokens == nil)
        #expect(arguments.task.slug == HarnessExperienceLibrary.slugify("Audit the targets") && arguments.task.expectsJSON)
    }

    @Test("BasaltError codes: each case keeps its JSON-RPC code", arguments: [
        (BasaltError.modelNotFound(identifier: "m"), -32001),
        (.invalidRequestStructure(details: "d"), -32602),
        (.modelNotCached(identifier: "m"), -32010),
        (.serverBusy(retryAfterSeconds: 1), -32012),
        (.modelConfigurationInvalid(path: "p", reason: "r"), -32012),
        (.memoryPressure(retryAfterSeconds: 1), -32013),
    ])
    func errorCodes(error: BasaltError, code: Int) {
        #expect(MCPFailure(error) == MCPFailure(code: code, message: error.localizedDescription))
    }

    @Test("failures: library and optimizer errors map to their codes")
    func failureMapping() {
        #expect(MCPFailure.optimization(HarnessBudgetError.leakInHarnessFile(path: "p", token: "t")).code == -32602)
        #expect(MCPFailure.optimization(BasaltError.modelNotFound(identifier: "m")).code == -32001)
        #expect(MCPFailure.optimization(CocoaError(.fileNoSuchFile)).message.hasPrefix("Harness optimization failed: "))
        #expect(MCPFailure.knowledge(CocoaError(.fileNoSuchFile)).code == -32603)
    }

    @Test("wire: the id is escaped, a missing id is omitted")
    func envelopes() {
        #expect(MCPWire.error(id: .string("a/\"b"), .parseError)
            == #"{"jsonrpc":"2.0","id":"a\/\"b","error":{"code":-32700,"message":"Parse error"}}"#)
        #expect(MCPWire.result(id: nil, ToolResult.text("ok"))
            == #"{"jsonrpc":"2.0","result":{"content":[{"text":"ok","type":"text"}]}}"#)
        #expect(MCPWire.result(id: .null, ToolResult.text("x", isError: true))
            == #"{"jsonrpc":"2.0","id":null,"result":{"content":[{"text":"x","type":"text"}],"isError":true}}"#)
    }

    @Test("AnyCodable: decoded JSON keeps its Foundation shape and encodes back unchanged")
    func anyCodableRoundTrip() throws {
        let json = #"{"a":[1,2.5,true,"s",null],"b":{}}"#
        let value = try JSONDecoder().decode(AnyCodable.self, from: Data(json.utf8))
        let raw = try #require(value.rawValue as? [String: Any])
        let array = try #require(raw["a"] as? [Any])
        #expect(array[0] is Int && array[1] is Double && array[2] is Bool && array[4] is NSNull)
        #expect(String(data: try MCPWire.encoder.encode(value), encoding: .utf8) == json)
        #expect(AnyCodable([String: Any]()).rawValue is [String: Any])
    }
}
