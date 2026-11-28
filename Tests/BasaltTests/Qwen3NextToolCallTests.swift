import Foundation
import Testing
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

enum Qwen3NextTools {
    struct Parameter: Sendable {
        let name: String
        let type: String
        var required = false
    }

    struct Tool: Sendable {
        let name: String
        let description: String
        let parameters: [Parameter]

        var definition: ToolDefinition {
            let properties = parameters.map { #""\#($0.name)": {"type": "\#($0.type)"}"# }.joined(separator: ", ")
            let required = parameters.filter(\.required).map { #""\#($0.name)""# }.joined(separator: ", ")
            return ToolDefinition(
                name: name, description: description,
                parametersJSON: #"{"type": "object", "properties": {\#(properties)}, "required": [\#(required)]}"#)
        }
    }

    struct Parsed: Equatable {
        var text = ""
        var calls: [ToolCall.Function] = []
        var rejections: [RejectedToolCall.Reason] = []
    }

    static let read = Tool(name: "Read", description: "Read a file from the local filesystem.", parameters: [
        Parameter(name: "file_path", type: "string", required: true),
        Parameter(name: "offset", type: "integer"),
        Parameter(name: "limit", type: "integer"),
    ])

    static let edit = Tool(name: "Edit", description: "Replace old_string with new_string in a file.", parameters: [
        Parameter(name: "file_path", type: "string", required: true),
        Parameter(name: "old_string", type: "string", required: true),
        Parameter(name: "new_string", type: "string", required: true),
        Parameter(name: "replace_all", type: "boolean"),
    ])

    static let bash = Tool(name: "Bash", description: "Run a shell command.", parameters: [
        Parameter(name: "command", type: "string", required: true),
        Parameter(name: "timeout", type: "integer"),
    ])

    static let all = [read, edit, bash]

    static func call(_ name: String, _ parameters: [(String, String)]) -> String {
        let body = parameters.map { "<parameter=\($0.0)>\n\($0.1)\n</parameter>\n" }.joined()
        return "<tool_call>\n<function=\(name)>\n\(body)</function>\n</tool_call>"
    }

    static func function(_ name: String, _ arguments: [String: JSONValue]) -> ToolCall.Function {
        ToolCall.Function(name: name, arguments: arguments)
    }

    static func parse(_ chunks: [String], tools: [Tool] = all) -> Parsed {
        let processor = ToolCallProcessor(format: .xmlFunction, tools: MLXInferenceEngine.toolSpecs(tools.map(\.definition)))
        var outputs: [ToolCallProcessor.Output] = []
        for chunk in chunks { outputs += processor.processChunkOutputs(chunk) }
        outputs += processor.processEOSOutputs()
        var parsed = Parsed()
        for output in outputs {
            switch output {
            case .response(let text): parsed.text += text
            case .toolCall(let call): parsed.calls.append(call.function)
            case .rejectedToolCall(let rejection): parsed.rejections.append(rejection.reason)
            }
        }
        return parsed
    }

    static func pieces(_ text: String, size: Int) -> [String] {
        stride(from: 0, to: text.count, by: size).map { String(text.dropFirst($0).prefix(size)) }
    }

    static func violations(_ arguments: [String: JSONValue], of tool: Tool) -> [String] {
        var found: [String] = []
        for parameter in tool.parameters {
            guard let value = arguments[parameter.name] else {
                if parameter.required { found.append("missing \(parameter.name)") }
                continue
            }
            if !matches(value, parameter.type) { found.append("\(parameter.name) is not \(parameter.type)") }
        }
        let known = Set(tool.parameters.map(\.name))
        return found + arguments.keys.filter { !known.contains($0) }.sorted().map { "unknown \($0)" }
    }

    static func matches(_ value: JSONValue, _ type: String) -> Bool {
        switch (type, value) {
        case ("string", .string), ("integer", .int), ("boolean", .bool): true
        default: false
        }
    }
}

@Suite("Qwen3-Coder-Next tool calls in the qwen3_coder XML form decode to typed arguments through the fork's xmlFunction processor fed with the engine's tool specs (#642 P4)")
struct Qwen3NextToolCallTests {
    static let code = "\n    func budget(_ items: [Int]) -> Int {\n\tlet total = items.reduce(0, +) \n\n        return total * 2\n    }\n"

    static let scalars: [(type: String, text: String, value: JSONValue)] = [
        ("integer", "42", .int(42)),
        ("number", "42.5", .double(42.5)),
        ("number", "42", .int(42)),
        ("boolean", "true", .bool(true)),
        ("boolean", "false", .bool(false)),
        ("string", "42", .string("42")),
        ("string", "true", .string("true")),
        ("integer", "42.5", .string("42.5")),
    ]

    static let literalMarkup = [
        "echo '</function>' && cat notes.txt",
        "grep -n '<parameter=x>' Sources",
        "printf '%s' '</tool_call>'",
        "sed -n '1p' <function=Bash>",
    ]

    @Test("The real chat template and the checkpoint's tool_parser_type both select the xmlFunction format")
    func formatIsSelected() {
        #expect(ToolCallFormat.inferred(fromChatTemplate: Qwen3CoderNextTemplate.jinja) == .xmlFunction)
        #expect(ToolCallFormat(toolParserType: "qwen3_coder") == .xmlFunction)
    }

    @Test("A parameter written as newline, 42, newline takes the JSON type its schema declares; a string schema keeps the text and an integer schema keeps text it cannot read as an integer", arguments: scalars)
    func scalarTakesTheSchemaType(type: String, text: String, value: JSONValue) {
        let probe = Qwen3NextTools.Tool(name: "probe", description: "Probe", parameters: [.init(name: "line", type: type, required: true)])
        let parsed = Qwen3NextTools.parse([Qwen3NextTools.call("probe", [("line", text)])], tools: [probe])
        #expect(parsed == Qwen3NextTools.Parsed(calls: [Qwen3NextTools.function("probe", ["line": value])]))
    }

    @Test("A string parameter holding code keeps indentation, tabs, a trailing space and inner blank lines; exactly one leading and one trailing newline are stripped, so a value that starts and ends with a blank line keeps one of each")
    func codeIsKeptByteForByte() {
        let payload = Qwen3NextTools.call("Edit", [("file_path", "Sources/App/Budget.swift"), ("old_string", "return total"), ("new_string", Self.code)])
        let parsed = Qwen3NextTools.parse([payload])
        #expect(parsed == Qwen3NextTools.Parsed(calls: [Qwen3NextTools.function("Edit", [
            "file_path": .string("Sources/App/Budget.swift"), "old_string": .string("return total"), "new_string": .string(Self.code),
        ])]))
    }

    @Test("A value containing a literal </function>, <parameter=x>, </tool_call> or <function=…> stays inside its parameter", arguments: literalMarkup)
    func literalMarkupStaysInTheValue(command: String) {
        let parsed = Qwen3NextTools.parse([Qwen3NextTools.call("Bash", [("command", command)])])
        #expect(parsed == Qwen3NextTools.Parsed(calls: [Qwen3NextTools.function("Bash", ["command": .string(command)])]))
    }

    @Test("A value ends at the first </parameter> whether or not a newline precedes it, so a value containing that literal makes the payload malformed: no call, one malformed_syntax rejection at end of output", arguments: [
        "echo '</parameter>' done", "first line\n</parameter>\nsecond line",
    ])
    func literalParameterCloseIsNotRepresentable(command: String) {
        let payload = Qwen3NextTools.call("Bash", [("command", command)])
        let specs = MLXInferenceEngine.toolSpecs(Qwen3NextTools.all.map(\.definition))
        #expect(ToolCallFormat.xmlFunction.createParser().parse(content: payload, tools: specs) == nil)
        #expect(Qwen3NextTools.parse([payload]) == Qwen3NextTools.Parsed(rejections: [.malformedSyntax]))
    }

    @Test("Two calls in one message decode in order, plain text before a call and text after it stay response text")
    func callsAndSurroundingText() {
        let read = Qwen3NextTools.call("Read", [("file_path", "a.swift"), ("offset", "42"), ("limit", "20")])
        let bash = Qwen3NextTools.call("Bash", [("command", "swift build"), ("timeout", "60000")])
        let expected = [
            Qwen3NextTools.function("Read", ["file_path": .string("a.swift"), "offset": .int(42), "limit": .int(20)]),
            Qwen3NextTools.function("Bash", ["command": .string("swift build"), "timeout": .int(60000)]),
        ]
        #expect(Qwen3NextTools.parse([read + "\n" + bash]) == Qwen3NextTools.Parsed(text: "\n", calls: expected))
        #expect(Qwen3NextTools.parse(["Let me read the file.\n" + read]) == Qwen3NextTools.Parsed(text: "Let me read the file.\n", calls: [expected[0]]))
        #expect(Qwen3NextTools.parse([read + "\nDone."]) == Qwen3NextTools.Parsed(text: "\nDone.", calls: [expected[0]]))
    }

    @Test("Prose with an opening quote before a call does not hide the call; an unmatched backtick still does, which is a known defect of the fork's recovery scanner (#887)")
    func proseBeforeACallDoesNotHideIt() {
        let read = Qwen3NextTools.call("Read", [("file_path", "a.swift")])
        let expected = [Qwen3NextTools.function("Read", ["file_path": .string("a.swift")])]
        let quoted = Qwen3NextTools.parse(["He said \"go.\n" + read])
        #expect(quoted.calls == expected && quoted.text == "He said \"go.\n")
        withKnownIssue("an unmatched backtick puts the fork's TextToolCallRecoveryScanner in a protected context, so the call is returned as text") {
            #expect(Qwen3NextTools.parse(["Use a ` here.\n" + read]).calls == expected)
        }
    }

    @Test("At end of output a call cut inside a parameter is rejected as incomplete_output with no text, and a call that closed </function> but not </tool_call> is accepted")
    func unterminatedCallAtEndOfOutput() {
        let cut = "<tool_call>\n<function=Read>\n<parameter=file_path>\nSources/App/Budget.swift"
        #expect(Qwen3NextTools.parse([cut]) == Qwen3NextTools.Parsed(rejections: [.incompleteOutput]))
        let unframed = "<tool_call>\n<function=Read>\n<parameter=file_path>\na.swift\n</parameter>\n</function>"
        #expect(Qwen3NextTools.parse([unframed]) == Qwen3NextTools.Parsed(calls: [Qwen3NextTools.function("Read", ["file_path": .string("a.swift")])]))
    }

    @Test("Array and object parameters hold JSON text that decodes under array and object schemas, and an optional parameter that is not written is absent")
    func arraysObjectsAndMissingOptional() {
        let plan = Qwen3NextTools.Tool(name: "Plan", description: "Plan", parameters: [
            .init(name: "steps", type: "array", required: true), .init(name: "options", type: "object", required: true),
            .init(name: "note", type: "string"),
        ])
        let payload = Qwen3NextTools.call("Plan", [("steps", #"["read", "edit"]"#), ("options", "{\n  \"depth\": 2,\n  \"follow\": true\n}")])
        #expect(Qwen3NextTools.parse([payload], tools: [plan]) == Qwen3NextTools.Parsed(calls: [Qwen3NextTools.function("Plan", [
            "steps": .array([.string("read"), .string("edit")]), "options": .object(["depth": .int(2), "follow": .bool(true)]),
        ])]))
    }

    @Test("Fed in two chunks split at every offset, one character at a time and in fixed chunks of 2, 3, 5 and 7 characters, the stream gives the calls and text of the whole message and no markup in the text")
    func streamingMatchesTheWholeMessage() {
        let message = [
            "Let me fix it.\n",
            Qwen3NextTools.call("Edit", [("file_path", "Sources/App/Budget.swift"), ("old_string", "print(\"</function>\")"), ("new_string", Self.code)]),
            "\n", Qwen3NextTools.call("Read", [("file_path", "Sources/App/Budget.swift"), ("offset", "42")]), "\nDone.",
        ].joined()
        let whole = Qwen3NextTools.parse([message])
        #expect(whole.calls.map(\.name) == ["Edit", "Read"])
        #expect(whole.calls.first?.arguments["new_string"] == .string(Self.code))
        #expect(whole.calls.last?.arguments["offset"] == .int(42))
        #expect(whole.text == "Let me fix it.\n\n\nDone.")
        #expect(whole.rejections.isEmpty)
        for offset in 0 ... message.count {
            let chunks = [String(message.prefix(offset)), String(message.dropFirst(offset))]
            #expect(Qwen3NextTools.parse(chunks) == whole, "split at \(offset): \(chunks[0].suffix(16).debugDescription) | \(chunks[1].prefix(16).debugDescription)")
        }
        for size in [1, 2, 3, 5, 7] {
            #expect(Qwen3NextTools.parse(Qwen3NextTools.pieces(message, size: size)) == whole, "chunks of \(size)")
        }
    }

    @Test("The schema check used by the real-model gate reports a missing required key, a wrong type and an unknown key, and accepts a call without its optional keys")
    func schemaCheck() {
        #expect(Qwen3NextTools.violations(["file_path": .string("a")], of: Qwen3NextTools.read).isEmpty)
        #expect(Qwen3NextTools.violations(["offset": .string("42"), "path": .string("a")], of: Qwen3NextTools.read)
            == ["missing file_path", "offset is not integer", "unknown path"])
        #expect(Qwen3NextTools.violations(["command": .string("ls"), "timeout": .double(1.5)], of: Qwen3NextTools.bash) == ["timeout is not integer"])
    }
}
