import Foundation
import Testing
import Jinja
import Hub
import MLXLMCommon
import Tokenizers
@testable import BasaltCore
@testable import BasaltInference

@Suite("Chat templates render as Python Jinja2 under HF transformers does (#104)", .serialized)
struct ChatTemplateFidelityTests {
    static func rendered(_ template: String, _ context: [String: Value] = ["messages": ["q", "r"]]) throws -> String {
        let compatible = MLXInferenceEngine.compatibleChatTemplate(in: Config(["chat_template": Config(template)]))
        return try Template(try #require(compatible.chatTemplate.string()), with: .init(lstripBlocks: true, trimBlocks: true))
            .render(context)
    }

    static let jinja2GroundTruth: [(String, String, String)] = [
        ("whitespace-controlled tag after a literal {",
         "a:{ {{- 'x' -}} }",
         "a:{x}"),
        ("whitespace-controlled block on the next line after a literal {",
         "a:{\n    {%- if true -%}x{%- endif -%}\n}",
         "a:{x}"),
        ("Gemma 4 format_parameters shape",
         "{%- macro f(p) -%}{%- for k, v in p | dictsort -%}{{ k }}:{\n    {%- if v -%}d{%- endif -%}\n}{%- endfor -%}{%- endmacro -%}parameters:{\n  {%- if true -%}\n    properties:{ {{- f({'city': 1}) -}} },\n  {%- endif -%}\n}",
         "parameters:{properties:{city:{d}},}"),
        ("comment delimiters inside string literals are text",
         "{{ '{#- x #}' }}|{{ \"{# y -#}\" }}|{{ 'a -%} b' }}",
         "{#- x #}|{# y -#}|a -%} b"),
        ("tag delimiters inside a comment are ignored",
         "{# a {%- if %} b -}} #}c",
         "c"),
        ("CRLF and CR newlines are normalized, and the newline after a comment is trimmed",
         "a\r\nb\r\n{# c #}\r\nd\re",
         "a\nb\nd\ne"),
        ("a +#} comment keeps the newline after it",
         "a\n  {#- c +#}\n  b",
         "a\n  b"),
        ("an indented block after a stripped comment is still lstripped",
         "x\n{#- c #}\n    {% if true %}y{% endif %}\n",
         "xy"),
        ("{%+ keeps the indentation and +%} the newline",
         "a\n  {%+ if true +%}\nX{% endif %}",
         "a\n  \nX"),
        ("{%+ endif +%} keeps both",
         "a\n  {%+ if true %}\nX\n  {%+ endif +%}\nb",
         "a\n  X\n  \nb"),
        ("raw block content is literal text",
         "{% raw %}{#- keep #}{{ x }}{% endraw %}|",
         "{#- keep #}{{ x }}|"),
        ("raw block under lstrip_blocks and trim_blocks",
         "a\n  {% raw %}\nX\n  {% endraw %}\nb",
         "a\n\nX\nb"),
        ("raw block with whitespace control",
         "a\n  {%- raw -%}\n X \n  {%- endraw -%}\n b",
         "aXb"),
        ("raw block after a literal {",
         "{ {%- raw %}{ {%- x -%} }{% endraw -%}  \n z",
         "{{ {%- x -%} }z"),
        ("a multi-line string literal is not lstripped or trimmed",
         "{{ 'a\n  {% b' }}|{{ 'c%}\nd' }}",
         "a\n  {% b|c%}\nd"),
        ("tojson spacing matches json.dumps",
         "{{ {'city': 'Paris', 'n': [1, 2.5], 'ok': true, 'none': none} | tojson }}",
         "{\"city\": \"Paris\", \"n\": [1, 2.5], \"ok\": true, \"none\": null}"),
        ("tojson keeps non-ASCII and HTML characters",
         "{{ {'c': 'Zürich <&> \\'q\\''} | tojson }}",
         "{\"c\": \"Zürich <&> 'q'\"}"),
        ("tojson indent",
         "{{ {'a': {'b': [1]}} | tojson(indent=2) }}",
         "{\n  \"a\": {\n    \"b\": [\n      1\n    ]\n  }\n}"),
        ("an unmodified template renders as before",
         "{% for m in messages %}{{ m }}\n{% endfor %}  {% if true %}\n  x\n{% endif %}",
         "q\nr\n    x\n"),
    ]

    @Test("Synthetic templates render byte for byte as Jinja2 with trim_blocks and lstrip_blocks", arguments: jinja2GroundTruth)
    func matchesJinja2(name: String, template: String, expected: String) throws {
        #expect(try Self.rendered(template) == expected, "\(name)")
    }

    static let cachedTemplateRepos = [
        "mlx-community/Qwen3-4B-4bit", "mlx-community/gemma-4-12B-it-qat-4bit",
        "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit", "mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits",
        "LiquidAI/LFM2.5-8B-A1B-MLX-4bit", "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
        "mlx-community/Ministral-3-8B-Instruct-2512-4bit", "mlx-community/Ministral-3-14B-Reasoning-2512-4bit",
        "ornith-ai/Ornith-1.5-9B-MLX-4bit", "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly",
    ]

    static let weatherTool = ToolDefinition(
        name: "get_weather", description: "Get the weather for a city",
        parametersJSON: #"{"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}"#)

    static let conversations: [(name: String, tools: [ToolDefinition], turns: [ChatMessageInput])] = [
        ("plain", [], [
            ChatMessageInput(role: .system, text: "You are a coding agent."),
            ChatMessageInput(role: .user, text: "list the files"),
            ChatMessageInput(role: .assistant, text: "Here they are."),
            ChatMessageInput(role: .user, text: "thanks"),
        ]),
        ("tool round trip", [weatherTool], [
            ChatMessageInput(role: .system, text: "You are helpful."),
            ChatMessageInput(role: .user, text: "Weather in Paris?"),
            ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "call_1", name: "get_weather", arguments: #"{"city": "Paris", "unit": "celsius"}"#)]),
            ChatMessageInput(role: .tool, text: #"{"temp": 21}"#, toolCallId: "call_1", toolName: "get_weather"),
        ]),
        ("non-ASCII and markup in text and arguments", [weatherTool], [
            ChatMessageInput(role: .user, text: "Погода в Zürich? <b>&amp;</b> \"quoted\""),
            ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "call_2", name: "get_weather", arguments: #"{"city": "Zürich <&> \"q\"\n"}"#)]),
            ChatMessageInput(role: .tool, text: "ok", toolCallId: "call_2", toolName: "get_weather"),
        ]),
    ]

    static let transformersRenderer = """
        import json, sys
        from transformers.utils.chat_template_utils import render_jinja_template
        jobs = json.load(sys.stdin)
        json.dump([render_jinja_template(conversations=[j["messages"]], tools=j["tools"], chat_template=j["template"],
                                         add_generation_prompt=True, **j["special"])[0][0] for j in jobs], sys.stdout)
        """

    static let hasTransformers: Bool = (try? python(["-c", "import transformers"], input: Data())) != nil

    static func python(_ arguments: [String], input: Data) throws -> Data {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "-I"] + arguments
        let stdin = Pipe(), stdout = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = FileHandle.nullDevice
        try process.run()
        stdin.fileHandleForWriting.write(input)
        try stdin.fileHandleForWriting.close()
        let output = stdout.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw CocoaError(.executableLoad) }
        return output
    }

    @Test("Cached model templates produce the same tokens as HF transformers' apply_chat_template",
          .enabled(if: cachedTemplateRepos.contains { ControlTokenGuardTests.cachedModel($0) != nil }, "needs a cached model template"),
          .enabled(if: hasTransformers, "needs python3 with transformers"))
    func cachedTemplatesMatchTransformers() async throws {
        var jobs: [[String: Any]] = []
        var expectations: [(label: String, tokenizer: any Tokenizers.Tokenizer, tokens: [Int])] = []
        for repo in Self.cachedTemplateRepos {
            guard let directory = ControlTokenGuardTests.cachedModel(repo) else { continue }
            let config = try #require(try await LanguageModelConfigurationFromHub(modelFolder: directory).tokenizerConfig)
            let tokenizer = try await MLXInferenceEngine.upstreamTokenizer(at: directory)
            var special: [String: String] = [:]
            for key in ["bos_token", "eos_token", "unk_token", "pad_token"] {
                let value: Config = config[dynamicMember: key]
                special[key] = value.string() ?? (value[dynamicMember: "content"] as Config).string()
            }
            for conversation in Self.conversations {
                let tools = MLXInferenceEngine.toolSpecs(conversation.tools)
                let messages = DefaultMessageGenerator().generate(
                    from: UserInput(chat: MLXInferenceEngine.chat(conversation.turns, lastText: conversation.turns.last!.text), tools: tools))
                expectations.append(("\(repo) \(conversation.name)", tokenizer, try tokenizer.applyChatTemplate(messages: messages, tools: tools)))
                jobs.append([
                    "template": try #require(config.chatTemplate.string()),
                    "messages": messages,
                    "tools": tools ?? NSNull(),
                    "special": special,
                ])
            }
        }
        let output = try Self.python(["-c", Self.transformersRenderer], input: JSONSerialization.data(withJSONObject: jobs, options: [.sortedKeys]))
        let groundTruth = try #require(try JSONSerialization.jsonObject(with: output) as? [String])
        #expect(groundTruth.count == expectations.count)
        for (expected, reference) in zip(expectations, groundTruth) {
            let referenceTokens = expected.tokenizer.encode(text: reference, addSpecialTokens: false)
            #expect(expected.tokens == referenceTokens,
                    "\(expected.label)\nswift: \(expected.tokenizer.decode(tokens: expected.tokens))\npython: \(reference)")
        }
    }
}
