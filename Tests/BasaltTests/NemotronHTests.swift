import Testing
import Foundation
import MLXLLM
import MLXLMCommon
import Jinja
import Hub
import Tokenizers
@testable import BasaltCore
@testable import BasaltInference

@Suite("Nemotron-H")
struct NemotronHTests {

    static let denseNano4BConfig = """
    {"architectures": ["NemotronHForCausalLM"], "attention_bias": false, "chunk_size": 256, "conv_kernel": 4,
     "expand": 2, "head_dim": 128, "hidden_size": 3136,
     "hybrid_override_pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-", "intermediate_size": 12544,
     "layer_norm_epsilon": 1e-05, "mamba_head_dim": 80, "mamba_num_heads": 96, "mamba_proj_bias": false,
     "max_position_embeddings": 262144, "mlp_bias": false, "model_type": "nemotron_h", "n_groups": 8,
     "num_attention_heads": 40, "num_hidden_layers": 42, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
     "ssm_state_size": 128, "tie_word_embeddings": false, "time_step_limit": [0.0, Infinity],
     "use_bias": false, "use_conv_bias": true, "vocab_size": 131072}
    """

    @Test("a dense checkpoint without MoE keys decodes")
    func denseConfigDecodes() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronHConfiguration.self, from: Data(Self.denseNano4BConfig.utf8))
        #expect(config.nRoutedExperts == 0)
        #expect(config.hiddenSize == 3136)
    }

    static func modelDirectory(config: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("nemotron-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(config.utf8).write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    @Test("a config.json carrying Infinity is a usable config")
    func infinityConfigIsUsable() throws {
        let dir = try Self.modelDirectory(config: Self.denseNano4BConfig)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelCache.hasUsableConfig(at: dir))
        #expect(ModelCache.config(at: dir)?["model_type"] as? String == "nemotron_h")
        #expect(KVCacheGeometry.load(modelDirectory: dir) != nil)
    }

    static let nanoV2ToolCallsSnippet = """
    {%- for message in messages %}{%- set content = message['content'] %}{%- if message['role'] == 'assistant' -%}\
    {{- '<SPECIAL_11>Assistant\\n' + content.strip() }}{%- if message.tool_calls -%}{{- '<TOOLCALL>[' -}}\
    {%- for call in message.tool_calls -%}{%- set fn = call.function if call.function is defined else call -%}\
    {{- '{"name": "' + fn.name + '", "arguments": ' -}}{%- if fn.arguments is string -%}{{- fn.arguments -}}\
    {%- else -%}{{- fn.arguments | tojson -}}{%- endif -%}{{- '}' + (', ' if not loop.last else '') -}}{%- endfor -%}\
    {{- ']</TOOLCALL>' -}}{%- endif -%}{{- '\\n<SPECIAL_12>\\n' -}}{%- endif -%}{%- endfor -%}
    """

    static let assistantWithCalls: [String: Jinja.Value] = [
        "messages": [[
            "role": "assistant", "content": "",
            "tool_calls": [
                ["function": ["name": "get_weather", "arguments": ["city": "Paris"]]],
                ["function": ["name": "get_time", "arguments": "{\"tz\":\"CET\"}"]],
            ],
        ]],
    ]

    @Test("swift-jinja refuses the Nano v2 template as shipped")
    func shippedTemplateIsRefused() {
        #expect(throws: (any Error).self) { try Template(Self.nanoV2ToolCallsSnippet) }
    }

    @Test("renaming the reserved loop variable makes the Nano v2 template render its tool calls")
    func renamedTemplateRenders() throws {
        let compatible = ChatTemplateCompatibility.renamingReservedLoopVariables(Self.nanoV2ToolCallsSnippet)
        let rendered = try Template(compatible).render(Self.assistantWithCalls)
        #expect(rendered.contains(#"<TOOLCALL>[{"name": "get_weather", "arguments": {"city":"Paris"}}, {"name": "get_time", "arguments": {"tz":"CET"}}]</TOOLCALL>"#))
    }

    @Test("outside the loop, as an attribute and inside a quoted word call keeps its name")
    func renameLeavesOtherNamesAlone() {
        let template = "{{ call }}{% for call in xs %}{{ call.name ~ 'call' ~ x.call }}{% endfor %}{{ call }} call"
        #expect(ChatTemplateCompatibility.renamingReservedLoopVariables(template)
            == "{{ call }}{% for call_item in xs %}{{ call_item.name ~ 'call' ~ x.call }}{% endfor %}{{ call }} call")
    }

    @Test("an inner unpacking loop does not end the renaming of the outer call loop")
    func unpackingLoopKeepsTheOuterRename() throws {
        let template = "{% for call in calls %}{% for k, v in call.args.items() %}{{ k }}={{ v }};{% endfor %}{{ call.name }}{% endfor %}"
        let compatible = ChatTemplateCompatibility.renamingReservedLoopVariables(template)
        #expect(compatible == "{% for call_item in calls %}{% for k, v in call_item.args.items() %}{{ k }}={{ v }};{% endfor %}{{ call_item.name }}{% endfor %}")
        let rendered = try Template(compatible).render(["calls": [["name": "ls", "args": ["path": "/"]]]])
        #expect(rendered == "path=/;ls")
    }

    @Test("an unpacking loop that binds call is renamed too")
    func unpackingLoopBindingCallIsRenamed() {
        #expect(ChatTemplateCompatibility.renamingReservedLoopVariables("{% for i, call in xs %}{{ call }}{% endfor %}")
            == "{% for i, call_item in xs %}{{ call_item }}{% endfor %}")
    }

    @Test("the Nano 9B tool instructions keep their wording inside a call loop")
    func stringLiteralsKeepCall() {
        let template = #"{%- for call in calls -%}{{- 'If you decide to call any tool(s), use the following format:\n' -}}{{- "call \"call\" " ~ call -}}{%- endfor -%}"#
        #expect(ChatTemplateCompatibility.renamingReservedLoopVariables(template)
            == #"{%- for call_item in calls -%}{{- 'If you decide to call any tool(s), use the following format:\n' -}}{{- "call \"call\" " ~ call_item -}}{%- endfor -%}"#)
    }

    @Test("a {%+ loop is renamed and closed like any other")
    func plusWhitespaceControlIsRecognised() {
        #expect(ChatTemplateCompatibility.renamingReservedLoopVariables("{%+ for call in xs %}{{ call }}{%+ endfor %}{{ call }}")
            == "{%+ for call_item in xs %}{{ call_item }}{%+ endfor %}{{ call }}")
    }

    @Test("the tokenizer config carries the compatible template, and an untouched one is returned as is")
    func tokenizerConfigIsRewritten() {
        let shipped = Config(["chat_template": Config(Self.nanoV2ToolCallsSnippet), "eos_token": Config("<SPECIAL_12>")])
        let rewritten = MLXInferenceEngine.compatibleChatTemplate(in: shipped)
        #expect(rewritten.chatTemplate.string()?.contains("for call_item in message.tool_calls") == true)
        #expect(rewritten.eosToken.string() == "<SPECIAL_12>")
        let plain = Config(["chat_template": Config("{{ messages }}")])
        #expect(MLXInferenceEngine.compatibleChatTemplate(in: plain).chatTemplate.string() == "{{ messages }}")
    }

    @Test("a Llama tokenizer_class with legacy false loads the same plain tokenizer the model folder loader builds")
    func tokenizerClassMatchesTheModelFolderLoader() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"tokenizer_class": "LlamaTokenizerFast", "legacy": false}"#.utf8)
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))
        try Data(#"{"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "▁": 3, "▁ab": 4}, "merges": ["a b", "▁ ab"]}, "added_tokens": []}"#.utf8)
            .write(to: dir.appendingPathComponent("tokenizer.json"))
        let loaded = try await MLXInferenceEngine.upstreamTokenizer(at: dir)
        let reference = try await AutoTokenizer.from(modelFolder: dir)
        #expect(type(of: loaded) == type(of: reference))
        #expect(loaded.encode(text: "ab", addSpecialTokens: false) == reference.encode(text: "ab", addSpecialTokens: false))
    }

    static let nanoV2ThinkingSnippet = """
    {%- set ns = namespace(enable_thinking=true) %}{%- for message in messages -%}{%- set content = message['content'] -%}\
    {%- if message['role'] == 'user' or message['role'] == 'system' -%}{%- if '/think' in content -%}\
    {%- set ns.enable_thinking = true -%}{%- elif '/no_think' in content -%}{%- set ns.enable_thinking = false -%}\
    {%- endif -%}{%- endif -%}{%- endfor -%}{%- if messages[0]['role'] != 'system' -%}{{- '<SPECIAL_10>System\\n' -}}\
    {%- else -%}{{- '<SPECIAL_10>System\\n' + messages[0]['content'].replace('/think', '').replace('/no_think', '').strip() }}\
    {%- endif -%}{{- '\\n' -}}{%- if messages[0]['role'] == 'system' -%}{%- set messages = messages[1:] -%}{%- endif -%}\
    {%- for message in messages %}{{- '<SPECIAL_11>User\\n' + message['content'] + '\\n' }}{%- endfor -%}\
    {{- '<SPECIAL_11>Assistant\\n' -}}{%- if ns.enable_thinking is defined and ns.enable_thinking is false -%}\
    {{- '<think></think>' -}}{%- else -%}{{- '<think>\\n' -}}{%- endif -%}
    """

    static func renderNanoV2(_ turns: [ChatMessageInput]) throws -> String {
        let messages = DefaultMessageGenerator().generate(messages: MLXInferenceEngine.chat(turns, lastText: turns.last!.text))
        return try Template(Self.nanoV2ThinkingSnippet).render(["messages": try Value(any: messages)])
    }

    static let nanoV2 = NemotronProfile(switchesThinkingInConversation: true)
    static let question = ChatMessageInput(role: .user, text: "What is 2+2?")

    @Test("thinking off on Nano v2 adds a system turn whose /no_think the template honours and then hides")
    func thinkingOffWithoutSystemTurn() throws {
        let turns = Self.nanoV2.turns([Self.question], for: .disabled)
        #expect(turns.map(\.role) == [.system, .user])
        let rendered = try Self.renderNanoV2(turns)
        #expect(rendered == "<SPECIAL_10>System\n\n<SPECIAL_11>User\nWhat is 2+2?\n<SPECIAL_11>Assistant\n<think></think>")
    }

    @Test("thinking off on Nano v2 keeps the caller's system text")
    func thinkingOffWithSystemTurn() throws {
        let turns = Self.nanoV2.turns([ChatMessageInput(role: .system, text: "Be brief."), Self.question], for: .disabled)
        let rendered = try Self.renderNanoV2(turns)
        #expect(rendered.hasPrefix("<SPECIAL_10>System\nBe brief.\n"))
        #expect(rendered.hasSuffix("<think></think>"))
    }

    @Test("an explicit effort turns Nano v2 thinking on, the model default leaves the turns alone")
    func thinkingOnAndDefault() throws {
        let off = [ChatMessageInput(role: .system, text: "/no_think"), Self.question]
        #expect(try Self.renderNanoV2(Self.nanoV2.turns(off, for: .effort("high"))).hasSuffix("<think>\n"))
        #expect(Self.nanoV2.turns([Self.question], for: .modelDefault).map(\.text) == [Self.question.text])
    }

    @Test("Nano 4B switches thinking through enable_thinking and never sees a /no_think marker")
    func nano4BUsesTemplateKwarg() {
        let nano4B = NemotronProfile(switchesThinkingInConversation: false)
        #expect(nano4B.templateContext(for: .disabled)["enable_thinking"] as? Bool == false)
        #expect(nano4B.turns([Self.question], for: .disabled).map(\.text) == [Self.question.text])
    }

    @Test("the registry tells the two Nemotron-H templates apart")
    func registryDetectsTheThinkingSwitch() throws {
        let dir = try Self.modelDirectory(config: Self.denseNano4BConfig)
        defer { try? FileManager.default.removeItem(at: dir) }
        let tokenizerConfig = try JSONSerialization.data(withJSONObject: ["chat_template": Self.nanoV2ThinkingSnippet])
        try tokenizerConfig.write(to: dir.appendingPathComponent("tokenizer_config.json"))
        let v2 = ModelProfileRegistry.profile(forModelAt: dir.path) as? NemotronProfile
        #expect(v2?.switchesThinkingInConversation == true)

        try Data("{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}".utf8)
            .write(to: dir.appendingPathComponent("chat_template.jinja"))
        let nano4B = ModelProfileRegistry.profile(forModelAt: dir.path) as? NemotronProfile
        #expect(nano4B?.switchesThinkingInConversation == false)
    }

    static func scan(_ pieces: [String]) -> (text: String, calls: [ToolCallOutput]) {
        var scanner = NemotronToolCallScanner()
        let chunks = pieces.flatMap { scanner.feed($0) } + scanner.finish()
        var text = ""
        var calls: [ToolCallOutput] = []
        for chunk in chunks {
            switch chunk {
            case .text(let t): text += t
            case .toolCall(let call): calls.append(call)
            default: break
            }
        }
        return (text, calls)
    }

    static let twoCalls = #"Checking. <TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_time", "arguments": "{\"tz\": \"CET\"}"}]</TOOLCALL>"#

    @Test("a <TOOLCALL> array becomes one tool call per entry, whole or streamed a character at a time")
    func toolCallArrayIsParsed() {
        for pieces in [[Self.twoCalls], Self.twoCalls.map(String.init)] {
            let (text, calls) = Self.scan(pieces)
            #expect(text == "Checking. ")
            #expect(calls.map(\.name) == ["get_weather", "get_time"])
            #expect(calls.map(\.arguments) == [#"{"city":"Paris"}"#, #"{"tz": "CET"}"#])
        }
    }

    @Test("a call cut off before </TOOLCALL> is still parsed at the end of the stream")
    func unterminatedCallIsParsedAtFinish() {
        let (text, calls) = Self.scan([#"<TOOLCALL>[{"name": "ls", "arguments": {}}]"#])
        #expect(text.isEmpty)
        #expect(calls.map(\.name) == ["ls"])
    }

    @Test("text that only looks like a tool call stays text")
    func nonCallsStayText() {
        let malformed = #"<TOOLCALL>[{"name": ]</TOOLCALL> and a < TOOL sign <TOOL"#
        #expect(Self.scan([malformed]).text == malformed)
        #expect(Self.scan([malformed]).calls.isEmpty)
    }

    @Test("a config.json that does not decode fails the load with the decoding error, not as corrupted weights")
    func undecodableConfigFailsTheLoad() async throws {
        let withoutHiddenSize = Self.denseNano4BConfig.replacingOccurrences(of: #""hidden_size": 3136,"#, with: "")
        let dir = try Self.modelDirectory(config: withoutHiddenSize)
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            try await MLXInferenceEngine().load(modelPath: dir.path)
            Issue.record("load succeeded")
        } catch {
            guard case .modelConfigurationInvalid(_, let reason) = error else {
                Issue.record("unexpected \(error)")
                return
            }
            #expect(reason.contains("hidden_size"))
            #expect(error.httpStatus == 422)
        }
    }

    @Test("only a config that does not decode or names a type no factory knows is an invalid configuration")
    func loadFailureMapping() async {
        func failure(_ error: any Error) async -> BasaltError {
            await MLXInferenceEngine.loadFailure(error, modelPath: "/m")
        }
        let decoding = DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "missing hidden_size"))
        guard case .modelConfigurationInvalid = await failure(ModelFactoryError.configurationDecodingError("config.json", "m", decoding)),
              case .modelConfigurationInvalid = await failure(ModelFactoryError.unsupportedModelType("no_such_arch")),
              case .weightsCorrupted = await failure(ModelFactoryError.unsupportedModelType("qwen2_vl")),
              case .weightsCorrupted = await failure(ModelFactoryError.invalidConfiguration("bad")),
              case .weightsCorrupted = await failure(CocoaError(.fileReadCorruptFile))
        else {
            Issue.record("a load failure was mapped to the wrong error")
            return
        }
    }

    @Test("only the Nemotron profile has its text scanned for <TOOLCALL> blocks")
    func onlyNemotronWritesToolCallsAsText() {
        #expect(NemotronProfile(switchesThinkingInConversation: true).writesToolCallsAsText)
        #expect(!QwenProfile().writesToolCallsAsText)
        #expect(!GemmaProfile().writesToolCallsAsText)
        #expect(!GenericProfile().writesToolCallsAsText)
    }

    @Test("with parallel tool calls off, nothing after the first admitted call of a <TOOLCALL> array is emitted")
    func singleCallStopsTheScannedStream() {
        let tools = [ToolDefinition(name: "get_weather"), ToolDefinition(name: "get_time")]
        var scanner = NemotronToolCallScanner()
        let scanned = scanner.feed(Self.twoCalls + " Done.") + scanner.finish()
        var single = ToolCallGate(request: GenerationRequest(prompt: "p", tools: tools, toolChoice: ToolChoice(parallel: false)))
        let emitted = single.admit(scanned)
        #expect(single.isSaturated)
        #expect(emitted.count == 2)
        guard case .text("Checking. ") = emitted.first, case .toolCall(let call) = emitted.last else {
            Issue.record("unexpected \(emitted)")
            return
        }
        #expect(call.name == "get_weather")
        var parallel = ToolCallGate(request: GenerationRequest(prompt: "p", tools: tools))
        #expect(parallel.admit(scanned).count == 4)
    }

    @Test("object arguments keep their slashes unescaped")
    func objectArgumentsKeepSlashes() {
        let (_, calls) = Self.scan([#"<TOOLCALL>[{"name": "read", "arguments": {"path": "/tmp/a.txt"}}]</TOOLCALL>"#])
        #expect(calls.map(\.arguments) == [#"{"path":"/tmp/a.txt"}"#])
    }

    @Test("a string argument that is not a JSON object becomes the input field instead of leaking the call as text")
    func freeTextArgumentBecomesInput() {
        let (text, calls) = Self.scan([#"<TOOLCALL>[{"name": "apply_patch", "arguments": "*** Begin Patch"}]</TOOLCALL>"#])
        #expect(text.isEmpty)
        #expect(calls.map(\.arguments) == [#"{"input":"*** Begin Patch"}"#])
    }

    @Test("null arguments are an empty object; an array or a scalar becomes the input field")
    func nonObjectArgumentsStayCalls() {
        let (text, calls) = Self.scan([
            #"<TOOLCALL>[{"name": "ls", "arguments": null}, {"name": "sum", "arguments": [1, 2]}, {"name": "sleep", "arguments": 5}]</TOOLCALL>"#,
        ])
        #expect(text.isEmpty)
        #expect(calls.map(\.arguments) == ["{}", #"{"input":[1,2]}"#, #"{"input":5}"#])
    }

    @Test("generation starts inside a thinking span exactly when the rendered prompt ends with an opener or the assistant prefix leaves one open")
    func renderedTailOpensThinking() {
        let longPrefix = String(repeating: "and then some more reasoning ", count: 20)
        let opened: [(rendered: String, prefix: String)] = [
            ("<SPECIAL_11>User\nWhat is 17*23?\n<SPECIAL_11>Assistant\n<think>\n", ""),
            ("<|im_start|>user\nWhat is 17*23?<|im_end|>\n<|im_start|>assistant\n<think>\n", ""),
            ("<|turn>user\nWhat is 17*23?<turn|>\n<|turn>model\n<|channel>thought\n", ""),
            ("<|start|>assistant<|channel|>thought", ""),
            ("<|im_start|>assistant\n", "<think>\nLet me"),
            ("<|im_start|>assistant\n<think>\n", "Let me"),
            ("<|im_start|>assistant\n<think>\n", longPrefix),
            ("<|im_start|>assistant\n", "<think>\nfirst</think>\n<think>\nagain"),
        ]
        let closed: [(rendered: String, prefix: String)] = [
            ("<SPECIAL_11>Assistant\n<think></think>", ""),
            ("<|im_start|>assistant\n<think></think>", ""),
            ("<|im_start|>assistant\n<think>\n\n</think>\n\n", ""),
            ("<|im_start|>user\nWhat is 17*23?<|im_end|>\n<|im_start|>assistant\n", ""),
            ("<|im_start|>user\nwhat does <think> mean<|im_end|>\n<|im_start|>assistant\n", ""),
            ("<|start_header_id|>user<|end_header_id|>\n\nwhat does <think> mean<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", ""),
            ("<|turn>user\nwhat does <|channel>thought mean<turn|>\n<|turn>model\n", ""),
            ("<|turn>model\n<|channel>thought\nshort<channel|>", ""),
            ("<|im_start|>assistant\n<think>\n", "done</think>\nThe answer"),
        ]
        #expect(opened.filter { !MLXInferenceEngine.endsInsideThinking($0.rendered, assistantPrefix: $0.prefix) }.map(\.rendered) == [])
        #expect(closed.filter { MLXInferenceEngine.endsInsideThinking($0.rendered, assistantPrefix: $0.prefix) }.map(\.rendered) == [])
    }

    @Test("the engine turns a Nemotron <TOOLCALL> into one structured call and stops generating once parallel calls are off, with and without a prompt cache")
    func engineScansNemotronCallsAndStopsAfterOne() async throws {
        let tools = [ToolDefinition(name: "get_weather"), ToolDefinition(name: "get_time")]
        let single = try await Self.scriptedChunks(
            NemotronProfile(switchesThinkingInConversation: false), tools: tools, toolChoice: ToolChoice(parallel: false))
        #expect(single.calls.map(\.name) == ["get_weather"])
        #expect(single.text == "Checking. ")
        #expect(single.completionTokens < ScriptedTokenizer.script.count)
        let cached = try await Self.scriptedChunks(
            NemotronProfile(switchesThinkingInConversation: false), tools: tools, toolChoice: ToolChoice(parallel: false), cacheKey: "t")
        #expect(cached.calls.map(\.name) == ["get_weather"])
        #expect(cached.text == "Checking. ")
        #expect(cached.completionTokens > 0 && cached.completionTokens < ScriptedTokenizer.script.count)
        let parallel = try await Self.scriptedChunks(NemotronProfile(switchesThinkingInConversation: false), tools: tools)
        #expect(parallel.calls.map(\.name) == ["get_weather", "get_time"])
        let generic = try await Self.scriptedChunks(GenericProfile(), tools: tools)
        #expect(generic.calls.isEmpty)
        #expect(generic.text.hasPrefix("Checking. <TOOLCALL>"))
    }

    struct ScriptedTokenizer: MLXLMCommon.Tokenizer {
        static let script = #"Checking. <TOOLCALL>[{"name": "get_weather", "arguments": {}}, {"name": "get_time", "arguments": {}}]</TOOLCALL> Done, and a long tail that a single-call request never generates."#

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { ByteTokenizer().encode(text: text, addSpecialTokens: addSpecialTokens) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { String(Self.script.prefix(tokenIds.count)) }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try ByteTokenizer().applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
        }
    }

    static func scriptedChunks(
        _ profile: any ModelProfile, tools: [ToolDefinition], toolChoice: ToolChoice? = nil, cacheKey: String? = nil
    ) async throws -> (text: String, calls: [ToolCallOutput], completionTokens: Int) {
        let engine = MLXInferenceEngine()
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "scripted"), model: try TinyModel.make(),
            processor: ByteInputProcessor(), tokenizer: ScriptedTokenizer())), profile: profile)
        let request = GenerationRequest(
            prompt: "weather and time", maxTokens: ScriptedTokenizer.script.count + 20, temperature: 0,
            tools: tools, toolChoice: toolChoice, cacheKey: cacheKey)
        var text = ""
        var calls: [ToolCallOutput] = []
        var completionTokens = 0
        for await chunk in try await engine.generateStream(request: request) {
            switch chunk {
            case .text(let piece): text += piece
            case .toolCall(let call): calls.append(call)
            case .metadata(let response): completionTokens = response.completionTokens
            default: break
            }
        }
        return (text, calls, completionTokens)
    }

    @Test("chat_template.json sits between chat_template.jinja and tokenizer_config.json")
    func chatTemplateSourcePrecedence() throws {
        let dir = try Self.modelDirectory(config: Self.denseNano4BConfig)
        defer { try? FileManager.default.removeItem(at: dir) }
        try JSONSerialization.data(withJSONObject: ["chat_template": "from tokenizer_config"])
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))
        #expect(ModelProfileRegistry.chatTemplate(at: dir) == "from tokenizer_config")
        try JSONSerialization.data(withJSONObject: ["chat_template": "from chat_template.json"])
            .write(to: dir.appendingPathComponent("chat_template.json"))
        #expect(ModelProfileRegistry.chatTemplate(at: dir) == "from chat_template.json")
        try Data("from chat_template.jinja".utf8).write(to: dir.appendingPathComponent("chat_template.jinja"))
        #expect(ModelProfileRegistry.chatTemplate(at: dir) == "from chat_template.jinja")
    }
}

