import Testing
import Foundation
import MLXLLM
import MLXLMCommon
import Jinja
import Hub
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

    @Test("only the loop binding call is renamed, text and other names are left alone")
    func renameIsScopedToTheLoop() {
        let template = "{{ call }}{% for call in xs %}{{ call.name ~ 'call' ~ x.call }}{% endfor %}{{ call }} call"
        #expect(ChatTemplateCompatibility.renamingReservedLoopVariables(template)
            == "{{ call }}{% for call_item in xs %}{{ call_item.name ~ 'call' ~ x.call }}{% endfor %}{{ call }} call")
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
}
