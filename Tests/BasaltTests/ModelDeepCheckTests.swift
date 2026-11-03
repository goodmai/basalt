import Foundation
import Testing
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("models check --deep: header-only fetch, shape-only build, template render (#227)", .serialized)
struct ModelDeepCheckTests {
    static let tinyConfig = #"""
        {"model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
         "vocab_size": 256, "rope_theta": 10000, "tie_word_embeddings": true, "max_position_embeddings": 512}
        """#

    static let qwenTemplate = """
        {%- for m in messages %}<|im_start|>{{ m.role }}
        {{ m.content }}<|im_end|>
        {% endfor %}{%- if add_generation_prompt %}<|im_start|>assistant
        {% endif %}
        """

    static func tinyTensors(config: String = tinyConfig) async throws -> [String: ModelInspector.Tensor] {
        let model = try await LLMTypeRegistry.shared.createModel(configuration: Data(config.utf8), modelType: "qwen3")
        return Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { name, array in
            (name, ModelInspector.Tensor(dtype: "F32", shape: array.shape, end: 0))
        })
    }

    static func header(_ tensors: [String: ModelInspector.Tensor]) -> (prefix: Data, json: Data, fileSize: Int64) {
        var offset = 0
        var entries: [String: Any] = ["__metadata__": ["format": "mlx"]]
        for (name, tensor) in tensors.sorted(by: { $0.key < $1.key }) {
            let size = tensor.shape.reduce(4, *)
            entries[name] = ["dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, offset + size]]
            offset += size
        }
        let json = try! JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
        let length = UInt64(json.count)
        return (Data((0 ..< 8).map { UInt8((length >> ($0 * 8)) & 0xff) }), json, Int64(8 + json.count + offset))
    }

    @Test("A checkpoint whose tensors match the model passes")
    func matchingCheckpoint() async throws {
        let problems = await ModelDeepCheck.loadProblems(configuration: Data(Self.tinyConfig.utf8), tensors: try await Self.tinyTensors(), metadata: [:])
        #expect(problems == [])
    }

    @Test("A renamed tensor is reported as the missing key and the unexpected one")
    func renamedTensor() async throws {
        var tensors = try await Self.tinyTensors()
        tensors["model.final_norm.weight"] = tensors.removeValue(forKey: "model.norm.weight")
        let problems = await ModelDeepCheck.loadProblems(configuration: Data(Self.tinyConfig.utf8), tensors: tensors, metadata: [:])
        #expect(problems == ["missing model.norm.weight", "unexpected model.final_norm.weight"])
    }

    @Test("A tensor of the wrong shape is reported with both shapes")
    func shapeMismatch() async throws {
        var tensors = try await Self.tinyTensors()
        tensors["model.norm.weight"] = ModelInspector.Tensor(dtype: "F32", shape: [32], end: 0)
        let problems = await ModelDeepCheck.loadProblems(configuration: Data(Self.tinyConfig.utf8), tensors: tensors, metadata: [:])
        #expect(problems == ["shape mismatch model.norm.weight: checkpoint [32], model [64]"])
    }

    @Test("Weights quantized at another bit width than config.json says do not load")
    func quantizationMismatch() async throws {
        let fourBit = Self.tinyConfig.replacingOccurrences(of: #""model_type""#, with: #""quantization": {"group_size": 32, "bits": 4}, "model_type""#)
        let model = try await LLMTypeRegistry.shared.createModel(configuration: Data(fourBit.utf8), modelType: "qwen3")
        quantize(model: model, groupSize: 32, bits: 8)
        let eightBitTensors = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { name, array in
            (name, ModelInspector.Tensor(dtype: "F32", shape: array.shape, end: 0))
        })
        let problems = await ModelDeepCheck.loadProblems(configuration: Data(fourBit.utf8), tensors: eightBitTensors, metadata: [:])
        #expect(problems.contains("shape mismatch model.layers.0.mlp.down_proj.weight: checkpoint [64, 32], model [64, 16]"))
    }

    @Test("An unsupported model_type is named")
    func unsupportedModelType() async throws {
        let config = Self.tinyConfig.replacingOccurrences(of: #""qwen3""#, with: #""no_such_arch""#)
        let problems = await ModelDeepCheck.loadProblems(configuration: Data(config.utf8), tensors: try await Self.tinyTensors(), metadata: [:])
        #expect(problems == ["model_type 'no_such_arch' is not supported by this build"])
    }

    @Test("A template that renders passes; chat_template.jinja wins over tokenizer_config.json")
    func templateRenders() throws {
        let config = Data(#"{"eos_token": "<|im_end|>", "chat_template": "{{ broken"}"#.utf8)
        #expect(ModelDeepCheck.templateProblems(tokenizerConfig: config, chatTemplateJinja: Data(Self.qwenTemplate.utf8), chatTemplateJSON: nil) == [])
        #expect(ModelDeepCheck.templateProblems(tokenizerConfig: nil, chatTemplateJinja: nil, chatTemplateJSON: Data(#"{"chat_template": "{{ eos_token }}"}"#.utf8)) == [])
    }

    @Test("A broken or missing template is reported")
    func brokenTemplate() throws {
        let broken = ModelDeepCheck.templateProblems(
            tokenizerConfig: nil, chatTemplateJinja: Data("{% for m in messages %}{{ m.content }".utf8), chatTemplateJSON: nil)
        #expect(broken.count == 1 && broken[0].hasPrefix("chat template: "))
        let raising = ModelDeepCheck.templateProblems(
            tokenizerConfig: nil, chatTemplateJinja: Data("{{ raise_exception('tools are not supported') }}".utf8), chatTemplateJSON: nil)
        #expect(raising.count == 1 && raising[0].contains("tools are not supported"))
        #expect(ModelDeepCheck.templateProblems(tokenizerConfig: Data("{}".utf8), chatTemplateJinja: nil, chatTemplateJSON: nil)
            == ["no chat template in tokenizer_config.json, chat_template.jinja or chat_template.json"])
    }

    @Test("Gaps and overlaps between tensor data offsets are refused", arguments: [
        ([[0, 8], [16, 24]], "Tensor data of 'b' starts at 16, expected 8 (gap or overlap)"),
        ([[0, 8], [4, 12]], "Tensor data of 'b' starts at 4, expected 8 (gap or overlap)"),
        ([[8, 16], [16, 24]], "Tensor data of 'a' starts at 8, expected 0 (gap or overlap)"),
    ])
    func offsetsMustBeContiguous(offsets: [[Int]], message: String) throws {
        let json = try JSONSerialization.data(withJSONObject: [
            "a": ["dtype": "U8", "shape": [8], "data_offsets": offsets[0]],
            "b": ["dtype": "U8", "shape": [8], "data_offsets": offsets[1]],
        ])
        let result = ModelInspector.parseSafetensorsHeader(json, fileSize: Int64(8 + json.count + 24))
        guard case .failure(let problem) = result else { Issue.record("accepted \(offsets)"); return }
        #expect(problem.message == message)
    }

    struct Repo {
        let host: String
        let hub: HuggingFaceHub
    }

    static func stubRepo(tensors: [String: ModelInspector.Tensor], reportedSize: (Int64) -> Int64 = { $0 }, template: String = qwenTemplate) -> Repo {
        let (prefix, json, fileSize) = header(tensors)
        let headerBytes = prefix + json
        let total = reportedSize(fileSize)
        let siblings = ["config.json", "model.safetensors", "tokenizer_config.json", "chat_template.jinja", "README.md"]
        let host = StubURLProtocol.host { request, _ in
            let path = request.url?.path ?? ""
            switch path {
            case "/api/models/org/tiny":
                let body = try! JSONSerialization.data(withJSONObject: ["id": "org/tiny", "siblings": siblings.map { ["rfilename": $0] }])
                return (200, [:], body)
            case "/org/tiny/resolve/main/config.json":
                return (200, [:], Data(tinyConfig.utf8))
            case "/org/tiny/resolve/main/tokenizer_config.json":
                return (200, [:], Data(#"{"eos_token": "<|im_end|>"}"#.utf8))
            case "/org/tiny/resolve/main/chat_template.jinja":
                return (200, [:], Data(template.utf8))
            case "/org/tiny/resolve/main/model.safetensors":
                let range = request.value(forHTTPHeaderField: "Range")?.dropFirst("bytes=".count).split(separator: "-").compactMap { Int($0) }
                guard let range, range.count == 2, range[1] < headerBytes.count else { return (500, [:], Data()) }
                return (206, ["Content-Range": "bytes \(range[0])-\(range[1])/\(total)"], headerBytes.subdata(in: range[0] ..< range[1] + 1))
            default:
                return (404, [:], Data())
            }
        }
        return Repo(host: host, hub: HuggingFaceHub(configuration: StubURLProtocol.configuration, endpoint: "https://\(host)"))
    }

    @Test("A sound repo passes from its headers alone: every weight request is a small Range request")
    func soundRepo() async throws {
        let tensors = try await Self.tinyTensors()
        let repo = Self.stubRepo(tensors: tensors)
        let check = try await ModelDeepCheck.run(repoId: "org/tiny", hub: repo.hub)
        #expect(check.passed, "\(check.load) \(check.template)")
        #expect(check.modelType == "qwen3" && check.shards == 1 && check.tensors == tensors.count)

        let weightRequests = StubURLProtocol.requests(for: repo.host).map(\.0).filter { $0.url?.path.hasSuffix(".safetensors") == true }
        let headerLength = Self.header(tensors).json.count
        #expect(weightRequests.compactMap { $0.value(forHTTPHeaderField: "Range") } == ["bytes=0-7", "bytes=8-\(7 + headerLength)"])
        #expect(!StubURLProtocol.requests(for: repo.host).contains { $0.0.url?.path.hasSuffix("README.md") == true })
    }

    @Test("A shard shorter than its header says is refused, naming the shard, and a broken template is reported")
    func truncatedShardAndBrokenTemplate() async throws {
        let repo = Self.stubRepo(tensors: try await Self.tinyTensors(), reportedSize: { $0 - 4 }, template: "{% if %}")
        let check = try await ModelDeepCheck.run(repoId: "org/tiny", hub: repo.hub)
        #expect(!check.passed)
        #expect(check.load.count == 1 && check.load[0].hasPrefix("model.safetensors: Truncated or oversized shard"))
        #expect(check.template.count == 1 && check.template[0].hasPrefix("chat template: "))
    }

    @Test("The index picks the shards when it names files the repo has, the model*.safetensors convention otherwise")
    func weightFileSelection() {
        let index = ["a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"]
        let sharded: Set = ["4bit/config.json", "4bit/model-00001-of-00002.safetensors", "4bit/model-00002-of-00002.safetensors", "4bit/optiq/mtp.safetensors"]
        #expect(ModelDeepCheck.weightFiles(in: sharded, prefix: "4bit/", weightMap: index)
            == ["4bit/model-00001-of-00002.safetensors", "4bit/model-00002-of-00002.safetensors"])
        let staleIndex: Set = ["model.safetensors", "consolidated.safetensors", "sub/model.safetensors"]
        #expect(ModelDeepCheck.weightFiles(in: staleIndex, prefix: "", weightMap: index) == ["model.safetensors"])
    }
}
