import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXVLM
import MLXLMCommon
import Jinja
import BasaltCore

package struct ModelDeepCheck: Codable, Sendable {
    package let repoId: String
    package let folder: String?
    package let modelType: String?
    package let shards: Int
    package let tensors: Int
    package let bytesFetched: Int
    package let load: [String]
    package let template: [String]

    package var passed: Bool { load.isEmpty && template.isEmpty }

    package static func run(repoId: String, hub: HuggingFaceHub) async throws -> ModelDeepCheck {
        let files = Set(try await hub.modelDetails(repoId: repoId).siblings?.map(\.rfilename) ?? [])
        guard let prefix = ([""] + ModelCache.quantFolders.map { $0 + "/" }).first(where: { files.contains($0 + "config.json") }) else {
            return ModelDeepCheck(
                repoId: repoId, folder: nil, modelType: nil, shards: 0, tensors: 0, bytesFetched: await hub.bytesFetched,
                load: ["config.json not found in \(repoId)"], template: [])
        }
        func fetch(_ name: String) async throws -> Data? {
            files.contains(prefix + name) ? try await hub.fetch(repoId: repoId, path: prefix + name).data : nil
        }

        let configuration = try await fetch("config.json") ?? Data()
        let index = try await fetch("model.safetensors.index.json")
        let weightMap = index.flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] }?["weight_map"] as? [String: String]
        let shards = weightFiles(in: files, prefix: prefix, weightMap: weightMap)
        let headers = try await withThrowingTaskGroup(of: (String, Result<ModelInspector.SafetensorsHeader, ModelInspector.ShardProblem>).self) { group in
            for shard in shards {
                group.addTask { (shard, try await hub.safetensorsHeader(repoId: repoId, path: shard)) }
            }
            return try await group.reduce(into: [:]) { $0[$1.0] = $1.1 }
        }

        var load = shards.isEmpty ? ["no .safetensors weights in \(repoId)"] : []
        if index != nil, weightMap == nil { load.append("model.safetensors.index.json has no readable weight_map") }
        var tensors: [String: ModelInspector.Tensor] = [:]
        var metadata: [String: String] = [:]
        for shard in shards {
            switch headers[shard] {
            case .success(let header)?:
                tensors.merge(header.tensors) { _, new in new }
                if metadata.isEmpty { metadata = header.metadata }
            case .failure(let problem)?:
                load.append("\(shard): \(problem.message)")
            case nil:
                load.append("\(shard): header not fetched")
            }
        }
        if load.isEmpty {
            load = await loadProblems(configuration: configuration, tensors: tensors, metadata: metadata)
        }

        let template = templateProblems(
            tokenizerConfig: try await fetch("tokenizer_config.json"),
            chatTemplateJinja: try await fetch("chat_template.jinja"),
            chatTemplateJSON: try await fetch("chat_template.json"))

        return ModelDeepCheck(
            repoId: repoId, folder: prefix.isEmpty ? nil : String(prefix.dropLast()),
            modelType: try? JSONDecoder.json5().decode(BaseConfiguration.self, from: configuration).modelType,
            shards: shards.count, tensors: tensors.count, bytesFetched: await hub.bytesFetched,
            load: load, template: template)
    }

    static func weightFiles(in files: Set<String>, prefix: String, weightMap: [String: String]?) -> [String] {
        let indexed = Set(weightMap?.values.map { prefix + $0 } ?? []).sorted()
        if !indexed.isEmpty, indexed.allSatisfy(files.contains) { return indexed }
        let present = files.filter { $0.hasPrefix(prefix) && $0.hasSuffix(".safetensors") && !$0.dropFirst(prefix.count).contains("/") }.sorted()
        for name in ["model", "weight"] {
            let matches = present.filter { $0.dropFirst(prefix.count).hasPrefix(name) }
            if !matches.isEmpty { return matches }
        }
        return present
    }

    static func loadProblems(configuration: Data, tensors: [String: ModelInspector.Tensor], metadata: [String: String]) async -> [String] {
        await MLXInferenceEngine.registerAliases()
        let base: BaseConfiguration
        do {
            base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configuration)
        } catch {
            return ["config.json: \(MLXInferenceEngine.briefly(error))"]
        }
        var candidates: [[String]] = []
        for registry in [VLMTypeRegistry.shared, LLMTypeRegistry.shared] {
            guard await registry.contains(base.modelType) else { continue }
            do {
                let model = try await registry.createModel(configuration: configuration, modelType: base.modelType)
                let problems = parameterProblems(model, tensors: tensors, metadata: metadata, quantization: base.perLayerQuantization)
                if problems.isEmpty { return [] }
                candidates.append(problems)
            } catch {
                candidates.append(["config.json: \(MLXInferenceEngine.briefly(error))"])
            }
        }
        return candidates.min { $0.count < $1.count } ?? ["model_type '\(base.modelType)' is not supported by this build"]
    }

    static func parameterProblems(
        _ model: LanguageModel, tensors: [String: ModelInspector.Tensor], metadata: [String: String],
        quantization: BaseConfiguration.PerLayerQuantization?
    ) -> [String] {
        let weights = model.sanitize(weights: tensors.mapValues { MLXArray.zeros($0.shape, dtype: dtype($0.dtype)) }, metadata: metadata)
        if let quantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] == nil ? nil : quantization.quantization(layer: path)?.asTuple
            }
        }
        let expected = Dictionary(model.parameters().flattened().map { ($0.0, $0.1.shape) }) { first, _ in first }
        do {
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            return []
        } catch {
            let missing = expected.keys.filter { weights[$0] == nil }.sorted().map { "missing \($0)" }
            let unexpected = weights.keys.filter { expected[$0] == nil }.sorted().map { "unexpected \($0)" }
            let mismatched = expected.compactMap { key, shape in
                weights[key].flatMap { $0.shape == shape ? nil : "shape mismatch \(key): checkpoint \($0.shape), model \(shape)" }
            }.sorted()
            let diff = missing + unexpected + mismatched
            return diff.isEmpty ? [MLXInferenceEngine.briefly(error)] : diff
        }
    }

    static func dtype(_ safetensors: String) -> DType {
        switch safetensors {
        case "BOOL": .bool
        case "I8": .int8
        case "I16": .int16
        case "I32": .int32
        case "I64": .int64
        case "U16": .uint16
        case "U32": .uint32
        case "U64": .uint64
        case "F16": .float16
        case "BF16": .bfloat16
        case "F32": .float32
        case "F64": .float64
        default: .uint8
        }
    }

    static let cannedTools = [ToolDefinition(
        name: "get_weather", description: "Get the weather for a city",
        parametersJSON: #"{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}"#)]

    static let cannedTurns = [
        ChatMessageInput(role: .system, text: "You are a coding agent."),
        ChatMessageInput(role: .user, text: "Weather in Paris?"),
        ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "call_1", name: "get_weather", arguments: #"{"city": "Paris"}"#)]),
        ChatMessageInput(role: .tool, text: #"{"temp": 21}"#, toolCallId: "call_1", toolName: "get_weather"),
    ]

    static func templateProblems(tokenizerConfig: Data?, chatTemplateJinja: Data?, chatTemplateJSON: Data?) -> [String] {
        func object(_ data: Data?) -> [String: Any] {
            data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        }
        var config = object(tokenizerConfig)
        if let jinja = chatTemplateJinja.flatMap({ String(data: $0, encoding: .utf8) }) {
            config["chat_template"] = jinja
        } else if let fromJSON = object(chatTemplateJSON)["chat_template"] {
            config["chat_template"] = fromJSON
        }
        let named = (config["chat_template"] as? [[String: Any]] ?? []).reduce(into: [String: String]()) { named, entry in
            if let name = entry["name"] as? String, let template = entry["template"] as? String { named[name] = template }
        }
        guard let template = config["chat_template"] as? String ?? named["tool_use"] ?? named["default"] else {
            return ["no chat template in tokenizer_config.json, chat_template.jinja or chat_template.json"]
        }

        let tools = MLXInferenceEngine.toolSpecs(cannedTools)
        let messages = DefaultMessageGenerator().generate(
            from: UserInput(chat: MLXInferenceEngine.chat(cannedTurns, lastText: cannedTurns[cannedTurns.count - 1].text), tools: tools))
        do {
            var context: [String: Value] = [
                "messages": try Value(any: messages), "tools": try Value(any: tools), "add_generation_prompt": .boolean(true),
            ]
            for key in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "additional_special_tokens"] {
                switch config[key] {
                case let token as String: context[key] = .string(token)
                case let token as [String: Any]: context[key] = (token["content"] as? String).map(Value.string)
                case let tokens as [String]: context[key] = .array(tokens.map(Value.string))
                default: break
                }
            }
            _ = try Template(ChatTemplateCompatibility.compatible(template), with: .init(lstripBlocks: true, trimBlocks: true)).render(context)
            return []
        } catch {
            return ["chat template: \((error as? LocalizedError)?.errorDescription ?? MLXInferenceEngine.briefly(error))"]
        }
    }
}
