import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine: ModelResearch {
    public func anatomy() async throws(BasaltError) -> ModelAnatomy {
        guard let container else { throw .modelNotLoaded }
        let scan = await container.perform { context in
            AnatomyScan(model: context.model, caches: (try? context.model.newCache(parameters: nil)) ?? [])
        }
        let directory = modelDescription.isEmpty ? nil : URL(fileURLWithPath: modelDescription)
        return scan.anatomy(config: directory.flatMap(ModelCache.config(at:)) ?? [:])
    }
}

enum LayerMixer: String, Sendable {
    case mamba
    case linearAttention = "linear attention"
    case conv
    case attention

    init?(modules: String) {
        if modules.contains("mamba") {
            self = .mamba
        } else if modules.contains("gateddelta") || modules.contains("linearattention") {
            self = .linearAttention
        } else if modules.contains("conv") {
            self = .conv
        } else if modules.contains("attention") {
            self = .attention
        } else {
            return nil
        }
    }
}

struct AnatomyScan: Sendable {
    struct Layer: Sendable, Equatable {
        var mixer: LayerMixer?
        var experts: Int?
        var cache: String?
        var window: Int?
    }

    var layers: [Layer]
    var parameterCount: Int
    var bytes: ByteCount

    init(model: any LanguageModel, caches: [any KVCache]) {
        let parameters = model.parameters().flattened()
        var modules: [Int: String] = [:]
        var groupSizes: [String: Int] = [:]
        for (path, module) in model.namedModules() {
            if let quantized = module as? any Quantized { groupSizes[path] = quantized.groupSize }
            if let index = Self.layerIndex(path) {
                modules[index, default: ""] += String(describing: type(of: module)).lowercased() + " "
            }
        }
        var experts: [Int: Int] = [:]
        for (path, array) in parameters where array.ndim == 3 && path.contains("switch") {
            if let index = Self.layerIndex(path) { experts[index] = array.dim(0) }
        }
        var layers = (0 ..< ((modules.keys.max() ?? -1) + 1)).map {
            Layer(mixer: modules[$0].flatMap(LayerMixer.init(modules:)), experts: experts[$0])
        }
        Self.assign(caches, to: &layers)
        self.layers = layers
        parameterCount = Self.parameterCount(parameters, groupSizes: groupSizes)
        bytes = ByteCount(parameters.reduce(0) { $0 + $1.1.nbytes })
    }

    static func layerIndex(_ path: String) -> Int? {
        let parts = path.split(separator: ".")
        guard let at = parts.firstIndex(of: "layers"), at + 1 < parts.count else { return nil }
        return Int(parts[at + 1])
    }

    static func assign(_ caches: [any KVCache], to layers: inout [Layer]) {
        let owners = caches.count == layers.count ? Array(layers.indices) : layers.indices.filter { layers[$0].mixer != nil }
        for (index, cache) in zip(owners, caches) {
            layers[index].cache = PromptCacheDiskStore.layoutName(cache)
            layers[index].window = cache.maxSize
        }
    }

    static func parameterCount(_ parameters: [(String, MLXArray)], groupSizes: [String: Int]) -> Int {
        let byName = Dictionary(parameters, uniquingKeysWith: { first, _ in first })
        return parameters.reduce(0) { total, entry in
            let owner = entry.0.split(separator: ".").dropLast().joined(separator: ".")
            guard let group = groupSizes[owner] else { return total + entry.1.size }
            if entry.0.hasSuffix(".weight"), let scales = byName[owner + ".scales"] { return total + scales.size * group }
            return entry.0.hasSuffix(".scales") || entry.0.hasSuffix(".biases") ? total : total + entry.1.size
        }
    }

    func anatomy(config json: [String: Any]) -> ModelAnatomy {
        let config = AnatomyConfig(json: json)
        var anatomy = ModelAnatomy(
            modelType: config.string("model_type") ?? "unknown",
            architecture: (json["architectures"] as? [String])?.first ?? "unknown",
            layers: layers.indices.map { config.layer($0, layers[$0]) },
            parameterCount: parameterCount,
            bytes: bytes)
        anatomy.vocabulary = config.int("vocab_size") ?? 0
        anatomy.hiddenSize = config.int("hidden_size") ?? 0
        anatomy.quantization = json.isEmpty ? nil : architectureLines(config: json)
            .first { $0.hasPrefix(Self.quantizationPrefix) }.map { String($0.dropFirst(Self.quantizationPrefix.count)) }
        return anatomy
    }

    static let quantizationPrefix = "Quantization: "
}

struct AnatomyConfig {
    let json: [String: Any]

    func int(_ key: String) -> Int? {
        (json["text_config"] as? [String: Any])?[key] as? Int ?? json[key] as? Int
    }

    func string(_ key: String) -> String? {
        json[key] as? String ?? (json["text_config"] as? [String: Any])?[key] as? String
    }

    func slides(_ index: Int) -> Bool {
        let types = (json["text_config"] as? [String: Any])?["layer_types"] as? [String] ?? json["layer_types"] as? [String]
        return types.map { $0.indices.contains(index) && $0[index] == "sliding_attention" } ?? false
    }

    func layer(_ index: Int, _ scan: AnatomyScan.Layer) -> LayerAnatomy {
        var layer = LayerAnatomy(index: index, kind: kind(index, scan), cache: scan.cache ?? "none")
        layer.window = scan.window
        layer.experts = scan.experts
        layer.expertsPerToken = scan.experts == nil ? nil : int("num_experts_per_tok") ?? int("top_k_experts") ?? int("moe_topk")
        switch scan.mixer {
        case .attention?:
            attention(&layer, global: scan.window == nil && !slides(index))
        case .linearAttention?:
            (layer.heads, layer.kvHeads, layer.headDim) = (int("linear_num_value_heads"), int("linear_num_key_heads"), int("linear_key_head_dim"))
        case .mamba?:
            (layer.heads, layer.headDim) = (int("mamba_num_heads"), int("mamba_head_dim"))
        case .conv?, nil:
            break
        }
        return layer
    }

    func kind(_ index: Int, _ scan: AnatomyScan.Layer) -> String {
        switch scan.mixer {
        case .attention?: scan.window != nil || slides(index) ? "sliding window" : "full attention"
        case let mixer?: mixer.rawValue
        case nil: scan.experts == nil ? "mlp" : "moe"
        }
    }

    func attention(_ layer: inout LayerAnatomy, global: Bool) {
        let heads = int("num_attention_heads")
        layer.heads = heads
        layer.kvHeads = (global ? int("num_global_key_value_heads") : nil) ?? int("num_key_value_heads") ?? heads
        let derived = heads.flatMap { heads in heads > 0 ? int("hidden_size").map { $0 / heads } : nil }
        layer.headDim = (global ? int("global_head_dim") : nil) ?? int("head_dim") ?? derived
        if !global, layer.window == nil { layer.window = int("sliding_window") }
    }
}
