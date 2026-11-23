import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct PEFTLoRAConfiguration: Decodable, Equatable {
    var peftType: String
    var rank: Int
    var alpha: Float
    var rsLoRA: Bool
    var dora: Bool
    var modulesToSave: [String]?
    var rankPattern: [String: Int]?
    var alphaPattern: [String: Float]?
    var initialization: String?

    enum CodingKeys: String, CodingKey {
        case peftType = "peft_type", rank = "r", alpha = "lora_alpha", rsLoRA = "use_rslora", dora = "use_dora"
        case modulesToSave = "modules_to_save", rankPattern = "rank_pattern", alphaPattern = "alpha_pattern", initialization = "init_lora_weights"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        peftType = try c.decode(String.self, forKey: .peftType)
        (rank, alpha) = (try c.decode(Int.self, forKey: .rank), try c.decode(Float.self, forKey: .alpha))
        rsLoRA = try c.decodeIfPresent(Bool.self, forKey: .rsLoRA) ?? false
        dora = try c.decodeIfPresent(Bool.self, forKey: .dora) ?? false
        modulesToSave = try c.decodeIfPresent([String].self, forKey: .modulesToSave)
        rankPattern = try c.decodeIfPresent([String: Int].self, forKey: .rankPattern)
        alphaPattern = try c.decodeIfPresent([String: Float].self, forKey: .alphaPattern)
        initialization = try? c.decodeIfPresent(String.self, forKey: .initialization)
    }

    static let baseChangingInitializations = ["pissa", "olora", "corda", "loftq"]

    var scale: Float { rsLoRA ? alpha / Float(rank).squareRoot() : alpha / Float(rank) }

    var refusal: String? {
        if peftType.uppercased() != "LORA" { return "peft_type \(peftType) is not LORA" }
        if dora { return "use_dora is set; DoRA adapters are not supported" }
        if !(modulesToSave ?? []).isEmpty { return "modules_to_save \(modulesToSave ?? []) would replace whole modules, which a LoRA adapter cannot carry" }
        if !(rankPattern ?? [:]).isEmpty || !(alphaPattern ?? [:]).isEmpty { return "rank_pattern or alpha_pattern gives layers their own rank or alpha, which is not supported" }
        if let initialization, Self.baseChangingInitializations.contains(where: initialization.lowercased().hasPrefix) {
            return "init_lora_weights \(initialization) trains against a changed base model; convert it to a plain LoRA with PEFT first"
        }
        return rank > 0 && alpha > 0 && alpha.isFinite ? nil : "r \(rank) and lora_alpha \(alpha) must be positive and finite"
    }
}

enum PEFTLoRAReader {
    struct Refused: Error, Equatable {
        let reason: String
    }

    static let weightsFile = "adapter_model.safetensors"

    static func isPEFT(_ directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appending(component: "adapter_config.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return object["peft_type"] != nil
    }

    static func container(at directory: URL, for model: Module) throws(Refused) -> LoRAContainer {
        let configuration: PEFTLoRAConfiguration
        let raw: [String: MLXArray]
        do {
            configuration = try JSONDecoder().decode(PEFTLoRAConfiguration.self, from: Data(contentsOf: directory.appending(component: "adapter_config.json")))
        } catch {
            throw Refused(reason: "adapter_config.json is not a PEFT LoRA config: \(error)")
        }
        do {
            raw = try MLX.loadArrays(url: directory.appending(component: weightsFile))
        } catch {
            throw Refused(reason: "\(weightsFile) is unreadable: \(error.localizedDescription)")
        }
        if let refusal = configuration.refusal { throw Refused(reason: refusal) }
        guard let lora = model as? LoRAModel else { throw Refused(reason: "the \(type(of: model)) architecture takes no LoRA adapter") }
        let linears = Dictionary(model.namedModules().compactMap { path, module in (module as? Linear).map { (path, $0) } }, uniquingKeysWith: { first, _ in first })
        let (parameters, keys) = try mapped(raw, linears: linears)
        let loraConfiguration = LoRAConfiguration(
            numLayers: lora.loraLayers.count, fineTuneType: .lora,
            loraParameters: .init(rank: configuration.rank, scale: configuration.scale, keys: keys))
        return LoRAContainer(configuration: loraConfiguration, parameters: ModuleParameters.unflattened(parameters))
    }

    static func mapped(_ raw: [String: MLXArray], linears: [String: Linear]) throws(Refused) -> ([String: MLXArray], [String]) {
        var parameters = [String: MLXArray]()
        var keys = Set<String>()
        for (key, tensor) in raw {
            let side: (suffix: String, name: String)
            if key.hasSuffix(".lora_A.weight") { side = (".lora_A.weight", "lora_a") }
            else if key.hasSuffix(".lora_B.weight") { side = (".lora_B.weight", "lora_b") }
            else { throw Refused(reason: "\(key) is neither a lora_A nor a lora_B weight") }
            let peftPath = String(key.dropLast(side.suffix.count).trimmingPrefix("base_model.model."))
            guard let path = modulePath(peftPath, in: linears), let linear = linears[path], let relative = layerRelative(path) else {
                throw Refused(reason: "\(key) belongs to no linear layer of this model")
            }
            let dtype = computeType(of: linear)
            parameters["\(path).\(side.name)"] = tensor.transposed().asType(dtype)
            keys.insert(relative)
        }
        let paths = Set(parameters.keys.map { $0.split(separator: ".").dropLast().joined(separator: ".") })
        if let unpaired = paths.first(where: { parameters["\($0).lora_a"] == nil || parameters["\($0).lora_b"] == nil }) {
            throw Refused(reason: "\(unpaired) has only one of lora_A and lora_B")
        }
        guard !parameters.isEmpty else { throw Refused(reason: "\(weightsFile) holds no LoRA weights") }
        return (parameters, keys.sorted())
    }

    static func modulePath(_ peftPath: String, in linears: [String: Linear]) -> String? {
        let rewrites: [(from: String, to: String)] = [
            ("", ""), ("model.language_model.", "language_model.model."), ("model.language_model.", "model."),
            ("model.language_model.", "language_model."), ("model.", "language_model.model."),
        ]
        return rewrites.lazy.compactMap { rewrite -> String? in
            guard peftPath.hasPrefix(rewrite.from) else { return nil }
            let candidate = rewrite.to + peftPath.dropFirst(rewrite.from.count)
            return linears[candidate] == nil ? nil : candidate
        }.first
    }

    static func layerRelative(_ path: String) -> String? {
        let parts = path.split(separator: ".")
        guard let index = parts.indices.dropLast(2).first(where: { parts[$0] == "layers" && Int(parts[$0 + 1]) != nil }) else { return nil }
        return parts[(index + 2)...].joined(separator: ".")
    }

    static func computeType(of linear: Linear) -> DType {
        guard let quantized = linear as? QuantizedLinear else { return linear.weight.dtype }
        let floats: Set<DType> = [.float16, .bfloat16, .float32]
        return [quantized.biases?.dtype, quantized.scales.dtype].compactMap { $0 }.first(where: floats.contains) ?? .bfloat16
    }
}
