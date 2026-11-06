import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXVLM
import MLXLMCommon
import BasaltCore

struct SafetensorsShard {
    let url: URL
    let headerLength: Int
    let fileSize: Int64
    let entries: [String: Any]

    init(url: URL) throws(BasaltError) {
        self.url = url
        let size = (try? FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)[.size] as? Int64) ?? 0
        guard let handle = try? FileHandle(forReadingFrom: url), let prefix = try? handle.read(upToCount: 8),
              let length = ModelInspector.safetensorsHeaderLength(prefix, fileSize: size),
              let header = try? handle.read(upToCount: length), header.count == length,
              let entries = (try? JSONSerialization.jsonObject(with: header)) as? [String: Any] else {
            throw .inferenceError("expert streaming: \(url.lastPathComponent) has no readable safetensors header")
        }
        headerLength = length
        fileSize = size
        self.entries = entries
    }

    func span(of key: String) -> (offset: Int64, length: Int64)? {
        guard let offsets = (entries[key] as? [String: Any])?["data_offsets"] as? [Int], offsets.count == 2, offsets[0] <= offsets[1] else { return nil }
        let start = Int64(8 + headerLength + offsets[0]), length = Int64(offsets[1] - offsets[0])
        return start + length <= fileSize ? (start, length) : nil
    }

    func header(placeholders: [String: Int], dropping dropped: Set<String>) throws(BasaltError) -> Data {
        var entries = self.entries.filter { !dropped.contains($0.key) }
        for (key, itemSize) in placeholders {
            guard var entry = entries[key] as? [String: Any], let begin = (entry["data_offsets"] as? [Int])?.first else { continue }
            entry["shape"] = [1]
            entry["data_offsets"] = [begin, begin + itemSize]
            entries[key] = entry
        }
        guard let data = try? JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys, .withoutEscapingSlashes]), data.count <= headerLength else {
            throw .inferenceError("expert streaming: the resident header of \(url.lastPathComponent) does not fit its original length")
        }
        return data + Data(repeating: 0x20, count: headerLength - data.count)
    }
}

struct ExpertPlan {
    var layouts: [ExpertLayout] = []
    var placeholders: [Int: [String: Int]] = [:]
    var dropped: [Int: Set<String>] = [:]

    var expertBytes: ByteCount { layouts.reduce(.zero) { $0 + $1.totalBytes } }

    init(model: any LanguageModel, shards: [SafetensorsShard], quantization: BaseConfiguration.PerLayerQuantization?) throws(BasaltError) {
        var sources: [ObjectIdentifier: (shard: Int, key: String)] = [:]
        var weights: [String: MLXArray] = [:]
        var metadata: [String: String] = [:]
        for (index, shard) in shards.enumerated() {
            guard let loaded = try? loadArraysAndMetadata(url: shard.url) else {
                throw .inferenceError("expert streaming: \(shard.url.lastPathComponent) did not open as safetensors")
            }
            for (key, array) in loaded.0 { sources[ObjectIdentifier(array)] = (index, key) }
            weights.merge(loaded.0) { _, new in new }
            if metadata.isEmpty { metadata = loaded.1 }
        }
        let sanitized = model.sanitize(weights: weights, metadata: metadata)
        for (path, module) in model.namedModules() where module is SwitchLinear {
            try add(path, sanitized: sanitized, sources: sources, shards: shards, quantization: quantization)
        }
    }

    private mutating func add(
        _ path: String, sanitized: [String: MLXArray], sources: [ObjectIdentifier: (shard: Int, key: String)], shards: [SafetensorsShard],
        quantization: BaseConfiguration.PerLayerQuantization?
    ) throws(BasaltError) {
        func tensor(_ suffix: String, experts: Int) throws(BasaltError) -> ExpertTensor? {
            guard let array = sanitized["\(path).\(suffix)"] else { return nil }
            guard let source = sources[ObjectIdentifier(array)] else {
                throw .inferenceError("expert streaming: \(path).\(suffix) is rewritten by the model's sanitize, so it cannot be read in place; convert the checkpoint with mlx-lm first")
            }
            guard let span = shards[source.shard].span(of: source.key), span.length == Int64(array.nbytes), array.nbytes % experts == 0 else {
                throw .inferenceError("expert streaming: \(source.key) in \(shards[source.shard].url.lastPathComponent) does not hold \(array.nbytes) bytes for \(experts) experts inside the file (truncated shard or data_offsets that disagree with the shape)")
            }
            let offset = span.offset
            if suffix == "weight" { placeholders[source.shard, default: [:]][source.key] = array.itemSize } else { dropped[source.shard, default: []].insert(source.key) }
            return ExpertTensor(file: source.shard, offset: offset, expertBytes: array.nbytes / experts, shape: Array(array.shape.dropFirst()), dtype: array.dtype)
        }
        guard let experts = sanitized["\(path).weight"]?.dim(0), experts > 0, let weight = try tensor("weight", experts: experts) else {
            throw .inferenceError("expert streaming: the checkpoint has no stacked expert weight for \(path)")
        }
        var layout = ExpertLayout(path: path, numExperts: experts, weight: weight)
        (layout.scales, layout.biases, layout.bias) = (try tensor("scales", experts: experts), try tensor("biases", experts: experts), try tensor("bias", experts: experts))
        if layout.scales != nil {
            guard let config = quantization?.quantization(layer: path), config.quantMethod != NF4.method else {
                throw .inferenceError("expert streaming: \(path) is quantized without a quantization config this build can stream (NF4 is not streamed)")
            }
            layout.quantization = config.asTuple
        }
        layouts.append(layout)
    }
}

enum ExpertStreamingLoader {
    struct Family {
        let types: ModelTypeRegistry<any LanguageModel>
        let factory: @Sendable (ModelTypeRegistry<any LanguageModel>) -> any ModelFactory
    }

    static let families = [
        Family(types: VLMModelFactory.shared.typeRegistry) { registry in
            let stock = VLMModelFactory.shared
            return VLMModelFactory(
                typeRegistry: registry, processorRegistry: stock.processorRegistry, modelRegistry: stock.modelRegistry,
                conventionsRegistry: stock.conventionsRegistry, processorLoadingRegistry: stock.processorLoadingRegistry)
        },
        Family(types: LLMModelFactory.shared.typeRegistry) { LLMModelFactory(typeRegistry: $0, modelRegistry: LLMRegistry.shared) },
    ]

    static func loadContainer(
        from directory: URL, streaming: ExpertStreaming, tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContainer? {
        let configuration = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configuration)
        var failure: (any Error)?
        for family in families where await family.types.contains(base.modelType) {
            do {
                return try await load(directory, family: family, base: base, configuration: configuration, streaming: streaming, tokenizerLoader: tokenizerLoader)
            } catch {
                failure = error
            }
        }
        throw failure ?? BasaltError.inferenceError("expert streaming: no model factory knows model_type \(base.modelType)")
    }

    static func load(
        _ directory: URL, family: Family, base: BaseConfiguration, configuration: Data, streaming: ExpertStreaming,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContainer? {
        let model = try await family.types.createModel(configuration: configuration, modelType: base.modelType)
        try refuseCompiledTraces(in: model)
        let shards = try shardURLs(in: directory).map { try SafetensorsShard(url: $0) }
        let plan = try ExpertPlan(model: model, shards: shards, quantization: base.perLayerQuantization)
        guard !plan.layouts.isEmpty else { return nil }
        let store = try ExpertStore(files: shards.map(\.url), layouts: plan.layouts, capacity: streaming.cache(forExperts: plan.expertBytes))
        try install(store, in: model)
        let view = try residentView(of: directory, shards: shards, plan: plan)
        defer { try? FileManager.default.removeItem(at: view) }
        let registry = ModelTypeRegistry<any LanguageModel>(creators: [base.modelType: { _ in model }])
        return try await family.factory(registry).loadContainer(from: view, using: tokenizerLoader)
    }

    static func refuseCompiledTraces(in model: Module) throws(BasaltError) {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        for (path, module) in modules where module is SwitchLinear {
            let parts = path.split(separator: ".").map(String.init)
            for depth in 0 ..< parts.count {
                guard let owner = depth == 0 ? model : modules[parts.prefix(depth).joined(separator: ".")], holdsCompiledTrace(owner) else { continue }
                throw .invalidRequestStructure(
                    details: "--experimental-expert-streaming: \(path) runs inside a compiled trace of \(type(of: owner)), where the router's expert indices cannot be read back to load experts; this architecture (Qwen3.5/3.6 MoE) cannot stream its experts yet")
            }
        }
    }

    static func holdsCompiledTrace(_ module: Module) -> Bool {
        module.items().flattenedValues().contains { value in
            if case .other(let held) = value { held is CompiledTraceInvalidating } else { false }
        }
    }

    static func store(in model: Module) -> ExpertStore? {
        model.namedModules().lazy.compactMap { ($0.1 as? StreamedSwitchLinear)?.store }.first
    }

    static func shardURLs(in directory: URL) throws -> [URL] {
        let index = directory.appending(component: "model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
           let map = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["weight_map"] as? [String: String] {
            let urls = Set(map.values).sorted().map { directory.appending(component: $0) }
            if !urls.isEmpty, urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) { return urls }
        }
        return try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    static func install(_ store: ExpertStore, in model: Module) throws {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        for (index, layout) in store.layouts.enumerated() {
            var parts = layout.path.split(separator: ".").map(String.init)
            guard let key = parts.popLast(), let parent = modules[parts.joined(separator: ".")] else {
                throw BasaltError.inferenceError("expert streaming: \(layout.path) has no parent module")
            }
            try parent.update(modules: ModuleChildren.unflattened([(key, StreamedSwitchLinear(store: store, module: index))]), verify: .noUnusedKeys)
        }
    }

    static func residentView(of directory: URL, shards: [SafetensorsShard], plan: ExpertPlan, in scratch: URL? = nil) throws -> URL {
        let files = FileManager.default
        let view = try scratch ?? files.url(for: .itemReplacementDirectory, in: .userDomainMask, appropriateFor: directory, create: true)
        do {
            try fill(view, from: directory, shards: shards, plan: plan)
        } catch {
            try? files.removeItem(at: view)
            throw error
        }
        return view
    }

    static func fill(_ view: URL, from directory: URL, shards: [SafetensorsShard], plan: ExpertPlan) throws {
        let files = FileManager.default
        let rewritten = Dictionary(uniqueKeysWithValues: shards.enumerated().map { ($1.url.lastPathComponent, $0) })
        for item in try files.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            let target = view.appending(component: item.lastPathComponent)
            guard let index = rewritten[item.lastPathComponent] else {
                try files.createSymbolicLink(at: target, withDestinationURL: item)
                continue
            }
            try files.copyItem(at: item.resolvingSymlinksInPath(), to: target)
            let header = try shards[index].header(placeholders: plan.placeholders[index] ?? [:], dropping: plan.dropped[index] ?? [])
            let handle = try FileHandle(forWritingTo: target)
            try handle.seek(toOffset: 8)
            try handle.write(contentsOf: header)
            try handle.close()
        }
    }
}
