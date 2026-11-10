import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    static let gateTokens = 32
    static let targetTokens = 16

    public func gate(_ request: GateRequest) async throws(BasaltError) -> GateReport {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        try Self.validate(request)
        let live = modelFacts.deltas.filter { !$0.value.slice.isEmpty || $0.value.pack != nil }
        let family = (modelDescription.isEmpty ? nil : URL(fileURLWithPath: modelDescription)).flatMap(ModelCache.config(at:))?["model_type"] as? String
        return try await Self.labRun(container, "the quality gate") { context in try Self.gate(request, context: context, live: live, family: family) }
    }

    public func applyAdapter(_ source: AdapterSource, gate request: GateRequest?) async throws(BasaltError) -> AdapterVerdict {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        if let request { try Self.validate(request) }
        let key = Self.describe(source)
        guard modelFacts.deltas[key] == nil else { throw .invalidRequestStructure(details: "adapter_apply: \(key) is already applied; undo it first") }
        record(.learned(modelEpoch) { $0.deltas[key] = AppliedDelta() })
        let modules: [String]
        do throws(BasaltError) {
            modules = try await Self.labRun(container, "the adapter") { context in try Self.attach(source, to: context.model) }
        } catch {
            record(.learned(modelEpoch) { $0.deltas[key] = nil })
            throw error
        }
        guard self.container === container else {
            throw .inferenceHardwareFailure(reason: "the model was replaced while the adapter was applied; nothing is applied")
        }
        let info = register(nil, description: key, lifetime: request == nil ? OverlayRegistry.defaultLifetime : nil)
        weightsChanged { $0.deltas[key] = AppliedDelta(info: info, modules: modules) }
        scheduleUndo(of: info)
        guard let request else { return AdapterVerdict(overlay: info, report: nil) }
        return try await verdict(on: info, key: key, request)
    }

    private func verdict(on info: OverlayInfo, key: String, _ request: GateRequest) async throws(BasaltError) -> AdapterVerdict {
        let report: GateReport
        do throws(BasaltError) {
            report = try await gate(request)
        } catch {
            try? await undo(info.id)
            throw error
        }
        guard modelFacts.deltas[key]?.info?.id == info.id else {
            throw .inferenceError("the adapter was undone or the model replaced while the gate ran; nothing is kept")
        }
        guard report.passed else {
            try await undo(info.id)
            return AdapterVerdict(overlay: nil, report: report)
        }
        return AdapterVerdict(overlay: info, report: report)
    }

    static func validate(_ request: GateRequest) throws(BasaltError) {
        guard request.maxKL.isFinite, request.maxKL >= 0, request.maxPerplexityRise.isFinite, request.maxPerplexityRise >= 0 else {
            throw .invalidRequestStructure(details: "gate: maxKL and maxPerplexityRise must be finite and at least 0")
        }
        guard request.target.allSatisfy({ !$0.expected.isEmpty }) else {
            throw .invalidRequestStructure(details: "gate: every target needs a non-empty expected answer")
        }
    }

    static func describe(_ source: AdapterSource) -> String {
        switch source {
        case .directory(let path): "adapter \(path)"
        case .rankOne(let module, _, _, let scale): "rank-1 adapter on \(module) × \(scale)"
        }
    }

    static func gate(_ request: GateRequest, context: ModelContext, live: [String: AppliedDelta], family: String?) throws -> GateReport {
        let pristine = try lifted(live, in: context.model) { try ModelProbe.facts(context) }
        let current = try ModelProbe.facts(context)
        let rows = try request.heldOut.map { try compareRow($0, context: context, maxTokens: gateTokens, pristine: live) { nil } }
        let targets = try request.target.map { target in
            let answer = try ModelProbe.known(target.prompt, expected: target.expected.lowercased(), context: context, tokens: targetTokens)
            let hit: Float = ModelProbe.answer(answer) == nil ? 1 : 0
            return GateCheck(name: "target \(target.prompt) -> \(target.expected)", passed: hit == 1, value: hit, limit: 1)
        }
        let checks = probeChecks(current, pristine: pristine, family: family, rise: request.maxPerplexityRise) + heldOutChecks(rows, request) + targets
        return GateReport(checks: checks, compare: rows.isEmpty ? nil : CompareReport(rows: rows.map(finite)))
    }

    static func probeChecks(_ current: ModelProbe.Facts, pristine: ModelProbe.Facts, family: String?, rise: Float) -> [GateCheck] {
        let ceiling = ModelProbe.perplexityCeiling(family: family)
        let limit = pristine.perplexity.isFinite ? max(ceiling, pristine.perplexity * (1 + rise)) : ceiling
        let perplexity = GateCheck(name: "probe perplexity", passed: current.perplexity <= limit, value: finite(current.perplexity), limit: limit)
        let answers = zip(current.answers, pristine.answers).map { now, before in
            let held: Float = ModelProbe.answer(now) == nil ? 1 : 0
            let had: Float = ModelProbe.answer(before) == nil ? 1 : 0
            return GateCheck(name: "probe known answer \(now.expected)", passed: held >= had, value: held, limit: had)
        }
        return [perplexity] + answers
    }

    static func heldOutChecks(_ rows: [CompareRow], _ request: GateRequest) -> [GateCheck] {
        guard !rows.isEmpty else { return [] }
        let count = Float(rows.count)
        let kl = rows.map(\.klDivergence).reduce(0, +) / count
        let rise = rows.map { $0.overlayPerplexity / $0.baselinePerplexity - 1 }.reduce(0, +) / count
        return [
            GateCheck(name: "held-out mean KL", passed: kl <= request.maxKL, value: finite(kl), limit: request.maxKL),
            GateCheck(name: "held-out perplexity rise", passed: rise <= request.maxPerplexityRise, value: finite(rise), limit: request.maxPerplexityRise),
        ]
    }

    static func finite(_ value: Float) -> Float { value.isFinite ? value : .greatestFiniteMagnitude }

    static func finite(_ row: CompareRow) -> CompareRow {
        var row = row
        (row.klDivergence, row.baselinePerplexity, row.overlayPerplexity) = (finite(row.klDivergence), finite(row.baselinePerplexity), finite(row.overlayPerplexity))
        return row
    }

    static func lifted<T>(_ deltas: [String: AppliedDelta], in model: Module, _ body: () throws -> T) throws -> T {
        var restore: [String: AppliedDelta] = [:]
        model.setLoRAEnabled(false)
        model.invalidateCompiledTraces()
        defer {
            for (name, applied) in restore { try? Self.restore(applied, of: name, in: model) }
            model.setLoRAEnabled(true)
            model.invalidateCompiledTraces()
        }
        for (name, applied) in deltas { restore[name] = try swapped(applied, of: name, in: model) }
        return try body()
    }

    static func swapped(_ applied: AppliedDelta, of name: String, in model: Module) throws(BasaltError) -> AppliedDelta {
        if let pack = applied.pack {
            let (_, current) = try TensorPacker.linear(name, in: model)
            try pack.restore(in: model)
            return AppliedDelta(pack: PackedLayer(path: pack.path, layer: current))
        }
        let tensor = try writableTensor(name, in: model)
        let indices: [any MLXArrayIndex] = applied.slice
        var current = applied
        current.original = tensor[indices].asType(.float32).asArray(Float.self)
        try restore(applied, of: name, in: model)
        return current
    }
}

extension MLXInferenceEngine {
    static func attach(_ source: AdapterSource, to model: any LanguageModel) throws(BasaltError) -> [String] {
        switch source {
        case .directory(let path): try attachDirectory(URL(fileURLWithPath: path), to: model)
        case .rankOne(let module, let input, let output, let scale): try attachRankOne(module, input: input, output: output, scale: scale, to: model)
        }
    }

    static func attachRankOne(_ path: String, input: [Float], output: [Float], scale: Float, to model: any LanguageModel) throws(BasaltError) -> [String] {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "adapter_apply on \(path): \(reason)") }
        guard let linear = model.namedModules().first(where: { $0.0 == path })?.1 as? Linear, restorable(linear) else {
            throw refused("not a plain Linear or QuantizedLinear layer without an adapter; model_tensors lists the layers (the tensor name without .weight)")
        }
        let (outputs, inputs) = linear.shape
        guard input.count == inputs, output.count == outputs else {
            throw refused("input needs \(inputs) values and output \(outputs), got \(input.count) and \(output.count)")
        }
        guard scale.isFinite, input.allSatisfy(\.isFinite), output.allSatisfy(\.isFinite) else { throw refused("input, output and scale must be finite") }
        let lora = LoRALinear.from(linear: linear, rank: 1, scale: scale)
        lora.update(parameters: ModuleParameters.unflattened(["lora_a": MLXArray(input, [inputs, 1]), "lora_b": MLXArray(output, [1, outputs])]))
        try replace(path, with: lora, in: model)
        settle(model)
        return [path]
    }

    static func attachDirectory(_ directory: URL, to model: any LanguageModel) throws(BasaltError) -> [String] {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "adapter_apply \(directory.path): \(reason)") }
        return try attachContainer(try loraContainer(at: directory, refused), to: model, refused)
    }

    static func loraContainer(at directory: URL, _ refused: (String) -> BasaltError) throws(BasaltError) -> LoRAContainer {
        do {
            return try LoRAContainer.from(directory: directory)
        } catch {
            throw refused("not an mlx-lm adapter directory (adapter_config.json and adapters.safetensors): \(briefly(error))")
        }
    }

    static func attachContainer(_ adapter: LoRAContainer, to model: any LanguageModel, _ refused: (String) -> BasaltError) throws(BasaltError) -> [String] {
        guard model is LoRAModel else { throw refused("the \(type(of: model)) architecture takes no LoRA adapter") }
        guard loraModules(of: model).isEmpty else { throw refused("another adapter is applied; undo it first") }
        try checkShapes(adapter, in: model, refused)
        let plain = Dictionary(model.namedModules().map { ($0.0, restorable($0.1)) }, uniquingKeysWith: { first, _ in first })
        let loaded = Result { try model.load(adapter: adapter) }
        let modules = loraModules(of: model)
        if case .failure(let error) = loaded {
            try detach(modules, from: model)
            throw refused(briefly(error))
        }
        guard modules.allSatisfy({ plain[$0] == true }) else {
            try detach(modules, from: model)
            throw refused("it wraps a layer that is not a plain Linear or QuantizedLinear, which undo could not restore exactly")
        }
        guard !modules.isEmpty else { throw refused("it matches no linear layer of this model") }
        return modules
    }

    static func checkShapes(_ adapter: LoRAContainer, in model: Module, _ refused: (String) -> BasaltError) throws(BasaltError) {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        let rank = adapter.configuration.loraParameters.rank
        for (key, value) in adapter.parameters.flattened() {
            guard let linear = modules[key.split(separator: ".").dropLast().joined(separator: ".")] as? Linear else {
                throw refused("\(key) belongs to no linear layer of this model")
            }
            let (outputs, inputs) = linear.shape
            let expected = key.hasSuffix(".lora_a") ? [inputs, rank] : key.hasSuffix(".lora_b") ? [rank, outputs] : value.shape
            guard value.shape == expected else { throw refused("\(key) has shape \(value.shape), the layer needs \(expected)") }
        }
    }

    static func loraModules(of model: Module) -> [String] {
        model.namedModules().filter { $0.1 is LoRALayer }.map(\.0)
    }

    static func detach(_ paths: [String], from model: Module) throws(BasaltError) {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        for path in paths {
            guard let lora = modules[path] as? LoRALayer else { continue }
            try replace(path, with: lora.reverted(), in: model)
        }
        settle(model)
    }

    static func replace(_ path: String, with module: Module, in model: Module) throws(BasaltError) {
        var parts = path.split(separator: ".").map(String.init)
        guard let key = parts.popLast(), let parent = model.namedModules().first(where: { $0.0 == parts.joined(separator: ".") })?.1 else {
            throw .invalidRequestStructure(details: "adapter: \(path) has no parent module to hold a replacement")
        }
        do {
            try parent.update(modules: ModuleChildren.unflattened([(key, module)]), verify: .noUnusedKeys)
            settle(model)
        } catch {
            throw .inferenceError("adapter: \(path) could not be swapped: \(briefly(error))")
        }
    }

    static func restorable(_ module: Module) -> Bool {
        guard (module as? QuantizedLinear)?.globalScale == nil else { return false }
        return [ObjectIdentifier(Linear.self), ObjectIdentifier(QuantizedLinear.self)].contains(ObjectIdentifier(type(of: module)))
    }

    static func settle(_ model: Module) {
        model.invalidateCompiledTraces()
        do {
            try (model as? any LanguageModel)?.prepare()
        } catch {
            BasaltLogger(module: "MLXInferenceEngine").warn("The model did not rebuild its derived state after an adapter change: \(briefly(error))")
        }
    }
}
