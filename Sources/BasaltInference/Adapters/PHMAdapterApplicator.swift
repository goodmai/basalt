import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

public enum PHMAdapterApplicator {
    struct TargetProjection {
        let fullPath: String
        let relativeKey: String
        let linear: Linear
    }

    @discardableResult
    public static func apply(
        to model: Module,
        spec: AdapterSpec,
        targetSuffixes: [String]? = nil
    ) throws(BasaltError) -> [String] {
        try spec.validate(prefix: "adapter")
        guard !model.namedModules().contains(where: { $0.1 is LoRALayer }) else {
            throw .invalidRequestStructure(details: "adapter: an adapter is already applied; revert it before applying another")
        }
        let targets = try resolveTargets(in: model, spec: spec, targetSuffixes: targetSuffixes)
        for target in targets {
            let (outputDim, inputDim) = target.linear.shape
            try spec.method.validate(inputDim: inputDim, outputDim: outputDim, prefix: "adapter (\(target.fullPath))")
        }
        model.freeze()
        let replacements = try buildReplacements(for: targets, method: spec.method)
        updateLeafModules(in: model, updates: replacements)
        MLXInferenceEngine.settle(model)
        return replacements.map(\.0)
    }

    @discardableResult
    public static func revert(in model: Module) -> [String] {
        let updates: [(String, Module)] = model.namedModules().compactMap { path, module in
            guard !path.isEmpty, let layer = module as? LoRALayer else { return nil }
            return (path, layer.reverted())
        }
        updateLeafModules(in: model, updates: updates)
        model.unfreeze()
        MLXInferenceEngine.settle(model)
        return updates.map(\.0)
    }

    @discardableResult
    public static func fuse(in model: Module) -> [String] {
        let updates: [(String, Module)] = model.namedModules().compactMap { path, module in
            guard !path.isEmpty, let layer = module as? LoRALayer else { return nil }
            return (path, layer.fused())
        }
        updateLeafModules(in: model, updates: updates)
        MLXInferenceEngine.settle(model)
        return updates.map(\.0)
    }

    static func updateLeafModules(in model: Module, updates: [(String, Module)]) {
        guard !updates.isEmpty else { return }
        var parents: [String: Module] = ["": model]
        for (path, module) in model.namedModules() where !path.isEmpty {
            parents[path] = module
        }
        var grouped: [String: [(String, Module)]] = [:]
        var order: [String] = []
        for (fullPath, replacement) in updates {
            let parentPath: String
            let leafKey: String
            if let dot = fullPath.lastIndex(of: ".") {
                parentPath = String(fullPath[..<dot])
                leafKey = String(fullPath[fullPath.index(after: dot)...])
            } else {
                parentPath = ""
                leafKey = fullPath
            }
            if grouped[parentPath] == nil { order.append(parentPath) }
            grouped[parentPath, default: []].append((leafKey, replacement))
        }
        for parentPath in order {
            if let parent = parents[parentPath], let leafUpdates = grouped[parentPath] {
                parent.update(modules: .unflattened(leafUpdates))
            }
        }
    }

    public static func parameterCount(in model: Module) -> Int {
        model.trainableParameters().flattened().reduce(0) { $0 + $1.1.size }
    }

    public static func projectionDims(
        in model: Module,
        spec: AdapterSpec,
        targetSuffixes: [String]? = nil
    ) throws(BasaltError) -> [(inputDim: Int, outputDim: Int)] {
        try spec.validate(prefix: "adapter")
        return try resolveTargets(in: model, spec: spec, targetSuffixes: targetSuffixes).map { target in
            let (outputDim, inputDim) = target.linear.shape
            return (inputDim: inputDim, outputDim: outputDim)
        }
    }
}

extension PHMAdapterApplicator {
    static func resolveTargets(
        in model: Module,
        spec: AdapterSpec,
        targetSuffixes: [String]?
    ) throws(BasaltError) -> [TargetProjection] {
        let blocks = LeJEPABackbone.layers(of: model)
        let explicitSuffixes = targetSuffixes ?? spec.targetSuffixes
        if !blocks.isEmpty {
            return try resolveBlockTargets(in: model, blocks: blocks, layers: spec.layers, suffixes: explicitSuffixes)
        }
        if let requested = spec.layers, requested != [0] {
            throw .invalidRequestStructure(details: "adapter: model has no transformer layer array for layers \(requested)")
        }
        let candidates = model.namedModules()
            .filter { !$0.0.isEmpty }
            .sorted { $0.0 < $1.0 }
            .compactMap { path, module -> TargetProjection? in
                guard let linear = module as? Linear, MLXInferenceEngine.restorable(linear) else { return nil }
                if let explicitSuffixes, !matchesKey(path, suffixes: explicitSuffixes) { return nil }
                return TargetProjection(fullPath: path, relativeKey: path, linear: linear)
            }
        guard !candidates.isEmpty else {
            throw .invalidRequestStructure(details: "adapter: no plain or quantized Linear layer matched target suffixes")
        }
        return candidates
    }

    private static func resolveBlockTargets(
        in model: Module,
        blocks: [Module],
        layers: [Int]?,
        suffixes: [String]?
    ) throws(BasaltError) -> [TargetProjection] {
        let chosen = (layers ?? Array(blocks.indices)).sorted()
        guard !chosen.isEmpty, chosen.allSatisfy(blocks.indices.contains) else {
            throw .invalidRequestStructure(details: "adapter: layers must be a non-empty subset of 0..<\(blocks.count)")
        }
        let defaultKeys = (model as? LoRAModel)?.loraDefaultKeys
            ?? Array(Set(blocks.flatMap { $0.namedModules() }.compactMap { $0.1 is Linear ? $0.0 : nil }))
        let activeSuffixes = (suffixes ?? defaultKeys).sorted()
        let blockPaths = Dictionary(
            model.namedModules().map { (ObjectIdentifier($0.1), $0.0) },
            uniquingKeysWith: { first, _ in first }
        )
        var targets: [TargetProjection] = []
        for index in Set(chosen).sorted() {
            let block = blocks[index]
            let prefix = blockPaths[ObjectIdentifier(block)] ?? ""
            let blockModules = block.namedModules().filter { !$0.0.isEmpty }.sorted { $0.0 < $1.0 }
            for (key, child) in blockModules {
                guard matchesKey(key, suffixes: activeSuffixes),
                      let linear = child as? Linear,
                      MLXInferenceEngine.restorable(linear) else { continue }
                let fullPath = prefix.isEmpty ? key : "\(prefix).\(key)"
                targets.append(TargetProjection(fullPath: fullPath, relativeKey: key, linear: linear))
            }
        }
        guard !targets.isEmpty else {
            throw .invalidRequestStructure(details: "adapter: the chosen layers have no plain or quantized linear projection to adapt")
        }
        return targets
    }

    private static func matchesKey(_ key: String, suffixes: [String]) -> Bool {
        suffixes.contains { suffix in
            suffix.contains(".") ? key == suffix : (key == suffix || key.hasSuffix(".\(suffix)"))
        }
    }

    private static func buildReplacements(
        for targets: [TargetProjection],
        method: AdapterMethod
    ) throws(BasaltError) -> [(String, Module)] {
        switch method {
        case .lora(let rank, let scale):
            if let reason = targets.lazy.compactMap({ MLXInferenceEngine.microscaledLoRARefusal($0.linear, path: $0.fullPath) }).first {
                throw .invalidRequestStructure(details: "adapter: \(reason)")
            }
            return targets.map { target in
                (target.fullPath, LoRALinear.from(linear: target.linear, rank: rank, scale: scale))
            }
        case .phm(let hypercomplexDim, let factorRank, let sharedBasis, let scale):
            let sharedBank = sharedBasis ? PHMBasisBank(hypercomplexDim: hypercomplexDim) : nil
            var replacements: [(String, Module)] = []
            replacements.reserveCapacity(targets.count)
            for (offset, target) in targets.enumerated() {
                let mode: PHMBasisMode = if let sharedBank {
                    offset == 0 ? .owned(sharedBank) : .borrowed(sharedBank)
                } else {
                    .independent
                }
                let adapted = try PHMLinear.from(
                    linear: target.linear,
                    hypercomplexDim: hypercomplexDim,
                    factorRank: factorRank,
                    scale: scale,
                    basisMode: mode
                )
                replacements.append((target.fullPath, adapted))
            }
            return replacements
        case .oft(let blockSize, let parameterization, let blockShared):
            var replacements: [(String, Module)] = []
            replacements.reserveCapacity(targets.count)
            for target in targets {
                let adapted = try OFTLinear.from(
                    linear: target.linear,
                    blockSize: blockSize,
                    parameterization: parameterization,
                    blockShared: blockShared
                )
                replacements.append((target.fullPath, adapted))
            }
            return replacements
        }
    }
}
