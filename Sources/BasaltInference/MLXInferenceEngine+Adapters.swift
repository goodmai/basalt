import Foundation
import MLXNN
import MLXLMCommon
import BasaltCore

struct ServedAdapter: Sendable {
    let weights: LoRAContainer
    let namespace: String
}

extension MLXInferenceEngine {
    public var adapterNames: [String] { modelFacts.adapters.keys.sorted() }
    public var activeAdapter: String? { modelFacts.adapter }
    var adapterNamespace: String? { modelFacts.adapter.flatMap { modelFacts.adapters[$0]?.namespace } }

    public func loadAdapters(_ directories: [String: URL]) async throws(BasaltError) {
        guard let container else { throw .modelNotLoaded }
        guard modelFacts.adapter == nil, !modelModified else {
            throw .invalidRequestStructure(details: "adapters register on the plain base model; undo every overlay first")
        }
        let epoch = modelEpoch
        var served: [String: ServedAdapter] = [:]
        for (name, directory) in directories.sorted(by: { $0.key < $1.key }) {
            let weights = try Self.loraContainer(at: directory, Self.adapterRefusal(name))
            try await Self.labRun(container, "checking adapter \(name)") { context in
                _ = try Self.attachContainer(weights, to: context.model, Self.adapterRefusal(name))
                weights.unload(from: context.model)
                Self.settle(context.model)
            }
            served[name] = ServedAdapter(weights: weights, namespace: "\(name)@\(PromptCacheDiskStore.fingerprint(modelDirectory: directory))")
        }
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the adapters were checked; none is registered") }
        record(.learned(epoch) { [served] in $0.adapters = served })
    }

    public func useAdapter(_ name: String?) async throws(BasaltError) {
        guard name != modelFacts.adapter else { return }
        guard let container else { throw .modelNotLoaded }
        guard !modelModified else { throw .invalidRequestStructure(details: "adapter: an overlay changes the live model; undo it before switching adapters") }
        let current = modelFacts.adapter.flatMap { modelFacts.adapters[$0] }
        let next = name.flatMap { modelFacts.adapters[$0] }
        guard name == nil || next != nil else { throw .invalidRequestStructure(details: "unknown adapter \(name ?? "")") }
        let epoch = modelEpoch
        record(.learned(epoch) { $0.adapter = nil })
        try await Self.labRun(container, "the adapter switch") { context in
            try Self.swapAdapter(current?.weights, for: next?.weights, in: context.model)
        }
        record(.learned(epoch) { $0.adapter = name })
    }

    static func swapAdapter(_ current: LoRAContainer?, for next: LoRAContainer?, in model: any LanguageModel) throws {
        current?.unload(from: model)
        settle(model)
        guard let next else { return }
        do {
            try model.load(adapter: next)
        } catch {
            next.unload(from: model)
            settle(model)
            throw error
        }
    }

    static func adapterRefusal(_ name: String) -> (String) -> BasaltError {
        { .invalidRequestStructure(details: "adapter \(name): \($0)") }
    }
}
