import Foundation
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    public func trainOfflineDPO(_ config: OfflineDPOConfig) async throws(BasaltError) -> OfflineDPOReport {
        guard let container else { throw .modelNotLoaded }
        guard !modelModified else {
            throw .invalidRequestStructure(details: "dpo: an overlay or weights delta changes the live model; undo it before training on the base weights")
        }
        try config.validate(prefix: "dpo")
        if let gate = config.gate { try Self.validate(gate) }
        let pairs = try PreferencePairsLoader.load(from: config.pairsPath)
        let directory = try adapterDirectory(config.output)
        let epoch = modelEpoch
        let family = trainingModelFamily()
        var report = try await Self.labRun(container, "offline DPO training") { context in
            try OfflineDPOTrainer.train(pairs: pairs, config: config, directory: directory, family: family, context: context)
        }
        guard self.container === container, modelEpoch == epoch else {
            throw .inferenceError("dpo: the model changed during training; the adapter in \(directory.path) was not applied")
        }
        if config.applyOverlay {
            let verdict = try await applyAdapter(.directory(directory.path), gate: config.gate)
            report.gatePassed = verdict.report?.passed ?? (verdict.overlay != nil)
        }
        return report
    }

    public func trainGRPO(_ config: GRPOConfig) async throws(BasaltError) -> GRPOReport {
        guard let container else { throw .modelNotLoaded }
        guard !modelModified else {
            throw .invalidRequestStructure(details: "grpo: an overlay or weights delta changes the live model; undo it before training on the base weights")
        }
        try config.validate(prefix: "grpo")
        try Self.validate(config.gate)
        let tasks = try TrainingTask.load(config.tasks)
        let directory = try adapterDirectory(config.output)
        let epoch = modelEpoch
        let family = trainingModelFamily()
        var report = try await Self.labRun(container, "GRPO training") { context in
            try GRPOTrainer.train(tasks: tasks, config: config, directory: directory, family: family, context: context)
        }
        guard self.container === container, modelEpoch == epoch else {
            throw .inferenceError("grpo: the model changed during training; the adapter in \(directory.path) was not applied")
        }
        if config.applyOverlay {
            let verdict = try await applyAdapter(.directory(directory.path), gate: config.gate)
            report.gatePassed = verdict.report?.passed ?? (verdict.overlay != nil)
        }
        return report
    }

    private func trainingModelFamily() -> String? {
        (modelDescription.isEmpty ? nil : URL(fileURLWithPath: modelDescription))
            .flatMap(ModelCache.config(at:))?["model_type"] as? String
    }
}
