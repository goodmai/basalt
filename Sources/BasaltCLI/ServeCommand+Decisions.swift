import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

extension ServeCommand {
    func loadDecisions(chat: MLXInferenceEngine) async throws -> (any DecisionEngine)? {
        guard let decisionModel = decision.localModel else { return nil }
        do {
            return try await loadDecisions(decisionModel, chat: chat)
        } catch let error as BasaltError {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func loadDecisions(_ decisionModel: String, chat: MLXInferenceEngine) async throws -> any DecisionEngine {
        if decisionModel == "self" {
            log("Serving decisions (/v1/systemone) by option-letter readout of the chat model")
            return ReadoutDecisionEngine(engine: chat, modelID: "self", readout: .uncalibrated)
        }
        let directory = URL(fileURLWithPath: try await resolvedPath(decisionModel, quant: nil))
        if CoreAIDecisionEngine.isBundle(directory) {
            log("Specializing the Core AI decision bundle for the Neural Engine (first start only; cached afterwards)…")
            let engine = try await CoreAIDecisionEngine(directory: directory)
            log("Serving \(engine.modelID) (decision, /v1/systemone, Core AI on the Neural Engine)")
            return engine
        }
        if MultiHeadDecisionEngine.isBundle(directory) {
            return try await loadBankDecisions(directory, chat: chat)
        }
        guard JevEvaluator.isBundle(directory) else { return try await readoutDecisions(decisionModel, directory: directory) }
        let base = URL(fileURLWithPath: try await resolvedPath(try JevEvaluator.baseModel(of: directory), quant: nil))
        let evaluator = try await JevEvaluator(directory: directory, base: base)
        log("Serving \(evaluator.modelID) (decision, /v1/systemone) on base \(base.lastPathComponent)")
        return evaluator
    }

    func loadBankDecisions(_ directory: URL, chat: MLXInferenceEngine) async throws -> any DecisionEngine {
        let baseModel = try MultiHeadDecisionEngine.baseModel(of: directory)
        if baseModel == "self" || baseModel == targetModel {
            let fallback = ReadoutDecisionEngine(engine: chat, modelID: "self", readout: .uncalibrated)
            let bank = try MultiHeadDecisionEngine(directory: directory, engine: chat, ownsEngine: false, fallback: fallback)
            log("Serving \(bank.modelID) (multi-head bank, /v1/systemone, \(bank.manifest.slots.count) slots) sharing chat model")
            return bank
        }
        let base = URL(fileURLWithPath: try await resolvedPath(baseModel, quant: nil))
        let bank = try await MultiHeadDecisionEngine(directory: directory, base: base)
        log("Serving \(bank.modelID) (multi-head bank, /v1/systemone, \(bank.manifest.slots.count) slots) on base \(base.lastPathComponent)")
        return bank
    }

    func readoutDecisions(_ model: String, directory: URL) async throws(BasaltError) -> any DecisionEngine {
        var configuration = EngineConfiguration.standard
        configuration.promptCacheBytes = 0
        let engine = MLXInferenceEngine(configuration)
        try await engine.load(modelPath: directory.path)
        let readout = SystemOneReadout.calibration(for: model, in: directory)
        log("Serving \(model) (decision, /v1/systemone) by option-letter readout\(readout == .uncalibrated ? ", uncalibrated" : ", calibrated T \(readout.temperature)")")
        return ReadoutDecisionEngine(engine: engine, modelID: model, readout: readout)
    }
}
