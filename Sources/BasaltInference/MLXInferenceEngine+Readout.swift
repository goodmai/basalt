import Foundation
import MLX
import MLXLMCommon
import BasaltCore

public struct ReadoutDecisionEngine: DecisionEngine {
    public let modelID: String
    let engine: MLXInferenceEngine
    let readout: SystemOneReadout

    public init(engine: MLXInferenceEngine, modelID: String, readout: SystemOneReadout) {
        (self.engine, self.modelID, self.readout) = (engine, modelID, readout)
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        guard request.questions.values.allSatisfy({ SystemOneReadout.options($0).count <= SystemOneReadout.letters.count }) else {
            throw .invalidRequestStructure(details: "systemone: the letter readout takes at most \(SystemOneReadout.letters.count) options per question")
        }
        let names = request.questions.keys.sorted()
        let questions = names.map { request.questions[$0]! }
        let (logits, usage) = try await engine.letterLogits(state: request.state, questions: questions)
        let calibration = request.calibrated == false ? SystemOneReadout.uncalibrated : readout
        var answers: [String: SystemOneAnswer] = [:]
        for (index, name) in names.enumerated() {
            answers[name] = SystemOneCore.gated(
                calibration.answer(questions[index], letterLogits: logits[index]), abstainBelow: request.abstainBelow, minMargin: request.minMargin)
        }
        return SystemOneResponse(model: modelID, answers: answers, usage: usage, calibrated: request.calibrated != false && calibration != .uncalibrated)
    }

    public func unload() async {
        if modelID != "self" { await engine.unload() }
    }
}

extension MLXInferenceEngine {
    func letterLogits(state: String, questions: [SystemOneQuestion]) async throws(BasaltError) -> ([[Double]], SystemOneUsage) {
        guard let container else { throw .modelNotLoaded }
        do {
            return try await container.perform { context in try Self.letterLogits(state: state, questions: questions, context: context) }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("systemone: \(Self.briefly(error))")
        }
    }

    static func letterIds(_ tokenizer: any Tokenizer, count: Int) throws -> [Int] {
        let ids = SystemOneReadout.letters.prefix(count).map { tokenizer.encode(text: $0, addSpecialTokens: false) }
        guard ids.allSatisfy({ $0.count == 1 }), Set(ids.map { $0[0] }).count == ids.count else {
            throw BasaltError.modelConfigurationInvalid(path: "tokenizer", reason: "option letters do not map to distinct single tokens, so the letter readout cannot run on this model")
        }
        return ids.map { $0[0] }
    }

    static func readoutTemplate(_ tokenizer: any Tokenizer) throws -> (header: String, tail: String) {
        let ids = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": JevBundle.marker]], tools: nil, additionalContext: ["enable_thinking": false])
        let parts = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false).components(separatedBy: JevBundle.marker)
        guard parts.count == 2 else { throw BasaltError.modelConfigurationInvalid(path: "chat template", reason: "the template does not place the user message exactly once") }
        return (parts[0], parts[1])
    }

    static func letterLogits(state: String, questions: [SystemOneQuestion], context: ModelContext) throws -> ([[Double]], SystemOneUsage) {
        let (header, userTail) = try readoutTemplate(context.tokenizer)
        let prefix = context.tokenizer.encode(text: header + SystemOneReadout.prefix(state: state), addSpecialTokens: false)
        let cache = try context.model.newCache(parameters: nil)
        let warm = context.model(LMInput.Text(tokens: MLXArray(prefix.map(Int32.init))[.newAxis]), cache: cache, state: nil)
        eval(cache)
        var (rows, suffixTokens) = ([[Double]](), 0)
        for question in questions {
            try Task.checkCancellation()
            let tail = context.tokenizer.encode(text: SystemOneReadout.tail(question) + userTail, addSpecialTokens: false)
            let letters = try letterIds(context.tokenizer, count: SystemOneReadout.options(question).count)
            let branch = cache.map { $0.copy() }
            var state = warm.state
            if tail.count > 1 {
                state = context.model(LMInput.Text(tokens: MLXArray(tail.dropLast().map(Int32.init))[.newAxis]), cache: branch, state: state).state
                eval(branch)
            }
            let output = context.model(LMInput.Text(tokens: MLXArray([Int32(tail[tail.count - 1])])[.newAxis]), cache: branch, state: state)
            let last = output.logits[0, -1].asType(.float32)
            rows.append(last[MLXArray(letters.map(Int32.init))].asArray(Float.self).map(Double.init))
            suffixTokens += tail.count
        }
        let usage = SystemOneUsage(prefixTokens: prefix.count, candidateBranches: questions.count, prefixEvaluations: 1, branchForwardCalls: questions.count, suffixTokens: suffixTokens)
        return (rows, usage)
    }
}
