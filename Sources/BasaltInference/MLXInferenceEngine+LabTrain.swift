import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import MLXOptimizers
import BasaltCore

struct TrainingRun: Sendable {
    var losses: [Float]
    var accuracyAfter: Float
    var emaApplied: Bool?
}

extension MLXInferenceEngine {
    static let answerTokens = 64
    static let reportedLosses = 20
    static let trainingRanks = 1...64
    static let trainingIterations = 1...10_000

    public func train(_ request: TrainRequest) async throws(BasaltError) -> TrainReport {
        guard let container else { throw .modelNotLoaded }
        try Self.validate(request)
        if let adpo = request.adpo { return try await trainADPO(request, adpo: adpo, container: container) }
        let tasks = try TrainingTask.load(request.tasks)
        let heldOut = Array(tasks.suffix(max(1, tasks.count / 5)))
        let practice = Array(tasks.dropLast(heldOut.count))
        let directory = try adapterDirectory(request.output)
        var teacher: TeacherClient?
        if let url = request.teacher { teacher = try TeacherClient(url, model: request.teacherModel) }
        let own = try await Self.labRun(container) { context in try tasks.map { try Self.labAnswer($0.prompt, context) } }
        var pairs = zip(practice, own).filter { $0.accepts($1) }.map { TrainingTask(prompt: $0.prompt, answer: $1) }
        var collected = practice.count
        if let teacher {
            for task in practice {
                let answer = try await teacher.answer(task.prompt)
                collected += 1
                if task.accepts(answer) { pairs.append(TrainingTask(prompt: task.prompt, answer: answer)) }
            }
        }
        let before = Self.accuracy(heldOut, Array(own.suffix(heldOut.count)))
        var report = TrainReport(
            collected: collected, verified: pairs.count, losses: [], accuracyBefore: before, accuracyAfter: before,
            verdict: AdapterVerdict(overlay: nil, report: nil))
        guard !pairs.isEmpty else { return report }
        guard self.container === container else { throw .inferenceError("the model changed while answers were collected; nothing was trained") }
        let run = try await Self.labRun(container) { [pairs] context in
            try Self.trainAdapter(pairs, heldOut: heldOut, request: request, directory: directory, context: context)
        }
        guard self.container === container else { throw .inferenceError("the model changed during training; the adapter in \(directory.path) was not applied") }
        report.losses = run.losses
        report.accuracyAfter = run.accuracyAfter
        report.emaApplied = run.emaApplied
        report.adapterDirectory = directory.path
        do {
            report.verdict = try await applyAdapter(.directory(directory.path), gate: request.gate)
        } catch {
            throw .inferenceError("the adapter was trained and saved in \(directory.path) but not applied: \(Self.briefly(error))")
        }
        return report
    }

    static func validate(_ request: TrainRequest) throws(BasaltError) {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "train: \(reason)") }
        guard request.experts == nil else {
            throw refused("per-expert LoRA is not available: the fork's SwitchLinear expert layers take no LoRA. Omit experts; LoRA then trains the attention and dense or shared-expert projections of the chosen layers")
        }
        if let method = request.method {
            try method.validate(prefix: "train")
        } else {
            guard trainingRanks.contains(request.rank) else { throw refused("rank must be in \(trainingRanks)") }
        }
        guard trainingIterations.contains(request.iterations) else { throw refused("iterations must be in \(trainingIterations)") }
        guard request.learningRate.isFinite, request.learningRate > 0, request.learningRate <= 1 else { throw refused("learningRate must be in (0, 1]") }
        if let adpo = request.adpo { try adpo.validate(prefix: "train") }
        if let ema = request.emaDecay {
            guard ema.isFinite, ema > 0, ema < 1 else { throw refused("emaDecay must lie in (0, 1)") }
        }
    }

    func adapterDirectory(_ output: String?) throws(BasaltError) -> URL {
        let fingerprint = modelFingerprint ?? PromptCacheDiskStore.fingerprint(modelDirectory: URL(fileURLWithPath: modelDescription))
        let stamp = String(Int(Date().timeIntervalSince1970 * 1000))
        let directory = output.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/adapters/\(fingerprint)/\(stamp)")
        try refuseInsideModel(directory)
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw .invalidRequestStructure(details: "train: \(directory.path) already exists; the adapter goes to a new directory, nothing is overwritten")
        }
        return directory
    }

    func refuseInsideModel(_ directory: URL) throws(BasaltError) {
        guard modelDescription.isEmpty || !Self.isInside(directory, URL(fileURLWithPath: modelDescription)) else {
            throw .invalidRequestStructure(details: "train: the adapter is never written next to the source weights; choose an output outside \(modelDescription)")
        }
    }

    static func isInside(_ target: URL, _ directory: URL) -> Bool {
        func identity(_ url: URL) -> NSObject? { try? url.resourceValues(forKeys: [.fileResourceIdentifierKey]).fileResourceIdentifier as? NSObject }
        guard let model = identity(resolved(directory)) else { return false }
        var url = resolved(target)
        for _ in url.pathComponents {
            if identity(url)?.isEqual(model) == true { return true }
            url.deleteLastPathComponent()
        }
        return false
    }

    static func resolved(_ url: URL) -> URL {
        let components = url.absoluteURL.pathComponents
        func path(_ count: Int) -> URL { URL(fileURLWithPath: NSString.path(withComponents: Array(components.prefix(count)))) }
        let existing = (1 ... max(1, components.count)).last { FileManager.default.fileExists(atPath: path($0).path) } ?? 1
        return components.dropFirst(existing).reduce(path(existing).resolvingSymlinksInPath()) { $0.appending(component: $1) }.standardizedFileURL
    }
}

extension MLXInferenceEngine {
    static func labRun<R: Sendable>(
        _ container: ModelContainer, _ what: String = "training", _ body: @Sendable (ModelContext) throws -> R
    ) async throws(BasaltError) -> R {
        do {
            return try await container.performEvaluating(nonSendable: ()) { context, _ in try withError { try body(context) } }
        } catch {
            throw labFailure(error, what)
        }
    }

    static func labFailure(_ error: any Error, _ what: String) -> BasaltError {
        switch error {
        case let error as BasaltError: error
        case is CancellationError: .inferenceError("\(what) was cancelled; the model is unchanged")
        default: .inferenceError("\(what) failed: \(briefly(error))")
        }
    }

    static func trainAdapter(
        _ pairs: [TrainingTask], heldOut: [TrainingTask], request: TrainRequest, directory: URL, context: ModelContext
    ) throws -> TrainingRun {
        let method = configuredMethod(explicit: request.method, fallbackRank: request.rank)
        let configuration = try attachConfiguredAdapter(to: context.model, layers: request.layers, method: method)
        defer { _ = PHMAdapterApplicator.revert(in: context.model) }
        let texts = try pairs.map { try trainingText($0, context) }
        let parameters = LoRATrain.Parameters(
            batchSize: min(4, texts.count), iterations: request.iterations, stepsPerReport: 1, stepsPerEval: Int.max, validationBatches: 1)
        var ema = request.emaDecay.map { EMAWeightShadow(decay: $0, model: context.model) }
        var losses: [Float] = []
        try LoRATrain.train(
            model: context.model, train: texts, validate: Array(texts.prefix(1)), optimizer: Adam(learningRate: request.learningRate),
            loss: { _, inputs, targets, lengths in labLoss(context.model, inputs, targets, lengths) }, tokenizer: context.tokenizer, parameters: parameters
        ) { progress in
            if case .train(_, let loss, _, _) = progress {
                losses.append(loss)
                ema?.update(from: context.model)
            }
            return Task.isCancelled ? .stop : .more
        }
        try Task.checkCancellation()
        var emaApplied: Bool?
        if let ema, ema.updates > 0 {
            ema.apply(to: context.model)
            emaApplied = true
        }
        try saveConfiguredAdapter(configuration, model: context.model, to: directory)
        let after = accuracy(heldOut, try heldOut.map { try labAnswer($0.prompt, context) })
        return TrainingRun(losses: reported(losses), accuracyAfter: after, emaApplied: emaApplied)
    }

    static func loraBlocks(of model: Module) -> [Module] {
        LeJEPABackbone.layers(of: model)
    }

    static func attachLoRA(
        to model: Module, blocks: [Module]? = nil, layers: [Int]? = nil, rank: Int, keys: [String]? = nil
    ) throws(BasaltError) -> LoRAConfiguration {
        let blocks = blocks ?? loraBlocks(of: model)
        guard !blocks.isEmpty else { throw .invalidRequestStructure(details: "train: this architecture has no LoRA layers in the fork") }
        guard !model.namedModules().contains(where: { $0.1 is LoRALayer }) else {
            throw .invalidRequestStructure(details: "train: an adapter is applied; undo it before training another")
        }
        let chosen = layers ?? Array(blocks.indices)
        guard let first = chosen.min(), chosen.allSatisfy(blocks.indices.contains) else {
            throw .invalidRequestStructure(details: "train: layers must be a non-empty subset of 0..<\(blocks.count)")
        }
        let defaultKeys = (model as? LoRAModel)?.loraDefaultKeys
            ?? Array(Set(blocks.flatMap { $0.namedModules() }.compactMap { $0.1 is Linear ? $0.0 : nil }))
        let targetKeys = (keys ?? defaultKeys).sorted()
        let parameters = LoRAConfiguration.LoRAParameters(rank: rank, keys: targetKeys)
        model.freeze()
        replaceLoRA(in: blocks, indices: Set(chosen), parameters: parameters)
        model.invalidateCompiledTraces()
        guard !model.trainableParameters().flattened().isEmpty else {
            detachLoRA(from: model, blocks: blocks)
            throw .invalidRequestStructure(details: "train: the chosen layers have no plain or quantized linear projection to adapt")
        }
        return LoRAConfiguration(numLayers: blocks.count - first, loraParameters: parameters)
    }

    static func replaceLoRA(
        in blocks: [Module], indices: Set<Int>, parameters: LoRAConfiguration.LoRAParameters,
        fineTuneType: LoRAConfiguration.FineTuneType = .lora
    ) {
        let keys = parameters.keys ?? []
        for index in indices where blocks.indices.contains(index) {
            let update: [(String, Module)] = blocks[index].namedModules().compactMap { key, module -> (String, Module)? in
                guard keys.contains(key), let linear = module as? Linear, restorable(linear) else { return nil }
                let replacement: Module = switch fineTuneType {
                case .lora: LoRALinear.from(linear: linear, rank: parameters.rank, scale: parameters.scale, dropout: parameters.dropout)
                case .dora: DoRALinear.from(linear: linear, rank: parameters.rank, scale: parameters.scale, dropout: parameters.dropout)
                }
                return (key, replacement)
            }
            if !update.isEmpty { blocks[index].update(modules: .unflattened(update)) }
        }
    }

    static func detachLoRA(from model: Module, blocks: [Module]? = nil) {
        for block in blocks ?? loraBlocks(of: model) {
            let update: [(String, Module)] = block.namedModules().compactMap { key, module -> (String, Module)? in
                (module as? LoRALayer).map { (key, $0.reverted()) }
            }
            if !update.isEmpty { block.update(modules: .unflattened(update)) }
        }
        model.unfreeze()
        model.invalidateCompiledTraces()
    }

    static func labLoss(_ model: any LanguageModel, _ inputs: MLXArray, _ targets: MLXArray, _ lengths: MLXArray) -> (MLXArray, MLXArray) {
        let logits = model(inputs, cache: nil).asType(.float32)
        let mask = MLXArray(0 ..< inputs.dim(1))[.newAxis, 0...] .< lengths[0..., .newAxis]
        let tokens = mask.sum()
        return ((crossEntropy(logits: logits, targets: targets) * mask).sum() / tokens, tokens)
    }

    static func labPrompt(_ prompt: String, _ context: ModelContext, thinking: Bool = false) throws -> [Int] {
        try labPrompt(messages: [["role": "user", "content": prompt]], context.tokenizer, thinking: thinking)
    }

    static func labPrompt(messages: [[String: String]], _ tokenizer: any Tokenizer, thinking: Bool = false) throws -> [Int] {
        try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: ["enable_thinking": thinking])
    }

    static func labAnswer(_ prompt: String, _ context: ModelContext) throws -> String {
        try Task.checkCancellation()
        let stops = stopTokenIds(context)
        let tokens = try greedy(try labPrompt(prompt, context), context: context, probe: nil, maxTokens: answerTokens, stops: stops)
        return context.tokenizer.decode(tokenIds: tokens.filter { !stops.contains($0) }, skipSpecialTokens: true)
    }

    static func trainingText(_ pair: TrainingTask, _ context: ModelContext) throws -> String {
        let rendered = context.tokenizer.decode(tokenIds: try labPrompt(pair.prompt, context), skipSpecialTokens: false)
        let bos = context.tokenizer.bosToken ?? ""
        let prompt = !bos.isEmpty && rendered.hasPrefix(bos) ? String(rendered.dropFirst(bos.count)) : rendered
        return prompt + pair.answer + (context.tokenizer.eosToken ?? "")
    }

    static func accuracy(_ tasks: [TrainingTask], _ answers: [String]) -> Float {
        Float(zip(tasks, answers).filter { $0.accepts($1) }.count) / Float(max(tasks.count, 1))
    }

    static func reported(_ losses: [Float]) -> [Float] {
        let size = max(1, (losses.count + reportedLosses - 1) / reportedLosses)
        return stride(from: 0, to: losses.count, by: size).map { start in
            let chunk = losses[start ..< min(start + size, losses.count)]
            return chunk.reduce(0, +) / Float(chunk.count)
        }
    }
}

extension MLXInferenceEngine {
    func trainADPO(
        _ request: TrainRequest,
        adpo: ADPOConfig,
        container: ModelContainer
    ) async throws(BasaltError) -> TrainReport {
        let epoch = modelEpoch
        let tasks = try TrainingTask.load(request.tasks)
        let heldOut = Array(tasks.suffix(max(1, tasks.count / 5)))
        let practice = Array(tasks.dropLast(heldOut.count))
        let directory = try adapterDirectory(request.output)
        var teacher: TeacherClient?
        if let url = request.teacher { teacher = try TeacherClient(url, model: request.teacherModel) }
        let generations = try await Self.labRun(container) { context in
            let greedyAnswers = try tasks.map { try Self.labAnswer($0.prompt, context) }
            let sampledAnswers = try zip(practice, greedyAnswers).enumerated().map { index, item -> String? in
                guard item.0.accepts(item.1) else { return nil }
                return try Self.labSampledAnswer(item.0.prompt, seed: UInt64(0xAD90_0000 + index), context)
            }
            return (greedyAnswers, sampledAnswers)
        }
        let before = Self.accuracy(heldOut, Array(generations.0.suffix(heldOut.count)))
        let collected = try await Self.collectADPOPairs(
            practice: practice,
            own: Array(generations.0.prefix(practice.count)),
            sampled: generations.1,
            teacher: teacher,
            adpo: adpo
        )
        var report = TrainReport(
            collected: collected.collected,
            verified: collected.pairs.count,
            losses: [],
            accuracyBefore: before,
            accuracyAfter: before,
            verdict: AdapterVerdict(overlay: nil, report: nil)
        )
        guard !collected.pairs.isEmpty else { return report }
        guard self.container === container, modelEpoch == epoch else {
            throw .inferenceError("the model changed while preference pairs were collected; nothing was trained")
        }
        let pairs = collected.pairs
        let session = try await Self.labRun(container) { context in
            try Self.runADPOSession(pairs, heldOut: heldOut, request: request, adpo: adpo, directory: directory, context: context)
        }
        guard self.container === container, modelEpoch == epoch else {
            throw .inferenceError("the model changed during training; the adapter in \(directory.path) was not applied")
        }
        Self.populateADPOReport(&report, outcome: session.0, accuracyAfter: session.1, directory: directory)
        do {
            report.verdict = try await applyAdapter(.directory(directory.path), gate: request.gate)
        } catch {
            throw .inferenceError("the adapter was trained and saved in \(directory.path) but not applied: \(Self.briefly(error))")
        }
        return report
    }

    static func labSampledAnswer(_ prompt: String, seed: UInt64, _ context: ModelContext) throws -> String {
        try Task.checkCancellation()
        let stops = stopTokenIds(context)
        let promptIds = try labPrompt(prompt, context)
        let tokens = try withRandomState(MLXRandom.RandomState(seed: seed)) {
            let params = GenerateParameters(maxTokens: answerTokens, temperature: 0.9)
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(promptIds.map(Int32.init))),
                model: context.model,
                parameters: params
            )
            var out: [Int] = []
            while let token = iterator.next() {
                if stops.contains(token) { break }
                out.append(token)
            }
            return out
        }
        return context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)
    }

    private static func populateADPOReport(
        _ report: inout TrainReport,
        outcome: ADPOTrainOutcome,
        accuracyAfter: Float,
        directory: URL
    ) {
        report.losses = reported(outcome.losses)
        report.accuracyAfter = accuracyAfter
        report.preferenceMarginBefore = outcome.marginBefore
        report.preferenceMarginAfter = outcome.marginAfter
        report.entropyBefore = outcome.entropyBefore
        report.entropyAfter = outcome.entropyAfter
        report.sharpeningTax = outcome.sharpeningTax
        report.passAtKDiversity = outcome.passAtKDiversity
        report.sharpeningGuardTriggered = outcome.sharpeningGuardTriggered
        report.earlyStoppedStep = outcome.earlyStoppedStep
        report.emaApplied = outcome.emaApplied
        report.adapterDirectory = directory.path
    }
}

extension MLXInferenceEngine {
    static func collectADPOPairs(
        practice: [TrainingTask],
        own: [String],
        sampled: [String?],
        teacher: TeacherClient?,
        adpo: ADPOConfig
    ) async throws(BasaltError) -> (pairs: [ADPOPair], collected: Int) {
        var pairs = adpo.pairs
        var count = practice.count + adpo.pairs.count
        for (index, (task, ownAnswer)) in zip(practice, own).enumerated() {
            if task.accepts(ownAnswer) {
                if let alt = sampled[index], !alt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                   !task.accepts(alt), alt != ownAnswer {
                    count += 1
                    pairs.append(ADPOPair(prompt: task.prompt, chosen: ownAnswer, rejected: alt))
                }
                continue
            }
            var chosen: String?
            if let teacher {
                let candidate = try await teacher.answer(task.prompt)
                count += 1
                if task.accepts(candidate) { chosen = candidate }
            }
            if chosen == nil, adpo.useGroundTruthChosen {
                chosen = task.answer
            }
            let rejected = ownAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
            if let chosen, !rejected.isEmpty, chosen != ownAnswer {
                pairs.append(ADPOPair(prompt: task.prompt, chosen: chosen, rejected: ownAnswer))
            }
        }
        return (pairs, count)
    }

    static func runADPOSession(
        _ pairs: [ADPOPair],
        heldOut: [TrainingTask],
        request: TrainRequest,
        adpo: ADPOConfig,
        directory: URL,
        context: ModelContext
    ) throws -> (ADPOTrainOutcome, Float) {
        let tokenized = try tokenizeADPOPairs(pairs, context: context)
        guard !tokenized.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "train: no valid tokenized preference pairs after vocabulary filtering")
        }
        let method = configuredMethod(explicit: request.method, fallbackRank: request.rank)
        let configuration = try attachConfiguredAdapter(to: context.model, layers: request.layers, method: method)
        defer { _ = PHMAdapterApplicator.revert(in: context.model) }
        let options = ADPOOptimizeOptions(
            iterations: request.iterations,
            learningRate: request.learningRate,
            batchSize: min(4, tokenized.count),
            emaDecay: request.emaDecay ?? adpo.emaDecay
        )
        let outcome = try ADPOTrainer.optimize(
            model: context.model,
            pairs: tokenized,
            config: adpo,
            options: options,
            stops: stopTokenIds(context)
        )
        try Task.checkCancellation()
        try saveConfiguredAdapter(configuration, model: context.model, to: directory)
        let after = accuracy(heldOut, try heldOut.map { try labAnswer($0.prompt, context) })
        return (outcome, after)
    }

    static func tokenizeADPOPairs(_ pairs: [ADPOPair], context: ModelContext) throws -> [TokenizedPreferencePair] {
        let vocab = vocabularySize(context.model)
        let eos = context.tokenizer.eosTokenId.map { [$0] } ?? []
        func valid(_ ids: [Int]) -> Bool { !ids.isEmpty && ids.allSatisfy { $0 >= 0 && $0 < vocab } }
        var tokenized: [TokenizedPreferencePair] = []
        for pair in pairs {
            let promptIds = try labPrompt(pair.prompt, context)
            let chosenIds = context.tokenizer.encode(text: pair.chosen, addSpecialTokens: false) + eos
            let rejectedIds = context.tokenizer.encode(text: pair.rejected, addSpecialTokens: false) + eos
            if valid(promptIds), valid(chosenIds), valid(rejectedIds), chosenIds != rejectedIds {
                tokenized.append(TokenizedPreferencePair(promptIds: promptIds, chosenIds: chosenIds, rejectedIds: rejectedIds))
            }
        }
        return tokenized
    }
}
