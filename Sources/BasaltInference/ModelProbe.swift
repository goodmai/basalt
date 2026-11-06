import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import MLXVLM
import BasaltCore

package struct ModelProbe: Codable, Sendable {
    package enum Check: String, Codable, Sendable {
        case chatTemplate = "chat template"
        case roundTrip = "tokenizer round trip"
        case stopTokens = "stop tokens"
        case finiteLogits = "finite logits"
        case entropy
        case perplexity
        case knownAnswer = "known answer"
    }

    package enum Verdict: Codable, Sendable, Comparable {
        case pass, warn, fail
    }

    package struct Finding: Codable, Sendable, Equatable {
        package let check: Check
        package let verdict: Verdict
        package let detail: String
    }

    package enum Template: Sendable, Equatable {
        case rendered(tokens: Int)
        case missing
        case failed(String)
    }

    package struct StopToken: Sendable, Equatable {
        let id: Int
        let name: String?
    }

    package struct Answer: Sendable, Equatable {
        let expected: String
        let got: String
    }

    package struct Facts: Sendable, Equatable {
        var template: Template
        var echoed: String
        var stopTokens: [StopToken]
        var vocabularySize: Int
        var nonFiniteLogits: Int
        var meanEntropy: Float
        var perplexity: Float
        var answers: [Answer]
    }

    static let conversation: [[String: any Sendable]] = [
        ["role": "user", "content": "Hi"],
        ["role": "assistant", "content": "Hello. How can I help?"],
        ["role": "user", "content": "Repeat exactly: blue cat"],
    ]
    static let roundTripText = "The quick brown fox jumps over the lazy dog, 42 times."
    static let knownAnswers = [
        (prompt: "Repeat the input exactly.\nInput: green tree\nOutput: green tree\nInput: blue cat\nOutput:", expected: "blue cat"),
        (prompt: "2 + 3 = 5\n4 + 4 =", expected: "8"),
    ]
    static let answerTokens = 4
    static let perplexityText = """
        It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, \
        it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, \
        it was the spring of hope, it was the winter of despair.
        """
    static let entropyFloor: Float = 0.01
    static let uniformEntropyShare: Float = 0.9
    static let perplexityCeiling: Float = 40
    static let familyPerplexityCeilings: [String: Float] = [
        "qwen3_5": 30, "prism_hadamard_qwen35": 5, "gemma4": 24, "gemma4_unified": 99, "lfm2_moe": 8,
    ]

    package var findings: [Finding]

    package var verdict: Verdict { findings.map(\.verdict).max() ?? .pass }
    package var failure: Finding? { findings.first { $0.verdict == .fail } }

    package init(_ facts: Facts, family: String? = nil) {
        findings = [
            Self.template(facts.template), Self.roundTrip(facts.echoed),
            Self.stops(facts.stopTokens, vocabularySize: facts.vocabularySize), Self.finite(facts.nonFiniteLogits),
        ].compactMap { $0 }
        guard facts.nonFiniteLogits == 0 else { return }
        findings += [Self.entropy(facts.meanEntropy, vocabularySize: facts.vocabularySize), Self.perplexity(facts.perplexity, family: family)].compactMap { $0 }
        findings += facts.answers.compactMap(Self.answer)
    }

    static func template(_ template: Template) -> Finding? {
        switch template {
        case .rendered(let tokens) where tokens > 0:
            nil
        case .rendered:
            Finding(check: .chatTemplate, verdict: .fail, detail: "the chat template rendered a three-turn conversation to no tokens")
        case .missing:
            Finding(check: .chatTemplate, verdict: .warn, detail: "the model has no chat template; chat requests are joined as plain text")
        case .failed(let error):
            Finding(check: .chatTemplate, verdict: .fail, detail: "the chat template does not render a three-turn conversation: \(error)")
        }
    }

    static func roundTrip(_ echoed: String) -> Finding? {
        echoed.trimmingCharacters(in: .whitespacesAndNewlines) == roundTripText ? nil
            : Finding(check: .roundTrip, verdict: .warn, detail: "encoding then decoding \"\(roundTripText)\" gave \"\(echoed)\"")
    }

    static func stops(_ tokens: [StopToken], vocabularySize: Int) -> Finding? {
        guard !tokens.isEmpty else {
            return Finding(check: .stopTokens, verdict: .fail, detail: "no EOS or stop token, so a generation only ends at max_tokens")
        }
        let outside = tokens.filter { !(0 ..< vocabularySize).contains($0.id) }.map(\.id).sorted()
        let unnamed = tokens.filter { $0.name == nil }.map(\.id).sorted()
        if !outside.isEmpty {
            return Finding(check: .stopTokens, verdict: .fail, detail: "stop token ids \(outside) are not in the vocabulary of \(vocabularySize) tokens")
        }
        return unnamed.isEmpty ? nil
            : Finding(check: .stopTokens, verdict: .warn, detail: "the tokenizer has no name for stop token ids \(unnamed)")
    }

    static func finite(_ nonFinite: Int) -> Finding? {
        nonFinite == 0 ? nil
            : Finding(check: .finiteLogits, verdict: .fail, detail: "a forward over a short fixed text gave \(nonFinite) NaN or +Inf logits")
    }

    static func entropy(_ mean: Float, vocabularySize: Int) -> Finding? {
        let ceiling = uniformEntropyShare * Float(log(Double(max(vocabularySize, 2))))
        guard !(entropyFloor ... ceiling).contains(mean) else { return nil }
        return Finding(
            check: .entropy, verdict: .warn,
            detail: "mean next-token entropy \(Self.rounded(mean)) nats is outside \(entropyFloor)...\(Self.rounded(ceiling)), 90% of uniform over \(vocabularySize) tokens: one token takes all the mass, or none stands out")
    }

    static func perplexityCeiling(family: String?) -> Float {
        family.flatMap { familyPerplexityCeilings[$0] } ?? perplexityCeiling
    }

    static func perplexity(_ value: Float, family: String?) -> Finding? {
        let ceiling = perplexityCeiling(family: family)
        return value.isFinite && value <= ceiling ? nil
            : Finding(check: .perplexity, verdict: .warn, detail: "perplexity \(Self.rounded(value)) on a well-known fixed text is above \(Self.rounded(ceiling)), the ceiling for \(family ?? "an unknown family")")
    }

    static func answer(_ answer: Answer) -> Finding? {
        answer.got.lowercased().contains(answer.expected) ? nil
            : Finding(check: .knownAnswer, verdict: .warn, detail: "greedy output \"\(answer.got)\" does not contain \"\(answer.expected)\"")
    }

    static func rounded(_ value: Float) -> String { String(format: "%.2f", value) }
}

extension ModelProbe {
    static let logger = BasaltLogger(module: "ModelProbe")

    static func facts(_ context: ModelContext) throws -> Facts {
        let tokenizer = context.tokenizer
        let text = encoded(perplexityText, tokenizer)
        let input = LMInput.Text(tokens: MLXArray(text.map(Int32.init)).reshaped(1, -1))
        let logits = context.model(input, cache: try context.model.newCache(parameters: nil), state: nil).logits[0].asType(.float32)
        let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
        let entropy = EntropyDynamicTemperature.entropy(logits).mean()
        let targets = MLXArray(text.dropFirst().map(Int32.init))[0..., .newAxis]
        let perplexity = exp(-takeAlong(logp[0 ..< (text.count - 1)], targets, axis: -1).mean())
        let nonFinite = isNaN(logits).sum() + isPosInf(logits).sum()
        eval(entropy, perplexity, nonFinite)
        return Facts(
            template: template(tokenizer),
            echoed: tokenizer.decode(tokenIds: tokenizer.encode(text: roundTripText, addSpecialTokens: false), skipSpecialTokens: true),
            stopTokens: MLXInferenceEngine.stopTokenIds(context).sorted().map { StopToken(id: $0, name: tokenizer.convertIdToToken($0)) },
            vocabularySize: logits.dim(-1), nonFiniteLogits: nonFinite.item(Int.self),
            meanEntropy: entropy.item(Float.self), perplexity: perplexity.item(Float.self),
            answers: try knownAnswers.map { try known($0.prompt, expected: $0.expected, context: context) })
    }

    static func encoded(_ text: String, _ tokenizer: any MLXLMCommon.Tokenizer) -> [Int] {
        let ids = tokenizer.encode(text: text, addSpecialTokens: true)
        guard let bos = tokenizer.bosToken.flatMap(tokenizer.convertTokenToId), ids.first != bos else { return ids }
        return [bos] + ids
    }

    static func template(_ tokenizer: any MLXLMCommon.Tokenizer) -> Template {
        do {
            return .rendered(tokens: try tokenizer.applyChatTemplate(messages: conversation, tools: nil, additionalContext: nil).count)
        } catch MLXLMCommon.TokenizerError.missingChatTemplate {
            return .missing
        } catch {
            return .failed(MLXInferenceEngine.briefly(error))
        }
    }

    static func known(_ prompt: String, expected: String, context: ModelContext, tokens: Int = answerTokens) throws -> Answer {
        let chat = try? context.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]], tools: nil, additionalContext: ["enable_thinking": false])
        var answer = Answer(expected: expected, got: "")
        for ids in [chat, encoded(prompt, context.tokenizer)].compactMap({ $0 }) where Self.answer(answer) != nil {
            answer = Answer(expected: expected, got: try greedy(ids, context: context, tokens: tokens))
        }
        return answer
    }

    static func greedy(_ ids: [Int], context: ModelContext, tokens: Int = answerTokens) throws -> String {
        let generated = try TokenIterator(
            input: LMInput(tokens: MLXArray(ids.map(Int32.init))), model: context.model,
            parameters: GenerateParameters(maxTokens: tokens, temperature: 0))
        return context.tokenizer.decode(tokenIds: Array(generated), skipSpecialTokens: true)
    }

    static func run(_ container: ModelContainer, family: String? = nil) async throws -> ModelProbe {
        let gathered = try await container.performEvaluating(nonSendable: ()) { context, _ in try facts(context) }
        logger.info("Model probe measured \(family ?? "unknown family"): perplexity \(rounded(gathered.perplexity)), mean entropy \(rounded(gathered.meanEntropy)) nats, known answers \(gathered.answers.map(\.got))")
        return ModelProbe(gathered, family: family)
    }

    static func stored(fingerprint: String) -> URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/model-probes/\(fingerprint).json")
    }

    static func loadOrRun(directory: URL, container: ModelContainer) async throws -> ModelProbe {
        let url = stored(fingerprint: PromptCacheDiskStore.fingerprint(modelDirectory: directory))
        if let data = try? Data(contentsOf: url), let probe = try? JSONDecoder().decode(ModelProbe.self, from: data) { return probe }
        let probe = try await run(container, family: ModelCache.config(at: directory)?["model_type"] as? String)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? JSONEncoder().encode(probe).write(to: url, options: .atomic)
        return probe
    }
}

extension MLXInferenceEngine {
    func probe(_ container: ModelContainer, directory: URL, modelPath: String) async throws(BasaltError) {
        guard configuration.probesModel else { return }
        let started = ContinuousClock.now
        let probe: ModelProbe
        do {
            probe = try await ModelProbe.loadOrRun(directory: directory, container: container)
        } catch {
            throw .modelProbeFailed(path: modelPath, check: "forward", reason: Self.briefly(error))
        }
        for finding in probe.findings {
            ModelProbe.logger.warn("Model probe, \(finding.check.rawValue): \(finding.detail)")
        }
        if let failure = probe.failure {
            throw .modelProbeFailed(path: modelPath, check: failure.check.rawValue, reason: failure.detail)
        }
        ModelProbe.logger.info("Model probe: \(probe.verdict) in \(ContinuousClock.now - started) (--no-model-probe skips it)")
    }

    static func loadFailure(_ error: any Error, modelPath: String) async -> BasaltError {
        if let error = error as? BasaltError { return error }
        if error is CancellationError { return .cancelled }
        if let error = error as? Tokenizers.TokenizerError {
            return .modelConfigurationInvalid(path: modelPath, reason: "tokenizer: \(error.localizedDescription)")
        }
        switch error as? ModelFactoryError {
        case .configurationDecodingError?:
            return .modelConfigurationInvalid(path: modelPath, reason: error.localizedDescription)
        case .unsupportedModelType(let modelType)? where await !VLMTypeRegistry.shared.contains(modelType):
            return .modelConfigurationInvalid(path: modelPath, reason: error.localizedDescription)
        default:
            return .weightsCorrupted(path: modelPath, reason: error.localizedDescription)
        }
    }
}
