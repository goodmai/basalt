import Foundation
import MLXLMCommon
import MLXGuidedGeneration
import BasaltCore

struct GrammarPool: Sendable {
    private(set) var idle: [String: [GrammarConstraint]] = [:]
    private var recency: [String] = []

    mutating func take(_ key: String) -> GrammarConstraint? {
        idle[key]?.popLast()
    }

    mutating func put(_ grammar: GrammarConstraint, key: String) {
        recency.removeAll { $0 == key }
        recency.append(key)
        if recency.count > MLXInferenceEngine.grammarSchemaLimit { idle[recency.removeFirst()] = nil }
        guard idle[key, default: []].count < MLXInferenceEngine.idleGrammarsPerSchema else { return }
        idle[key, default: []].append(grammar)
    }
}

extension MLXInferenceEngine {
    struct OutputConstraint: Sendable {
        let session: GrammarSession
        let vocabulary: GrammarVocabulary
        let toolName: String?
        let fastForward: Bool
    }

    struct ConstraintUnavailable: Error {
        let reason: String
    }

    static let grammarSchemaLimit = 16
    static let idleGrammarsPerSchema = 4

    var idleGrammars: [String: [GrammarConstraint]] { grammarPool.idle }

    func outputConstraint(for request: GenerationRequest, allowsFastForward: Bool = true) async throws(BasaltError) -> OutputConstraint? {
        let forcedTool = request.toolChoice?.forcedTool.flatMap { name in request.tools?.first { $0.name == name } }
        let format = forcedTool.map { $0.parametersJSON.map { .jsonSchema($0) } ?? .jsonObject } ?? request.responseFormat
        guard let format else { return nil }
        func unconstrained(_ reason: String) throws(BasaltError) -> OutputConstraint? {
            guard let forcedTool else { throw .invalidRequestStructure(details: "response_format: \(reason)") }
            logger.info("tool_choice \(forcedTool.name): arguments left unconstrained, \(reason)")
            return nil
        }
        let schema: String
        do { schema = try format.grammarSource() } catch { return try unconstrained(error.message) }
        let epoch = modelEpoch
        let vocabulary: GrammarVocabulary
        switch await loadedGrammarVocabulary() {
        case .success(let loaded): vocabulary = loaded
        case .failure(let error): return try unconstrained(error.reason)
        }
        guard !vocabulary.stopTokens.isEmpty else { return try unconstrained("the model declares no stop token to end the document") }
        let fastForwardRequested = allowsFastForward && (request.grammarJumpForward ?? grammarJumpForward)
        let fastForward: Bool
        if fastForwardRequested, let container {
            fastForward = await supportsMultiTokenForward(container, epoch: epoch)
        } else {
            fastForward = false
        }
        let poolKey = format.kind + ":" + schema + (fastForward ? ":ff" : "")
        let grammar: GrammarConstraint
        if epoch == modelEpoch, let idle = grammarPool.take(poolKey) {
            grammar = idle
        } else {
            do {
                grammar = try Self.compile(format, source: schema, vocabulary: vocabulary, fastForward: fastForward)
            } catch GrammarError.invalidJSONSchema(let message) {
                return try unconstrained(message)
            } catch GrammarError.constraintCompilationFailed(let message) {
                return try unconstrained("the \(format.kind) does not compile: \(message)")
            } catch {
                return try unconstrained("\(error)")
            }
        }
        return OutputConstraint(
            session: GrammarSession(schema: schema, grammar: grammar, poolKey: poolKey), vocabulary: vocabulary,
            toolName: forcedTool?.name, fastForward: fastForward)
    }

    static func compile(
        _ format: ResponseFormat, source: String, vocabulary: GrammarVocabulary, fastForward: Bool
    ) throws -> GrammarConstraint {
        let (tokenizer, host) = (vocabulary.tokenizer, fastForward ? vocabulary.hostTokenizer : nil)
        switch format {
        case .jsonObject, .jsonSchema:
            return try GrammarConstraint(tokenizer: tokenizer, jsonSchema: source, fastForward: fastForward, hostTokenizer: host)
        case .regex:
            return try GrammarConstraint(tokenizer: tokenizer, regex: source, fastForward: fastForward, hostTokenizer: host)
        case .grammar:
            return try GrammarConstraint(tokenizer: tokenizer, grammar: source, fastForward: fastForward, hostTokenizer: host)
        case .structuralTag:
            return try GrammarConstraint(tokenizer: tokenizer, structuralTag: source, fastForward: fastForward, hostTokenizer: host)
        }
    }

    func recycle(_ session: GrammarSession, epoch: ModelEpoch) {
        guard epoch == modelEpoch, let grammar = session.rewound() else { return }
        grammarPool.put(grammar, key: session.poolKey)
    }

    func loadedGrammarVocabulary() async -> Result<GrammarVocabulary, ConstraintUnavailable> {
        if let grammarVocabulary { return grammarVocabulary }
        guard let container else { return .failure(ConstraintUnavailable(reason: "no model is loaded")) }
        let epoch = modelEpoch
        let loaded: Result<GrammarVocabulary, ConstraintUnavailable> = await container.perform { context in
            do { return .success(try GrammarVocabulary(context: context)) } catch {
                return .failure(ConstraintUnavailable(reason: "constrained decoding is unavailable for this model: \(error)"))
            }
        }
        if epoch == modelEpoch { grammarVocabulary = loaded }
        return loaded
    }

    static func constraining(
        _ components: GenerationComponents, with constraint: OutputConstraint?, phase: JSONConstraintProcessor.Phase
    ) -> GenerationComponents {
        guard let constraint else { return components }
        return components.appendingBasaltProcessor {
            JSONConstraintProcessor(session: constraint.session, vocabulary: constraint.vocabulary, phase: phase)
        }
    }

    static func batchStopTokens(_ context: ModelContext) -> Set<Int> {
        stopTokenIds(context).union(context.tokenizer.unknownTokenId.map { [$0] } ?? [])
    }

    static func stopTokenIds(_ context: ModelContext) -> Set<Int> {
        var ids = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { ids.insert(eos) }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        return ids
    }
}
