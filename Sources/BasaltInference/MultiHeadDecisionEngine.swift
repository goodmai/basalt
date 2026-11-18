import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    func requireContainer() throws(BasaltError) -> ModelContainer {
        guard let container else { throw .modelNotLoaded }
        return container
    }
}

public actor MultiHeadDecisionEngine: DecisionEngine {
    enum Backbone: Sendable {
        case container(ModelContainer, owns: Bool)
        case engine(MLXInferenceEngine, owns: Bool)
    }

    struct BankSnapshot: Sendable {
        let modelID: String
        let manifest: SystemOneBankManifest
        let weight: [Float]
        let bias: [Float]?
    }

    struct BankRun: Sendable {
        let coveredIDs: [String]
        let answers: [String: SystemOneAnswer]
        let usage: SystemOneUsage
        let latencyMs: Int
    }

    struct FallbackRun: Sendable {
        let uncoveredIDs: [String]
        let response: SystemOneResponse
        let latencyMs: Int
    }

    public nonisolated let modelID: String
    public nonisolated let manifest: SystemOneBankManifest

    public nonisolated var sharesPerceived: Bool {
        fallback?.sharesPerceived ?? true
    }

    public nonisolated var onDevice: (any DecisionEngine)? {
        guard let fallback else { return self }
        return fallback.onDevice != nil ? self : nil
    }

    private var backbone: Backbone?
    private var generation: UInt64 = 0
    private let snapshot: BankSnapshot
    private let fallback: (any DecisionEngine)?

    public static func isBundle(_ directory: URL) -> Bool {
        SystemOneBankBundle.isBundle(directory)
    }

    public static func baseModel(of directory: URL) throws(BasaltError) -> String {
        try SystemOneBankBundle.loadManifest(from: directory).baseModelID
    }

    public init(
        manifest: SystemOneBankManifest,
        weight: [Float],
        bias: [Float]? = nil,
        container: ModelContainer,
        ownsContainer: Bool = false,
        fallback: (any DecisionEngine)? = nil
    ) throws(BasaltError) {
        let resolvedID = manifest.modelID ?? "bank/\(manifest.baseModelID)"
        try Self.validateBank(manifest: manifest, weight: weight, bias: bias, path: resolvedID)
        self.modelID = resolvedID
        self.manifest = manifest
        self.backbone = .container(container, owns: ownsContainer)
        self.snapshot = BankSnapshot(modelID: resolvedID, manifest: manifest, weight: weight, bias: bias)
        self.fallback = fallback
    }

    public init(
        directory: URL,
        container: ModelContainer,
        ownsContainer: Bool = false,
        fallback: (any DecisionEngine)? = nil
    ) throws(BasaltError) {
        let loaded = try SystemOneBankBundle.loadManifest(from: directory)
        let head = try SystemOneBankBundle.loadHead(from: directory, manifest: loaded)
        let resolvedID = loaded.modelID ?? directory.lastPathComponent
        self.modelID = resolvedID
        self.manifest = loaded
        self.backbone = .container(container, owns: ownsContainer)
        self.snapshot = BankSnapshot(modelID: resolvedID, manifest: loaded, weight: head.weight, bias: head.bias)
        self.fallback = fallback
    }

    public init(
        directory: URL,
        engine: MLXInferenceEngine,
        ownsEngine: Bool = false,
        fallback: (any DecisionEngine)? = nil
    ) throws(BasaltError) {
        let loaded = try SystemOneBankBundle.loadManifest(from: directory)
        let head = try SystemOneBankBundle.loadHead(from: directory, manifest: loaded)
        let resolvedID = loaded.modelID ?? directory.lastPathComponent
        self.modelID = resolvedID
        self.manifest = loaded
        self.backbone = .engine(engine, owns: ownsEngine)
        self.snapshot = BankSnapshot(modelID: resolvedID, manifest: loaded, weight: head.weight, bias: head.bias)
        self.fallback = fallback
    }

    public init(directory: URL, base: URL) async throws(BasaltError) {
        let loaded = try SystemOneBankBundle.loadManifest(from: directory)
        let head = try SystemOneBankBundle.loadHead(from: directory, manifest: loaded)
        var configuration = EngineConfiguration.standard
        configuration.promptCacheBytes = 0
        let engine = MLXInferenceEngine(configuration)
        try await engine.load(modelPath: base.path)
        let calibration = SystemOneReadout.calibration(for: loaded.baseModelID, in: base)
        let readoutFallback = ReadoutDecisionEngine(engine: engine, modelID: loaded.baseModelID, readout: calibration)
        let resolvedID = loaded.modelID ?? directory.lastPathComponent
        self.modelID = resolvedID
        self.manifest = loaded
        self.backbone = .engine(engine, owns: true)
        self.snapshot = BankSnapshot(modelID: resolvedID, manifest: loaded, weight: head.weight, bias: head.bias)
        self.fallback = readoutFallback
    }

    public func unload() async {
        generation &+= 1
        let held = backbone
        backbone = nil
        switch held {
        case .container(_, let owns):
            if owns { Memory.clearCache() }
        case .engine(let engine, let owns):
            if owns { await engine.unload() }
        case nil:
            break
        }
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        let epoch = generation
        guard backbone != nil else { throw .modelNotLoaded }
        let partition = SystemOneBankCore.partition(request.questions, manifest: manifest)
        if !partition.uncovered.isEmpty, fallback == nil {
            let names = partition.uncovered.keys.sorted().joined(separator: ", ")
            throw .invalidRequestStructure(details: "systemone bank \(modelID) has no trained slot for: \(names)")
        }
        var bankRun: BankRun?
        if !partition.covered.isEmpty {
            let started = ContinuousClock.now
            let container = try await activeContainer(epoch: epoch)
            let evaluated = try await Self.runBank(container: container, state: request.state, snapshot: snapshot)
            guard generation == epoch, backbone != nil else { throw .modelNotLoaded }
            let bankMs = max(0, Int((ContinuousClock.now - started).inSeconds * 1000))
            let answers = SystemOneBankCore.decode(partition.covered, logits: evaluated.logits.span, request: request)
            let usage = SystemOneUsage(
                prefixTokens: evaluated.prefixTokens,
                candidateBranches: partition.covered.count,
                prefixEvaluations: 1,
                branchForwardCalls: 0,
                suffixTokens: 0
            )
            bankRun = BankRun(coveredIDs: partition.covered.map(\.id), answers: answers, usage: usage, latencyMs: bankMs)
        }
        guard !partition.uncovered.isEmpty, let fallback else {
            let usage = bankRun?.usage ?? SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0)
            return SystemOneResponse(model: modelID, answers: bankRun?.answers ?? [:], usage: usage, calibrated: request.calibrated != false)
        }
        return try await mergeFallback(fallback, request: request, uncovered: partition.uncovered, bankRun: bankRun)
    }

    private func activeContainer(epoch: UInt64) async throws(BasaltError) -> ModelContainer {
        guard generation == epoch, let current = backbone else { throw .modelNotLoaded }
        switch current {
        case .container(let container, _):
            return container
        case .engine(let engine, _):
            let container = try await engine.requireContainer()
            guard generation == epoch, backbone != nil else { throw .modelNotLoaded }
            return container
        }
    }

    private func mergeFallback(
        _ fallback: any DecisionEngine,
        request: SystemOneRequest,
        uncovered: [String: SystemOneQuestion],
        bankRun: BankRun?
    ) async throws(BasaltError) -> SystemOneResponse {
        let epoch = generation
        let started = ContinuousClock.now
        var subRequest = SystemOneRequest(
            state: request.state,
            questions: uncovered,
            calibrated: request.calibrated,
            abstainBelow: request.abstainBelow,
            minMargin: request.minMargin
        )
        subRequest.media = request.media
        let tail = try await fallback.decide(subRequest)
        guard generation == epoch, backbone != nil else { throw .modelNotLoaded }
        let tailMs = max(0, Int((ContinuousClock.now - started).inSeconds * 1000))
        let fallbackRun = FallbackRun(uncoveredIDs: uncovered.keys.sorted(), response: tail, latencyMs: tailMs)
        return Self.combineResponses(
            modelID: modelID,
            calibrated: request.calibrated != false,
            bankRun: bankRun,
            fallbackRun: fallbackRun
        )
    }
}

extension MultiHeadDecisionEngine {
    static func combineResponses(
        modelID: String,
        calibrated: Bool,
        bankRun: BankRun?,
        fallbackRun: FallbackRun
    ) -> SystemOneResponse {
        guard let bankRun else {
            return SystemOneResponse(model: modelID, answers: fallbackRun.response.answers, usage: fallbackRun.response.usage, calibrated: fallbackRun.response.calibrated)
        }
        var merged = bankRun.answers
        for (key, value) in fallbackRun.response.answers { merged[key] = value }
        var usage = SystemOneUsage(
            prefixTokens: bankRun.usage.prefixTokens + fallbackRun.response.usage.prefixTokens,
            candidateBranches: bankRun.usage.candidateBranches + fallbackRun.response.usage.candidateBranches,
            prefixEvaluations: bankRun.usage.prefixEvaluations + fallbackRun.response.usage.prefixEvaluations,
            branchForwardCalls: bankRun.usage.branchForwardCalls + fallbackRun.response.usage.branchForwardCalls,
            suffixTokens: bankRun.usage.suffixTokens + fallbackRun.response.usage.suffixTokens
        )
        usage.outputDecodingTokens = bankRun.usage.outputDecodingTokens + fallbackRun.response.usage.outputDecodingTokens
        usage.cost = fallbackRun.response.usage.cost
        usage.stages = [
            SystemOneStage(model: modelID, questions: bankRun.coveredIDs, latencyMs: bankRun.latencyMs),
            SystemOneStage(
                model: fallbackRun.response.model,
                questions: fallbackRun.uncoveredIDs,
                latencyMs: fallbackRun.latencyMs,
                cost: fallbackRun.response.usage.cost,
                fallback: "uncovered_slot"
            ),
        ]
        return SystemOneResponse(model: modelID, answers: merged, usage: usage, calibrated: calibrated || fallbackRun.response.calibrated)
    }
    static func validateBank(
        manifest: SystemOneBankManifest,
        weight: [Float],
        bias: [Float]?,
        path: String
    ) throws(BasaltError) {
        try manifest.validate(path: path)
        let expected = manifest.totalOutputs * manifest.hiddenSize
        guard weight.count == expected else {
            throw .modelConfigurationInvalid(path: path, reason: "bank weight count \(weight.count) != \(expected)")
        }
        if let bias, bias.count != manifest.totalOutputs {
            throw .modelConfigurationInvalid(path: path, reason: "bank bias count \(bias.count) != \(manifest.totalOutputs)")
        }
    }

    static func runBank(
        container: ModelContainer,
        state: String,
        snapshot: BankSnapshot
    ) async throws(BasaltError) -> (logits: [Float], prefixTokens: Int) {
        do {
            return try await container.performEvaluating(nonSendable: ()) { context, _ in
                try Self.evaluateBank(state: state, snapshot: snapshot, context: context)
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("systemone bank: \(error.localizedDescription)")
        }
    }

    static func evaluateBank(
        state: String,
        snapshot: BankSnapshot,
        context: ModelContext
    ) throws -> (logits: [Float], prefixTokens: Int) {
        let tokens = encodeState(state, tokenizer: context.tokenizer)
        let hidden = try lastHidden(context.model, tokens: tokens)
        guard hidden.dim(0) == snapshot.manifest.hiddenSize else {
            throw BasaltError.modelConfigurationInvalid(
                path: snapshot.modelID,
                reason: "backbone hidden size \(hidden.dim(0)) != bank_manifest hiddenSize \(snapshot.manifest.hiddenSize)"
            )
        }
        let weightMatrix = MLXArray(snapshot.weight, [snapshot.manifest.totalOutputs, snapshot.manifest.hiddenSize])
        var logits = matmul(hidden, weightMatrix.T)
        if let bias = snapshot.bias {
            logits = logits + MLXArray(bias, [snapshot.manifest.totalOutputs])
        }
        let flat = logits.asType(.float32)
        eval(flat)
        return (flat.asArray(Float.self), tokens.count)
    }

    static func encodeState(_ state: String, tokenizer: any Tokenizer) -> [Int] {
        let prompt = SystemOneReadout.prefix(state: state)
        if let templated = try? tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]],
            tools: nil,
            additionalContext: ["enable_thinking": false]
        ), !templated.isEmpty {
            return templated
        }
        return tokenizer.encode(text: prompt, addSpecialTokens: true)
    }

    static func lastHidden(_ model: any LanguageModel, tokens: [Int]) throws -> MLXArray {
        guard !tokens.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "systemone bank prompt encoded to zero tokens")
        }
        guard let norm = finalNorm(of: model) else {
            throw BasaltError.inferenceError("systemone bank: backbone exposes no final normalization layer")
        }
        let probe = FinalResidual()
        let cache = try model.newCache(parameters: nil)
        let input = LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)).reshaped(1, -1))
        _ = withForwardProbe(probe) { model(input, cache: cache, state: nil) }
        guard let residual = probe.hidden else {
            throw BasaltError.inferenceError("systemone bank: backbone reported no residual to ForwardProbe")
        }
        let lastSlice = residual[0 ..< 1, (tokens.count - 1) ..< tokens.count]
        return norm(lastSlice)[0, 0].asType(.float32)
    }

    static func finalNorm(of model: Module) -> (any UnaryLayer)? {
        let modules: [(path: String, name: String?, module: Module)] = model.namedModules()
            .map { (path: $0.0, name: $0.0.split(separator: ".").last.map(String.init), module: $0.1) }
            .sorted { $0.path.split(separator: ".").count < $1.path.split(separator: ".").count }
        let normNames = ["norm", "norm_f", "embedding_norm"]
        if let embedding = ["embed_tokens", "embeddings"].lazy.compactMap({ target in
            modules.first { $0.name == target && $0.module is Embedding }
        }).first {
            let parent = embedding.path.split(separator: ".").dropLast().map { $0 + "." }.joined()
            if let matched = normNames.lazy.compactMap({ target in
                modules.first { $0.path == parent + target }?.module as? any UnaryLayer
            }).first {
                return matched
            }
        }
        return normNames.lazy.compactMap { target in
            modules.first { $0.name == target }?.module as? any UnaryLayer
        }.first
    }
}
