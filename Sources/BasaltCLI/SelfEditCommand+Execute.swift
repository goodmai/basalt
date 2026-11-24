import ArgumentParser
import BasaltBench
import BasaltCloud
import BasaltCore
import BasaltInference
import Foundation

extension SelfEditCommand {
    func parseSeeds() throws -> [UInt64] {
        let parsed = seeds.split(separator: ",").compactMap { UInt64($0.trimmingCharacters(in: .whitespaces)) }
        guard !parsed.isEmpty else {
            throw ValidationError("--seeds must be a non-empty comma-separated list of unsigned integers")
        }
        return parsed
    }

    func resolveModelPath() throws -> String {
        guard let model, !model.isEmpty else {
            throw ValidationError("--model is required to run a self-editing session")
        }
        if FileManager.default.fileExists(atPath: model) {
            return model
        }
        if let cached = ModelCache.resolve(repoId: model)?.path {
            return cached
        }
        throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
    }

    func buildEditor(
        spec: SelfEditEditorSpec,
        privateStrings: Set<String>,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> (any EditSessionEditor)? {
        switch spec {
        case .local:
            return nil
        case .cloudGemini(let modelName):
            let geminiKey = environment["GEMINI_API_KEY"] ?? ""
            let openRouterKey = environment["OPENROUTER_API_KEY"] ?? ""
            let useGemini = !geminiKey.isEmpty
            let apiKey = useGemini ? geminiKey : openRouterKey
            let baseURL = useGemini
                ? GeminiCloudEditorConfig.defaultGeminiBaseURL
                : GeminiCloudEditorConfig.defaultOpenRouterBaseURL
            let cloudConfig = GeminiCloudEditorConfig(
                model: modelName,
                apiKey: apiKey,
                baseURL: baseURL,
                allowCloudEditor: allowCloudEditor,
                maxSessionUSD: maxCloudUsd,
                maxTokensPerTurn: max(256, maxEditorTokens)
            )
            let gate = CloudGate(limits: [modelName: maxCloudUsd], storage: nil)
            return try GeminiCloudEditor(config: cloudConfig, cloudGate: gate, privateStrings: privateStrings)
        }
    }

    func buildLiveRequest(
        sessionId: String,
        prepared: PreparedHarness,
        modelPath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> (config: SelfRewriteLiveConfig, grader: any PairedAnswerGrader, editor: (any EditSessionEditor)?) {
        let seedList = try parseSeeds()
        let topology = MLXInferenceEngine.loadGemma4Topology(fromModelDirectory: URL(fileURLWithPath: modelPath))
        let sessionConfig = SelfRewriteSessionConfig(
            sessionId: sessionId,
            targetScope: prepared.scope,
            maxAlpha: maxAlpha,
            budget: prepared.budget,
            outputDirectory: output.map { URL(fileURLWithPath: $0) },
            topology: topology
        )
        .withTranscriptURL(prepared.transcriptURL)
        .withMaxEditorTokens(maxEditorTokens)

        let evalConfig = PairedTestRunConfig(
            seeds: seedList,
            maxTokens: max(16, maxEvalTokens),
            temperature: 0.0,
            stopTokenIds: [],
            budgetCost: 1
        )
        let liveConfig = SelfRewriteLiveConfig(
            session: sessionConfig,
            taskMode: prepared.taskMode,
            evalConfig: evalConfig,
            reportURL: report.map { URL(fileURLWithPath: $0) }
        )
        let selectedGrader: any PairedAnswerGrader = switch grader {
        case .math: MathAnswerGrader()
        case .translation: TranslationPairedGrader()
        }
        let externalEditor = try buildEditor(
            spec: prepared.editorSpec,
            privateStrings: prepared.split.reservedFenceStrings(),
            environment: environment
        )
        return (liveConfig, selectedGrader, externalEditor)
    }

    func executeLiveSession(
        sessionId: String,
        prepared: PreparedHarness
    ) async throws -> SelfRewriteLiveReport {
        let modelPath = try resolveModelPath()
        let request = try buildLiveRequest(sessionId: sessionId, prepared: prepared, modelPath: modelPath)
        return try await GPULock.withLock(timeout: .seconds(300)) {
            let engine = MLXInferenceEngine()
            let orchestrator = ModelOrchestratorActor(engine: engine)
            defer { await withTaskCancellationShield { await orchestrator.shutdown() } }
            try await orchestrator.loadModel(path: modelPath)
            return try await engine.runSelfRewrite(
                config: request.config,
                split: prepared.split,
                grader: request.grader,
                editor: request.editor
            )
        }
    }
}
