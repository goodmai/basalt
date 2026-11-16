import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import BasaltCore

extension AdapterProgress {
    init(_ request: CorpusTrainRequest, model: String, data: String, layerCount: Int) {
        (self.model, self.data, self.layerCount) = (model, data, layerCount)
        layers = request.layers ?? Array(0 ..< layerCount)
        let resolved = request.effectiveMethod
        let effectiveRank: Int = switch resolved {
        case .lora(let r, _): r
        case .phm(_, let r, _, _): r
        case .oft(let b, _, _): b
        }
        (rank, method, batchSize, learningRate) = (effectiveRank, resolved, request.batchSize, request.learningRate)
    }
}

extension MLXInferenceEngine {
    public func trainCorpus(_ request: CorpusTrainRequest, lease: StepLease?) async throws(BasaltError) -> CorpusTrainReport {
        try await trainCorpus(
            request,
            transport: URLSessionCorpusTransport(),
            home: FileManager.default.homeDirectoryForCurrentUser,
            environment: ProcessInfo.processInfo.environment,
            lease: lease
        )
    }

    public func trainCorpus(
        _ request: CorpusTrainRequest,
        transport: any CorpusHTTPTransport,
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        lease: StepLease? = nil
    ) async throws(BasaltError) -> CorpusTrainReport {
        try Self.validateCorpusRequest(request)
        let resolvedMethod = Self.configuredMethod(explicit: request.method, fallbackRank: request.rank)
        let steps = try adapterSteps(layers: request.layers, method: resolvedMethod, lease: lease, what: "corpus training")
        let fetched = try await HFDatasetStreamer.load(request.corpus, transport: transport, home: home, environment: environment)
        let parsedRows = try NemotronRowMapper.parseJSONLines(fetched.jsonl)
        let split = try NemotronRowMapper.prepare(rows: parsedRows, spec: request.corpus, heldOutFraction: request.heldOutFraction)
        let directory = request.resume ? try resumedDirectory(request.output) : try adapterDirectory(request.output)
        do {
            return try await runCorpusTraining(request, fetched: fetched, split: split, directory: directory, steps: steps)
        } catch {
            throw AdapterCheckpoint.withResumeHint(error, in: directory)
        }
    }

    private func runCorpusTraining(
        _ request: CorpusTrainRequest,
        fetched: FetchedCorpus,
        split: PreparedCorpusSplit,
        directory: URL,
        steps: AdapterSteps
    ) async throws(BasaltError) -> CorpusTrainReport {
        let resolvedMethod = Self.configuredMethod(explicit: request.method, fallbackRank: request.rank)
        var report = CorpusTrainReport()
        (report.fetchedRows, report.keptRows, report.heldOutRows, report.rejected) = (fetched.rowCount, split.train.count, split.heldOut.count, split.rejected)
        report.cacheFile = fetched.cacheFile?.path
        let markers = profile.thinkingMarkers
        let (run, digest) = try await reserving(directory, resume: request.resume) { () async throws(BasaltError) in
            try await CorpusAdapterRunner.train(split: split, markers: markers, request: request, directory: directory, steps: steps)
        }
        (report.losses, report.nllBefore, report.nllAfter, report.resumedAt) = (run.losses, run.nllBefore, run.nllAfter, run.resumedAt)
        report.rejected["too_long", default: 0] += run.tooLong
        report.adapterDirectory = directory.path
        var provenance = AdapterProvenance(
            corpus: request.corpus,
            datasetRepo: fetched.repo,
            baseModelFingerprint: steps.fingerprint,
            method: resolvedMethod
        )
        (provenance.subset, provenance.split) = (fetched.subset, fetched.split)
        (provenance.trainRows, provenance.heldOutRows, provenance.rowDigest) = (split.train.count, split.heldOut.count, digest)
        (provenance.iterations, provenance.nllBefore, provenance.nllAfter) = (request.iterations, run.nllBefore, run.nllAfter)
        if request.applyOverlay {
            let verdict = try await applyDistilled(directory, gate: request.gate, steps: steps)
            report.verdict = verdict
            provenance.gatePassed = verdict.overlay != nil
        }
        try CorpusAdapterRunner.writeProvenance(provenance, to: directory)
        report.provenance = provenance
        return report
    }

    private static func validateCorpusRequest(_ request: CorpusTrainRequest) throws(BasaltError) {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "train: \(reason)") }
        try request.corpus.validate(prefix: "train")
        guard request.heldOutFraction.isFinite, request.heldOutFraction > 0, request.heldOutFraction < 1 else {
            throw refused("heldOutFraction must lie in (0, 1)")
        }
        try request.effectiveMethod.validate(prefix: "train")
        guard trainingIterations.contains(request.iterations) else { throw refused("iterations must be in \(trainingIterations)") }
        guard (1 ... 64).contains(request.batchSize) else { throw refused("batchSize must be in 1...64") }
        guard (16 ... 32_768).contains(request.maxSeqLen) else { throw refused("maxSeqLen must be in 16...32768") }
        guard request.learningRate.isFinite, request.learningRate > 0, request.learningRate <= 1 else { throw refused("learningRate must lie in (0, 1]") }
        guard request.checkpointEvery >= 0 else { throw refused("checkpointEvery must be 0 or positive") }
    }
}

enum CorpusAdapterRunner {
    static func train(
        split: PreparedCorpusSplit,
        markers: ThinkingMarkers,
        request: CorpusTrainRequest,
        directory: URL,
        steps: AdapterSteps
    ) async throws(BasaltError) -> (DistillRun, String) {
        let (rows, layerCount) = try await steps.read { context in
            let prepared = try prepareRows(split: split, markers: markers, request: request, context: context)
            return (prepared, MLXInferenceEngine.loraBlocks(of: context.model).count)
        }
        let digest = AdapterProgress.digest(rows.train + rows.heldOut)
        let identity = AdapterProgress(request, model: steps.fingerprint, data: digest, layerCount: layerCount)
        var state = AdapterState(optimizer: Adam(learningRate: request.learningRate))
        var progress = try await startState(&state, identity: identity, request: request, directory: directory, steps: steps)
        let resumedAt = request.resume ? progress.step : nil
        var before: Float = 0
        for row in rows.heldOut { before += try await steps.read { context in MLXInferenceEngine.nll([row], context.model) } }
        while progress.step < request.iterations {
            let next = try await steps.section(&state) { [progress, rows] context, state in
                progress.advanced(by: try steps.withAdapter(&state, context: context) { context, optimizer, _ in
                    try MLXInferenceEngine.distillStep(rows.train, index: progress.step, batchSize: request.batchSize, optimizer: optimizer, model: context.model)
                })
            }
            try await steps.confirm()
            guard next.losses.last?.isFinite != false else {
                AdapterCheckpoint.discard(in: directory)
                throw .inferenceError("train: loss became non-finite at step \(progress.step); checkpoint discarded")
            }
            if next.checkpointDue(every: request.checkpointEvery, of: request.iterations) {
                try MLXInferenceEngine.checkpoint(state, next, to: directory)
            }
            progress = next
        }
        var after: Float = 0
        for row in rows.heldOut { after += try await steps.trainedAdapter(&state) { context, _, _ in MLXInferenceEngine.nll([row], context.model) } }
        try await steps.trainedAdapter(&state) { context, _, configuration in
            try MLXInferenceEngine.saveConfiguredAdapter(configuration, model: context.model, to: directory)
        }
        AdapterCheckpoint.discard(in: directory)
        let denom = Float(max(1, rows.heldOut.count))
        return (DistillRun(losses: MLXInferenceEngine.reported(progress.losses), nllBefore: before / denom, nllAfter: after / denom, tooLong: rows.tooLong, resumedAt: resumedAt), digest)
    }

    private static func startState(
        _ state: inout sending AdapterState,
        identity: AdapterProgress,
        request: CorpusTrainRequest,
        directory: URL,
        steps: AdapterSteps
    ) async throws(BasaltError) -> AdapterProgress {
        if request.resume {
            let progress = try AdapterCheckpoint.progress(in: directory, as: identity, iterations: request.iterations)
            try await steps.section(&state) { context, state in try steps.restore(&state, from: directory, context: context) }
            return progress
        }
        if request.checkpointEvery > 0 {
            try await steps.section(&state) { context, state in try steps.withAdapter(&state, context: context) { _, _, _ in } }
            try await steps.confirm()
            try MLXInferenceEngine.checkpoint(state, identity, to: directory)
        }
        return identity
    }

    private static func prepareRows(
        split: PreparedCorpusSplit,
        markers: ThinkingMarkers,
        request: CorpusTrainRequest,
        context: ModelContext
    ) throws -> (train: [DistillRow], heldOut: [DistillRow], tooLong: Int) {
        let all = try (split.train + split.heldOut).map {
            try encodeExample($0, includeReasoning: request.corpus.includeReasoning, markers: markers, context: context)
        }
        let fits = all.map { $0.tokens.count <= request.maxSeqLen }
        let trainRows = zip(all.prefix(split.train.count), fits.prefix(split.train.count)).filter(\.1).map(\.0)
        let heldOutRows = zip(all.suffix(split.heldOut.count), fits.suffix(split.heldOut.count)).filter(\.1).map(\.0)
        guard !trainRows.isEmpty, !heldOutRows.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "train: every training or held-out example exceeds maxSeqLen \(request.maxSeqLen) tokens")
        }
        return (trainRows, heldOutRows, fits.filter { !$0 }.count)
    }

    private static func encodeExample(
        _ example: CorpusExample,
        includeReasoning: Bool,
        markers: ThinkingMarkers,
        context: ModelContext
    ) throws -> DistillRow {
        let formatted = NemotronRowMapper.formatTarget(
            reasoning: example.reasoning,
            answer: example.answer,
            includeReasoning: includeReasoning,
            markers: markers
        )
        let thinking = MLXInferenceEngine.opensThinking(formatted)
        let prompt = try MLXInferenceEngine.labPrompt(messages: example.contextMessages, context.tokenizer, thinking: thinking)
        var answer = Substring(formatted)
        if thinking, MLXInferenceEngine.tailOpensThinking(context.tokenizer.decode(tokenIds: prompt, skipSpecialTokens: false)),
           let opener = MLXInferenceEngine.thinkingOpeners.first(where: { answer.hasPrefix($0) }) {
            answer = answer.dropFirst(opener.count).drop(while: \.isNewline)
        }
        let eos = context.tokenizer.eosTokenId.map { [$0] } ?? []
        let completion = context.tokenizer.encode(text: String(answer), addSpecialTokens: false) + eos
        return DistillRow(tokens: prompt + completion, promptCount: prompt.count)
    }

    static func writeProvenance(_ provenance: AdapterProvenance, to directory: URL) throws(BasaltError) {
        do {
            try withTaskCancellationShield {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(provenance)
                let file = directory.appending(component: "adapter_provenance.json")
                try data.write(to: file, options: .atomic)
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
            }
        } catch {
            throw .inferenceError("train: could not write adapter_provenance.json in \(directory.path): \(error.localizedDescription)")
        }
    }
}
