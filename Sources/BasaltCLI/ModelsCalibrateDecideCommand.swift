import ArgumentParser
import BasaltCloud
import BasaltCore
import BasaltInference
import Foundation

struct CalibrateDecideSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "calibrate-decide",
        abstract: "Fit System One readout calibration (basalt_readout.json) against teacher probabilities or gold labels"
    )

    @Argument(help: "Model repo id from the local cache, or a local model directory")
    var model: String

    @Option(name: .long, help: "Path to JSONL calibration dataset")
    var dataset: String

    @Option(name: .long, help: "Optional teacher model spec, e.g. cloud:gemini/gemini-decision-1")
    var teacher: String?

    @Option(name: .customLong("api-key-env"), help: "Environment variable holding the cloud teacher API key")
    var keyEnvName: String = CloudDecisionProvider.gemini.keyEnvironment

    @Option(name: .customLong("max-usd"), help: "Maximum cloud teacher spend in USD")
    var maxUsd: Double = 1.0

    @Option(name: .customLong("cache-dir"), help: "Directory caching teacher probability distributions")
    var cacheDir: String = "~/.basalt/calibration-cache"

    @Option(name: .customLong("holdout-fraction"), help: "Fraction of dataset held out for pre/post metric evaluation")
    var holdoutFraction: Double = 0.2

    @Option(name: .long, help: "Output path for basalt_readout.json (default: <resolved-model-dir>/basalt_readout.json)")
    var out: String?

    func validate() throws {
        guard holdoutFraction.isFinite, (0 ..< 1).contains(holdoutFraction) else {
            throw ValidationError("--holdout-fraction must lie in 0..<1, got \(holdoutFraction)")
        }
        guard maxUsd.isFinite, maxUsd >= 0 else {
            throw ValidationError("--max-usd must be non-negative, got \(maxUsd)")
        }
        if let teacher, teacher.trimmingCharacters(in: .whitespaces).isEmpty {
            throw ValidationError("--teacher cannot be blank")
        }
    }

    mutating func run() async throws {
        let expanded = (model as NSString).expandingTildeInPath
        guard let modelPath = FileManager.default.fileExists(atPath: expanded) ? expanded : ModelCache.resolve(repoId: model)?.path else {
            throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
        }
        let config = CalibrateDecideRunner.Config(
            modelPath: modelPath,
            datasetPath: (dataset as NSString).expandingTildeInPath,
            teacherSpec: teacher,
            keyEnvName: keyEnvName,
            maxUSD: maxUsd,
            cacheDir: URL(fileURLWithPath: (cacheDir as NSString).expandingTildeInPath),
            holdoutFraction: holdoutFraction,
            outputURL: out.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
                ?? URL(fileURLWithPath: modelPath).appending(path: SystemOneReadout.bundleFile)
        )
        let result = try await CalibrateDecideRunner.run(config)
        print(CalibrateDecideRunner.summary(result))
    }
}

struct CalibrateDecideRunner {
    struct Config: Sendable {
        var modelPath: String
        var datasetPath: String
        var teacherSpec: String?
        var keyEnvName: String
        var maxUSD: Double
        var cacheDir: URL
        var holdoutFraction: Double
        var outputURL: URL
    }

    struct Result: Sendable, Equatable {
        var fit: CalibrationFit
        var outputURL: URL
        var totalSamples: Int
        var teacherCalls: Int
        var cachedHits: Int
        var spentUSD: Double
    }

    struct ResolvedTeacher {
        var id: String
        var engine: (any DecisionEngine)?
    }

    struct Collected {
        var samples: [CalibrationSample]
        var calls: Int
        var cached: Int
        var spentUSD: Double
    }

    static func run(
        _ config: Config,
        student injectedStudent: MLXInferenceEngine? = nil,
        teacher injectedTeacher: (any DecisionEngine)? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) async throws -> Result {
        let entries = try CalibrateDatasetIO.parseDataset(at: config.datasetPath)
        guard !entries.isEmpty else { throw ValidationError("Calibration dataset at '\(config.datasetPath)' is empty") }
        let needsStudent = entries.contains { if case .prompt = $0 { return true } else { return false } }
        let ownedStudent = needsStudent && injectedStudent == nil ? uncachedEngine() : nil
        defer {
            if let ownedStudent { await withTaskCancellationShield { await ownedStudent.unload() } }
        }
        if let ownedStudent { try await ownedStudent.load(modelPath: config.modelPath) }
        let resolvedTeacher = try await makeTeacher(config: config, injected: injectedTeacher, environment: environment)
        defer {
            if injectedTeacher == nil, let engine = resolvedTeacher.engine {
                await withTaskCancellationShield { await engine.unload() }
            }
        }
        let collected = try await collectSamples(entries: entries, config: config, student: injectedStudent ?? ownedStudent, teacher: resolvedTeacher)
        let fit = SystemOneCalibrationCore.fit(collected.samples, holdoutFraction: config.holdoutFraction)
        let encoded = try fit.readout.encode()
        try withTaskCancellationShield {
            try FileManager.default.createDirectory(at: config.outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoded.write(to: config.outputURL, options: .atomic)
        }
        return Result(
            fit: fit, outputURL: config.outputURL, totalSamples: collected.samples.count,
            teacherCalls: collected.calls, cachedHits: collected.cached, spentUSD: collected.spentUSD)
    }

    static func summary(_ result: Result) -> String {
        let f = result.fit
        let r = f.readout
        return [
            "samples: \(result.totalSamples) (evaluated: \(f.sampleCount)), teacher calls: \(result.teacherCalls) (\(result.cachedHits) cached, $\(String(format: "%.4f", result.spentUSD)))",
            String(format: "KL:    %.4f -> %.4f", f.klBefore, f.klAfter),
            String(format: "ECE:   %.4f -> %.4f", f.eceBefore, f.eceAfter),
            String(format: "Brier: %.4f -> %.4f", f.brierBefore, f.brierAfter),
            String(format: "fitted: temperature=%.6f, noulTemperature=%.6f, noulBias=%.6f", r.temperature, r.noulTemperature, r.noulBias),
            "out:   \(result.outputURL.path)"
        ].joined(separator: "\n")
    }
}

extension CalibrateDecideRunner {
    static func makeTeacher(
        config: Config, injected: (any DecisionEngine)?, environment: [String: String]
    ) async throws -> ResolvedTeacher {
        if let injected { return ResolvedTeacher(id: config.teacherSpec ?? injected.modelID, engine: injected) }
        guard let spec = config.teacherSpec else { return ResolvedTeacher(id: "", engine: nil) }
        if spec.hasPrefix(DecisionOptions.cloudPrefix) {
            let ref = String(spec.dropFirst(DecisionOptions.cloudPrefix.count))
            let parsed: (provider: CloudDecisionProvider, model: String) = ref.hasPrefix("openrouter/")
                ? (.openRouter, String(ref.dropFirst("openrouter/".count)))
                : CloudDecisionProvider.parse(ref)
            let primary = environment[config.keyEnvName] ?? ""
            let key = primary.isEmpty && config.keyEnvName == CloudDecisionProvider.gemini.keyEnvironment ? (environment[parsed.provider.keyEnvironment] ?? "") : primary
            let engine: (any DecisionEngine)? = key.isEmpty ? nil : try RemoteDecisionEngine(model: parsed.model, apiKey: key, provider: parsed.provider)
            return ResolvedTeacher(id: "\(parsed.provider.rawValue)/\(parsed.model)", engine: engine)
        }
        let expanded = (spec as NSString).expandingTildeInPath
        guard let path = FileManager.default.fileExists(atPath: expanded) ? expanded : ModelCache.resolve(repoId: spec)?.path else {
            throw ValidationError("Teacher model '\(spec)' is not in the local cache")
        }
        let engine = uncachedEngine()
        try await engine.load(modelPath: path)
        let readout = SystemOneReadout.calibration(for: spec, in: URL(fileURLWithPath: path))
        return ResolvedTeacher(id: spec, engine: ReadoutDecisionEngine(engine: engine, modelID: spec, readout: readout))
    }

    private static func uncachedEngine() -> MLXInferenceEngine {
        var configuration = EngineConfiguration.standard
        configuration.promptCacheBytes = 0
        return MLXInferenceEngine(configuration)
    }

    static func collectSamples(
        entries: [CalibrateDatasetIO.Entry], config: Config, student: MLXInferenceEngine?, teacher: ResolvedTeacher
    ) async throws -> Collected {
        var (samples, calls, cached, spent) = ([CalibrationSample](), 0, 0, 0.0)
        for entry in entries {
            try Task.checkCancellation()
            switch entry {
            case .sample(let sample):
                samples.append(sample)
            case .prompt(let row):
                guard let student else { throw BasaltError.modelNotLoaded }
                let (logits, _) = try await student.readoutLogits(SystemOneRequest(state: row.state, questions: row.questions))
                let batch = try await resolvePromptRow(row: row, logits: logits, config: config, teacher: teacher, remainingUSD: config.maxUSD - spent)
                samples.append(contentsOf: batch.samples)
                (calls, cached, spent) = (calls + batch.calls, cached + batch.cached, spent + batch.spentUSD)
            }
        }
        return Collected(samples: samples, calls: calls, cached: cached, spentUSD: spent)
    }

    static func resolvePromptRow(
        row: CalibrateDatasetIO.PromptRow, logits: [String: [Double]], config: Config, teacher: ResolvedTeacher, remainingUSD: Double
    ) async throws -> Collected {
        let stateHash = SystemOneCalibrationCore.stateHash(for: row.state)
        let names = row.questions.keys.sorted()
        var (probsByID, goldByID, missing, cachedHits) = ([String: [Double]](), [String: Int](), [String: SystemOneQuestion](), 0)
        for id in names {
            guard let question = row.questions[id] else { continue }
            if let gold = CalibrateDatasetIO.resolveGold(id: id, question: question, row: row) { goldByID[id] = gold }
            if let probs = row.teacherProbabilities?[id], !probs.isEmpty {
                probsByID[id] = probs
            } else if !teacher.id.isEmpty, let hit = CalibrateDatasetIO.readCached(teacherID: teacher.id, stateHash: stateHash, question: question, cacheDir: config.cacheDir) {
                (probsByID[id], cachedHits) = (hit, cachedHits + 1)
            } else if goldByID[id] == nil {
                missing[id] = question
            }
        }
        var (calls, spend) = (0, 0.0)
        if !missing.isEmpty {
            let queried = try await queryTeacher(state: row.state, missing: missing, logits: logits, config: config, teacher: teacher, remainingUSD: remainingUSD)
            for (id, probs) in queried.probs { probsByID[id] = probs }
            (calls, spend) = (1, queried.spend)
        }
        let built = names.compactMap { id -> CalibrationSample? in
            guard let question = row.questions[id], let rawScores = logits[id] else { return nil }
            return CalibrationSample(stateHash: stateHash, questionID: id, kind: question.type, rawScores: rawScores, teacherProbabilities: probsByID[id] ?? [], goldIndex: goldByID[id])
        }
        return Collected(samples: built, calls: calls, cached: cachedHits, spentUSD: spend)
    }

    private static func queryTeacher(
        state: String, missing: [String: SystemOneQuestion], logits: [String: [Double]],
        config: Config, teacher: ResolvedTeacher, remainingUSD: Double
    ) async throws -> (probs: [String: [Double]], spend: Double) {
        guard config.teacherSpec != nil || teacher.engine != nil else {
            throw ValidationError("Dataset row is missing teacherProbabilities/goldIndex for \(missing.keys.sorted().joined(separator: ", ")); pass --teacher <spec>")
        }
        guard let engine = teacher.engine else {
            throw BasaltError.authenticationFailed(details: "teacher requires API key in \(config.keyEnvName)")
        }
        guard remainingUSD > 0 else {
            throw ValidationError("--max-usd \(config.maxUSD) reached before completing uncached teacher calls")
        }
        let response = try await engine.decide(SystemOneRequest(state: state, questions: missing, calibrated: true))
        let spend = response.usage.cost ?? 0
        if spend > remainingUSD {
            throw ValidationError("--max-usd \(config.maxUSD) exceeded ($\(String(format: "%.4f", config.maxUSD - remainingUSD + spend)) spent)")
        }
        let stateHash = SystemOneCalibrationCore.stateHash(for: state)
        var out: [String: [Double]] = [:]
        for id in missing.keys.sorted() {
            guard let question = missing[id], let probs = CalibrateDatasetIO.extractProbabilities(answer: response.answers[id], question: question) else {
                throw BasaltError.modelInferenceError(details: "teacher returned no probabilities for question '\(id)'")
            }
            out[id] = probs
            let cached = CalibrationSample(stateHash: stateHash, questionID: id, kind: question.type, rawScores: logits[id] ?? [], teacherProbabilities: probs)
            try CalibrateDatasetIO.writeCached(cached, teacherID: teacher.id, question: question, cacheDir: config.cacheDir)
        }
        return (out, spend)
    }
}

enum CalibrateDatasetIO {
    struct PromptRow: Decodable, Sendable {
        var state: String
        var questions: [String: SystemOneQuestion]
        var teacherProbabilities: [String: [Double]]?
        var goldIndex: [String: Int]?
        var gold: [String: String]?

        enum CodingKeys: String, CodingKey {
            case state, questions, gold
            case teacherProbabilities, teacherProbabilitiesSnake = "teacher_probabilities"
            case goldIndex, goldIndexSnake = "gold_index"
        }

        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            state = try c.decode(String.self, forKey: .state)
            questions = try c.decode([String: SystemOneQuestion].self, forKey: .questions)
            gold = try c.decodeIfPresent([String: String].self, forKey: .gold)
            let camelProbs = try c.decodeIfPresent([String: [Double]].self, forKey: .teacherProbabilities)
            teacherProbabilities = try camelProbs ?? c.decodeIfPresent([String: [Double]].self, forKey: .teacherProbabilitiesSnake)
            let camelGold = try c.decodeIfPresent([String: Int].self, forKey: .goldIndex)
            goldIndex = try camelGold ?? c.decodeIfPresent([String: Int].self, forKey: .goldIndexSnake)
        }
    }

    enum Entry: Sendable {
        case sample(CalibrationSample)
        case prompt(PromptRow)
    }

    static func parseDataset(at path: String) throws -> [Entry] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let decoder = JSONDecoder()
        var entries: [Entry] = []
        for (lineNumber, raw) in String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline).enumerated() {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            let bytes = Data(line.utf8)
            if let sample = try? decoder.decode(CalibrationSample.self, from: bytes) {
                entries.append(.sample(sample))
            } else if let prompt = try? decoder.decode(PromptRow.self, from: bytes) {
                entries.append(.prompt(prompt))
            } else {
                throw ValidationError("Malformed dataset JSONL on line \(lineNumber + 1)")
            }
        }
        return entries
    }

    static func resolveGold(id: String, question: SystemOneQuestion, row: PromptRow) -> Int? {
        if let idx = row.goldIndex?[id] { return idx }
        guard let label = row.gold?[id] else { return nil }
        if question.type == .noul {
            switch label.lowercased() {
            case "yes", "true", "1": return 0
            case "no", "false", "0": return 1
            default: return nil
            }
        }
        let options = SystemOneReadout.options(question)
        return options.firstIndex { $0.key == label } ?? options.firstIndex { $0.description == label }
    }

    static func extractProbabilities(answer: SystemOneAnswer?, question: SystemOneQuestion) -> [Double]? {
        guard let answer else { return nil }
        if question.type == .noul {
            guard let yes = answer.noul, yes.isFinite else { return nil }
            let clamped = min(max(yes, 0), 1)
            return [clamped, 1.0 - clamped]
        }
        let keys = SystemOneReadout.options(question).map(\.key)
        if let map = answer.probabilities, keys.allSatisfy({ map[$0] != nil }) {
            return keys.map { max(map[$0] ?? 0, 0) }
        }
        if let choice = answer.choice, let idx = keys.firstIndex(of: choice) {
            return keys.indices.map { $0 == idx ? 1.0 : 0.0 }
        }
        return nil
    }

    static func readCached(teacherID: String, stateHash: String, question: SystemOneQuestion, cacheDir: URL) -> [Double]? {
        let key = SystemOneCalibrationCore.cacheKey(teacherModel: teacherID, stateHash: stateHash, question: question)
        guard let data = try? Data(contentsOf: cacheDir.appending(path: "\(key).jsonl")),
              let sample = try? SystemOneCalibrationCore.decodeJSONL(data).first,
              !sample.teacherProbabilities.isEmpty else { return nil }
        return sample.teacherProbabilities
    }

    static func writeCached(_ sample: CalibrationSample, teacherID: String, question: SystemOneQuestion, cacheDir: URL) throws {
        guard !teacherID.isEmpty else { return }
        let key = SystemOneCalibrationCore.cacheKey(teacherModel: teacherID, stateHash: stateHash(of: sample), question: question)
        let data = try SystemOneCalibrationCore.encodeJSONL([sample])
        try withTaskCancellationShield {
            try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
            try data.write(to: cacheDir.appending(path: "\(key).jsonl"), options: .atomic)
        }
    }

    private static func stateHash(of sample: CalibrationSample) -> String { sample.stateHash }
}
