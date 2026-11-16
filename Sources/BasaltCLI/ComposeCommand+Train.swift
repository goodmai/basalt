import ArgumentParser
import BasaltBench
import BasaltCore
import BasaltInference
import Foundation

struct ComposeTrainEnvironment: Sendable {
    var force: Bool = false
    var resume: Bool = false
    var transport: any CorpusHTTPTransport = URLSessionCorpusTransport()
    var home: URL = FileManager.default.homeDirectoryForCurrentUser
    var environment: [String: String] = ProcessInfo.processInfo.environment
    var makeEngine: (@Sendable (String) async throws -> MLXInferenceEngine)? = nil
}

struct ComposeTrainOutcome: Sendable, Equatable {
    let name: String
    let base: String
    let directory: String
    let skipped: Bool
    let keptRows: Int
    let nllBefore: Float?
    let nllAfter: Float?
}

enum ComposeTrainRunner {
    static func selectAdapters(
        in spec: ComposeSpec,
        filter: [String]
    ) throws -> [(name: String, adapter: ComposeAdapterSpec)] {
        guard !spec.adapters.isEmpty else {
            throw ValidationError("compose \(spec.name) declares no adapters under adapters:")
        }
        if filter.isEmpty {
            return spec.adapters.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
        }
        var chosen: [(name: String, adapter: ComposeAdapterSpec)] = []
        for name in filter {
            guard let found = spec.adapters[name] else {
                throw ValidationError("adapter '\(name)' is not declared under adapters in \(spec.name)")
            }
            chosen.append((name, found))
        }
        return chosen
    }

    static func trainSelected(
        _ items: [(name: String, adapter: ComposeAdapterSpec)],
        in spec: ComposeSpec,
        env: ComposeTrainEnvironment = ComposeTrainEnvironment()
    ) async throws -> [ComposeTrainOutcome] {
        var grouped: [String: [(name: String, adapter: ComposeAdapterSpec)]] = [:]
        for item in items { grouped[item.adapter.base, default: []].append(item) }
        var outcomes: [ComposeTrainOutcome] = []
        for baseName in grouped.keys.sorted() {
            let entries = grouped[baseName] ?? []
            guard let modelSource = spec.local(baseName) else {
                throw ValidationError("base model '\(baseName)' is not a local model in compose \(spec.name)")
            }
            let modelPath = env.makeEngine != nil ? modelSource : try resolveModelPath(modelSource)
            outcomes += try await trainGroup(entries, baseName: baseName, modelPath: modelPath, composeName: spec.name, env: env)
        }
        return outcomes
    }

    private struct PendingAdapter: Sendable {
        let name: String
        let adapter: ComposeAdapterSpec
        let dir: URL
    }

    private static func trainGroup(
        _ entries: [(name: String, adapter: ComposeAdapterSpec)],
        baseName: String,
        modelPath: String,
        composeName: String,
        env: ComposeTrainEnvironment
    ) async throws -> [ComposeTrainOutcome] {
        var pending: [PendingAdapter] = []
        var outcomes: [ComposeTrainOutcome] = []
        for entry in entries {
            let dir = entry.adapter.resolvedDirectory(composeName: composeName, adapterName: entry.name, home: env.home)
            let exists = FileManager.default.fileExists(atPath: dir.appending(component: "adapters.safetensors").path)
            if exists, !env.force, !env.resume {
                outcomes.append(ComposeTrainOutcome(name: entry.name, base: baseName, directory: dir.path, skipped: true, keptRows: 0, nllBefore: nil, nllAfter: nil))
            } else {
                if env.force, !env.resume { _ = try? FileManager.default.removeItem(at: dir) }
                pending.append(PendingAdapter(name: entry.name, adapter: entry.adapter, dir: dir))
            }
        }
        guard !pending.isEmpty else { return outcomes }
        let (engine, orchestrator) = try await acquireEngine(modelPath: modelPath, env: env)
        defer {
            if let orchestrator {
                await orchestrator.shutdown()
            }
        }
        for item in pending {
            outcomes.append(try await trainOne(item, baseName: baseName, composeName: composeName, engine: engine, env: env))
        }
        return outcomes
    }

    private static func acquireEngine(
        modelPath: String,
        env: ComposeTrainEnvironment
    ) async throws -> (MLXInferenceEngine, ModelOrchestratorActor<MLXInferenceEngine>?) {
        if let factory = env.makeEngine {
            return (try await factory(modelPath), nil)
        }
        let fresh = MLXInferenceEngine()
        let actor = ModelOrchestratorActor(engine: fresh)
        try await actor.loadModel(path: modelPath)
        return (fresh, actor)
    }

    private static func trainOne(
        _ item: PendingAdapter,
        baseName: String,
        composeName: String,
        engine: MLXInferenceEngine,
        env: ComposeTrainEnvironment
    ) async throws -> ComposeTrainOutcome {
        var request = item.adapter.trainRequest(composeName: composeName, adapterName: item.name, resume: env.resume, home: env.home)
        request.applyOverlay = false
        let report = try await engine.trainCorpus(request, transport: env.transport, home: env.home, environment: env.environment, lease: nil)
        if let gateSpec = item.adapter.gate {
            let verdict = try await engine.applyAdapter(.directory(item.dir.path), gate: gateSpec.gateRequest())
            if let overlay = verdict.overlay {
                _ = try? await engine.undo(overlay.id)
            } else {
                throw BasaltError.invalidRequestStructure(details: "compose train: adapter \(item.name) failed quality gate")
            }
        }
        return ComposeTrainOutcome(
            name: item.name,
            base: baseName,
            directory: item.dir.path,
            skipped: false,
            keptRows: report.keptRows,
            nllBefore: report.nllBefore,
            nllAfter: report.nllAfter
        )
    }

    private static func resolveModelPath(_ source: String) throws -> String {
        let expanded = (source as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) { return expanded }
        if let cached = ModelCache.resolve(repoId: source)?.path { return cached }
        throw ValidationError("Model '\(source)' is not in the local cache; run `basalt compose pull` or `basalt models download \(source)` first")
    }
}

extension ComposeCommand {
    struct Train: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "train",
            abstract: "Train declared LoRA, PHM or OFT adapters from a compose v2 file on Hugging Face Nemotron or local corpora (#510)"
        )

        @OptionGroup var source: Source

        @Option(name: .customLong("adapter"), help: "Train only the named adapter(s); repeatable")
        var adapterNames: [String] = []

        @Flag(name: .customLong("force"), help: "Overwrite an existing trained adapter directory")
        var force = false

        @Flag(name: .customLong("resume"), help: "Resume from an existing checkpoint inside the adapter directory")
        var resume = false

        func run() async throws {
            let spec = try source.spec()
            let selected = try ComposeTrainRunner.selectAdapters(in: spec, filter: adapterNames)
            let env = ComposeTrainEnvironment(force: force, resume: resume)
            let results = try await GPULock.withLock(timeout: .seconds(600)) {
                try await ComposeTrainRunner.trainSelected(selected, in: spec, env: env)
            }
            for row in results {
                let status = row.skipped ? "skipped (already trained)" : String(
                    format: "trained (%d rows, NLL %.3f -> %.3f)",
                    row.keptRows,
                    row.nllBefore ?? 0,
                    row.nllAfter ?? 0
                )
                print("  \(row.name.padding(toLength: 16, withPad: " ", startingAt: 0)) base=\(row.base)  \(status)  -> \(row.directory)")
            }
        }
    }
}
