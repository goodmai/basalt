import Foundation

public struct ComposeAdapterGateSpec: Sendable, Codable, Equatable {
    public var maxKL: Float?
    public var maxPPLRise: Float?
    public var heldOut: [String]?

    public init(maxKL: Float? = nil, maxPPLRise: Float? = nil, heldOut: [String]? = nil) {
        (self.maxKL, self.maxPPLRise, self.heldOut) = (maxKL, maxPPLRise, heldOut)
    }

    public func gateRequest() -> GateRequest {
        var req = GateRequest(heldOut: heldOut ?? CorpusTrainRequest.defaultHeldOutPrompts)
        if let maxKL { req.maxKL = maxKL }
        if let maxPPLRise { req.maxPerplexityRise = maxPPLRise }
        return req
    }
}

public struct ComposeAdapterSpec: Sendable, Codable, Equatable {
    public var base: String
    public var method: AdapterMethod
    public var corpus: CorpusSourceSpec
    public var output: String?
    public var layers: [Int]?
    public var iterations: Int = 200
    public var batchSize: Int = 2
    public var learningRate: Float = 1e-4
    public var maxSeqLen: Int = 2048
    public var checkpointEvery: Int = 50
    public var gate: ComposeAdapterGateSpec?

    public init(
        base: String,
        method: AdapterMethod = .lora(rank: 16),
        corpus: CorpusSourceSpec
    ) {
        self.base = base
        self.method = method
        self.corpus = corpus
    }

    public func resolvedDirectory(composeName: String, adapterName: String, home: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        if let output, !output.isEmpty {
            return URL(fileURLWithPath: (output as NSString).expandingTildeInPath)
        }
        let safeCompose = composeName.isEmpty ? "default" : composeName
        return home
            .appending(path: ".basalt/adapters/compose")
            .appending(component: safeCompose)
            .appending(component: adapterName)
    }

    public func trainRequest(composeName: String, adapterName: String, resume: Bool = false, home: URL = FileManager.default.homeDirectoryForCurrentUser) -> CorpusTrainRequest {
        let gateReq = gate?.gateRequest() ?? GateRequest(heldOut: CorpusTrainRequest.defaultHeldOutPrompts)
        var req = CorpusTrainRequest(corpus: corpus, gate: gateReq)
        req.method = method
        req.layers = layers
        req.iterations = iterations
        req.batchSize = batchSize
        req.learningRate = learningRate
        req.maxSeqLen = maxSeqLen
        req.checkpointEvery = checkpointEvery
        req.resume = resume
        req.applyOverlay = gate != nil
        req.output = resolvedDirectory(composeName: composeName, adapterName: adapterName, home: home).path
        return req
    }
}

extension ComposeSpec {
    public var adapterBaseModels: Set<String> {
        Set(adapters.values.map(\.base).filter { models[$0] != nil })
    }

    public func adapters(forBase modelName: String, home: URL = FileManager.default.homeDirectoryForCurrentUser) -> [String: URL] {
        var result: [String: URL] = [:]
        for (adapterName, spec) in adapters where spec.base == modelName {
            let dir = spec.resolvedDirectory(composeName: name, adapterName: adapterName, home: home)
            if FileManager.default.fileExists(atPath: dir.appending(component: "adapters.safetensors").path) {
                result[adapterName] = dir
            }
        }
        return result
    }

    var adapterIssues: [Issue] {
        var issues: [Issue] = []
        let hasRouteAdapters = views.contains { _, view in
            view.answerAdapter != nil || view.routes.contains { $0.adapter != nil }
        }
        if (!adapters.isEmpty || hasRouteAdapters), version < 2 {
            issues.append(Issue("adapters", "needs version: 2"))
        }
        for (adapterName, spec) in adapters.sorted(by: { $0.key < $1.key }) {
            issues += singleAdapterIssues(name: adapterName, spec: spec)
        }
        for (pipeline, view) in views {
            let prefix = pipeline == Self.defaultPipeline ? "pipeline" : "pipelines.\(pipeline)"
            issues += viewAdapterRouteIssues(view, prefix: prefix)
        }
        return issues
    }

    private func singleAdapterIssues(name: String, spec: ComposeAdapterSpec) -> [Issue] {
        let path = "adapters.\(name)"
        var issues: [Issue] = []
        if name.isEmpty || !name.allSatisfy({ $0.isLowercase || $0.isNumber || $0 == "-" || $0 == "_" }) {
            issues.append(Issue(path, "an adapter name uses lowercase letters, digits, - and _"))
        }
        issues += adapterOutputIssues(spec.output, path: "\(path).output")
        if let baseModel = models[spec.base] {
            if baseModel.isCloud {
                issues.append(Issue("\(path).base", "\(spec.base) is a cloud model; adapters train and run on local models only"))
            }
        } else {
            issues.append(Issue("\(path).base", "\(spec.base) is not a declared model"))
        }
        do throws(BasaltError) {
            try spec.method.validate(prefix: path)
        } catch {
            issues.append(Issue("\(path).method", error.localizedDescription))
        }
        issues += adapterCorpusIssues(spec.corpus, path: "\(path).corpus")
        issues += adapterHyperparamIssues(spec, path: path)
        return issues
    }

    private func adapterOutputIssues(_ output: String?, path: String) -> [Issue] {
        guard let output else { return [] }
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        let meaningful = parts.filter { !$0.isEmpty && $0 != "." }
        if trimmed.isEmpty || trimmed.hasPrefix("/") || trimmed.hasPrefix("~") || parts.contains("..") || meaningful.isEmpty {
            return [Issue(path, "output must be a relative path without '..' or '~'")]
        }
        return []
    }

    private func adapterCorpusIssues(_ corpus: CorpusSourceSpec, path: String) -> [Issue] {
        var issues: [Issue] = []
        do throws(BasaltError) {
            try corpus.validate(prefix: path)
        } catch {
            issues.append(Issue(path, error.localizedDescription))
        }
        let rawPrefixes = ["sk-", "hf_", "ghp_", "github_pat_", "AIza", "AKIA", "Bearer ", "bearer "]
        let trimmedSource = corpus.source.trimmingCharacters(in: .whitespacesAndNewlines)
        if rawPrefixes.contains(where: { trimmedSource.hasPrefix($0) }) || SecretRedactor.redact(trimmedSource) != trimmedSource {
            issues.append(Issue("\(path).source", "looks like a secret token; pass a dataset preset, <org>/<repo>, or local path"))
        }
        return issues
    }

    private func adapterHyperparamIssues(_ spec: ComposeAdapterSpec, path: String) -> [Issue] {
        var issues: [Issue] = []
        if !(1 ... 10_000).contains(spec.iterations) {
            issues.append(Issue("\(path).iterations", "must be between 1 and 10000"))
        }
        if !(1 ... 64).contains(spec.batchSize) {
            issues.append(Issue("\(path).batch_size", "must be between 1 and 64"))
        }
        if !spec.learningRate.isFinite || spec.learningRate <= 0 || spec.learningRate > 1 {
            issues.append(Issue("\(path).learning_rate", "must lie in (0, 1]"))
        }
        if !(16 ... 32_768).contains(spec.maxSeqLen) {
            issues.append(Issue("\(path).max_seq_len", "must be between 16 and 32768"))
        }
        if let gate = spec.gate {
            if let kl = gate.maxKL, (!kl.isFinite || kl <= 0) {
                issues.append(Issue("\(path).gate.max_kl", "must be positive"))
            }
            if let rise = gate.maxPPLRise, (!rise.isFinite || rise <= 0) {
                issues.append(Issue("\(path).gate.max_ppl_rise", "must be positive"))
            }
        }
        return issues
    }

    private func viewAdapterRouteIssues(_ view: ComposeSpec, prefix: String) -> [Issue] {
        var issues: [Issue] = []
        if let ansAdapter = view.answerAdapter {
            if let declared = adapters[ansAdapter] {
                if declared.base != view.answer {
                    issues.append(Issue("\(prefix).answer.adapter", "adapter \(ansAdapter) belongs to base \(declared.base), not \(view.answer)"))
                }
            } else {
                issues.append(Issue("\(prefix).answer.adapter", "\(ansAdapter) is not declared under adapters"))
            }
        }
        for (index, route) in view.routes.enumerated() {
            guard let routeAdapter = route.adapter else { continue }
            let routePath = "\(prefix).answer.routes[\(index)].adapter"
            if let declared = adapters[routeAdapter] {
                if declared.base != route.model {
                    issues.append(Issue(routePath, "adapter \(routeAdapter) belongs to base \(declared.base), not \(route.model)"))
                }
            } else {
                issues.append(Issue(routePath, "\(routeAdapter) is not declared under adapters"))
            }
        }
        return issues
    }
}
