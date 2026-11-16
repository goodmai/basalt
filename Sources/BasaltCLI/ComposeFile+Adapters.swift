import Foundation
import BasaltCore
import Yams

extension ComposeFile {
    static func relativeAdapters(_ adapters: [String: ComposeAdapterSpec], base: URL?) -> [String: ComposeAdapterSpec] {
        var updated = adapters
        for (adapterName, var adapterSpec) in adapters {
            if case .localFile(let localPath) = adapterSpec.corpus.location {
                adapterSpec.corpus.source = resolvedPath(localPath, base: base)
            }
            if let out = adapterSpec.output {
                adapterSpec.output = resolvedPath(out, base: base)
            }
            updated[adapterName] = adapterSpec
        }
        return updated
    }
}

extension ComposeFile.Walk {
    mutating func adapters(_ node: Node?, answerNode: Node?, into spec: inout ComposeSpec) {
        spec.answerAdapter = answerNode?.mapping?["adapter"].flatMap { text($0, "pipeline.answer.adapter") }
        guard let node else { return }
        lines["adapters"] = node.mark?.line
        guard node.mapping != nil else {
            issue("adapters", node, "must be a mapping of adapter name to adapter specification")
            return
        }
        for (name, item) in named(node, path: "adapters") {
            if let parsed = composeAdapter(item, path: "adapters.\(name)") {
                spec.adapters[name] = parsed
            }
        }
    }

    mutating func composeAdapter(_ node: Node, path: String) -> ComposeAdapterSpec? {
        lines[path] = node.mark?.line
        guard let field = fields(node, path: path, section: "compose_adapter"),
              included(field["profiles"], path: "\(path).profiles") else { return nil }
        let base = text(field["base"], "\(path).base") ?? ""
        if field["base"] == nil { issue("\(path).base", node, "an adapter needs base: <model>") }
        guard let corpusNode = field["corpus"], let corpus = adapterCorpus(corpusNode, path: "\(path).corpus") else {
            if field["corpus"] == nil { issue("\(path).corpus", node, "an adapter needs corpus: { source: ... }") }
            return nil
        }
        guard let method = parseAdapterMethod(field, path: path, node: node) else { return nil }
        var out = ComposeAdapterSpec(base: base, method: method, corpus: corpus)
        out.output = text(field["output"], "\(path).output")
        if let layersNode = field["layers"] { out.layers = integerList(layersNode, path: "\(path).layers") }
        if let iters = integer(field["iterations"], "\(path).iterations") { out.iterations = iters }
        if let bs = integer(field["batch_size"], "\(path).batch_size") { out.batchSize = bs }
        if let lr = number(field["learning_rate"], "\(path).learning_rate") { out.learningRate = Float(lr) }
        if let seq = integer(field["max_seq_len"], "\(path).max_seq_len") { out.maxSeqLen = seq }
        if let ckpt = integer(field["checkpoint_every"], "\(path).checkpoint_every") { out.checkpointEvery = ckpt }
        if let gateNode = field["gate"] { out.gate = adapterGate(gateNode, path: "\(path).gate") }
        return out
    }

    private mutating func parseAdapterMethod(_ field: [String: Node], path: String, node: Node) -> AdapterMethod? {
        let methodName = text(field["method"], "\(path).method") ?? "lora"
        let rank = integer(field["rank"], "\(path).rank") ?? 16
        let scale = Float(number(field["scale"], "\(path).scale") ?? 10.0)
        switch methodName {
        case "lora":
            return .lora(rank: rank, scale: scale)
        case "phm":
            let dim = integer(field["hypercomplex_dim"], "\(path).hypercomplex_dim") ?? 8
            let factorRank = integer(field["factor_rank"], "\(path).factor_rank") ?? 1
            let shared = flag(field["shared_basis"], "\(path).shared_basis") ?? true
            return .phm(hypercomplexDim: dim, factorRank: factorRank, sharedBasis: shared, scale: scale)
        case "oft":
            let blockSize = integer(field["block_size"], "\(path).block_size") ?? 16
            let paramText = text(field["parameterization"], "\(path).parameterization") ?? OrthogonalParameterization.cayley.rawValue
            guard let param = OrthogonalParameterization(rawValue: paramText) else {
                issue("\(path).parameterization", field["parameterization"] ?? node, "must be cayley or matrixExp")
                return nil
            }
            let blockShared = flag(field["block_shared"], "\(path).block_shared") ?? false
            return .oft(blockSize: blockSize, parameterization: param, blockShared: blockShared)
        default:
            issue("\(path).method", field["method"] ?? node, "must be lora, phm or oft")
            return nil
        }
    }

    private mutating func adapterCorpus(_ node: Node, path: String) -> CorpusSourceSpec? {
        lines[path] = node.mark?.line
        guard let field = fields(node, path: path, section: "adapter_corpus") else { return nil }
        guard let sourceNode = field["source"], let source = text(sourceNode, "\(path).source") else {
            issue("\(path).source", node, "corpus needs source (nemotron:*, hf:<org>/<repo>, or path)")
            return nil
        }
        var spec = CorpusSourceSpec(
            source: source,
            subset: text(field["subset"], "\(path).subset"),
            split: text(field["split"], "\(path).split"),
            maxRows: integer(field["max_rows"], "\(path).max_rows") ?? 512,
            includeReasoning: flag(field["include_reasoning"], "\(path).include_reasoning") ?? true
        )
        spec.minPassRate = number(field["min_pass_rate"], "\(path).min_pass_rate")
        if let rawSeed = integer(field["seed"], "\(path).seed") {
            if rawSeed < 0 {
                issue("\(path).seed", field["seed"]!, "must be non-negative")
            } else {
                spec.seed = UInt64(rawSeed)
            }
        }
        return spec
    }

    private mutating func adapterGate(_ node: Node, path: String) -> ComposeAdapterGateSpec? {
        lines[path] = node.mark?.line
        guard let field = fields(node, path: path, section: "adapter_gate") else { return nil }
        return ComposeAdapterGateSpec(
            maxKL: number(field["max_kl"], "\(path).max_kl").map(Float.init),
            maxPPLRise: number(field["max_ppl_rise"], "\(path).max_ppl_rise").map(Float.init),
            heldOut: field["held_out"].map { list($0, "\(path).held_out") }
        )
    }

    private mutating func integerList(_ node: Node, path: String) -> [Int]? {
        guard let seq = node.sequence else {
            issue(path, node, "must be a list of non-negative layer indices")
            return nil
        }
        var out: [Int] = []
        for item in seq {
            guard let val = item.int, val >= 0 else {
                issue(path, item, "each layer index must be a non-negative integer")
                return nil
            }
            out.append(val)
        }
        return out
    }
}
