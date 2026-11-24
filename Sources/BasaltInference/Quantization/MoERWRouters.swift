import BasaltCore
import MLX
import MLXLMCommon
import MLXNN

extension MoERWPipeline {
    static let routerSuffixes = [".mlp.gate", ".mlp.router", ".feed_forward.gate", ".feed_forward.router", ".router.proj"]

    static func discoverRouters(in root: Module) -> [Int: RouterSite] {
        var out: [Int: RouterSite] = [:]
        let modules = root.namedModules()
        let byPath = Dictionary(modules, uniquingKeysWith: { first, _ in first })
        for (path, mod) in modules {
            guard let bIdx = PVTuner.textBlockIndex(path), out[bIdx] == nil, routerSuffixes.contains(where: path.hasSuffix) else { continue }
            let dims: (out: Int, inp: Int)? = if let nf4 = mod as? NF4Linear {
                nf4.shape
            } else if let ql = mod as? QuantizedLinear {
                (ql.weight.dim(0), (ql.weight.dim(1) * 32) / max(1, ql.bits))
            } else if let lin = mod as? Linear {
                lin.shape
            } else {
                nil
            }
            if let dims, dims.out >= 2, dims.inp >= 16 {
                let parent = path.hasSuffix(".proj") ? byPath[String(path.dropLast(".proj".count))] : nil
                let scale = parent.flatMap { Dictionary(uniqueKeysWithValues: $0.parameters().flattened())["per_expert_scale"] }
                out[bIdx] = RouterSite(
                    block: bIdx, path: path, module: mod, numExperts: dims.out, inFeatures: dims.inp,
                    expertScale: scale?.size == dims.out ? scale?.asType(.float32) : nil)
            }
        }
        return out
    }
}

final class MoERWRouterProbe: ForwardProbe {
    let boxes: [Int: MoERouterGateBox]

    init(boxes: [Int: MoERouterGateBox]) {
        self.boxes = boxes
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray { hidden }
    func attentionWeights(layer: Int, _ weights: MLXArray) {}
    func wantsAttention(layer: Int) -> Bool { false }
    func headMask(layer: Int) -> MLXArray? { nil }
    func routerLogits(layer: Int, _ logits: MLXArray) {}
    func expertMask(layer: Int) -> MLXArray? { nil }

    func routerTopK(layer: Int, indices: MLXArray, gates: MLXArray) {
        boxes[layer]?.recordRouted(indices: indices, gates: gates)
    }

    static func forward(_ model: any LanguageModel, tokens: [Int32], probe: MoERWRouterProbe) -> MLXArray {
        let input = MLXArray(tokens)[.newAxis]
        if let probable = model as? any BasaltProbableModel {
            return probable.forwardWithProbe(input, cache: nil, probe: probe)
        }
        return withForwardProbe(probe) { model(LMInput.Text(tokens: input), cache: nil, state: nil).logits }
    }
}
