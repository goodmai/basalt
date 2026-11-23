import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Inkling2CompiledSteps {
    var pre: [Bool: ([MLXArray]) -> [MLXArray]] = [:]
    var post: [Bool: ([MLXArray]) -> [MLXArray]] = [:]
    var fingerprint: [ObjectIdentifier] = []

    func reset(to fingerprint: [ObjectIdentifier]) {
        (pre, post, self.fingerprint) = ([:], [:], fingerprint)
    }
}

enum Inkling2Fingerprint {
    static let linears: Set<ObjectIdentifier> = [ObjectIdentifier(Linear.self), ObjectIdentifier(QuantizedLinear.self)]
    static let switches: Set<ObjectIdentifier> = [ObjectIdentifier(SwitchLinear.self), ObjectIdentifier(QuantizedSwitchLinear.self)]

    static func linear(_ module: Linear?, into print: inout [ObjectIdentifier]) -> Bool {
        guard let module else { return true }
        guard linears.contains(ObjectIdentifier(type(of: module))) else { return false }
        print += [ObjectIdentifier(module), ObjectIdentifier(module.weight)]
        return true
    }

    static func expert(_ module: SwitchLinear, into print: inout [ObjectIdentifier]) -> Bool {
        guard switches.contains(ObjectIdentifier(type(of: module))) else { return false }
        print.append(ObjectIdentifier(module))
        print += module.parameters().flattened().map { ObjectIdentifier($0.1) }
        return true
    }
}

extension Inkling2DecoderLayer {
    func stepFingerprint() -> [ObjectIdentifier]? {
        var print: [ObjectIdentifier] = []
        let gates = [queryGate, keyGate, valueGate, mlpGate].compactMap { $0 }
        let linears: [Linear?] = [
            attention.qProj, attention.kProj, attention.vProj, attention.rProj, attention.oProj, attention.gateProj, attention.sinkProj,
            attention.keyConv.weightProj, attention.valueConv.weightProj,
        ] + gates.flatMap { [$0.downProj, $0.forgetProj, $0.messageProj] as [Linear?] }
        guard linears.allSatisfy({ Inkling2Fingerprint.linear($0, into: &print) }),
              Inkling2Fingerprint.expert(mlp.experts.gateUp, into: &print), Inkling2Fingerprint.expert(mlp.experts.down, into: &print)
        else { return nil }
        let norms = [postAttentionNorm, inputNorm, queryNorm, keyNorm, valueNorm].compactMap { $0 }
        let arrays: [MLXArray?] = norms.map(\.weight) + [
            mlp.router.weight, mlp.router.correction, attention.relativeLogits.proj, attention.queryClip.scale, attention.relativeClip.scale,
            attention.keyConv.conv?.weight, attention.keyConv.staticWeight, attention.valueConv.conv?.weight, attention.valueConv.staticWeight,
            postAttentionConv.conv.weight, postMLPConv.conv.weight, attentionInput?.query, mlpInput.query,
        ]
        return print + arrays.compactMap { $0.map(ObjectIdentifier.init) } + norms.map(ObjectIdentifier.init)
    }

    func compiledDecode(_ hidden: MLXArray, blocks: MLXArray?, cache: KVCache?, conv: ArraysCache?, messages: MLXArray?) -> (MLXArray, MLXArray?)? {
        guard let print = stepFingerprint() else { return nil }
        if print != steps.fingerprint { steps.reset(to: print) }
        let hasMessages = messages != nil
        let start = cache?.offset ?? 0
        let length = hidden.dim(1)
        let states: [MLXArray?] = (0 ..< 4).map { conv?[$0] }
        let kvDims = attention.kvHeads * attention.headDim
        let zeros = { (channels: Int) in MLXArray.zeros([hidden.dim(0), self.kernel - 1, channels], dtype: .float32) }
        var preArgs = [hidden] + (index > 0 ? [blocks!] : []) + (messages.map { [$0] } ?? [])
        preArgs += [states[0] ?? zeros(kvDims), states[1] ?? zeros(kvDims), Inkling2Attention.positions(start, length)]
        let pre = preStep(hasMessages)(preArgs)
        let (keys, values, keyStart) = Inkling2Attention.store(pre[1], pre[2], in: cache, start: start)
        let prepared = Inkling2Prepared(queries: pre[0], keys: keys, values: values, bias: pre[3], sinks: pre[4], gate: pre[5])
        let attended = attention.finish(prepared, keys: keys, values: values, distance: Inkling2Attention.distance(query: start, length: length, key: keyStart, count: keys.dim(2)))
        var postArgs = [attended, hidden, pre[8]]
        postArgs += (index > 0 ? [blocks!] : []) + (messages.map { [$0] } ?? [])
        postArgs += [states[2] ?? zeros(hidden.dim(-1)), states[3] ?? zeros(hidden.dim(-1))]
        let post = postStep(hasMessages)(postArgs)
        if let conv {
            let windows = [pre[6], pre[7], post[2], post[3]]
            Inkling2Ops.commit(windows.map { Inkling2Ops.convState($0, kernel: kernel) }, windows: windows, count: length, to: conv)
        }
        return (post[0], post[1])
    }

    private func preStep(_ hasMessages: Bool) -> ([MLXArray]) -> [MLXArray] {
        if let step = steps.pre[hasMessages] { return step }
        let step = compile(inputs: [], outputs: []) { [unowned self] (arrays: [MLXArray]) -> [MLXArray] in
            var rest = arrays[...]
            let hidden = rest.removeFirst()
            let blocks = self.index > 0 ? rest.removeFirst() : nil
            let messages = hasMessages ? rest.removeFirst() : nil
            let (keyState, valueState, positions) = (rest.removeFirst(), rest.removeFirst(), rest.removeFirst())
            let normed = Inkling2Ops.unitRMS(hidden)
            let inputs = self.attentionInputs(hidden, mixed: self.attentionInput.map { $0(blocks!, normed) }, messages: messages, dtype: hidden.dtype)
            let (prepared, keyWindow, valueWindow) = self.attention.prepare(inputs, keyState: keyState, valueState: valueState, positions: positions)
            return [
                prepared.queries, prepared.keys, prepared.values, prepared.bias, prepared.sinks, prepared.gate,
                keyWindow, valueWindow, normed,
            ]
        }
        steps.pre[hasMessages] = step
        return step
    }

    private func postStep(_ hasMessages: Bool) -> ([MLXArray]) -> [MLXArray] {
        if let step = steps.post[hasMessages] { return step }
        let step = compile(inputs: [], outputs: []) { [unowned self] (arrays: [MLXArray]) -> [MLXArray] in
            var rest = arrays[...]
            let (attended, hidden, normed) = (rest.removeFirst(), rest.removeFirst(), rest.removeFirst())
            var blocks = self.index > 0 ? rest.removeFirst() : nil
            let messages = hasMessages ? rest.removeFirst() : nil
            let (attentionState, mlpState) = (rest.removeFirst(), rest.removeFirst())
            let (afterAttention, attentionWindow) = self.postAttentionConv(attended, state: attentionState)
            var residual = afterAttention.asType(.float32)
            if self.blockBoundary {
                let block = normed.expandedDimensions(axis: 0)
                blocks = blocks.map { concatenated([$0, block], axis: 0) } ?? block
            } else {
                residual = hidden.asType(.float32) + residual
            }
            var x = self.mlpInput(blocks!, Inkling2Ops.unitRMS(residual))[.ellipsis, 0, 0...]
            if let mlpGate = self.mlpGate, let messages { x = mlpGate(x, message: messages[.ellipsis, 3, 0...]) }
            let (afterMLP, mlpWindow) = self.postMLPConv(self.mlp(self.postAttentionNorm(x).asType(hidden.dtype)), state: mlpState)
            return [
                (residual + afterMLP.asType(.float32)).asType(hidden.dtype), blocks!,
                attentionWindow, mlpWindow,
            ]
        }
        steps.post[hasMessages] = step
        return step
    }
}
