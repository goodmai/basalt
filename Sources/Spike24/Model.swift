import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

struct Bridge: MLXLMCommon.Tokenizer {
    let t: any Tokenizers.Tokenizer
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { t.encode(text: text, addSpecialTokens: addSpecialTokens) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { t.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { t.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { t.convertIdToToken(id) }
    var bosToken: String? { t.bosToken }
    var eosToken: String? { t.eosToken }
    var unknownToken: String? { t.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
}

struct Loader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        Bridge(t: try await AutoTokenizer.from(modelFolder: directory))
    }
}

final class AsymKVCache: KVCache, KVCacheAttentionProtocol {
    let keyBits: Int, valueBits: Int, rotateValues: Bool
    var offset = 0
    var maxSize: Int? { nil }
    var metaState: [String] { get { [""] } set {} }
    var isTrimmable: Bool { false }
    func trim(_ n: Int) -> Int { 0 }
    func copy() -> any KVCache { self }
    func innerState() -> [MLXArray] { state }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { fatalError("plain route") }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) { return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize)) }
        return .causal
    }
    var k: (MLXArray, MLXArray, MLXArray?)?
    var v: (MLXArray, MLXArray, MLXArray?)?

    init(keyBits: Int, valueBits: Int, rotateValues: Bool) {
        self.keyBits = keyBits; self.valueBits = valueBits; self.rotateValues = rotateValues
    }

    static func cat(_ a: (MLXArray, MLXArray, MLXArray?)?, _ b: (MLXArray, MLXArray, MLXArray?)) -> (MLXArray, MLXArray, MLXArray?) {
        guard let a else { return b }
        return (concatenated([a.0, b.0], axis: 2), concatenated([a.1, b.1], axis: 2), b.2.map { concatenated([a.2!, $0], axis: 2) })
    }

    func append(keys: MLXArray, values: MLXArray) {
        let vv = rotateValues ? hadamardTransform(values.asType(.float32)).asType(values.dtype) : values
        let qk = quantized(keys, groupSize: 64, bits: keyBits)
        let qv = quantized(vv, groupSize: 64, bits: valueBits)
        k = Self.cat(k, (qk.wq, qk.scales, qk.biases))
        v = Self.cat(v, (qv.wq, qv.scales, qv.biases))
        offset += keys.dim(2)
    }

    var state: [MLXArray] {
        get { [k!.0, k!.1, k!.2!, v!.0, v!.1, v!.2!] }
        set {}
    }

    func updateAndAttend(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        append(keys: keys, values: values)
        let (B, nQ, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let nKV = keys.dim(1), rep = nQ / nKV
        let e = { (t: (MLXArray, MLXArray, MLXArray?)) in
            (expandedDimensions(t.0, axis: -3), expandedDimensions(t.1, axis: -3), t.2.map { expandedDimensions($0, axis: -3) })
        }
        let (kk, vv) = (e(k!), e(v!))
        let qs = (queries * scale).reshaped([B, nKV, rep, L, D])
        var scores = quantizedMM(qs, kk.0, scales: kk.1, biases: kk.2, transpose: true, groupSize: 64, bits: keyBits).asType(.float32)
        switch mask {
        case .causal:
            let kL = scores.dim(-1)
            let qi = MLXArray(0 ..< L) + MLXArray(kL - L)
            let ki = MLXArray(0 ..< kL)
            scores = MLX.where(greaterEqual(expandedDimensions(qi, axis: -1), expandedDimensions(ki, axis: -2)), scores, MLXArray(-Float.infinity))
        case .array(let m):
            let mm = m.ndim == 4 ? expandedDimensions(m, axis: -3) : m
            scores = mm.dtype == .bool ? MLX.where(mm, scores, MLXArray(-Float.infinity)) : scores + mm.asType(.float32)
        default: break
        }
        let w = softmax(scores, axis: -1).asType(queries.dtype)
        var out = quantizedMM(w, vv.0, scales: vv.1, biases: vv.2, transpose: false, groupSize: 64, bits: valueBits)
            .reshaped([B, nQ, L, D])
        if rotateValues { out = hadamardTransform(out.asType(.float32)).asType(queries.dtype) }
        return out
    }
}

enum Scheme: String, CaseIterable { case fp16, affine8, affine4, turbo8v4, asymK8V4, asymK8V4H }

func convert(_ cache: inout [KVCache], _ s: Scheme) {
    switch s {
    case .fp16: break
    case .affine8: maybeQuantizeKVCache(cache: &cache, kvBits: 8)
    case .affine4: maybeQuantizeKVCache(cache: &cache, kvBits: 4)
    case .turbo8v4: maybeQuantizeKVCache(cache: &cache, kvBits: nil, kvScheme: "turbo8v4")
    case .asymK8V4, .asymK8V4H:
        cache = cache.map { c in
            guard let simple = c as? KVCacheSimple else { return c }
            let st = simple.state
            let a = AsymKVCache(keyBits: 8, valueBits: 4, rotateValues: s == .asymK8V4H)
            a.append(keys: st[0][.ellipsis, ..<simple.offset, 0...], values: st[1][.ellipsis, ..<simple.offset, 0...])
            return a
        }
    }
}

func kvBytes(_ cache: [KVCache]) -> Int {
    cache.reduce(0) { total, c in
        if let t = c as? TurboQuantKVCache { return total + t.state.reduce(0) { $0 + $1.nbytes } }
        return total + c.state.reduce(0) { $0 + $1.nbytes }
    }
}

func copyCache(_ cache: [KVCache]) -> [KVCache] {
    cache.map { c in
        let s = c as! KVCacheSimple
        let n = KVCacheSimple()
        n.state = s.state.map { $0[.ellipsis, ..<s.offset, 0...] + 0 }
        eval(n.state)
        return n
    }
}

func modelEval(_ dir: URL, schemes: [Scheme], pplTokens: Int, passkeyLengths: [Int]) async throws {
    let container = try await loadModelContainer(from: dir, using: Loader())
    let repo = URL(fileURLWithPath: CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : FileManager.default.currentDirectoryPath)
    let corpus = ["README.md", "CHANGELOG.md", "BENCHMARKING.md", "CONTRIBUTING.md", "AGENTS.md", "README.ru.md"]
        .compactMap { try? String(contentsOf: repo.appendingPathComponent($0), encoding: .utf8) }.joined(separator: "\n\n")
    try await container.perform { (context: ModelContext) in
        let model = context.model
        let tok = context.tokenizer
        let ids = Array(tok.encode(text: corpus).prefix(pplTokens + 1))
        print("== PPL on \(ids.count) tokens of repo docs; first 1024 prefilled fp16, rest scored after conversion")
        let all = MLXArray(ids).reshaped(1, ids.count)
        let head = 1024
        let base = try model.newCache(parameters: nil)
        let headLogits = try model(all[0..., ..<head], cache: base)
        eval(headLogits)
        for s in schemes {
            var cache = copyCache(base)
            convert(&cache, s)
            var last = headLogits[0..., -1, 0...]
            var nll = 0.0, n = 0
            var pos = head
            var step = 1
            while pos < ids.count - 1 {
                let end = min(pos + step, ids.count - 1)
                let chunk = all[0..., pos ..< end]
                let out = try model(chunk, cache: cache)
                let preds = concatenated([expandedDimensions(last, axis: 1), out[0..., ..<(end - pos - 1), 0...]], axis: 1).asType(.float32)
                let lp = preds - preds.logSumExp(axis: -1, keepDims: true)
                let picked = takeAlong(lp, expandedDimensions(chunk, axis: -1), axis: -1)
                nll -= Double(picked.sum().item(Float.self))
                n += end - pos
                last = out[0..., -1, 0...]
                pos = end
                step = 512
            }
            print("  \(s.rawValue): ppl \(String(format: "%.4f", exp(nll / Double(n)))) over \(n) tokens, kv \(String(format: "%.1f", Double(kvBytes(cache)) / 1e6)) MB")
            Memory.clearCache()
        }

        let filler = tok.encode(text: " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.", addSpecialTokens: false)
        let question = tok.encode(text: "\nWhat is the pass key? The pass key is", addSpecialTokens: false)
        for length in passkeyLengths {
            for depth in [0.1, 0.5, 0.9] {
                let key = String(10000 + (length / 7 + Int(depth * 1000)) % 89999)
                let needle = tok.encode(text: " The pass key is \(key). Remember it. \(key) is the pass key.", addSpecialTokens: false)
                let fillCount = length - needle.count - question.count
                var body: [Int] = []
                while body.count < fillCount { body += filler }
                body = Array(body.prefix(fillCount))
                let at = Int(Double(fillCount) * depth)
                let prompt = Array(body[..<at]) + needle + Array(body[at...]) + question
                let p = MLXArray(prompt).reshaped(1, prompt.count)
                let base = try model.newCache(parameters: nil)
                var logits = MLXArray(0)
                var i = 0
                while i < prompt.count {
                    let e = min(i + 2048, prompt.count)
                    logits = try model(p[0..., i ..< e], cache: base)
                    eval(logits)
                    i = e
                }
                var line = "  passkey len \(prompt.count) depth \(depth) key \(key):"
                for s in schemes {
                    var cache = copyCache(base)
                    convert(&cache, s)
                    Memory.peakMemory = 0
                    var next = logits[0..., -1, 0...].argMax(axis: -1).item(Int.self)
                    var outIds: [Int] = []
                    let t0 = Date()
                    for _ in 0 ..< 10 {
                        outIds.append(next)
                        let l = try model(MLXArray([next]).reshaped(1, 1), cache: cache)
                        next = l[0..., -1, 0...].argMax(axis: -1).item(Int.self)
                    }
                    let tps = 10 / Date().timeIntervalSince(t0)
                    let text = tok.decode(tokenIds: outIds, skipSpecialTokens: true)
                    line += " \(s.rawValue)=\(text.contains(key) ? "OK" : "MISS(\(text.prefix(12)))")/\(String(format: "%.0f", tps))t/s/\(String(format: "%.2f", Double(kvBytes(cache)) / 1e9))GB"
                    Memory.clearCache()
                }
                print(line)
            }
        }
    }
}
