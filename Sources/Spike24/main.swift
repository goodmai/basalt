import Foundation
import MLX
import MLXLMCommon

let B = 1, HKV = 8, HQ = 32, D = 128
let attnScale = 1 / Float(D).squareRoot()

func gb(_ x: Int) -> String { String(format: "%.3f GB", Double(x) / 1e9) }
func f(_ x: Float) -> String { String(format: "%.5f", x) }

enum Shape { case gaussian, channelOutliers, tokenOutliers }

func synth(_ T: Int, _ shape: Shape, seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    var x = MLXRandom.normal([B, HKV, T, D])
    switch shape {
    case .gaussian: break
    case .channelOutliers:
        let s = (0 ..< D).map { [3, 17, 64, 101].contains($0) ? Float(20) : Float(1) }
        x = x * MLXArray(s)
    case .tokenOutliers:
        let s = (0 ..< T).map { $0 < 4 || $0 % 997 == 0 ? Float(20) : Float(1) }
        x = x * MLXArray(s).reshaped([1, 1, T, 1])
    }
    return x.asType(.float16)
}

func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
    let d = (a.asType(.float32) - b.asType(.float32))
    return (sqrt((d * d).sum()) / sqrt((b.asType(.float32) * b.asType(.float32)).sum())).item(Float.self)
}

func deq(_ x: MLXArray, bits: Int) -> MLXArray {
    let q = quantized(x, groupSize: 64, bits: bits)
    return dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: bits)
}

func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
    MLXFast.scaledDotProductAttention(
        queries: q.asType(.float32), keys: k.asType(.float32), values: v.asType(.float32),
        scale: attnScale, mask: .none)
}

func asymSDPA(
    _ q: MLXArray, k: (MLXArray, MLXArray, MLXArray?), v: (MLXArray, MLXArray, MLXArray?),
    keyBits: Int, valueBits: Int
) -> MLXArray {
    let L = q.dim(2)
    let rep = HQ / HKV
    let qs = (q * attnScale).reshaped([B, HKV, rep, L, D])
    let e = { (t: (MLXArray, MLXArray, MLXArray?)) in
        (expandedDimensions(t.0, axis: -3), expandedDimensions(t.1, axis: -3), t.2.map { expandedDimensions($0, axis: -3) })
    }
    let (kk, vv) = (e(k), e(v))
    let scores = quantizedMM(qs, kk.0, scales: kk.1, biases: kk.2, transpose: true, groupSize: 64, bits: keyBits)
    let w = softmax(scores.asType(.float32), axis: -1).asType(q.dtype)
    let out = quantizedMM(w, vv.0, scales: vv.1, biases: vv.2, transpose: false, groupSize: 64, bits: valueBits)
    return out.reshaped([B, HQ, L, D])
}

func tup(_ x: MLXArray, bits: Int) -> (MLXArray, MLXArray, MLXArray?) {
    let q = quantized(x, groupSize: 64, bits: bits)
    return (q.wq, q.scales, q.biases)
}

func accuracy() {
    print("== attention-output relative error vs fp32 reference, T=4096, decode query")
    let T = 4096
    for (kShape, vShape) in [(Shape.gaussian, Shape.gaussian), (.channelOutliers, .gaussian), (.channelOutliers, .channelOutliers), (.gaussian, .tokenOutliers)] {
        let K = synth(T, kShape, seed: 1)
        let V = synth(T, vShape, seed: 2)
        MLXRandom.seed(3)
        let q = MLXRandom.normal([B, HQ, 1, D]).asType(.float16)
        let ref = sdpa(q, K, V)
        let a8 = sdpa(q, deq(K, bits: 8), deq(V, bits: 8))
        let a4 = sdpa(q, deq(K, bits: 4), deq(V, bits: 4))
        let k8v4 = sdpa(q, deq(K, bits: 8), deq(V, bits: 4))
        let Vr = hadamardTransform(V.asType(.float32)).asType(.float16)
        let k8v4h = hadamardTransform(sdpa(q, deq(K, bits: 8), deq(Vr, bits: 4)))
        let Kr = hadamardTransform(K.asType(.float32)).asType(.float16)
        let qr = hadamardTransform(q.asType(.float32)).asType(.float16)
        let k8hv4h = hadamardTransform(sdpa(qr, deq(Kr, bits: 8), deq(Vr, bits: 4)))
        let turbo = TurboQuantKVCache(bits: 4, keyBits: 8, valueBits: 4)
        _ = turbo.update(keys: K[.ellipsis, ..<(T - 1), 0...], values: V[.ellipsis, ..<(T - 1), 0...])
        let t84 = turbo.compressedAttention(
            queries: q, keys: K[.ellipsis, (T - 1)..., 0...], values: V[.ellipsis, (T - 1)..., 0...], scale: attnScale)
        let fused = asymSDPA(q, k: tup(K, bits: 8), v: tup(V, bits: 4), keyBits: 8, valueBits: 4)
        let vErr = rel(deq(V, bits: 4), V)
        let vErrH = rel(hadamardTransform(deq(Vr, bits: 4).asType(.float32)), V)
        print("K=\(kShape) V=\(vShape)")
        print("  out err: q8=\(f(rel(a8, ref))) q4=\(f(rel(a4, ref))) k8v4=\(f(rel(k8v4, ref))) k8v4+Hv=\(f(rel(k8v4h, ref))) k8+Hk,v4+Hv=\(f(rel(k8hv4h, ref))) turbo8v4=\(f(rel(t84, ref))) asymQMM=\(f(rel(fused, ref)))")
        print("  V4 recon err: plain=\(f(vErr)) hadamard=\(f(vErrH))")
    }
}

func bytes() {
    print("== bytes per layer at 16384 tokens (8 kv heads, head_dim 128)")
    let T = 16384
    let K = synth(T, .gaussian, seed: 5), V = synth(T, .gaussian, seed: 6)
    let fp16 = K.nbytes + V.nbytes
    let turbo = TurboQuantKVCache(bits: 4, keyBits: 8, valueBits: 4)
    _ = turbo.update(keys: K[.ellipsis, ..<(T - 1), 0...], values: V[.ellipsis, ..<(T - 1), 0...])
    let q = MLXRandom.normal([B, HQ, 1, D]).asType(.float16)
    eval(turbo.compressedAttention(queries: q, keys: K[.ellipsis, (T - 1)..., 0...], values: V[.ellipsis, (T - 1)..., 0...], scale: attnScale))
    let live = turbo.state.reduce(0) { $0 + $1.nbytes }
    let q8 = QuantizedKVCache(groupSize: 64, bits: 8); _ = q8.updateQuantized(keys: K, values: V)
    let q4 = QuantizedKVCache(groupSize: 64, bits: 4); _ = q4.updateQuantized(keys: K, values: V)
    let s8 = q8.state.reduce(0) { $0 + $1.nbytes }, s4 = q4.state.reduce(0) { $0 + $1.nbytes }
    let elems = Double(T * HKV * D)
    print("  fp16 \(fp16)  bits/elem-pair \(Double(fp16) * 8 / elems)")
    print("  affine8 \(s8)  bits/pair \(Double(s8) * 8 / elems)  ratio \(Double(s8) / Double(fp16))")
    print("  affine4 \(s4)  bits/pair \(Double(s4) * 8 / elems)  ratio \(Double(s4) / Double(fp16))")
    print("  turbo8v4 state \(live) memoryBytes(alloc) \(turbo.memoryBytes)  bits/pair \(Double(live) * 8 / elems)  ratio \(Double(live) / Double(fp16))")
    for (i, a) in turbo.state.enumerated() { print("    state[\(i)] \(a.shape) \(a.dtype) \(a.nbytes)") }
}

func peak(_ label: String, layers: Int, T: Int, convert: (inout [KVCache]) -> Void, decode: ([KVCache], MLXArray, MLXArray, MLXArray) -> [MLXArray]) {
    var caches: [KVCache] = (0 ..< layers).map { _ in KVCacheSimple() }
    for (i, c) in caches.enumerated() {
        let (k, v) = c.update(keys: synth(T, .gaussian, seed: UInt64(100 + i)), values: synth(T, .gaussian, seed: UInt64(200 + i)))
        eval(k, v)
    }
    Memory.clearCache()
    let before = Memory.activeMemory
    Memory.peakMemory = 0
    convert(&caches)
    MLXRandom.seed(9)
    let q = MLXRandom.normal([B, HQ, 1, D]).asType(.float16)
    let k1 = MLXRandom.normal([B, HKV, 1, D]).asType(.float16)
    let v1 = MLXRandom.normal([B, HKV, 1, D]).asType(.float16)
    eval(decode(caches, q, k1, v1))
    let pk = Memory.peakMemory
    Memory.clearCache()
    let after = Memory.activeMemory
    print("  \(label): fp16 resident \(gb(before)) → peak \(gb(pk)) (\(String(format: "%.2f", Double(pk) / Double(before)))× fp16) → steady \(gb(after)) (\(String(format: "%.3f", Double(after) / Double(before)))×)")
    caches.removeAll()
    Memory.clearCache()
}

func conversionPeak() {
    let layers = 36, T = 16384
    print("== conversion peak, \(layers) layers × \(T) tokens (Qwen3-4B-like geometry)")
    let quantDecode: ([KVCache], MLXArray, MLXArray, MLXArray) -> [MLXArray] = { caches, q, k, v in
        caches.map { c in
            attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: c, scale: attnScale, mask: .none)
        }
    }
    peak("affine8", layers: layers, T: T, convert: { maybeQuantizeKVCache(cache: &$0, kvBits: 8) }, decode: quantDecode)
    peak("affine4", layers: layers, T: T, convert: { maybeQuantizeKVCache(cache: &$0, kvBits: 4) }, decode: quantDecode)
    peak("turbo8v4", layers: layers, T: T, convert: { maybeQuantizeKVCache(cache: &$0, kvBits: nil, kvScheme: "turbo8v4") }, decode: quantDecode)
}

func trimCheck() {
    print("== turbo8v4 trim after compression")
    let T = 2048
    let K = synth(T, .gaussian, seed: 11), V = synth(T, .gaussian, seed: 12)
    MLXRandom.seed(13)
    let q = MLXRandom.normal([B, HQ, 1, D]).asType(.float16)
    let k1 = MLXRandom.normal([B, HKV, 1, D]).asType(.float16), v1 = MLXRandom.normal([B, HKV, 1, D]).asType(.float16)
    let a = TurboQuantKVCache(bits: 4, keyBits: 8, valueBits: 4)
    _ = a.update(keys: K, values: V)
    eval(a.compressedAttention(queries: q, keys: k1, values: v1, scale: attnScale))
    a.trim(101)
    let outA = a.compressedAttention(queries: q, keys: k1, values: v1, scale: attnScale)
    let b = TurboQuantKVCache(bits: 4, keyBits: 8, valueBits: 4)
    _ = b.update(keys: K[.ellipsis, ..<(T - 100), 0...], values: V[.ellipsis, ..<(T - 100), 0...])
    let outB = b.compressedAttention(queries: q, keys: k1, values: v1, scale: attnScale)
    let ref = sdpa(q, concatenated([K[.ellipsis, ..<(T - 100), 0...], k1], axis: 2), concatenated([V[.ellipsis, ..<(T - 100), 0...], v1], axis: 2))
    print("  offsets a=\(a.offset) b=\(b.offset)  trimmed-vs-fresh rel=\(f(rel(outA, outB)))  trimmed-vs-fp32=\(f(rel(outA, ref)))")
    let multi = a.compressedAttention(
        queries: MLXRandom.normal([B, HQ, 16, D]).asType(.float16),
        keys: MLXRandom.normal([B, HKV, 16, D]).asType(.float16), values: MLXRandom.normal([B, HKV, 16, D]).asType(.float16),
        scale: attnScale, mask: .causal)
    eval(multi)
    print("  16-token chunk after compression ok, offset \(a.offset), finite \(isNaN(multi).any().item(Bool.self) == false)")
}

func speed() {
    let T = 16384, layers = 36, steps = 64
    print("== decode attention time, \(layers) layers × \(T) context, \(steps) steps (attention only, no matmuls)")
    let K = synth(T, .gaussian, seed: 21), V = synth(T, .gaussian, seed: 22)
    MLXRandom.seed(23)
    let qs = (0 ..< steps).map { _ in MLXRandom.normal([B, HQ, 1, D]).asType(.float16) }
    let ks = (0 ..< steps).map { _ in MLXRandom.normal([B, HKV, 1, D]).asType(.float16) }
    func run(_ label: String, make: () -> KVCache) {
        let caches = (0 ..< layers).map { _ -> KVCache in
            let c = make()
            if let s = c as? KVCacheSimple { _ = s.update(keys: K, values: V); eval(s.innerState()) }
            else if let qc = c as? QuantizedKVCache { let r = qc.updateQuantized(keys: K, values: V); eval(r.0.0, r.1.0) }
            else if let t = c as? TurboQuantKVCache { _ = t.update(keys: K, values: V); eval(t.compressedAttention(queries: qs[0], keys: ks[0], values: ks[0], scale: attnScale)) }
            return c
        }
        for i in 0 ..< 4 { eval(caches.map { attentionWithCacheUpdate(queries: qs[i], keys: ks[i], values: ks[i], cache: $0, scale: attnScale) }) }
        let t0 = Date()
        for i in 0 ..< steps { eval(caches.map { attentionWithCacheUpdate(queries: qs[i], keys: ks[i], values: ks[i], cache: $0, scale: attnScale) }) }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(steps)
        print("  \(label): \(String(format: "%.2f", ms)) ms/token")
        Memory.clearCache()
    }
    run("fp16") { KVCacheSimple() }
    run("affine8") { QuantizedKVCache(groupSize: 64, bits: 8) }
    run("affine4") { QuantizedKVCache(groupSize: 64, bits: 4) }
    run("turbo8v4") { TurboQuantKVCache(bits: 4, keyBits: 8, valueBits: 4) }
    let kq = tup(K, bits: 8), vq = tup(V, bits: 4)
    let vqh = tup(hadamardTransform(V.asType(.float32)).asType(.float16), bits: 4)
    eval(kq.0, vq.0, vqh.0)
    for (label, rotate) in [("asymQMM k8v4 (static, no append)", false), ("asymQMM k8v4+H (static, no append)", true)] {
        let t0 = Date()
        for i in 0 ..< steps {
            eval((0 ..< layers).map { _ in
                rotate ? hadamardTransform(asymSDPA(qs[i], k: kq, v: vqh, keyBits: 8, valueBits: 4)) : asymSDPA(qs[i], k: kq, v: vq, keyBits: 8, valueBits: 4)
            })
        }
        print("  \(label): \(String(format: "%.2f", Date().timeIntervalSince(t0) * 1000 / Double(steps))) ms/token")
    }
}

let which = CommandLine.arguments.dropFirst().first ?? "all"
if which == "all" || which == "accuracy" { accuracy() }
if which == "all" || which == "bytes" { bytes() }
if which == "all" || which == "trim" { trimCheck() }
if which == "all" || which == "peak" { conversionPeak() }
if which == "all" || which == "speed" { speed() }
if which == "model" {
    let dir = URL(fileURLWithPath: CommandLine.arguments[2])
    let lengths = (ProcessInfo.processInfo.environment["PASSKEY"] ?? "16384").split(separator: ",").compactMap { Int($0) }
    try await modelEval(dir, schemes: Scheme.allCases, pplTokens: Int(ProcessInfo.processInfo.environment["PPL"] ?? "8192")!, passkeyLengths: lengths)
}
