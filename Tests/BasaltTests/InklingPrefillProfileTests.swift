import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("Inkling 2 P2c: where one prefill chunk spends its time, layer by part, on a random 3-bit layer of the real geometry (#650)", .serialized)
struct InklingPrefillProfileTests {
    static var enabled: Bool { ProcessInfo.processInfo.environment["BASALT_E2E_INKLING_PROFILE"] == "1" }
    static func config() throws -> Inkling2Configuration {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Inkling2/config.json")
        return try JSONDecoder().decode(Inkling2Configuration.self, from: Data(contentsOf: url))
    }

    static func time(_ label: String, _ n: Int = 5, _ f: () -> MLXArray) {
        for _ in 0 ..< 2 { eval(f()) }
        Memory.peakMemory = 0
        let start = ContinuousClock.now
        for _ in 0 ..< n { eval(f()) }
        let d = ContinuousClock.now - start
        let ms = (Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15) / Double(n)
        print("PREFILL \(label): \(String(format: "%.1f", ms)) ms, peak \(Memory.peakMemory / 1_048_576) MB")
    }

    static func randomized(_ m: Module) {
        let params = m.parameters().flattened().map { ($0.0, (MLXRandom.normal($0.1.shape) * 0.02).asType(.bfloat16)) }
        m.update(parameters: ModuleParameters.unflattened(params))
        quantize(model: m) { path, module in
            if module is SwitchLinear { return (64, 3, .affine) }
            if let l = module as? Linear, l.weight.dim(-1) % 64 == 0, !path.contains("message_gate") { return (64, 8, .affine) }
            return nil
        }
        eval(m)
    }

    @Test("A 512-token chunk after 4K (global and local layer) and 16K (global): full layer, MoE, attention, prepare, mask, masked SDPA and a causal SDPA lower bound",
          .enabled(if: enabled, "set BASALT_E2E_INKLING_PROFILE=1"), arguments: [(5, 4096), (3, 4096), (5, 16384)])
    func chunk(index: Int, context: Int) throws {
        let config = try Self.config()
        let chunk = 512
        let layer = Inkling2DecoderLayer(config, index: index)
        Self.randomized(layer)
        let kind = config.localLayers.contains(index) ? "local" : "global"
        let kv: KVCache = config.localLayers.contains(index) ? Inkling2WindowKVCache(window: config.slidingWindow) : KVCacheSimple()
        let d = config.hiddenSize
        let blocks = MLXRandom.normal([1, 1, chunk, d]).asType(.float32)
        let history = (MLXRandom.normal([1, context - chunk, d]) * 0.1).asType(.bfloat16)
        _ = layer(history, blocks: broadcast(blocks[0..., 0..., ..<1, 0...], to: [1, 1, context - chunk, d]), cache: kv, conv: ArraysCache(size: 4), messages: nil)
        eval(kv.innerState())
        let x = (MLXRandom.normal([1, chunk, d]) * 0.1).asType(.bfloat16)
        let label = "layer \(index) \(kind) chunk \(chunk) after \(context - chunk)"
        Self.time("\(label) full layer") {
            layer(x, blocks: blocks, cache: kv.copy(), conv: ArraysCache(size: 4), messages: nil).0
        }
        Self.time("\(label) moe") { layer.mlp(x) }
        let inputs = Inkling2AttentionInputs(query: x, key: x, value: x)
        Self.time("\(label) attention (prepare + mask + sdpa + o_proj)") {
            var states: [MLXArray?] = [nil, nil, nil, nil]
            return layer.attention(inputs, cache: kv.copy(), conv: &states)
        }
        let start = kv.offset
        let (prepared, _, _) = layer.attention.prepare(inputs, keyState: nil, valueState: nil, positions: Inkling2Attention.positions(start, chunk))
        let copy = kv.copy()
        let (keys, values, keyStart) = Inkling2Attention.store(prepared.keys, prepared.values, in: copy, start: start)
        eval(keys, values, prepared.queries, prepared.bias, prepared.sinks)
        let distance = Inkling2Attention.distance(query: start, length: chunk, key: keyStart, count: keys.dim(2))
        Self.time("\(label) prepare only") { layer.attention.prepare(inputs, keyState: nil, valueState: nil, positions: Inkling2Attention.positions(start, chunk)).0.queries }
        Self.time("\(label) mask only (\(keys.dim(2)) keys)") { layer.attention.additiveMask(bias: prepared.bias, length: chunk, keysCount: keys.dim(2), positions: Inkling2Positions(query: start, key: keyStart)) }
        let mask = layer.attention.additiveMask(bias: prepared.bias, length: chunk, keysCount: keys.dim(2), positions: Inkling2Positions(query: start, key: keyStart)).asType(prepared.queries.dtype)
        eval(mask)
        Self.time("\(label) sdpa with mask + sink column") { layer.attention.sdpa(prepared.queries, keys, values, mask: mask, sinks: prepared.sinks) }
        Self.time("\(label) sdpa causal, no mask (lower bound)") {
            MLXFast.scaledDotProductAttention(queries: prepared.queries, keys: keys, values: values, scale: layer.attention.scale, mask: .causal)
        }
        #expect(distance.dim(1) == keys.dim(2))
    }
}
