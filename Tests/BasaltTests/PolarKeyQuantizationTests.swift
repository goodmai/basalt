import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Polar key quantization spike (#351): self-checks on synthetic keys")
struct PolarKeyQuantizationTests {
    static let rope = PolarKeySpike.Rope(rotated: 64, theta: 1e7)

    @Test("Polar encode/decode of CN(0, σ²) pairs keeps every phase within π/2^b")
    func phaseBound() {
        let keys = MLXRandom.normal([1, 2, 512, 64], key: MLXRandom.key(1))
        let decoded = PolarKeySpike.polar(keys, rope: Self.rope, phaseBits: 4, modulusBits: 3, restBits: 3)
        let (re, im, _) = PolarKeySpike.split(keys, rope: Self.rope)
        let (reHat, imHat, _) = PolarKeySpike.split(decoded, rope: Self.rope)
        let difference = atan2(imHat, reHat) - atan2(im, re)
        let error = abs(atan2(sin(difference), cos(difference))).max().item(Float.self)
        #expect(error <= .pi / 16 + 1e-4, "max phase error \(error)")
    }

    @Test("A code shift by a rounded δ stays within π/2^b of the true rotation on top of the encode error, and is never exact for Δp > 0", arguments: [1, 17, 1000, 4097])
    func shiftBound(delta: Int) {
        let keys = MLXRandom.normal([1, 2, 256, 64], key: MLXRandom.key(2))
        let (maxError, _) = PolarKeySpike.shiftedPolarPhaseError(keys, rope: Self.rope, phaseBits: 4, delta: delta)
        #expect(maxError <= 2 * .pi / 16 + 1e-4 && maxError > 0, "Δp \(delta): \(maxError)")
    }

    @Test("Rotating the query by −Δ equals roping the keys by +Δ, within fp32 rounding")
    func queryRotation() {
        let keys = MLXRandom.normal([1, 2, 128, 64], key: MLXRandom.key(3))
        let query = MLXRandom.normal([1, 2, 1, 64], key: MLXRandom.key(4))
        let shifted = matmul(query, PolarKeySpike.rotate(keys, rope: Self.rope, by: 1000).swappedAxes(-1, -2))
        let rotatedQuery = matmul(PolarKeySpike.rotate(query, rope: Self.rope, by: -1000), keys.swappedAxes(-1, -2))
        #expect((abs(shifted - rotatedQuery).max() / abs(shifted).max()).item(Float.self) < 1e-5)
    }

    @Test("The affine row is bit-exact with ZonedKVCache's middle zone on the same input")
    func affineMatchesZoned() throws {
        let keys = MLXRandom.normal([1, 2, 64, 128], key: MLXRandom.key(5))
        var rows = ZonedKVCache.QuantizedRows(bits: 3, groupSize: 64)
        rows.append(keys)
        #expect(arrayEqual(try #require(rows.dequantizedRows()), PolarKeySpike.affine(keys, bits: 3)).item(Bool.self))
    }
}

@Suite("Polar key quantization spike on real keys (#351 step 2): polar per RoPE pair against affine and TurboQuant at 3.5 and 4.5 bits per coordinate", .serialized)
struct PolarKeyRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let models = env["BASALT_E2E"] == "1" ? (env["BASALT_E2E_POLAR_MODELS"] ?? "").split(separator: ",").map(String.init) : []

    static func rope(_ path: String) -> PolarKeySpike.Rope? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path).appending(component: "config.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let text = json["text_config"] as? [String: Any] ?? json
        let parameters = text["rope_parameters"] as? [String: Any]
        let theta = (parameters?["rope_theta"] ?? text["rope_theta"]) as? Double ?? 10000
        let factor = (parameters?["partial_rotary_factor"] ?? text["partial_rotary_factor"]) as? Double ?? 1
        let headDim = text["head_dim"] as? Int ?? ((text["hidden_size"] as? Int ?? 0) / max(text["num_attention_heads"] as? Int ?? 1, 1))
        return PolarKeySpike.Rope(rotated: Int(Double(headDim) * factor), theta: Float(theta))
    }

    static func methods(_ rope: PolarKeySpike.Rope) -> [(String, Double, (MLXArray) -> MLXArray)] {
        [
            ("affine 3-bit g64", 3.5, { PolarKeySpike.affine($0, bits: 3) }),
            ("turbo 3-bit", 3.0, { PolarKeySpike.turbo($0, bits: 3) }),
            ("polar φ4 r3", 3.5, { PolarKeySpike.polar($0, rope: rope, phaseBits: 4, modulusBits: 3, restBits: 3) }),
            ("affine 4-bit g64", 4.5, { PolarKeySpike.affine($0, bits: 4) }),
            ("turbo 4-bit", 4.0, { PolarKeySpike.turbo($0, bits: 4) }),
            ("polar φ5 r4", 4.5, { PolarKeySpike.polar($0, rope: rope, phaseBits: 5, modulusBits: 4, restBits: 4) }),
        ]
    }

    static func report(_ recorders: [KVRecordingCache], rope: PolarKeySpike.Rope, length: Int, log: ModelRunLog) {
        var kls: [String: [Float]] = [:]
        for (layer, recorder) in recorders.enumerated() {
            let keys = recorder.rows.keys
            for (name, bits, method) in methods(rope) {
                let approx = method(keys)
                let scores = PolarKeySpike.scores(queries: recorder.queries, keys: keys, approx: approx)
                let perCoordinate = bits + (name.hasPrefix("turbo") ? 32 / Double(keys.dim(-1)) : 0)
                kls[name, default: []].append(scores.kl)
                log.record("row", String(format: "{\"tokens\":%d,\"layer\":%d,\"method\":\"%@\",\"bits\":%.3f,\"logit_error\":%.5f,\"kl\":%.6f,\"top32\":%.4f}",
                    length, layer, name, perCoordinate, scores.logitError, scores.kl, scores.overlap))
            }
            if layer == recorders.count / 2 {
                let stats = PolarKeySpike.pairStatistics(keys, rope: rope, phaseBits: 4)
                let mean = { (values: [Float]) in values.isEmpty ? Float.nan : values.reduce(0, +) / Float(values.count) }
                log.record("pairs", String(format: "layer %d: %d slow pairs mean resultant %.3f phase entropy %.3f Rayleigh KS %.3f; %d fast pairs %.3f / %.3f / %.3f",
                    layer, stats.slow.count, mean(stats.slow), stats.entropySlow, stats.ksSlow, stats.fast.count, mean(stats.fast), stats.entropyFast, stats.ksFast))
                for delta in [1, 17, 1000, 4097] {
                    let shift = PolarKeySpike.shiftedPolarPhaseError(keys, rope: rope, phaseBits: 4, delta: delta)
                    log.record("relocate", String(format: "Δp %d: polar code shift (b=4) phase error max %.3f rms %.3f rad (bound %.3f per step); query rotation and re-RoPE are exact to fp16", delta, shift.maxError, shift.rmsError, Float.pi / 16))
                }
            }
        }
        for (name, _, _) in methods(rope) {
            let values = (kls[name] ?? []).sorted()
            log.record("RESULT", String(format: "%d tokens %@: median KL %.6f, worst layer %.6f", length, name, values.isEmpty ? .nan : values[values.count / 2], values.last ?? .nan))
        }
    }

    @Test("Median KL per method and budget over full-attention layers at 8k and 32k tokens; the go gate is polar 20% under both affine and TurboQuant at 3.5 bits",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_POLAR_MODELS (Qwen3.8-27B-3bit, LFM2.5-24B-A2B-4bit); run a release build"), arguments: models)
    func spike(model: String) async throws {
        let path = NF4Run.path(model)
        let rope = try #require(Self.rope(path), "no RoPE geometry in \(path)/config.json")
        let log = try await ModelRunLog("polar-keys", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        log.record("rope", "\(rope.rotated) rotated dims (\(rope.pairs) pairs), θ \(rope.theta), \((0 ..< rope.pairs).filter(rope.slow).count) pairs turn less than once in 32k tokens")
        let corpus = KVMergeRealModelTests.corpus
        try await container.performSending(nonSendable: corpus) { context, corpus in
            for length in [8192, 32768] {
                let recorders = try KVMergeRealModelTests.capture(context, tokens: length, corpus: corpus)
                log.record("capture", "\(length) tokens requested, \(recorders.first?.rows.keys.dim(2) ?? 0) recorded, \(recorders.count) full-attention layers")
                Self.report(recorders, rope: rope, length: length, log: log)
            }
        }
        await engine.unload()
    }
}
