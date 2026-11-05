import CryptoKit
import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

enum KeyTrajectorySignal {
    static let poles = [
        Cx.polar(0.999, 0.05), Cx.polar(0.995, 0.4), Cx.polar(0.99, 1.1), Cx.polar(0.98, -0.7),
        Cx.polar(0.997, 2.0), Cx.polar(0.985, -1.6), Cx.polar(0.993, 2.7), Cx.polar(0.975, -2.3),
    ]

    static func damped(_ poles: [Cx], length: Int = 1000, noise: Double, seed: UInt64 = 7) -> [Cx] {
        var generator = SplitMix64(seed: seed)
        let residues = poles.indices.map { Cx.polar(1 + 0.25 * Double($0), 0.9 * Double($0)) }
        return (0 ..< length).map { t in
            var value = Cx(re: noise * .random(in: -1 ... 1, using: &generator), im: noise * .random(in: -1 ... 1, using: &generator))
            for (pole, residue) in zip(poles, residues) { value = value + residue * Cx.polar(pow(pole.magnitude, Double(t)), atan2(pole.im, pole.re) * Double(t)) }
            return value
        }
    }

    static func worstPoleError(_ fitted: [Cx], _ truth: [Cx]) -> Double {
        truth.map { pole in fitted.map { ($0 - pole).magnitude }.min() ?? .infinity }.max() ?? .infinity
    }
}

@Suite("Prony fit of key trajectories: pole recovery and residual on synthetic signals (#364)")
struct KeyTrajectoryFitTests {
    @Test("A sum of M damped complex exponentials plus noise: the order-M fit recovers every pole and leaves a residual near the noise floor",
          arguments: [4, 8])
    func recoversPoles(order: Int) throws {
        let truth = Array(KeyTrajectorySignal.poles.prefix(order))
        let fit = try #require(KeyTrajectoryFit.fit(KeyTrajectorySignal.damped(truth, noise: 1e-4), order: order))
        #expect(KeyTrajectorySignal.worstPoleError(fit.poles, truth) < 1e-3, "\(fit.poles)")
        #expect(fit.relative < 1e-2)
    }

    @Test("Too low an order leaves a large residual on an 8-pole signal, so the error ranks trajectory order")
    func underfitShowsInResidual() throws {
        let signal = KeyTrajectorySignal.damped(KeyTrajectorySignal.poles, noise: 1e-4)
        let low = try #require(KeyTrajectoryFit.fit(signal, order: 4))
        let exact = try #require(KeyTrajectoryFit.fit(signal, order: 8))
        #expect(low.relative > 0.1)
        #expect(exact.relative < 1e-3)
    }

    @Test("Pure noise fits with a relative residual close to 1 at every order")
    func noiseResidualNearOne() throws {
        let signal = KeyTrajectorySignal.damped([], noise: 1)
        for order in KeyTrajectoryFit.orders {
            let fit = try #require(KeyTrajectoryFit.fit(signal, order: order))
            #expect(fit.relative > 0.9, "order \(order): \(fit.relative)")
        }
    }

    @Test("A RoPE rotation e^{iθt} on a channel pair leaves the fit error unchanged, so post-RoPE keys give the pre-RoPE order")
    func ropeRotationInvariant() throws {
        let signal = KeyTrajectorySignal.damped(Array(KeyTrajectorySignal.poles.prefix(6)), noise: 0.05)
        let rotated = signal.enumerated().map { $1 * Cx.polar(1, 0.3 * Double($0)) }
        let plain = try #require(KeyTrajectoryFit.fit(signal, order: 4))
        let turned = try #require(KeyTrajectoryFit.fit(rotated, order: 4))
        #expect(abs(plain.relative - turned.relative) < 1e-6)
    }

    @Test("Zero and too-short signals are skipped, not fitted")
    func degenerateInputsSkip() {
        #expect(KeyTrajectoryFit.fit([Cx](repeating: .zero, count: 100), order: 4) == nil)
        #expect(KeyTrajectoryFit.fit(Array(KeyTrajectorySignal.damped(KeyTrajectorySignal.poles, noise: 0).prefix(8)), order: 4) == nil)
    }

    @Test("Channel pairs follow RoPE's half split in the rotated block and cover each channel once",
          arguments: [(256, 32), (64, 32), (512, 256)])
    func channelPairsCoverHead(headDim: Int, ropeHalf: Int) {
        let pairs = KeyTrajectoryFit.channelPairs(headDim: headDim, ropeHalf: ropeHalf)
        #expect(Set(pairs.flatMap { [$0.re, $0.im] }) == Set(0 ..< headDim))
        #expect(pairs.prefix(ropeHalf).allSatisfy { $0.im == $0.re + ropeHalf })
    }

    @Test("A head of two low-order channel pairs scores below the low-error bar at M = 4; a noise head does not")
    func headErrorSeparatesHeads() {
        let signal = KeyTrajectorySignal.damped(Array(KeyTrajectorySignal.poles.prefix(2)), noise: 1e-4)
        let noise = KeyTrajectorySignal.damped([], noise: 1, seed: 11)
        let head = { (x: [Cx]) in ArraySlice(x.flatMap { [Float($0.re), Float($0.re), Float($0.im), Float($0.im)] }) }
        let plain = KeyTrajectoryFit.Rope(half: 0, rotated: 0, base: 1)
        let low = KeyTrajectoryFit.headError(head(signal), tokens: 1000, headDim: 4, rope: plain)
        let high = KeyTrajectoryFit.headError(head(noise), tokens: 1000, headDim: 4, rope: plain)
        #expect(low.relative[0] < KeyTrajectoryFit.lowError && low.skipped == 0)
        #expect(high.relative[0] > 0.9)
    }

    @Test("A pre-RoPE mean rotated by RoPE is one pole: inverting RoPE and removing the mean leaves only the noise")
    func centeredErrorRemovesRotatedMean() {
        var generator = SplitMix64(seed: 3)
        let pair = (0 ..< 1000).flatMap { t -> [Float] in
            let value = Cx(re: 5 + 0.1 * .random(in: -1 ... 1, using: &generator), im: 0.1 * .random(in: -1 ... 1, using: &generator))
                * Cx.polar(1, Double(t))
            return [Float(value.re), Float(value.im)]
        }
        let error = KeyTrajectoryFit.headError(ArraySlice(pair), tokens: 1000, headDim: 2,
                                               rope: KeyTrajectoryFit.Rope(half: 1, rotated: 1, base: 10_000))
        #expect(error.relative[0] < KeyTrajectoryFit.lowError)
        #expect(error.centered[0] > 0.9)
    }

    @Test("Two separated clusters of head errors are bimodal and clear the go bar; one tight cluster, or low errors that are only a mean, do not")
    func goVerdict() {
        let split = Array(repeating: 0.03, count: 10) + Array(repeating: 0.9, count: 30) + [0.05, 0.85]
        let single = (0 ..< 40).map { 0.8 + 0.001 * Double($0) }
        #expect(KeyTrajectoryFit.phaseOneGo(split, centered: split).go)
        #expect(!KeyTrajectoryFit.phaseOneGo(single, centered: single).go)
        #expect(!KeyTrajectoryFit.phaseOneGo(split, centered: split.map { _ in 0.95 }).go)
    }
}

struct KeyDump: Sendable {
    struct Layer: Sendable {
        let index: Int
        let heads: Int
        let tokens: Int
        let headDim: Int
        let keys: [Float]
    }

    let tokens: Int
    let layers: [Layer]

    static func capture(_ model: any LanguageModel, tokens: [Int]) throws -> KeyDump {
        let caches = try model.newCache(parameters: nil)
        eval([model(MLXArray(tokens.map(Int32.init))[.newAxis], cache: caches)] + caches.flatMap { $0.innerState() })
        let layers = caches.enumerated().compactMap { index, cache -> Layer? in
            guard type(of: cache) == KVCacheSimple.self, let keys = cache.state.first else { return nil }
            return Layer(index: index, heads: keys.dim(1), tokens: keys.dim(2), headDim: keys.dim(3),
                         keys: keys[0].asType(.float32).asArray(Float.self))
        }
        return KeyDump(tokens: tokens.count, layers: layers)
    }
}

@Suite("Key trajectory order on real models: Prony fit error per full-attention KV head (#364)", .serialized)
struct KeyTrajectoryRealModelTests {
    static let env = ProcessInfo.processInfo.environment

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_PADE_MODELS"] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    static func path(_ entry: String) -> String {
        entry.hasPrefix("/") ? entry : ModelCache.cacheDir(for: entry).path
    }

    static var corpus: URL {
        env["BASALT_PADE_CORPUS"].map(URL.init(fileURLWithPath:))
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("CONTRIBUTING.md")
    }

    static func rope(modelPath: String, headDim: Int) -> KeyTrajectoryFit.Rope {
        let config = (try? Data(contentsOf: URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        let text = config["text_config"] as? [String: Any] ?? config
        let parameters = text["rope_parameters"] as? [String: Any] ?? [:]
        let global = parameters["full_attention"] as? [String: Any] ?? parameters
        let base = (global["rope_theta"] ?? text["rope_theta"]) as? Double ?? 10_000
        let partial = (global["partial_rotary_factor"] ?? text["partial_rotary_factor"]) as? Double ?? 1
        let rotated = Int(partial * Double(headDim)) / 2
        return KeyTrajectoryFit.Rope(half: global["rope_type"] as? String == "proportional" ? headDim / 2 : rotated,
                                     rotated: rotated, base: base)
    }

    @Test("Per full-attention KV head: relative Prony residual at M = 4, 8, 16 over 1000-token windows, and the phase-1 go check",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_PADE_MODELS"), arguments: models)
    func trajectoryOrder(entry: String) async throws {
        let path = Self.path(entry)
        let log = try await ModelRunLog("key-trajectory", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine()
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let text = try String(contentsOf: Self.corpus, encoding: .utf8)
        let limit = Self.env["BASALT_PADE_TOKENS"].flatMap(Int.init) ?? 2 * KeyTrajectoryFit.window
        let digest = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
        let dump = try await log.step("dump") {
            try await container.perform { context in
                try KeyDump.capture(context.model, tokens: Array(context.tokenizer.encode(text: text).prefix(limit)))
            }
        }
        log.record("corpus", "\(Self.corpus.lastPathComponent) sha256 \(digest) tokens \(dump.tokens) full-attention layers \(dump.layers.map(\.index))")
        try #require(dump.tokens >= KeyTrajectoryFit.window, "corpus shorter than one window")
        var atSixteen: [Double] = []
        var centeredAtSixteen: [Double] = []
        for layer in dump.layers {
            let rope = Self.rope(modelPath: path, headDim: layer.headDim)
            for head in 0 ..< layer.heads {
                let size = layer.tokens * layer.headDim
                let error = KeyTrajectoryFit.headError(layer.keys[head * size ..< (head + 1) * size],
                                                       tokens: layer.tokens, headDim: layer.headDim, rope: rope)
                let cells = zip(KeyTrajectoryFit.orders, zip(error.relative, error.centered)).map {
                    String(format: "M%ld %.4f centered %.4f", $0, $1.0, $1.1)
                }
                log.record("head", "layer \(layer.index) kv-head \(head) dim \(layer.headDim) rope \(rope.rotated)/\(rope.half) base \(rope.base) \(cells.joined(separator: " ")) skipped \(error.skipped)")
                atSixteen.append(error.relative[KeyTrajectoryFit.orders.count - 1])
                centeredAtSixteen.append(error.centered[KeyTrajectoryFit.orders.count - 1])
            }
        }
        let verdict = KeyTrajectoryFit.phaseOneGo(atSixteen, centered: centeredAtSixteen)
        log.record("result", String(format: "PADE heads %ld below-%.2f-and-centered-at-most-%.2f-at-M16 %.3f bimodality %.3f %@", atSixteen.count,
                                    KeyTrajectoryFit.lowError, KeyTrajectoryFit.centeredCeiling, verdict.lowShare, verdict.bimodality,
                                    verdict.go ? "GO" : "NO-GO"))
        #expect(!atSixteen.isEmpty, "\(log.model): no full-attention KV head captured")
    }
}
