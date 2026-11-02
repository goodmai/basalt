import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Entropy-based dynamic temperature (#40)")
struct EntropyDynamicTemperatureTests {

    private func entropy(_ probs: [Float]) -> Float {
        EntropyDynamicTemperature.entropy(log(MLXArray(probs)).reshaped(1, -1)).item(Float.self)
    }

    private func temperature(_ edt: EntropyDynamicTemperature, h: Float, r: Float = 0) -> Float {
        edt.temperature(entropy: MLXArray([h]), repetition: r).item(Float.self)
    }

    @Test("Entropy of known distributions, in nats")
    func knownEntropies() {
        #expect(abs(entropy(Array(repeating: 1.0 / 8, count: 8)) - log(8)) < 1e-4)
        #expect(abs(entropy([0.5, 0.25, 0.25]) - 1.5 * log(2)) < 1e-4)
        #expect(entropy([1, 0, 0, 0]) == 0, "masked (-inf) tokens contribute 0, not NaN")
    }

    @Test("T_t rises with entropy and with the repetition index")
    func monotonic() {
        let edt = EntropyDynamicTemperature(baseTemperature: 0.7)
        let byEntropy = [Float(0.3), 0.5, 1, 2, 4, 8].map { temperature(edt, h: $0) }
        #expect(zip(byEntropy, byEntropy.dropFirst()).allSatisfy { $0 < $1 }, "\(byEntropy)")
        #expect(byEntropy.allSatisfy { $0 <= 0.7 }, "without repetition EDT only cools: N^(θ/H) ≤ 1")
        let byRepetition = [Float(0), 0.5, 1].map { temperature(edt, h: 1, r: $0) }
        #expect(zip(byRepetition, byRepetition.dropFirst()).allSatisfy { $0 < $1 }, "\(byRepetition)")
        #expect(abs(byRepetition[2] - byRepetition[0] - EntropyDynamicTemperature.defaultGammaRep) < 1e-5)
    }

    @Test("Paper formula: T0 · N^(θ/H)")
    func paperFormula() {
        let edt = EntropyDynamicTemperature(baseTemperature: 0.7)
        #expect(abs(temperature(edt, h: 1.5) - 0.7 * 0.8) < 1e-5)
        #expect(abs(temperature(edt, h: 3) - 0.7 * pow(0.8, 0.5)) < 1e-5)
    }

    @Test("Near-zero entropy is near-greedy and finite; nothing exceeds the clamp")
    func clamped() {
        let edt = EntropyDynamicTemperature(baseTemperature: 1.8)
        #expect(temperature(edt, h: 0) == EntropyDynamicTemperature.temperatureRange.lowerBound)
        #expect(temperature(edt, h: 0.05) == EntropyDynamicTemperature.temperatureRange.lowerBound)
        #expect(temperature(edt, h: 50, r: 1) == EntropyDynamicTemperature.temperatureRange.upperBound)
    }

    @Test("process() sharpens a confident distribution relative to the static temperature")
    func processSharpens() {
        let edt = EntropyDynamicTemperature(baseTemperature: 0.7)
        let logits = MLXArray([Float(2), 0, 0, 0]).reshaped(1, -1)
        let staticTop = softmax(logits / 0.7, axis: -1).max().item(Float.self)
        let edtTop = softmax(edt.process(logits: logits) / 0.7, axis: -1).max().item(Float.self)
        #expect(edtTop > staticTop + 0.05, "static \(staticTop), EDT \(edtTop)")
    }

    @Test("A looping token stream raises R_t and so the temperature; distinct tokens leave it at 0")
    func repetitionFeedsTemperature() {
        var looping = EntropyDynamicTemperature(baseTemperature: 0.7)
        var distinct = EntropyDynamicTemperature(baseTemperature: 0.7)
        let cycle = Array(100..<120)
        for token in Array(repeating: cycle, count: 8).flatMap({ $0 }) {
            looping.didSample(token: MLXArray([Int32(token)]))
        }
        for token in 0..<160 {
            distinct.didSample(token: MLXArray([Int32(token)]))
        }
        #expect(looping.repetition == 1)
        #expect(distinct.repetition == 0)
        let logits = MLXArray([Float(2), 1, 0, 0]).reshaped(1, -1)
        let hot = softmax(looping.process(logits: logits) / 0.7, axis: -1).max().item(Float.self)
        let cold = softmax(distinct.process(logits: logits) / 0.7, axis: -1).max().item(Float.self)
        #expect(hot < cold, "a loop flattens the distribution: \(hot) vs \(cold)")
    }
}

@Suite("Engine sampling setup (#40)")
struct EngineSamplingTests {

    @Test("min_p precedence: request, then --min-p, then the profile; 0 turns it off")
    func minPPrecedence() async {
        let engine = MLXInferenceEngine(minP: 0.1)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p")).0.minP == 0.1)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", minP: 0.02)).0.minP == 0.02)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", minP: 0)).0.minP == 0)
        #expect(await MLXInferenceEngine().sampling(for: GenerationRequest(prompt: "p")).0.minP == 0)
    }

    @Test("EDT processor only when dynamic_temperature is on and the temperature is non-zero")
    func edtComponent() async throws {
        let engine = MLXInferenceEngine()
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p")).1.logitProcessorFactory == nil)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", dynamicTemperature: false)).shaping == nil)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", temperature: 0, dynamicTemperature: true)).shaping == nil)

        let factory = try #require(await engine.sampling(
            for: GenerationRequest(prompt: "p", temperature: 0.9, dynamicTemperature: true)).shaping)
        let edt = factory()
        #expect(edt.baseTemperature == 0.9)

        let (params, _, shaping, _) = await MLXInferenceEngine(minP: 0.1).sampling(
            for: GenerationRequest(prompt: "p", temperature: 0.9, topP: 0.8, dynamicTemperature: true))
        #expect(params.topP == 1 && params.minP == 0)
        let moved = try #require(shaping?())
        #expect(moved.topP == 0.8 && moved.minP == 0.1)
    }

    @Test("The request's min_p masks the tail in the sampler the engine builds")
    func minPMasksTail() async {
        let logits = log(MLXArray([Float(0.5), 0.3, 0.2])).reshaped(1, -1)
        func draws(minP: Double) async -> Set<Int> {
            let request = GenerationRequest(prompt: "p", temperature: 1, topP: 1, minP: minP)
            let sampler = await MLXInferenceEngine().sampling(for: request).0.sampler()
            return Set((0..<300).map { _ in sampler.sample(logits: logits).item(Int.self) })
        }
        #expect(await draws(minP: 0.5) == [0, 1])
        #expect(await draws(minP: 0).contains(2), "without Min-P the tail is reachable")
    }

    @Test("EDT keeps min_p on the original distribution even when a loop heats T_t up")
    func edtKeepsMinPUnderHeat() {
        let logits = log(MLXArray([Float(0.5), 0.3, 0.2])).reshaped(1, -1)
        var hot = EntropyDynamicTemperature(baseTemperature: 0.7, minP: 0.5)
        for token in Array(repeating: Array(0..<16), count: 8).flatMap({ $0 }) {
            hot.didSample(token: MLXArray(Int32(token)))
        }
        #expect(hot.repetition > 0, "the loop must have raised R_t for this test to mean anything")
        let out = hot.process(logits: logits)
        #expect(out[0, 2].item(Float.self) == -Float.infinity)
        #expect(out[0, 0].item(Float.self).isFinite && out[0, 1].item(Float.self).isFinite)
        let plain = EntropyDynamicTemperature(baseTemperature: 0.7)
        let scaled = plain.process(logits: logits)
        #expect(scaled[0, 2].item(Float.self).isFinite)
    }

    @Test("EDT top-p drops the tail of the original distribution")
    func edtTopP() {
        let logits = log(MLXArray([Float(0.5), 0.3, 0.2])).reshaped(1, -1)
        let out = EntropyDynamicTemperature(baseTemperature: 0.7, topP: 0.7).process(logits: logits)
        #expect(out[0, 2].item(Float.self) == -Float.infinity)
        #expect(out[0, 1].item(Float.self).isFinite)
    }

    @Test("A request colder than the floor is never heated up; one-hot entropy is 0, not -0")
    func coldRequestStaysCold() {
        let cold = EntropyDynamicTemperature(baseTemperature: 0.01)
        #expect(cold.temperature(entropy: MLXArray([Float(0.0001)]), repetition: 0).item(Float.self) <= 0.01 + 1e-6)
        let oneHot = log(MLXArray([Float(1), 0, 0, 0])).reshaped(1, -1)
        let h = EntropyDynamicTemperature.entropy(oneHot).item(Float.self)
        #expect(h == 0 && h.sign == .plus)
        let t = cold.temperature(entropy: EntropyDynamicTemperature.entropy(oneHot), repetition: 0).item(Float.self)
        #expect(t <= 0.01 + 1e-6, "a confident token must not jump to the ceiling")
    }
}

@Suite("DRY penalty and XTC (#177)")
struct DRYAndXTCTests {

    private func dryLogits(_ history: [Int], breakers: Set<Int> = [], vocabulary: Int = 8) -> [Float] {
        var dry = DRYPenalty(multiplier: 0.8, base: 1.75, allowedLength: 2, breakers: breakers)
        dry.prompt(MLXArray(history.dropLast().map(Int32.init)))
        dry.didSample(token: MLXArray([Int32(history[history.count - 1])]))
        return dry.process(logits: MLXArray.zeros([1, vocabulary], type: Float32.self)).asArray(Float.self)
    }

    private func expectOnly(_ logits: [Float], token: Int, penalty: Float) {
        for (id, value) in logits.enumerated() {
            #expect(abs(value - (id == token ? -penalty : 0)) < 1e-5, "token \(id): \(value)")
        }
    }

    @Test("On A B C D E A B C, DRY lowers only D, by m·b^(3−2), reading the last sampled token after the fact")
    func penalisesTheContinuation() {
        expectOnly(dryLogits([1, 2, 3, 4, 5, 1, 2, 3]), token: 4, penalty: 0.8 * 1.75)
    }

    @Test("A breaker inside the repeated run shortens the match: between B and C it falls below the allowed length, between A and B it drops to b^0")
    func breakersCutTheMatch() {
        expectOnly(dryLogits([1, 2, 6, 3, 4, 5, 1, 2, 6, 3]), token: 4, penalty: 0.8 * 1.75 * 1.75)
        expectOnly(dryLogits([1, 2, 6, 3, 4, 5, 1, 2, 6, 3], breakers: [6]), token: 4, penalty: 0)
        expectOnly(dryLogits([1, 6, 2, 3, 4, 5, 1, 6, 2, 3], breakers: [6]), token: 4, penalty: 0.8)
    }

    @Test("A run shorter than the allowed length and a history without repeats are left alone")
    func shortRunsPass() {
        expectOnly(dryLogits([1, 4, 5, 1]), token: 4, penalty: 0)
        expectOnly(dryLogits([1, 2, 3, 4, 5, 6, 7]), token: 4, penalty: 0)
    }

    @Test("A copy keeps its own history")
    func copyIsIndependent() {
        var original = DRYPenalty(multiplier: 0.8, breakers: [])
        original.prompt(MLXArray([Int32(1), 2, 3, 4, 5, 1, 2]))
        var copy = original.copy()
        copy.didSample(token: MLXArray([Int32(7)]))
        original.didSample(token: MLXArray([Int32(3)]))
        let logits = MLXArray.zeros([1, 8], type: Float32.self)
        expectOnly(original.process(logits: logits).asArray(Float.self), token: 4, penalty: 0.8 * 1.75)
        expectOnly(copy.process(logits: logits).asArray(Float.self), token: 4, penalty: 0)
    }

    private let distribution = log(MLXArray([Float(0.4), 0.3, 0.2, 0.05, 0.05])).reshaped(1, -1)

    @Test("XTC at p = 1, τ = 0.1 masks every token above τ except the least likely of them")
    func excludesTopChoices() {
        let out = ExcludeTopChoices(threshold: 0.1, probability: 1, seed: 1).process(logits: distribution).asArray(Float.self)
        let original = distribution.asArray(Float.self)
        #expect(out[0] == -.infinity && out[1] == -.infinity)
        #expect(Array(out[2...]) == Array(original[2...]))
    }

    @Test("XTC at p = 0 leaves the logits unchanged; with one token above τ there is nothing to exclude")
    func probabilityZeroIsIdentity() {
        let original = distribution.asArray(Float.self)
        #expect(ExcludeTopChoices(threshold: 0.1, probability: 0, seed: 1).process(logits: distribution).asArray(Float.self) == original)
        #expect(ExcludeTopChoices(threshold: 0.35, probability: 1, seed: 1).process(logits: distribution).asArray(Float.self) == original)
    }

    @Test("XTC with the same seed excludes on the same steps; the draws are not constant")
    func seededDrawsRepeat() {
        func draws(seed: UInt64) -> [Bool] {
            let xtc = ExcludeTopChoices(threshold: 0.1, probability: 0.5, seed: seed)
            return (0 ..< 32).map { _ in xtc.process(logits: distribution)[0, 0].item(Float.self) == -.infinity }
        }
        let first = draws(seed: 9)
        #expect(first == draws(seed: 9))
        #expect(Set(first).count == 2)
    }

    @Test("With every DRY and XTC field unset no processor is appended; DRY needs a positive multiplier, XTC a positive probability and T > 0")
    func wiring() async throws {
        let engine = MLXInferenceEngine(seed: 5)
        let plain = await engine.sampling(for: GenerationRequest(prompt: "p"))
        #expect(plain.1.logitProcessorFactory == nil && plain.exclusion == nil)
        let inert = await engine.sampling(for: GenerationRequest(
            prompt: "p", dryMultiplier: 0, dryBase: 2, dryAllowedLength: 3, drySequenceBreakers: ["x"], xtcThreshold: 0.2))
        #expect(inert.1.logitProcessorFactory == nil && inert.exclusion == nil)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", temperature: 0, xtcProbability: 1)).exclusion == nil)

        let on = await engine.sampling(
            for: GenerationRequest(prompt: "p", temperature: 0.7, dryMultiplier: 0.8, xtcProbability: 0.5), breakers: [9])
        let dry = try #require(on.1.logitProcessorFactory?() as? DRYPenalty)
        #expect(dry.multiplier == 0.8 && dry.base == DRYPenalty.defaultBase
            && dry.allowedLength == DRYPenalty.defaultAllowedLength && dry.breakers == [9])
        let xtc = try #require(on.exclusion?())
        #expect(xtc.threshold == ExcludeTopChoices.defaultThreshold && xtc.probability == 0.5)
    }
}
