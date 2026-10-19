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
        #expect(temperature(edt, h: 0) == EntropyDynamicTemperature.range.lowerBound)
        #expect(temperature(edt, h: 0.05) == EntropyDynamicTemperature.range.lowerBound)
        #expect(temperature(edt, h: 50, r: 1) == EntropyDynamicTemperature.range.upperBound)
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
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", dynamicTemperature: false)).1.logitProcessorFactory == nil)
        #expect(await engine.sampling(for: GenerationRequest(prompt: "p", temperature: 0, dynamicTemperature: true)).1.logitProcessorFactory == nil)

        let factory = try #require(await engine.sampling(
            for: GenerationRequest(prompt: "p", temperature: 0.9, dynamicTemperature: true)).1.logitProcessorFactory)
        let edt = try #require(factory() as? EntropyDynamicTemperature)
        #expect(edt.baseTemperature == 0.9)

        let (params, edtComponents) = await MLXInferenceEngine(minP: 0.1).sampling(
            for: GenerationRequest(prompt: "p", temperature: 0.9, topP: 0.8, dynamicTemperature: true))
        #expect(params.topP == 1 && params.minP == 0)
        let moved = try #require(edtComponents.logitProcessorFactory?() as? EntropyDynamicTemperature)
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
