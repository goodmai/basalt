import CoreAI
import Foundation
import MLX
import Synchronization
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

struct GPUFeedForwardShare: FeedForwardShare {
    let columns: Int
    let lengths: Set<Int>
    let hidden: Int
    let activation: FeedForwardActivation
    let weights: [Int: [[Float]]]
    var fails = false
    let calls = Calls()

    final class Calls: Sendable {
        let count = Mutex(0)
    }

    init(_ model: any SplitFeedForwardPrefill & Module, columns: Int, lengths: Set<Int>, fails: Bool = false) throws {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        var weights: [Int: [[Float]]] = [:]
        var hidden = 0
        for layer in model.denseFeedForwardLayers {
            let path = model.feedForwardPath(layer: layer)
            let dense = try ["gate_proj", "up_proj", "down_proj"].map { name in
                let linear = try #require(modules[path + "." + name] as? QuantizedLinear)
                return dequantized(linear.weight, scales: linear.scales, biases: linear.biases, groupSize: linear.groupSize, bits: linear.bits).asType(.float32)
            }
            hidden = dense[0].dim(1)
            weights[layer] = [dense[0][..<columns], dense[1][..<columns], dense[2][0..., ..<columns]].map { $0.asArray(Float.self) }
        }
        (self.columns, self.lengths, self.hidden, self.activation, self.weights, self.fails) = (columns, lengths, hidden, .geluTanh, weights, fails)
    }

    func supports(layer: Int, tokens: Int) -> Bool { lengths.contains(tokens) && weights[layer] != nil }

    func run(layer: Int, planar: NDArray, tokens: Int) async throws -> NDArray {
        calls.count.withLock { $0 += 1 }
        if fails { throw BasaltError.inferenceError("the test share fails") }
        let w = try #require(weights[layer])
        let x = SplitFeedForward.unplanar(planar, tokens: tokens, hidden: hidden).reshaped([tokens, hidden]).asType(.float32)
        let (gate, up, down) = (MLXArray(w[0], [columns, hidden]), MLXArray(w[1], [columns, hidden]), MLXArray(w[2], [hidden, columns]))
        let y = matmul(activation(matmul(x, gate.T)) * matmul(x, up.T), down.T)
        return SplitFeedForward.planar(y.reshaped([1, tokens, hidden]))
    }
}

@Suite("Prefill FFN split (#274 step 3b): a dense layer's MLP as the GPU's columns plus a share computed elsewhere, through the model's own async prefill")
struct SplitFeedForwardTests {
    static func model() throws -> any SplitFeedForwardPrefill & Module {
        let model = try TinyModel.make(.gemma4Unified)
        quantize(model: model, groupSize: 64, bits: 4) { path, module in path.contains(".mlp.") && module is Linear && !(module is QuantizedLinear) }
        return try #require(model as? any SplitFeedForwardPrefill & Module)
    }

    static var tokens: MLXArray { MLXArray((0 ..< 12).map { Int32(($0 * 37) % 250 + 1) }).expandedDimensions(axis: 0) }

    static func stock(_ model: any SplitFeedForwardPrefill & Module) throws -> MLXArray {
        let language = try #require(model as? any LanguageModel)
        return language(tokens, cache: try language.newCache(parameters: nil)).asType(.float32)
    }

    static func split(_ model: any SplitFeedForwardPrefill & Module, share: any FeedForwardShare) async throws -> MLXArray {
        let language = try #require(model as? any LanguageModel)
        let split = try SplitFeedForward(model: model, share: share, activation: .geluTanh)
        return try await model.prefill(text: tokens, cache: try language.newCache(parameters: nil)) { layer, x in try await split(layer, x) }.asType(.float32)
    }

    @Test("Half the columns on the share: logits within 2% of the stock forward at the largest, the same next token at every position")
    func matchesStock() async throws {
        let model = try Self.model()
        let stock = try Self.stock(model)
        let split = try await Self.split(model, share: try GPUFeedForwardShare(model, columns: 64, lengths: [12]))
        let error = (abs(split - stock).max() / abs(stock).max()).item(Float.self)
        #expect(error < 0.02, "relative max error \(error)")
        #expect(argMax(split, axis: -1).asArray(Int32.self) == argMax(stock, axis: -1).asArray(Int32.self))
    }

    @Test("A length the share was not built for runs the layer's own MLP: exactly the stock forward")
    func unsupportedLength() async throws {
        let model = try Self.model()
        let split = try await Self.split(model, share: try GPUFeedForwardShare(model, columns: 64, lengths: [2048]))
        #expect(arrayEqual(split, try Self.stock(model)).item(Bool.self))
    }

    @Test("A failing share fails the prefill instead of returning a partial sum")
    func failingShare() async throws {
        let model = try Self.model()
        let share = try GPUFeedForwardShare(model, columns: 64, lengths: [12], fails: true)
        await #expect(throws: BasaltError.self) { _ = try await Self.split(model, share: share) }
    }

    @Test("Columns off the quantization group are refused when the split is built")
    func offGroupColumns() throws {
        let model = try Self.model()
        #expect(throws: BasaltError.self) { _ = try SplitFeedForward(model: model, share: try GPUFeedForwardShare(model, columns: 40, lengths: [12]), activation: .geluTanh) }
    }

    @Test("The exporter's manifest is read for the columns and lengths, a bundle without one is refused, and a missing layer model fails the load")
    func bundle() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "ffn-split-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(!CoreAIFeedForwardShare.isBundle(folder))
        await #expect(throws: BasaltError.self) { _ = try await CoreAIFeedForwardShare.load(folder) }
        let manifest = #"{"ffn_split": {"hidden": 64, "intermediate": 128, "columns": 64, "share": 0.5, "weights": "int8", "activation": "gelu_pytorch_tanh", "lengths": [512, 2048], "layers": {"0": "ffn-0-0.5-int8.aimodel"}}}"#
        try Data(manifest.utf8).write(to: folder.appending(component: "manifest.json"))
        let split = try #require(CoreAIFeedForwardShare.manifest(folder))
        #expect(split.columns == 64 && split.lengths == [512, 2048] && split.layers == ["0": "ffn-0-0.5-int8.aimodel"])
        #expect(FeedForwardActivation(split.activation) == .geluTanh && FeedForwardActivation("silu") == .silu)
        await #expect(throws: BasaltError.self) { _ = try await CoreAIFeedForwardShare.load(folder) }
    }

    static func reply(columns: Int?, fails: Bool = false) async throws -> (text: String, calls: Int, lengths: [Int]?) {
        let container = try TinyModel.container(.gemma4Unified)
        await container.update { quantize(model: $0.model, groupSize: 64, bits: 4) { path, module in path.contains(".mlp.") && module is Linear && !(module is QuantizedLinear) } }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        await engine.attach(container)
        var share: GPUFeedForwardShare?
        if let columns {
            let made = try await container.perform { context in
                try GPUFeedForwardShare(try #require(context.model as? any SplitFeedForwardPrefill & Module), columns: columns, lengths: [8], fails: fails)
            }
            await engine.attachFeedForwardSplit(made, lengths: [8], activation: .geluTanh, to: container, epoch: await engine.modelEpoch, source: "test")
            share = made
        }
        let reply = try await engine.generate(request: GenerationRequest(prompt: "Basalt splits the prefill of each dense layer.", maxTokens: 8, temperature: 0))
        return (reply.generatedText, share?.calls.count.withLock { $0 } ?? 0, await engine.modelFacts.feedForwardSplitLengths)
    }

    @Test("Through the engine: the prompt prefilled with the split decodes the plain engine's greedy text; a failing share falls back to the GPU and decodes it too")
    func engine() async throws {
        let plain = try await Self.reply(columns: nil)
        let split = try await Self.reply(columns: 64)
        #expect(split.lengths == [8] && split.calls > 0, "the split prefill did not run: \(split.calls) share calls")
        #expect(split.text == plain.text, "\(split.text) against \(plain.text)")
        let failed = try await Self.reply(columns: 64, fails: true)
        #expect(failed.calls > 0 && failed.text == plain.text)
    }
}
