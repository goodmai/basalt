import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("PaddleOCR-VL: decoder parity with the stock ERNIE-4.5 graph, image prefill against decode, processor and positions (#400, #228)", .serialized)
struct PaddleOCRVLTests {
    static let logitTolerance: Float = 1e-4
    static let frame = THW(1, 4, 6)

    static func stockErnie(matching model: PaddleOCRVL) throws -> Ernie45Model {
        let config = """
            {"hidden_size": 64, "intermediate_size": 128, "max_position_embeddings": 4096, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 16, "num_hidden_layers": 4, "rms_norm_eps": 1e-5, "vocab_size": 256,
             "rope_theta": 500000.0, "use_bias": false, "tie_word_embeddings": false}
            """
        let stock = Ernie45Model(try JSONDecoder.json5().decode(Ernie45Configuration.self, from: Data(config.utf8)))
        let half = model.config.headDim / 2
        let interleave = MLXArray((0 ..< half).flatMap { [Int32($0), Int32($0 + half)] })
        let weights = model.languageModel.parameters().flattened().map { key, value in
            guard key.hasSuffix("q_proj.weight") || key.hasSuffix("k_proj.weight") else { return (key, value) }
            let heads = value.dim(0) / model.config.headDim
            return (key, value.reshaped(heads, model.config.headDim, -1).take(interleave, axis: 1).reshaped(value.shape))
        }
        try stock.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(stock)
        return stock
    }

    static func imagePrompt(text: [Int] = [5, 9, 13]) -> [Int] {
        [1, 2, 101] + Array(repeating: 250, count: frame.product / 4) + [102] + text
    }

    static func imageInput(_ tokens: [Int]) -> LMInput {
        let pixels = withRandomState(MLXRandom.RandomState(seed: 7)) { MLXRandom.normal([frame.product, 14, 14, 3]) }
        return LMInput(text: .init(tokens: MLXArray(tokens.map(Int32.init))[.newAxis]), image: .init(pixels: pixels, frames: [frame]))
    }

    static func prefillLogits(_ model: PaddleOCRVL, _ tokens: [Int]) throws -> MLXArray {
        guard case .logits(let output) = try model.prepare(imageInput(tokens), cache: try model.newCache(parameters: nil), state: nil, prefill: .init()) else {
            Issue.record("an image prompt must prefill to logits")
            return MLXArray(0)
        }
        return output.logits[0, -1].asType(.float32)
    }

    @Test("The text decoder matches the stock MLXLLM ERNIE-4.5 graph with q/k rows interleaved for its traditional RoPE: logits within 1e-4, greedy tokens exact")
    func stockErnieParity() throws {
        let model = try #require(try TinyModel.make(.paddleOCR) as? PaddleOCRVL)
        let stock = try Self.stockErnie(matching: model)
        let tokens = MLXArray(LookupFixture.unit.map(Int32.init)).reshaped(1, -1)
        let difference = MLX.abs(model(tokens, cache: nil) - stock(tokens, cache: nil)).max().item(Float.self)
        #expect(difference < Self.logitTolerance)
        let prompt = TinyModel.tokens(Conversation.firstTurn)
        expectTokenSequence(try LookupFixture.baseline(model, prompt: prompt, maxTokens: 16), try LookupFixture.baseline(stock, prompt: prompt, maxTokens: 16))
    }

    @Test("Greedy decoding after an image prefill, with the M-RoPE delta carried in the state, picks the argmax of a fresh image prefill over the prompt plus every token so far")
    func imageDecodeMatchesPrefill() throws {
        let model = try #require(try TinyModel.make(.paddleOCR) as? PaddleOCRVL)
        let prompt = Self.imagePrompt()
        let iterator = try TokenIterator(input: Self.imageInput(prompt), model: model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 8)
        let tokens = Array(iterator)
        #expect(tokens.count == 8)
        let fresh = try (0 ..< tokens.count).map { try argMax(Self.prefillLogits(model, prompt + tokens[..<$0]), axis: -1).item(Int.self) }
        expectTokenSequence(fresh, tokens)
    }

    @Test("Image features change the answer, and an image prompt with a placeholder count that does not match the features is refused")
    func imageFeaturesMatter() throws {
        let model = try #require(try TinyModel.make(.paddleOCR) as? PaddleOCRVL)
        let prompt = Self.imagePrompt()
        let withImage = try Self.prefillLogits(model, prompt)
        let textOnly = model(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), cache: nil)[0, -1].asType(.float32)
        #expect(MLX.abs(withImage - textOnly).max().item(Float.self) > Self.logitTolerance)
        #expect(throws: BasaltError.self) { _ = try Self.prefillLogits(model, Array(prompt.dropFirst(4))) }
    }

    @Test("M-RoPE positions follow mlx-vlm get_rope_index: text runs count on, an image takes t/h/w grid indices, and the delta anchors decode")
    func positions() {
        let layout = PaddleOCRVLPositions(tokens: [7, 250, 250, 250, 250, 8], frames: [THW(1, 4, 4)], imageToken: 250, mergeSize: 2)
        #expect(layout.rows == [[0, 1, 1, 1, 1, 3], [0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 3]])
        #expect(layout.delta == -2)
        let text = PaddleOCRVLPositions(tokens: [7, 8, 9], frames: [], imageToken: 250, mergeSize: 2)
        #expect(text.rows == [[0, 1, 2], [0, 1, 2], [0, 1, 2]] && text.delta == 0)
    }

    @Test("smart_resize matches the PaddleOCR-VL image processor: multiples of 28 within the pixel budget, short sides scaled up first",
          arguments: [((1700, 2200), (868, 1120)), ((850, 1100), (840, 1092)), ((600, 20), (1848, 84)), ((200, 300), (280, 420)), ((1000, 1000), (980, 980))])
    func smartResize(_ size: (Int, Int), _ expected: (Int, Int)) throws {
        let config = PaddleOCRVLProcessorConfiguration(minPixels: 112_896, maxPixels: 1_003_520, patchSize: 14, mergeSize: 2, imageMean: [0.5, 0.5, 0.5], imageStd: [0.5, 0.5, 0.5])
        let resized = try config.resized(width: size.0, height: size.1)
        #expect(resized.width == expected.0 && resized.height == expected.1)
        #expect(throws: BasaltError.self) { try config.resized(width: 6000, height: 28) }
    }

    @Test("A page already at its target size reaches the encoder byte-exact: patches are (x/255 - 0.5)/0.5 in row-major 14x14 tiles")
    func patchesKeepPixels() throws {
        let rgb: [UInt8] = (0 ..< 28 * 56).flatMap { i -> [UInt8] in
            let (x, y) = (i % 56, i / 56)
            return [UInt8(x * 4), UInt8(y * 9), UInt8((x + y) % 256)]
        }
        let bytes: [UInt8] = stride(from: 0, to: rgb.count, by: 3).flatMap { Array(rgb[$0 ..< $0 + 3]) + [255] }
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        let image = try #require(CGImage(
            width: 56, height: 28, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 56 * 4, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let page = try #require(CIImage(data: OCRImage.png(image)))
        let config = PaddleOCRVLProcessorConfiguration(minPixels: 28 * 56, maxPixels: 28 * 56, patchSize: 14, mergeSize: 2, imageMean: [0.5, 0.5, 0.5], imageStd: [0.5, 0.5, 0.5])
        let processor = PaddleOCRVLProcessor(config, tokenizer: ByteTokenizer())
        let (patches, frame) = try processor.patches(page)
        #expect(frame.t == 1 && frame.h == 2 && frame.w == 4)
        let pixels = MLXArray(rgb.map(Float.init)).reshaped(2, 14, 4, 14, 3).transposed(0, 2, 1, 3, 4).reshaped(8, 14, 14, 3)
        let expected = pixels / 255 * 2 - 1
        #expect(MLX.abs(patches - expected).max().item(Float.self) < 1e-6)
    }

    @Test("Each image placeholder expands to t*h*w/4 tokens in order, and a count mismatch is refused")
    func placeholders() throws {
        let frames = [THW(1, 2, 4), THW(1, 4, 4)]
        #expect(try PaddleOCRVLProcessor.expanded([1, 9, 3, 9], token: 9, frames: frames, mergeSize: 2) == [1, 9, 9, 3, 9, 9, 9, 9])
        #expect(throws: BasaltError.self) { try PaddleOCRVLProcessor.expanded([1, 9], token: 9, frames: frames, mergeSize: 2) }
    }

    @Test("Position embeddings interpolate bilinearly with half-pixel centres and clamped edges, as mlx-vlm bilinear_interpolate")
    func bilinear() {
        let grid = MLXArray([Float(0), 1, 2, 3]).reshaped(2, 2, 1)
        let out = PaddleOCRVLVision.bilinear(grid, height: 3, width: 3).reshaped(-1)
        #expect(MLX.abs(out - MLXArray([Float(0), 0.5, 1, 1, 1.5, 2, 2, 2.5, 3])).max().item(Float.self) < 1e-6)
    }

    @Test("A user turn renders image parts before its text, as the PaddleOCR-VL chat template expects")
    func messages() throws {
        let input = UserInput(chat: [.user("OCR:", images: [.ciImage(CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2)))])])
        let content = try #require(PaddleOCRVLProcessor.messages(input).first?["content"] as? [[String: String]])
        #expect(content == [["type": "image"], ["type": "text", "text": "OCR:"]])
    }
}

@Suite("PaddleOCR-VL real checkpoint: greedy tokens match Python mlx-vlm on rendered pages (#400)", .serialized)
struct PaddleOCRVLRealModelTests {
    static var paths: (model: String, reference: String)? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1", let model = env["BASALT_E2E_PADDLE_MODEL_PATH"], let reference = env["BASALT_E2E_PADDLE_REFERENCE"] else { return nil }
        return (model, reference)
    }

    @Test("The prompt ids and the first 64 greedy tokens equal mlx-vlm's on every page of the reference file",
          .enabled(if: paths != nil, "set BASALT_E2E=1, BASALT_E2E_PADDLE_MODEL_PATH and BASALT_E2E_PADDLE_REFERENCE"))
    func matchesMLXVLM() async throws {
        let (model, referencePath) = try #require(Self.paths)
        let reference = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: referencePath))) as? [String: Any])
        await MLXInferenceEngine.registerAliases()
        let container = try await loadModelContainer(from: URL(fileURLWithPath: model), using: DefaultTokenizerLoader())
        let directory = URL(fileURLWithPath: referencePath).deletingLastPathComponent()
        for name in reference.keys.sorted() where name != "prompt" {
            let page = try #require(reference[name] as? [String: Any])
            let url = directory.appending(path: "\(name)-sized.png")
            let (prompt, tokens) = try await container.perform { context in
                let input = try await context.processor.prepare(input: UserInput(chat: [.user("OCR:", images: [.url(url)])]))
                let tokens = Array(try TokenIterator(input: input, model: context.model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 64))
                return (input.text.tokens.asArray(Int.self), tokens)
            }
            #expect(prompt == page["prompt_ids"] as? [Int], "\(name): prompt ids")
            expectTokenSequence(tokens, Array(try #require(page["tokens"] as? [Int]).prefix(64)))
            print("PADDLE-PARITY \(name) prompt=\(prompt.count) tokens=\(tokens.count) exact=\(Array(tokens) == Array((page["tokens"] as? [Int] ?? []).prefix(64)))")
        }
    }
}
