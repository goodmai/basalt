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
        #expect(throws: BasaltError.self) { _ = try Self.prefillLogits(model, prompt + [250]) }
    }

    @Test("Encoded features handed in place of pixels, as the Neural Engine tower does, prefill to exactly the logits of the pixels they came from (#224)")
    func encodedFeatures() throws {
        let model = try #require(try TinyModel.make(.paddleOCR) as? PaddleOCRVL)
        let prompt = Self.imagePrompt()
        let input = Self.imageInput(prompt)
        let features = model.visual(try #require(input.image).pixels, frames: [Self.frame])
        let encoded = LMInput(text: input.text, image: .init(pixels: features, frames: [Self.frame]))
        guard case .logits(let output) = try model.prepare(encoded, cache: try model.newCache(parameters: nil), state: nil, prefill: .init()) else {
            Issue.record("encoded features must prefill to logits")
            return
        }
        #expect(MLX.abs(output.logits[0, -1].asType(.float32) - (try Self.prefillLogits(model, prompt))).max().item(Float.self) == 0)
    }

    @Test("M-RoPE positions equal mlx-vlm get_rope_index on two non-square images (4x6 and 2x4 grids), and a short placeholder run is refused")
    func positions() throws {
        let frames = [THW(1, 4, 6), THW(1, 2, 4)]
        let layout = try PaddleOCRVLPositions(tokens: PaddleOCRVLGolden.tokens, frames: frames, imageToken: 250, mergeSize: 2)
        #expect(layout.rows == PaddleOCRVLGolden.positions)
        #expect(layout.delta == PaddleOCRVLGolden.delta)
        let text = try PaddleOCRVLPositions(tokens: [7, 8, 9], frames: [], imageToken: 250, mergeSize: 2)
        #expect(text.rows == [[0, 1, 2], [0, 1, 2], [0, 1, 2]] && text.delta == 0)
        #expect(throws: BasaltError.self) { try PaddleOCRVLPositions(tokens: [7, 250, 250], frames: [THW(1, 4, 4)], imageToken: 250, mergeSize: 2) }
        #expect(throws: BasaltError.self) { try PaddleOCRVLPositions(tokens: [7, 250, 250, 8, 250, 250], frames: [THW(1, 4, 4)], imageToken: 250, mergeSize: 2) }
        #expect(throws: BasaltError.self) { try PaddleOCRVLPositions(tokens: [7, 250, 250, 250, 250, 8, 250], frames: [THW(1, 4, 4)], imageToken: 250, mergeSize: 2) }
        #expect(throws: BasaltError.self) { try PaddleOCRVLPositions(tokens: [7, 250, 250, 250, 250, 8], frames: [THW(1, 4, 4), THW(1, 2, 2)], imageToken: 250, mergeSize: 2) }
    }

    static func goldenVision() throws -> PaddleOCRVLVision.Model {
        let model = try #require(try TinyModel.make(.paddleOCR) as? PaddleOCRVL)
        let parameters = model.visual.parameters().flattened().sorted { $0.0 < $1.0 }
        let weights = parameters.enumerated().map { t, entry in
            let shift: Float = entry.0.contains("norm") && entry.0.hasSuffix(".weight") ? 1 : 0
            let values = (0 ..< entry.1.size).map { Float(0.1 * Foundation.sin(0.7 * Double($0) + 0.3 * Double(t))) + shift }
            return (entry.0, MLXArray(values).reshaped(entry.1.shape))
        }
        try model.visual.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        return model.visual
    }

    @Test("The vision tower (embeddings, 2D RoPE, encoder, projector) matches mlx-vlm VisionModel on two images (4x6 and 2x4 grids) within 1e-4 (golden from scripts/paddleocr_reference.swift)")
    func visionGolden() throws {
        let vision = try Self.goldenVision()
        let pixels = MLXArray((0 ..< 32 * 14 * 14 * 3).map { Float(Foundation.sin(0.05 * Double($0))) }).reshaped(32, 14, 14, 3)
        let features = vision(pixels, frames: [THW(1, 4, 6), THW(1, 2, 4)])
        #expect(features.shape == PaddleOCRVLGolden.featureShape)
        #expect(MLX.abs(features.reshaped(-1) - MLXArray(PaddleOCRVLGolden.features)).max().item(Float.self) < Self.logitTolerance)
    }

    @Test("Vision attention split into query blocks gives the same output as one block")
    func blockedVisionAttention() throws {
        let vision = try Self.goldenVision()
        let x = MLXRandom.normal([24, 32], key: MLXRandom.key(3))
        let (cos, sin) = vision.rotary(THW(1, 4, 6))
        let attention = vision.layers[0].attention
        let whole = attention(x, cos: cos, sin: sin, queryBlock: 1024)
        #expect(MLX.abs(whole - attention(x, cos: cos, sin: sin, queryBlock: 5)).max().item(Float.self) < 1e-6)
    }

    @Test("smart_resize matches the PaddleOCR-VL image processor: multiples of 28 within the pixel budget, short sides scaled up first",
          arguments: [((1700, 2200), (868, 1120)), ((850, 1100), (840, 1092)), ((600, 20), (1848, 84)), ((200, 300), (280, 420)), ((1000, 1000), (980, 980))])
    func smartResize(_ size: (Int, Int), _ expected: (Int, Int)) throws {
        let config = PaddleOCRVLProcessorConfiguration(minPixels: 112_896, maxPixels: 1_003_520, patchSize: 14, mergeSize: 2, imageMean: [0.5, 0.5, 0.5], imageStd: [0.5, 0.5, 0.5])
        let resized = try config.resized(width: size.0, height: size.1)
        #expect(resized.width == expected.0 && resized.height == expected.1)
        #expect(throws: BasaltError.self) { try config.resized(width: 6000, height: 28) }
        #expect(throws: BasaltError.self) { try config.resized(width: 0, height: 28) }
        #expect(throws: BasaltError.self) { try config.resized(width: 0, height: 0) }
        let processor = PaddleOCRVLProcessor(config, tokenizer: ByteTokenizer())
        #expect(throws: BasaltError.self) { try processor.patches(CIImage(color: .white)) }
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

    @Test("Position embeddings interpolate bilinearly with half-pixel centres and clamped edges onto a non-square grid, as mlx-vlm bilinear_interpolate")
    func bilinear() {
        let grid = MLXArray([Float(0), 1, 2, 3]).reshaped(2, 2, 1)
        let out = PaddleOCRVLVision.bilinear(grid, height: 2, width: 3)
        #expect(out.shape == [2, 3, 1])
        #expect(MLX.abs(out.reshaped(-1) - MLXArray([Float(0), 0.5, 1, 2, 2.5, 3])).max().item(Float.self) < 1e-6)
    }

    struct ImageProcessor: UserInputProcessor {
        func prepare(input: UserInput) throws -> LMInput {
            input.images.isEmpty ? try ByteInputProcessor().prepare(input: input) : PaddleOCRVLTests.imageInput(PaddleOCRVLTests.imagePrompt())
        }
    }

    @Test("An image request through the engine with the prompt cache on reuses nothing, twice, and answers like TokenIterator")
    func engineImageRequest() async throws {
        let model = try #require(try TinyModel.make(.paddleOCR) as? PaddleOCRVL)
        let iterator = try TokenIterator(input: Self.imageInput(Self.imagePrompt()), model: model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 8)
        let expected = ByteTokenizer().decode(tokenIds: Array(iterator), skipSpecialTokens: false)
        let engine = try await TinyModel.engine(.paddleOCR, promptCacheBytes: 1 << 30, processor: ImageProcessor())
        let image = try ImageInput(data: try OCRFixture.png(width: 4, height: 4), mime: "image/png")
        let request = GenerationRequest(
            prompt: "OCR:", maxTokens: 8, temperature: 0, messages: [ChatMessageInput(role: .user, text: "OCR:", images: [image])]
        ).with { $0.cacheKey = "ns" }
        for _ in 0 ..< 2 {
            let reply = try await engine.generate(request: request)
            #expect(reply.cachedPromptTokens == 0)
            #expect(reply.generatedText == expected)
        }
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

    static let unsizedEditBudget = 0.05

    @Test("Pre-sized pages: prompt ids and the first 64 greedy tokens equal mlx-vlm's. Unsized pages (CoreGraphics resize against PIL bicubic): the 256-token text stays within 5% character edit distance",
          .enabled(if: paths != nil, "set BASALT_E2E=1, BASALT_E2E_PADDLE_MODEL_PATH and BASALT_E2E_PADDLE_REFERENCE"))
    func matchesMLXVLM() async throws {
        let (model, referencePath) = try #require(Self.paths)
        let reference = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: referencePath))) as? [String: Any])
        await MLXInferenceEngine.registerAliases()
        let container = try await loadModelContainer(from: URL(fileURLWithPath: model), using: DefaultTokenizerLoader())
        let directory = URL(fileURLWithPath: referencePath).deletingLastPathComponent()
        for name in reference.keys.sorted() {
            let page = try #require(reference[name] as? [String: Any])
            let (url, count) = (directory.appending(path: try #require(page["image"] as? String)), try #require(page["max_tokens"] as? Int))
            let (prompt, tokens, text) = try await container.perform { context in
                let input = try await context.processor.prepare(input: UserInput(chat: [.user("OCR:", images: [.url(url)])]))
                let tokens = Array(try TokenIterator(input: input, model: context.model, processor: nil, sampler: ArgMaxSampler(), maxTokens: count))
                return (input.text.tokens.asArray(Int.self), tokens, context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false))
            }
            let expected = Array(try #require(page["tokens"] as? [Int]).prefix(count))
            let expectedText = try #require(page["text"] as? String)
            let distance = MLXInferenceEngine.editDistance(text, expectedText)
            print("PADDLE-PARITY \(name) prompt=\(prompt.count)/\((page["prompt_ids"] as? [Int])?.count ?? 0) tokens=\(tokens.count) exact=\(tokens == expected) edit=\(distance)/\(expectedText.utf8.count)")
            if name.hasSuffix("-unsized") {
                #expect(Double(distance) <= Self.unsizedEditBudget * Double(expectedText.utf8.count), "\(name): edit distance \(distance)")
            } else {
                #expect(prompt == page["prompt_ids"] as? [Int], "\(name): prompt ids")
                expectTokenSequence(tokens, expected)
            }
        }
    }
}
