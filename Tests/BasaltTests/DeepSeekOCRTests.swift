import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

struct CharacterTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.unicodeScalars.map { Int($0.value % 200) + 3 } }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.map(String.init).joined(separator: ",") }
    func convertTokenToId(_ token: String) -> Int? { ["<s>": 0, DeepSeekOCRProcessor.imageToken: 250][token] }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { "<s>" }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
}

@Suite("DeepSeek-OCR: decoder and DeepEncoder parity with mlx-vlm goldens, image layout, processor and prefill against decode (#400, #228)", .serialized)
struct DeepSeekOCRTests {
    static let tolerance: Float = 1e-4
    static let featureTolerance: Float = 2e-4

    static func goldenModel() throws -> DeepSeekOCR {
        let model = try #require(try TinyModel.make(.deepSeekOCR) as? DeepSeekOCR)
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        let weights = parameters.enumerated().map { t, entry in
            let shift: Float = entry.0.contains("norm") && entry.0.hasSuffix(".weight") ? 1 : 0
            let values = (0 ..< entry.1.size).map { Float(0.1 * Foundation.sin(0.7 * Double($0) + 0.3 * Double(t))) + shift }
            return (entry.0, MLXArray(values).reshaped(entry.1.shape))
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    static func pixels(count: Int, side: Int, phase: Double) -> [Float] {
        (0 ..< count * side * side * 3).map { Float(Foundation.sin(0.05 * Double($0) + phase)) }
    }

    static func goldenInput(_ golden: DeepSeekOCRGolden.Case) -> (layout: DeepSeekOCRImageLayout, pixels: MLXArray, ids: [Int]) {
        let layout = DeepSeekOCRImageLayout(side: golden.side, columns: golden.tiles[0], rows: golden.tiles[1], tile: golden.tiles[2])
        let tiles = layout.tiled ? pixels(count: layout.columns * layout.rows, side: layout.tile, phase: 1) : []
        return (layout, MLXArray(pixels(count: 1, side: layout.side, phase: 0) + tiles), [1, 2] + Array(repeating: 250, count: layout.tokens) + [3, 4])
    }

    static func prefillLogits(_ model: DeepSeekOCR, _ ids: [Int], pixels: MLXArray, layout: DeepSeekOCRImageLayout) throws -> MLXArray {
        let input = LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init))[.newAxis]), image: .init(pixels: pixels, frames: layout.frames))
        guard case .logits(let output) = try model.prepare(input, cache: try model.newCache(parameters: nil), state: nil, prefill: .init()) else {
            Issue.record("an image prompt must prefill to logits")
            return MLXArray(0)
        }
        return output.logits[0, -1].asType(.float32)
    }

    @Test("The decoder (Llama attention, one dense layer, softmax top-2 of 4 routed experts plus a shared expert) matches mlx-vlm deepseekocr LanguageModel logits within 1e-4 (golden from scripts/deepseekocr_reference.swift)")
    func decoderGolden() throws {
        let model = try Self.goldenModel()
        let logits = model(MLXArray([Int32(5), 9, 13, 17, 21, 25, 29, 33]).reshaped(1, -1), cache: nil)[0]
        #expect(MLX.abs(logits.reshaped(-1) - MLXArray(DeepSeekOCRGolden.decoder)).max().item(Float.self) < Self.tolerance)
    }

    @Test("Image features (SAM with windowed and global attention, CLIP on SAM patches, projector, newline columns, local tiles before the global view, view separator) and the last logits match mlx-vlm get_input_embeddings (patched to the official DeepEncoder) within 2e-4 and 1e-4 on a 128 view, a 192 view with interpolated positions and a 2x1 grid of 64 tiles",
          arguments: DeepSeekOCRGolden.cases.indices)
    func imageGolden(_ index: Int) throws {
        let golden = DeepSeekOCRGolden.cases[index]
        let model = try Self.goldenModel()
        let (layout, pixels, ids) = Self.goldenInput(golden)
        let features = model.features(pixels, layout)
        #expect(features.shape == [layout.tokens, 64])
        #expect(MLX.abs(features.reshaped(-1) - MLXArray(golden.features)).max().item(Float.self) < Self.featureTolerance)
        let logits = try Self.prefillLogits(model, ids, pixels: pixels, layout: layout)
        #expect(MLX.abs(logits - MLXArray(golden.logits)).max().item(Float.self) < Self.tolerance)
    }

    @Test("SAM global attention split into query-row blocks gives the same output as one block")
    func blockedSAMAttention() throws {
        let model = try Self.goldenModel()
        let attention = model.sam.blocks[1].attn
        let x = MLXRandom.normal([2, 8, 8, 32], key: MLXRandom.key(3))
        #expect(MLX.abs(attention(x, queryRows: 1024) - attention(x, queryRows: 20)).max().item(Float.self) < 1e-5)
    }

    @Test("A config this port cannot route is refused at load: sigmoid scoring, normalized top-k, sparse MoE layers, grouped routing, multi-head latent attention",
          arguments: [#""scoring_func": "sigmoid","#, #""norm_topk_prob": true,"#, #""moe_layer_freq": 2,"#, #""n_group": 8, "topk_group": 4,"#, #""qk_nope_head_dim": 128,"#])
    func refusesUnsupportedRouting(_ key: String) throws {
        let decoder = JSONDecoder.json5()
        let config = TinyModel.Family.deepSeekOCR.config
        #expect(try decoder.decode(DeepSeekOCRConfiguration.self, from: Data(config.utf8)).validated().text.layers == 3)
        let edited = config.replacingOccurrences(of: #""qk_nope_head_dim": 0,"#, with: key)
        #expect(edited != config)
        #expect(throws: BasaltError.self) { try decoder.decode(DeepSeekOCRConfiguration.self, from: Data(edited.utf8)).validated() }
    }

    @Test("Frames parse into one layout per image, and views that are not squares of a multiple of 64 are refused")
    func layouts() throws {
        let gundam = DeepSeekOCRImageLayout(side: 1024, columns: 2, rows: 3, tile: 640)
        #expect(try DeepSeekOCRImageLayout.parse(gundam.frames + DeepSeekOCRImageLayout(side: 512, columns: 1, rows: 1, tile: 0).frames).count == 2)
        #expect(gundam.tokens == 273 + 21 * 30)
        #expect(throws: BasaltError.self) { try DeepSeekOCRImageLayout.parse([THW(1, 1024, 1024)]) }
        #expect(throws: BasaltError.self) { try DeepSeekOCRImageLayout.parse([THW(1, 1000, 1000), THW(1, 1, 0)]) }
        #expect(throws: BasaltError.self) { try DeepSeekOCRImageLayout.parse([THW(1, 1024, 512), THW(1, 1, 0)]) }
        #expect(throws: BasaltError.self) { try DeepSeekOCRImageLayout.parse([THW(1, 1024, 1024), THW(2, 1, 600)]) }
        #expect(throws: BasaltError.self) { try DeepSeekOCRImageLayout.parse([THW(1, 1024, 1024), THW(0, 1, 640)]) }
    }

    @Test("An image prompt whose placeholder count or pixel count does not match its layout is refused, as is an image after a cached prefix")
    func mismatchRefused() throws {
        let model = try #require(try TinyModel.make(.deepSeekOCR) as? DeepSeekOCR)
        let (layout, pixels, ids) = Self.goldenInput(DeepSeekOCRGolden.cases[0])
        #expect(throws: BasaltError.self) { _ = try Self.prefillLogits(model, Array(ids.dropFirst(3)), pixels: pixels, layout: layout) }
        #expect(throws: BasaltError.self) { _ = try Self.prefillLogits(model, ids + [250], pixels: pixels, layout: layout) }
        #expect(throws: BasaltError.self) { _ = try Self.prefillLogits(model, ids, pixels: pixels[..<100], layout: layout) }
        let cache = try model.newCache(parameters: nil)
        _ = model(MLXArray([Int32(1), 2]).reshaped(1, -1), cache: cache)
        let input = LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init))[.newAxis]), image: .init(pixels: pixels, frames: layout.frames))
        #expect(throws: BasaltError.self) { _ = try model.prepare(input, cache: cache, state: nil, prefill: .init()) }
    }

    @Test("Greedy decoding after an image prefill picks the argmax of a fresh image prefill over the prompt plus every token so far, and the image changes the answer")
    func imageDecodeMatchesPrefill() throws {
        let model = try #require(try TinyModel.make(.deepSeekOCR) as? DeepSeekOCR)
        let (layout, pixels, prompt) = Self.goldenInput(DeepSeekOCRGolden.cases[2])
        let input = LMInput(text: .init(tokens: MLXArray(prompt.map(Int32.init))[.newAxis]), image: .init(pixels: pixels, frames: layout.frames))
        let tokens = Array(try TokenIterator(input: input, model: model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 8))
        #expect(tokens.count == 8)
        let fresh = try (0 ..< tokens.count).map {
            try argMax(Self.prefillLogits(model, prompt + tokens[..<$0], pixels: pixels, layout: layout), axis: -1).item(Int.self)
        }
        expectTokenSequence(fresh, tokens)
        let textOnly = model(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), cache: nil)[0, -1].asType(.float32)
        #expect(MLX.abs(try Self.prefillLogits(model, prompt, pixels: pixels, layout: layout) - textOnly).max().item(Float.self) > Self.tolerance)
    }

    struct ModeCheckingProcessor: UserInputProcessor {
        let expected: OCRMode

        func prepare(input: UserInput) throws -> LMInput {
            guard !input.images.isEmpty else { return try ByteInputProcessor().prepare(input: input) }
            guard DeepSeekOCRProcessor.mode(input.additionalContext) == expected else {
                throw BasaltError.invalidRequestStructure(details: "the OCR mode did not reach the processor")
            }
            let (layout, pixels, ids) = DeepSeekOCRTests.goldenInput(DeepSeekOCRGolden.cases[0])
            return LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init))[.newAxis]), image: .init(pixels: pixels, frames: layout.frames))
        }
    }

    @Test("An OCR request through the engine hands its resolution and grounding to the processor, reuses no prompt cache across two image requests, and answers like TokenIterator")
    func engineImageRequest() async throws {
        let model = try #require(try TinyModel.make(.deepSeekOCR) as? DeepSeekOCR)
        let (layout, pixels, ids) = Self.goldenInput(DeepSeekOCRGolden.cases[0])
        let input = LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init))[.newAxis]), image: .init(pixels: pixels, frames: layout.frames))
        let expected = ByteTokenizer().decode(tokenIds: Array(try TokenIterator(input: input, model: model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 8)), skipSpecialTokens: false)
        let recipe = try OCRRecipe(family: .deepSeekOCR).with(resolution: .small)
        let engine = try await TinyModel.engine(
            .deepSeekOCR, promptCacheBytes: 1 << 30, processor: ModeCheckingProcessor(expected: OCRMode(resolution: .small, grounding: true)))
        let image = try ImageInput(data: try OCRFixture.png(width: 4, height: 4), mime: "image/png")
        let ocr = recipe.request(prompt: "<|grounding|>Convert the document to markdown.", image: image, stage: .recognition)
        let request = GenerationRequest(prompt: ocr.prompt, maxTokens: 8, temperature: 0, messages: ocr.messages).with {
            $0.cacheKey = "ns"
            $0.ocrMode = ocr.ocrMode
        }
        for _ in 0 ..< 2 {
            let reply = try await engine.generate(request: request)
            #expect(reply.cachedPromptTokens == 0)
            #expect(reply.generatedText == expected)
        }
    }
}

@Suite("DeepSeek-OCR processor: Gundam grid, padding box, prompt tokens, mode from the template context and pixel-exact views (#400)")
struct DeepSeekOCRProcessorTests {
    @Test("The Gundam tile grid equals mlx-vlm dynamic_preprocess (ties go to the larger grid only for large pages), and pages up to 640 on both sides take no tiles")
    func gundamGrid() {
        for (size, grid) in zip(DeepSeekOCRGolden.sizes, DeepSeekOCRGolden.grids) {
            let found = OCRResolution.gundam.grid(width: size[0], height: size[1])
            #expect([found.columns, found.rows] == grid, "\(size)")
            #expect(OCRResolution.base.grid(width: size[0], height: size[1]) == (1, 1))
        }
        #expect(OCRResolution.allCases.map { $0.visionTokens(width: 600, height: 600) } == [73, 111, 273, 421, 273])
        #expect(OCRResolution.gundam.visionTokens(width: 1700, height: 2200) == 273 + 21 * 30)
    }

    @Test("A page pads into a square view at the place PIL ImageOps.pad pastes it (banker's rounding, centred)")
    func padBox() {
        for (size, box) in zip(DeepSeekOCRGolden.sizes, DeepSeekOCRGolden.pads) {
            let rect = DeepSeekOCRProcessor.contained(width: size[0], height: size[1], side: 1024)
            #expect([Int(rect.minX), 1024 - Int(rect.maxY), Int(rect.maxX), 1024 - Int(rect.minY)] == box, "\(size)")
        }
        #expect(DeepSeekOCRProcessor.contained(width: 50_000, height: 1, side: 1024).height == 1)
    }

    @Test("The prompt follows mlx-vlm: images then the text of each turn plus a space, grounding on the last turn, BOS first, image runs sized by the layout and the trailing token dropped")
    func promptTokens() throws {
        let image = UserInput.Image.ciImage(CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2)))
        let input = UserInput(chat: [.user("Convert.", images: [image])])
        let prompt = DeepSeekOCRProcessor.prompt(input, grounding: true)
        #expect(prompt == "<image>\n<|grounding|>Convert. ")
        #expect(DeepSeekOCRProcessor.prompt(input, grounding: false) == "<image>\nConvert. ")
        let processor = DeepSeekOCRProcessor(.init(imageMean: [0.5, 0.5, 0.5], imageStd: [0.5, 0.5, 0.5]), tokenizer: CharacterTokenizer())
        let layout = DeepSeekOCRImageLayout(side: 512, columns: 1, rows: 1, tile: 0)
        let tokens = try processor.tokens("<image>\nAB ", layouts: [layout])
        let tokenizer = CharacterTokenizer()
        #expect(tokens == [0] + Array(repeating: 250, count: 73) + tokenizer.encode(text: "\nAB", addSpecialTokens: false))
        #expect(throws: BasaltError.self) { try processor.tokens("<image>\nAB ", layouts: []) }
    }

    @Test("The resolution and grounding reach the processor through the template context; without them it reads Gundam without grounding")
    func modeFromContext() {
        let mode = OCRMode(resolution: .tiny, grounding: true)
        #expect(DeepSeekOCRProcessor.mode(mode.templateContext) == mode)
        #expect(DeepSeekOCRProcessor.mode(nil) == OCRMode(resolution: .gundam, grounding: false))
        #expect(DeepSeekOCRProcessor.mode([OCRMode.resolutionKey: "huge"]).resolution == .gundam)
    }

    static func page(width: Int, height: Int) throws -> (image: CIImage, rgb: [UInt8]) {
        let rgb: [UInt8] = (0 ..< width * height).flatMap { i -> [UInt8] in
            let (x, y) = (i % width, i / width)
            return [UInt8(x % 251), UInt8(y % 241), UInt8((x / 7 + y / 5) % 256)]
        }
        let bytes: [UInt8] = stride(from: 0, to: rgb.count, by: 3).flatMap { Array(rgb[$0 ..< $0 + 3]) + [255] }
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        let image = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        return (try #require(CIImage(data: OCRImage.png(image))), rgb)
    }

    @Test("Gundam on a 1280x640 page: two 640 tiles cut in row-major order byte-exact from the page, and a 1024 global view with 256 gray rows above and below the page")
    func gundamViews() throws {
        let (image, rgb) = try Self.page(width: 1280, height: 640)
        let processor = DeepSeekOCRProcessor(.init(imageMean: [0.5, 0.5, 0.5], imageStd: [0.5, 0.5, 0.5]), tokenizer: CharacterTokenizer())
        let (pixels, layout) = try processor.view(image, .gundam)
        #expect(layout == DeepSeekOCRImageLayout(side: 1024, columns: 2, rows: 1, tile: 640))
        #expect(pixels.size == layout.values)
        let page = MLXArray(rgb.map(Float.init)).reshaped(640, 1280, 3) / 255 * 2 - 1
        let tiles = pixels[layout.globalValues...].reshaped(2, 640, 640, 3)
        #expect(MLX.abs(tiles[0] - page[0..., ..<640]).max().item(Float.self) < 1e-6)
        #expect(MLX.abs(tiles[1] - page[0..., 640...]).max().item(Float.self) < 1e-6)
        let global = pixels[..<layout.globalValues].reshaped(1024, 1024, 3)
        let gray = Float(127) / 255 * 2 - 1
        #expect(MLX.abs(global[..<256] - gray).max().item(Float.self) < 1e-6)
        #expect(MLX.abs(global[768...] - gray).max().item(Float.self) < 1e-6)
        #expect(MLX.abs(global[256 ..< 768] - gray).max().item(Float.self) > 0.1)
    }

    @Test("Non-tiled modes: a page already at the mode's side reaches the encoder byte-exact; a smaller page is stretched for tiny and small", arguments: [OCRResolution.tiny, .base])
    func singleViews(_ resolution: OCRResolution) throws {
        let side = resolution.side
        let (image, rgb) = try Self.page(width: side, height: side)
        let processor = DeepSeekOCRProcessor(.init(imageMean: [0.5, 0.5, 0.5], imageStd: [0.5, 0.5, 0.5]), tokenizer: CharacterTokenizer())
        let (pixels, layout) = try processor.view(image, resolution)
        #expect(layout == DeepSeekOCRImageLayout(side: side, columns: 1, rows: 1, tile: 0))
        #expect(MLX.abs(pixels - (MLXArray(rgb.map(Float.init)) / 255 * 2 - 1)).max().item(Float.self) < 1e-6)
        let (wide, _) = try Self.page(width: 128, height: 64)
        let view = try processor.view(wide, resolution).pixels.reshaped(side, side, 3)
        let gray = Float(127) / 255 * 2 - 1
        #expect((MLX.abs(view[0] - gray).max().item(Float.self) < 1e-6) == (resolution == .base))
        #expect(throws: BasaltError.self) { try processor.view(CIImage(color: .white), resolution) }
    }

}

@Suite("DeepSeek-OCR and DeepSeek-OCR-2 real checkpoints: greedy tokens match Python mlx-vlm on rendered pages per resolution mode (#400, #413; OCR-2 references from deepseekocr_reference.swift pages2)", .serialized)
struct DeepSeekOCRRealModelTests {
    static var paths: (model: String, reference: String)? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1", let model = env["BASALT_E2E_DEEPSEEK_MODEL_PATH"], let reference = env["BASALT_E2E_DEEPSEEK_REFERENCE"] else { return nil }
        return (model, reference)
    }

    static let gundamEditBudget = 0.05

    static func textBudget(_ resolution: OCRResolution) -> Double {
        switch resolution {
        case .tiny, .small: 0.25
        case .base, .large, .gundam: 0.05
        }
    }

    static func withoutBoxes(_ text: String) -> String {
        text.replacing(#/<\|det\|>.*?<\|/det\|>/#, with: "")
    }

    @Test("Prompt ids equal mlx-vlm's on every page and mode. Pages pre-sized to tiny, small, base and large: the 64-token text without its grounding boxes, over the length both have, stays within 5% character edit distance at base and large and 25% at tiny and small (box digits on the 0-999 grid are bf16 near-ties, and the lower resolutions amplify them). Gundam on the unsized page (CoreGraphics resize against PIL bicubic): the 256-token text stays within 5%. Every run decodes as many tokens as mlx-vlm",
          .enabled(if: paths != nil, "set BASALT_E2E=1, BASALT_E2E_DEEPSEEK_MODEL_PATH and BASALT_E2E_DEEPSEEK_REFERENCE"))
    func matchesMLXVLM() async throws {
        let (model, referencePath) = try #require(Self.paths)
        let reference = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: referencePath))) as? [String: Any])
        let log = try await ModelRunLog("deepseek-ocr-parity", modelPath: model)
        defer { log.releaseGPU() }
        await MLXInferenceEngine.registerAliases()
        let container = try await log.step("load") {
            try await loadModelContainer(from: URL(fileURLWithPath: model), using: DefaultTokenizerLoader())
        }
        let directory = URL(fileURLWithPath: referencePath).deletingLastPathComponent()
        for name in reference.keys.sorted() {
            let page = try #require(reference[name] as? [String: Any])
            let (url, count) = (directory.appending(path: try #require(page["image"] as? String)), try #require(page["max_tokens"] as? Int))
            let resolution = try #require((page["resolution"] as? String).flatMap(OCRResolution.init(rawValue:)))
            let context = OCRMode(resolution: resolution, grounding: true).templateContext
            let (prompt, tokens, text) = try await container.perform { context2 in
                let input = try await context2.processor.prepare(
                    input: UserInput(chat: [.user("Convert the document to markdown.", images: [.url(url)])], additionalContext: context))
                let tokens = Array(try TokenIterator(input: input, model: context2.model, processor: nil, sampler: ArgMaxSampler(), maxTokens: count))
                return (input.text.tokens.asArray(Int.self), tokens, context2.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false))
            }
            let expected = Array(try #require(page["tokens"] as? [Int]).prefix(count))
            let expectedText = try #require(page["text"] as? String)
            let distance = MLXInferenceEngine.editDistance(text, expectedText)
            let (plain, expectedPlain) = (Self.withoutBoxes(text), Self.withoutBoxes(expectedText))
            let common = min(plain.count, expectedPlain.count)
            let plainDistance = MLXInferenceEngine.editDistance(String(plain.prefix(common)), String(expectedPlain.prefix(common)))
            print("DEEPSEEK-PARITY \(name) prompt=\(prompt.count)/\((page["prompt_ids"] as? [Int])?.count ?? 0) tokens=\(tokens.count) exact=\(tokens == expected) first=\(zip(tokens, expected).enumerated().first { $1.0 != $1.1 }?.offset ?? -1) edit=\(distance)/\(expectedText.utf8.count) text-edit=\(plainDistance)/\(common)")
            log.record(name, "tokens \(tokens.count)/\(expected.count) exact \(tokens == expected) text-edit \(plainDistance)/\(common)")
            #expect(prompt == page["prompt_ids"] as? [Int], "\(name): prompt ids")
            #expect(tokens.count == expected.count, "\(name): \(tokens.count) tokens where mlx-vlm decoded \(expected.count)")
            if resolution == .gundam {
                #expect(Double(distance) <= Self.gundamEditBudget * Double(expectedText.utf8.count), "\(name): edit distance \(distance)")
            } else {
                #expect(Double(plainDistance) <= Self.textBudget(resolution) * Double(max(common, 1)), "\(name): text edit distance \(plainDistance)")
            }
        }
    }
}
