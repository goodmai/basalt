import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("DeepSeek-OCR-2: SAM into a Qwen2 encoder with learned queries, parity with mlx-vlm goldens, the official tile grid, its own processor and engine requests (#413, #228)", .serialized)
struct DeepSeekOCR2Tests {
    static let tolerance: Float = 1e-4
    static let featureTolerance: Float = 2e-4
    static let processorConfig = DeepSeekOCRProcessorConfiguration(imageMean: [0.5, 0.5, 0.5], imageStd: [0.5, 0.5, 0.5])

    static func goldenModel() throws -> DeepSeekOCR2 {
        let model = try #require(try TinyModel.make(.deepSeekOCR2) as? DeepSeekOCR2)
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

    static func goldenInput(_ golden: DeepSeekOCR2Golden.Case) -> (layout: DeepSeekOCRImageLayout, pixels: MLXArray, ids: [Int]) {
        let (columns, rows) = (golden.tiles[0], golden.tiles[1])
        let layout = DeepSeekOCRImageLayout(side: 1024, columns: columns, rows: rows, tile: columns * rows > 1 ? 768 : 0, newlines: false)
        let tiles = layout.tiled ? DeepSeekOCRTests.pixels(count: columns * rows, side: 768, phase: 1) : []
        return (layout, MLXArray(DeepSeekOCRTests.pixels(count: 1, side: 1024, phase: 0) + tiles), [1, 2] + Array(repeating: 250, count: layout.tokens) + [3, 4])
    }

    static func prefillLogits(_ model: DeepSeekOCR2, _ ids: [Int], pixels: MLXArray, layout: DeepSeekOCRImageLayout) throws -> MLXArray {
        let input = LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init))[.newAxis]), image: .init(pixels: pixels, frames: layout.frames))
        guard case .logits(let output) = try model.prepare(input, cache: try model.newCache(parameters: nil), state: nil, prefill: .init()) else {
            Issue.record("an image prompt must prefill to logits")
            return MLXArray(0)
        }
        return output.logits[0, -1].asType(.float32)
    }

    @Test("Image features (SAM at 1024 and at 768 with resized positions, the Qwen2 encoder over SAM tokens plus learned queries, projector, tiles before the global view, view separator) and the last logits match mlx-vlm deepseekocr_2 get_input_embeddings within 2e-4 and 1e-4, for a global view alone and beside a 2x1 grid of 768 tiles",
          arguments: DeepSeekOCR2Golden.cases.indices)
    func imageGolden(_ index: Int) throws {
        let golden = DeepSeekOCR2Golden.cases[index]
        let model = try Self.goldenModel()
        let (layout, pixels, ids) = Self.goldenInput(golden)
        let features = try model.features(pixels, layout)
        #expect(features.shape == [layout.tokens, 64])
        let kept = features[MLXArray(golden.rows.map(Int32.init))].reshaped(-1)
        #expect(MLX.abs(kept - MLXArray(golden.features)).max().item(Float.self) < Self.featureTolerance)
        let logits = try Self.prefillLogits(model, ids, pixels: pixels, layout: layout)
        #expect(MLX.abs(logits - MLXArray(golden.logits)).max().item(Float.self) < Self.tolerance)
    }

    @Test("The encoder mask lets SAM tokens see each other only, and each query every SAM token and the queries before it; a view without learned queries is refused")
    func encoderMask() throws {
        let mask = DeepSeekOCR2Encoder.Encoder.mask(images: 2).asArray(Bool.self)
        #expect(mask == [true, true, false, false, true, true, false, false, true, true, true, false, true, true, true, true])
        let model = try #require(try TinyModel.make(.deepSeekOCR2) as? DeepSeekOCR2)
        let layout = DeepSeekOCRImageLayout(side: 512, columns: 1, rows: 1, tile: 0, newlines: false)
        #expect(throws: BasaltError.self) { try model.features(MLXArray.zeros([layout.values]), layout) }
    }

    @Test("Layouts without newline columns count 256 global queries, a separator and 144 per 768 tile")
    func layouts() throws {
        let gundam = DeepSeekOCRImageLayout(side: 1024, columns: 2, rows: 3, tile: 768, newlines: false)
        #expect(gundam.tokens == 257 + 6 * 144)
        #expect(try DeepSeekOCRImageLayout.parse(gundam.frames, newlines: false) == [gundam])
        #expect(DeepSeekOCRImageLayout(side: 1024, columns: 1, rows: 1, tile: 0, newlines: false).tokens == 257)
    }

    @Test("The tiny config validates; a projector that does not take the encoder width, or heads that do not divide it, are refused at load")
    func refusesUnsupportedConfig() throws {
        let decoder = JSONDecoder.json5()
        let config = TinyModel.Family.deepSeekOCR2.config
        #expect(try decoder.decode(DeepSeekOCR2Configuration.self, from: Data(config.utf8)).validated().encoder.layers == 2)
        for (from, to) in [(#""input_dim": 64"#, #""input_dim": 128"#), (#""heads": 4, "kv_heads": 2"#, #""heads": 4, "kv_heads": 3"#), (#""topk_method": "greedy""#, #""topk_method": "noaux_tc""#)] {
            let edited = config.replacingOccurrences(of: from, with: to)
            #expect(edited != config)
            #expect(throws: BasaltError.self) { try decoder.decode(DeepSeekOCR2Configuration.self, from: Data(edited.utf8)).validated() }
        }
    }

    @Test("Greedy decoding after an image prefill picks the argmax of a fresh image prefill over the prompt plus every token so far, and the image changes the answer")
    func imageDecodeMatchesPrefill() throws {
        let model = try #require(try TinyModel.make(.deepSeekOCR2) as? DeepSeekOCR2)
        let (layout, pixels, prompt) = Self.goldenInput(DeepSeekOCR2Golden.cases[1])
        let input = LMInput(text: .init(tokens: MLXArray(prompt.map(Int32.init))[.newAxis]), image: .init(pixels: pixels, frames: layout.frames))
        let tokens = Array(try TokenIterator(input: input, model: model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 6))
        #expect(tokens.count == 6)
        let fresh = try (0 ..< tokens.count).map {
            try argMax(Self.prefillLogits(model, prompt + tokens[..<$0], pixels: pixels, layout: layout), axis: -1).item(Int.self)
        }
        expectTokenSequence(fresh, tokens)
        let textOnly = model(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), cache: nil)[0, -1].asType(.float32)
        #expect(MLX.abs(try Self.prefillLogits(model, prompt, pixels: pixels, layout: layout) - textOnly).max().item(Float.self) > Self.tolerance)
    }

    @Test("The tile grid follows the official OCR-2 preprocessing: 2 to 6 tiles of 768 (mlx-vlm dynamic_preprocess with min_num 2, max_num 6), none for pages up to 768 on both sides, and none in base mode")
    func tileGrid() {
        for (size, grid) in zip(DeepSeekOCR2Golden.sizes, DeepSeekOCR2Golden.grids) {
            let found = OCRResolution.gundam.grid(width: size[0], height: size[1], tile: 768, maxTiles: 6)
            #expect([found.columns, found.rows] == grid, "\(size)")
            #expect(OCRResolution.base.grid(width: size[0], height: size[1], tile: 768, maxTiles: 6) == (1, 1))
        }
    }

    @Test("The OCR-2 processor cuts a 1536x768 page into two 768 tiles byte-exact in row-major order beside a padded 1024 view, sizes the image run without newlines, and refuses the modes it has no queries for")
    func processorViews() throws {
        let (image, rgb) = try DeepSeekOCRProcessorTests.page(width: 1536, height: 768)
        let processor = DeepSeekOCRProcessor(Self.processorConfig, tokenizer: CharacterTokenizer(), geometry: .v2)
        let (pixels, layout) = try processor.view(image, .gundam)
        #expect(layout == DeepSeekOCRImageLayout(side: 1024, columns: 2, rows: 1, tile: 768, newlines: false))
        let page = MLXArray(rgb.map(Float.init)).reshaped(768, 1536, 3) / 255 * 2 - 1
        let tiles = pixels[layout.globalValues...].reshaped(2, 768, 768, 3)
        #expect(MLX.abs(tiles[0] - page[0..., ..<768]).max().item(Float.self) < 1e-6)
        #expect(MLX.abs(tiles[1] - page[0..., 768...]).max().item(Float.self) < 1e-6)
        let global = pixels[..<layout.globalValues].reshaped(1024, 1024, 3)
        #expect(MLX.abs(global[..<256] - (Float(127) / 255 * 2 - 1)).max().item(Float.self) < 1e-6)
        let small = try processor.view(try DeepSeekOCRProcessorTests.page(width: 700, height: 500).image, .gundam).layout
        #expect(small == DeepSeekOCRImageLayout(side: 1024, columns: 1, rows: 1, tile: 768, newlines: false) && !small.tiled)
        let tokens = try processor.tokens("<image>\nAB ", layouts: [layout])
        #expect(tokens == [0] + Array(repeating: 250, count: 257 + 288) + CharacterTokenizer().encode(text: "\nAB", addSpecialTokens: false))
        for resolution in [OCRResolution.tiny, .small, .large] {
            #expect(throws: BasaltError.self) { try processor.view(image, resolution) }
        }
    }

    static func checkpoint(_ family: TinyModel.Family) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "deepseek-ocr-\(family.rawValue)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(family.config.utf8).write(to: directory.appending(path: "config.json"))
        try Data(#"{"processor_class": "DeepseekVLV2Processor", "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}"#.utf8)
            .write(to: directory.appending(path: "processor_config.json"))
        try save(arrays: Dictionary(uniqueKeysWithValues: try TinyModel.make(family).parameters().flattened()), url: directory.appending(path: "model.safetensors"))
        return directory
    }

    @Test("Both checkpoints declare DeepseekVLV2Processor; loading from disk picks the OCR-2 processor by model_type deepseekocr_2 and keeps v1's for deepseekocr",
          arguments: [(TinyModel.Family.deepSeekOCR, "DeepSeek-OCR"), (.deepSeekOCR2, "DeepSeek-OCR-2")])
    func processorByModelType(_ family: TinyModel.Family, _ name: String) async throws {
        await MLXInferenceEngine.registerAliases()
        let directory = try Self.checkpoint(family)
        defer { try? FileManager.default.removeItem(at: directory) }
        let container = try await loadModelContainer(from: directory, using: ByteTokenizerLoader())
        let geometry = await container.perform { ($0.processor as? DeepSeekOCRProcessor)?.geometry.name }
        #expect(geometry == name)
    }

    @Test("Through the engine with the OCR-2 processor and model: a gundam page answers like TokenIterator on the processor's own input and reuses no prompt cache across two image requests; a tiny mode is refused")
    func engineImageRequest() async throws {
        let processor = DeepSeekOCRProcessor(Self.processorConfig, tokenizer: CharacterTokenizer(), geometry: .v2)
        let model = try #require(try TinyModel.make(.deepSeekOCR2) as? DeepSeekOCR2)
        let png = try OCRFixture.png(width: 900, height: 600)
        let recipe = try OCRRecipe(family: .deepSeekOCR).with(resolution: .gundam)
        let ocr = recipe.request(prompt: "Free OCR.", image: try ImageInput(data: png, mime: "image/png"), stage: .recognition)
        let input = try await processor.prepare(input: UserInput(
            chat: [.user("Free OCR.", images: [.ciImage(try #require(CIImage(data: png)))])], additionalContext: try #require(ocr.ocrMode).templateContext))
        let frames = DeepSeekOCRImageLayout(side: 1024, columns: 3, rows: 2, tile: 768, newlines: false).frames
        #expect(input.image?.frames?.map { [$0.t, $0.h, $0.w] } == frames.map { [$0.t, $0.h, $0.w] })
        let expected = ByteTokenizer().decode(tokenIds: Array(try TokenIterator(input: input, model: model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 6)), skipSpecialTokens: false)
        let engine = try await TinyModel.engine(.deepSeekOCR2, promptCacheBytes: 1 << 30, processor: processor)
        let request = GenerationRequest(prompt: ocr.prompt, maxTokens: 6, temperature: 0, messages: ocr.messages).with {
            $0.cacheKey = "ns"
            $0.ocrMode = ocr.ocrMode
        }
        for _ in 0 ..< 2 {
            let reply = try await engine.generate(request: request)
            #expect(reply.cachedPromptTokens == 0)
            #expect(reply.generatedText == expected)
        }
        let tiny = request.with { $0.ocrMode = OCRMode(resolution: .tiny, grounding: false) }
        await #expect(throws: (any Error).self) { _ = try await engine.generate(request: tiny) }
    }
}
