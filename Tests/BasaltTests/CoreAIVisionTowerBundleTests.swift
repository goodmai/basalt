import CoreAI
import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

@Suite("Core AI vision tower bundle: metadata, buckets and the merge-block patch order (#224, #400)")
struct CoreAIVisionTowerBundleTests {
    static var tower: [String: Any] {
        ["asset": "tower", "masked": -30000.0, "merge": 2, "hidden": 1152, "text_hidden": 1024, "head_dim": 72,
        "functions": [["function": "n5120", "tokens": 5120], ["function": "n1024", "tokens": 1024], ["function": "odd", "tokens": 1023]]]
    }

    @Test("Functions are sorted, counts that are not a multiple of merge² are dropped, and a grid takes the smallest bucket that holds it")
    func buckets() throws {
        let bundle = try CoreAIVisionTowerBundle(tower: Self.tower, path: "/b")
        #expect(bundle.functions.map(\.tokens) == [1024, 5120])
        #expect(bundle.function(for: THW(1, 32, 32))?.name == "n1024")
        #expect(bundle.function(for: THW(1, 62, 80))?.name == "n5120")
        #expect(bundle.function(for: THW(1, 80, 66)) == nil)
        #expect(bundle.function(for: THW(1, 31, 32)) == nil)
        #expect(bundle.function(for: THW(2, 16, 16)) == nil)
        #expect(bundle.bias(real: 2, length: 4) == [0, 0, -30000, -30000])
    }

    @Test("A bundle without functions or required fields is refused")
    func refusals() {
        var empty = Self.tower
        empty["functions"] = [["function": "odd", "tokens": 6]]
        #expect(throws: BasaltError.self) { try CoreAIVisionTowerBundle(tower: empty, path: "/b") }
        var missing = Self.tower
        missing["merge"] = nil
        #expect(throws: BasaltError.self) { try CoreAIVisionTowerBundle(tower: missing, path: "/b") }
        #expect(throws: BasaltError.self) { try CoreAIVisionTowerBundle(directory: URL(fileURLWithPath: "/nonexistent")) }
    }

    @Test("Rows in merge-block order reshape to exactly the projector's 2x2 merge on 4x6 and 2x4 grids", arguments: [(4, 6), (2, 4)])
    func mergeOrder(_ grid: (Int, Int)) throws {
        let bundle = try CoreAIVisionTowerBundle(tower: Self.tower, path: "/b")
        let (h, w, c) = (grid.0, grid.1, 3)
        let order = bundle.order(THW(1, h, w))
        #expect(Set(order).count == h * w)
        let x = MLXArray(0 ..< Int32(h * w * c)).reshaped(h * w, c)
        let projector = x.reshaped(1, h / 2, 2, w / 2, 2, c).transposed(0, 1, 3, 2, 4, 5).reshaped(h * w / 4, -1)
        #expect(arrayEqual(x.take(MLXArray(order), axis: 0).reshaped(h * w / 4, 4 * c), projector).item(Bool.self))
    }

    @Test("serve --experimental-ane-vision-tower takes a bundle directory, refuses anything else, and reaches the engine configuration")
    func serveFlag() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(component: "ane-tower-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory.appending(component: "tower.aimodel"), withIntermediateDirectories: true)
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--experimental-ane-vision-tower", directory.path]) }
        try Data("{}".utf8).write(to: directory.appending(component: "metadata.json"))
        let command = try ServeCommand.parse(["--experimental-ane-vision-tower", directory.path])
        #expect(command.experimental.aneVisionTower == directory.path)
        let engine = try command.makeEngine()
        #expect(await engine.configuration.visionTowerBundle == directory.path)
        #expect(try ServeCommand.parse([]).experimental.aneVisionTower == nil)
    }

    @Test("Hand-off on the tiny PaddleOCR-VL: tower features go to a padded fp16 (1, C, 1, N) array and back unchanged except for the fp16 cast, and prefill on them equals prefill on the cast features (#228)")
    func handOffRoundTrip() throws {
        let model = try #require(try TinyModel.make(.paddleOCR) as? PaddleOCRVL)
        let prompt = PaddleOCRVLTests.imagePrompt()
        let input = PaddleOCRVLTests.imageInput(prompt)
        let frame = PaddleOCRVLTests.frame
        let reference = model.visual(try #require(input.image).pixels, frames: [frame])
        let (rows, channels) = (reference.dim(0), reference.dim(1))
        let planar = try CoreAIVisionTower.planar(reference, length: rows + 4)
        #expect(planar.shape == [1, channels, 1, rows + 4])
        #expect(try CoreAIVisionTower.planar(reference, length: rows, perHead: true).shape == [1, 1, channels, rows])
        let back = try CoreAIVisionTower.features(
            [planar], encoded: [CoreAIVisionTower.Encoded(function: "n", inputs: [:], rows: rows)], dtype: reference.dtype)
        let cast = reference.asType(.float16).asType(reference.dtype)
        #expect(back.shape == reference.shape)
        #expect(arrayEqual(back, cast).item(Bool.self))
        func logits(_ features: MLXArray) throws -> MLXArray? {
            let encoded = LMInput(text: input.text, image: .init(pixels: features, frames: [frame]))
            guard case .logits(let output) = try model.prepare(encoded, cache: try model.newCache(parameters: nil), state: nil, prefill: .init()) else { return nil }
            return output.logits[0, -1]
        }
        let (handed, direct) = (try #require(try logits(back)), try #require(try logits(cast)))
        #expect(arrayEqual(handed, direct).item(Bool.self))
    }

    static func tinyTower(_ model: PaddleOCRVL, tokens: [Int], headDim: Int? = nil) throws -> CoreAIVisionTowerBundle {
        try CoreAIVisionTowerBundle(tower: [
            "asset": "tower", "masked": -30000.0, "merge": model.visual.projector.merge, "hidden": 0,
            "text_hidden": model.visual.projector.linear2.weight.dim(0), "head_dim": headDim ?? model.visual.headDim,
            "functions": tokens.map { ["function": "n\($0)", "tokens": $0] },
        ], path: "/b")
    }

    @Test("Inputs on the tiny PaddleOCR-VL: the smallest bucket that holds the grid, (1, C, 1, N) planes and per-head cos/sin padded to it, the mask open on the real patches; a bundle for another head size or with no bucket that fits gives no inputs")
    func tinyInputs() throws {
        let model = try #require(try TinyModel.make(.paddleOCR) as? PaddleOCRVL)
        let image = try #require(PaddleOCRVLTests.imageInput(PaddleOCRVLTests.imagePrompt()).image)
        let frame = PaddleOCRVLTests.frame
        let count = frame.h * frame.w
        let encoded = try #require(try Self.tinyTower(model, tokens: [8, 32, 64]).inputs(image, visual: model.visual))
        #expect(encoded.map(\.function) == ["n32"] && encoded.first?.rows == count / 4)
        let inputs = try #require(encoded.first?.inputs)
        #expect(Set(inputs.keys) == ["patches", "pos", "cos", "sin", "bias"])
        let (patches, cos) = (try #require(inputs["patches"]), try #require(inputs["cos"]))
        #expect(Array(patches.shape).last == 32)
        #expect(Array(cos.shape) == [1, 1, model.visual.headDim, 32])
        var bias = try #require(inputs["bias"])
        let mask = bias.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in Array(UnsafeBufferPointer(start: pointer, count: 32)) }
        #expect(mask.prefix(count).allSatisfy { $0 == 0 } && mask.dropFirst(count).allSatisfy { $0 < -1000 })
        #expect(try Self.tinyTower(model, tokens: [32], headDim: model.visual.headDim + 2).inputs(image, visual: model.visual) == nil)
        #expect(try Self.tinyTower(model, tokens: [8]).inputs(image, visual: model.visual) == nil)
    }

    @Test("A bundle Core AI cannot load leaves the vision tower on the GPU and the loaded model serving")
    func unloadableBundleStaysOnGPU() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(component: "ane-tower-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory.appending(component: "tower.aimodel"), withIntermediateDirectories: true)
        let metadata: [String: Any] = ["paddle_vision_tower": Self.tower]
        try JSONSerialization.data(withJSONObject: metadata).write(to: directory.appending(component: "metadata.json"))
        var configuration = EngineConfiguration(promptCacheBytes: 0)
        configuration.visionTowerBundle = directory.path
        let engine = MLXInferenceEngine(configuration)
        let container = try TinyModel.container(.paddleOCR)
        await engine.attach(container)
        await engine.loadVisionTower(container, epoch: await engine.modelEpoch)
        #expect(await engine.modelFacts.visionTower == nil)
        #expect(await engine.isLoaded)
    }
}
