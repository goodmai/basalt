import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("PaddleOCR-VL vision tower on the Neural Engine against the MLX tower: cosine, latency, hand-off and greedy output (#224, #400)", .serialized)
struct CoreAIVisionTowerRealModelTests {
    static var paths: (model: String, tower: String, pages: [(URL, String)])? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1", let model = env["BASALT_E2E_PADDLE_MODEL_PATH"], let tower = env["BASALT_E2E_ANE_TOWER"],
              let pages = env["BASALT_E2E_ANE_PAGES"] else { return nil }
        return (model, tower, pages.split(separator: ",").map { entry in
            let parts = entry.split(separator: "=", maxSplits: 1).map(String.init)
            return (URL(fileURLWithPath: parts[0]), parts.count > 1 ? parts[1] : "OCR:")
        })
    }

    static let samples = 5
    static let maxTokens = 1024

    static func ms(_ body: () async throws -> Void) async rethrows -> Double {
        var times: [Double] = []
        for _ in 0 ..< samples {
            let start = ContinuousClock.now
            try await body()
            times.append((ContinuousClock.now - start) / .milliseconds(1))
        }
        return times.sorted()[samples / 2]
    }

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> (all: Float, worstRow: Float) {
        let (x, y) = (a.asType(.float32), b.asType(.float32))
        let all = (x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))
        let rows = (x * y).sum(axis: -1) / (sqrt((x * x).sum(axis: -1)) * sqrt((y * y).sum(axis: -1)))
        return (all.item(Float.self), rows.min().item(Float.self))
    }

    static func greedy(_ context: ModelContext, _ input: LMInput) throws -> [Int] {
        let eos = context.tokenizer.eosTokenId
        var tokens: [Int] = []
        for token in try TokenIterator(input: input, model: context.model, processor: nil, sampler: ArgMaxSampler(), maxTokens: maxTokens) {
            if token == eos { break }
            tokens.append(token)
        }
        return tokens
    }

    static func handOff(_ tower: CoreAIVisionTower, _ image: LMInput.ProcessedImage, _ visual: PaddleOCRVLVision.Model) async throws -> Double {
        let frame = try #require(image.frames?.first)
        let function = try #require(tower.bundle.function(for: frame))
        let count = frame.h * frame.w
        let order = MLXArray(tower.bundle.order(frame))
        let (cos, sin) = visual.rotary(frame)
        let rows = [image.pixels[0 ..< count].reshaped(count, -1), visual.embeddings.positions(frame), cos.reshaped(count, -1), sin.reshaped(count, -1)]
            .map { $0.take(order, axis: 0) }
        eval(rows)
        let output = try CoreAIVisionTower.planar(MLXArray.zeros([function.tokens / 4, tower.bundle.textHidden]), length: function.tokens / 4)
        let handIn = try await ms { for row in rows { _ = try CoreAIVisionTower.planar(row, length: function.tokens) } }
        let handOut = try await ms { eval(try CoreAIVisionTower.rows(output, count: count / 4)) }
        return handIn + handOut
    }

    static func float32(_ visual: PaddleOCRVLVision.Model, _ image: LMInput.ProcessedImage) throws -> MLXArray {
        let original = visual.parameters()
        visual.update(parameters: original.mapValues { $0.dtype == .bfloat16 ? $0.asType(.float32) : $0 })
        defer { visual.update(parameters: original) }
        let features = visual(image.pixels.asType(.float32), frames: try #require(image.frames))
        eval(features)
        return features
    }

    static func firstDivergence(_ context: ModelContext, _ input: LMInput, _ reference: [Int], features: MLXArray, dtype: DType) throws -> (Int, Bool) {
        let tokens = try greedy(context, LMInput(text: input.text, image: LMInput.ProcessedImage(pixels: features.asType(dtype), frames: input.image?.frames)))
        return (zip(reference, tokens).prefix { $0 == $1 }.count, tokens == reference)
    }

    @Test("Per page: ANE and bf16 GPU features against a float32 tower and against each other, tower latency with the hand-off, greedy agreement against the shipped GPU path",
          .enabled(if: paths != nil, "set BASALT_E2E=1, BASALT_E2E_PADDLE_MODEL_PATH, BASALT_E2E_ANE_TOWER and BASALT_E2E_ANE_PAGES (png[=prompt],...)"))
    func aneMatchesGPU() async throws {
        let (path, towerPath, pages) = try #require(Self.paths)
        let log = try await ModelRunLog("ane-vision", modelPath: path)
        defer { log.releaseGPU() }
        await MLXInferenceEngine.registerAliases()
        let container = try await log.step("load") { try await loadModelContainer(from: URL(fileURLWithPath: path), using: DefaultTokenizerLoader()) }
        let start = ContinuousClock.now
        let tower = try await log.step("specialize") { try await CoreAIVisionTower(directory: URL(fileURLWithPath: towerPath)) }
        log.record("specialize", "\(Int((ContinuousClock.now - start) / .milliseconds(1))) ms, buckets \(tower.bundle.functions.map(\.tokens))")
        try await container.perform { context in
            let model = try #require(context.model as? PaddleOCRVL)
            for (url, prompt) in pages {
                try Task.checkCancellation()
                let input = try await context.processor.prepare(input: UserInput(chat: [.user(prompt, images: [.url(url)])]))
                let image = try #require(input.image)
                let gpu = model.visual(image.pixels.asType(model.visual.embeddings.patchEmbedding.weight.dtype), frames: try #require(image.frames))
                let ane = try #require(try await tower.features(image, visual: model.visual), "no bucket for \(image.frames ?? [])")
                eval(gpu, ane)
                let similarity = Self.cosine(gpu, ane)
                let gpuMs = await Self.ms { eval(model.visual(image.pixels.asType(model.visual.embeddings.patchEmbedding.weight.dtype), frames: image.frames ?? [])) }
                let aneMs = try await Self.ms { eval(try #require(try await tower.features(image, visual: model.visual))) }
                let handOff = try await Self.handOff(tower, image, model.visual)
                let exact = try Self.float32(model.visual, image)
                let (gpuExact, aneExact) = (Self.cosine(gpu, exact), Self.cosine(ane, exact))
                let reference = try Self.greedy(context, input)
                let dtype = model.visual.embeddings.patchEmbedding.weight.dtype
                let (aneAgree, aneSame) = try Self.firstDivergence(context, input, reference, features: ane, dtype: dtype)
                let (exactAgree, exactSame) = try Self.firstDivergence(context, input, reference, features: exact, dtype: dtype)
                log.record("result", String(
                    format: "%@ %@ grid %@ | ANE vs GPU cosine %.6f worst token %.6f | vs float32 tower: GPU bf16 %.6f (worst %.6f), ANE fp16 %.6f (worst %.6f) | tower GPU %.1f ms ANE %.1f ms (hand-off %.1f ms) | greedy against the GPU path over %d tokens: ANE first divergence %d identical %@, float32 tower first divergence %d identical %@ %@",
                    url.lastPathComponent, prompt, "\(image.frames ?? [])", similarity.all, similarity.worstRow, gpuExact.all, gpuExact.worstRow,
                    aneExact.all, aneExact.worstRow, gpuMs, aneMs, handOff, reference.count, aneAgree, aneSame ? "yes" : "no",
                    exactAgree, exactSame ? "yes" : "no", log.host()))
                #expect(aneExact.all >= gpuExact.all - 0.001, "\(url.lastPathComponent): the ANE tower is further from float32 than the bf16 GPU tower")
            }
        }
        Memory.clearCache()
    }
}
