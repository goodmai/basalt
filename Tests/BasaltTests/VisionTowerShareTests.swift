import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Vision tower share of TTFT for 1, 4 and 16 images on a real VLM: the #224 spike gate", .serialized)
struct VisionTowerShareTests {
    static var paths: (model: String, pages: [URL])? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1", let model = env["BASALT_E2E_VISION_SHARE_MODEL_PATH"], let pages = env["BASALT_E2E_VISION_SHARE_PAGES"] else { return nil }
        return (model, pages.split(separator: ",").map { URL(fileURLWithPath: String($0)) })
    }

    static let runs = 3

    struct Sample {
        var preprocess = 0.0
        var ttft = 0.0
        var textOnly = 0.0
        var tower: Double?
        var tokens = 0
    }

    static func ms(_ body: () throws -> Void) rethrows -> Double {
        let start = ContinuousClock.now
        try body()
        return (ContinuousClock.now - start) / .milliseconds(1)
    }

    static func imageToken(_ model: any LanguageModel) -> Int? {
        switch model {
        case let model as PaddleOCRVL: model.config.imageTokenId
        case let model as Gemma4: model.config.imageTokenId
        case let model as Gemma4Unified: model.config.imageTokenId
        default: nil
        }
    }

    static func prefill(_ model: any LanguageModel, _ input: LMInput) throws {
        let cache = try model.newCache(parameters: nil)
        switch try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters(chunking: .unchunked)) {
        case .logits(let output): eval(output.logits)
        case .tokens(let text): eval(model(text, cache: cache, state: nil).logits)
        }
    }

    static func tower(_ model: any LanguageModel, _ image: LMInput.ProcessedImage) -> Double? {
        guard let model = model as? PaddleOCRVL, let frames = image.frames else { return nil }
        return ms { eval(model.visual(image.pixels.asType(model.visual.embeddings.patchEmbedding.weight.dtype), frames: frames)) }
    }

    static func sample(_ context: ModelContext, images: [URL], filler: Int) async throws -> Sample {
        var sample = Sample()
        let start = ContinuousClock.now
        let prepared = try await context.processor.prepare(input: UserInput(chat: [.user("OCR:", images: images.map { .url($0) })]))
        let image = try #require(prepared.image)
        eval(image.pixels, prepared.text.tokens)
        sample.preprocess = (ContinuousClock.now - start) / .milliseconds(1)
        let token = try #require(imageToken(context.model))
        Memory.clearCache()
        sample.ttft = try ms { try prefill(context.model, prepared) }
        let ids = prepared.text.tokens
        let text = which(ids .== token, MLXArray(Int32(filler)), ids.asType(.int32))
        Memory.clearCache()
        sample.textOnly = try ms { try prefill(context.model, LMInput(text: .init(tokens: text, mask: prepared.text.mask))) }
        Memory.clearCache()
        sample.tower = tower(context.model, image)
        sample.tokens = ids.size
        return sample
    }

    static func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }

    static func summary(_ count: Int, _ samples: [Sample]) -> String {
        let (pre, ttft, text) = (median(samples.map(\.preprocess)), median(samples.map(\.ttft)), median(samples.map(\.textOnly)))
        let byDifference = max(ttft - text, 0)
        let direct = samples.compactMap(\.tower)
        let tower = direct.isEmpty ? byDifference : median(direct)
        return String(
            format: "images %d prompt %d preprocess %.0f ms prefill %.0f ms text-only prefill %.0f ms tower by difference %.0f ms tower direct %@ ttft %.0f ms tower share %.1f%% (gate 25%%)",
            count, samples[0].tokens, pre, ttft, text, byDifference, direct.isEmpty ? "n/a" : String(format: "%.0f ms", tower),
            pre + ttft, 100 * tower / (pre + ttft))
    }

    @Test("Preprocessing, image prefill, the same ids as text and the tower alone (where reachable), median of 3, release build",
          .enabled(if: paths != nil, "set BASALT_E2E=1, BASALT_E2E_VISION_SHARE_MODEL_PATH and BASALT_E2E_VISION_SHARE_PAGES (comma-separated PNGs)"))
    func share() async throws {
        let (path, pages) = try #require(Self.paths)
        let log = try await ModelRunLog("vision-share", modelPath: path)
        defer { log.releaseGPU() }
        await MLXInferenceEngine.registerAliases()
        let container = try await log.step("load") { try await loadModelContainer(from: URL(fileURLWithPath: path), using: DefaultTokenizerLoader()) }
        try await container.perform { context in
            let filler = try #require(context.tokenizer.encode(text: " the", addSpecialTokens: false).last)
            _ = try await log.step("warm-up") { try await Self.sample(context, images: [pages[0]], filler: filler) }
            for count in [1, 4, 16] {
                let images = (0 ..< count).map { pages[$0 % pages.count] }
                var samples: [Sample] = []
                for _ in 0 ..< Self.runs {
                    try Task.checkCancellation()
                    samples.append(try await log.step("images-\(count)") { try await Self.sample(context, images: images, filler: filler) })
                }
                log.record("result", "\(Self.summary(count, samples)) \(log.host())")
            }
        }
        Memory.clearCache()
    }
}
