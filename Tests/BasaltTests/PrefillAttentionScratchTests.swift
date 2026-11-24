import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltInference

@Suite("Prefill chunk cap where attention runs unfused, so long prompts keep their score scratch bounded (#741 P2)")
struct PrefillAttentionScratchTests {
    static var gemma4: [String: Any] { ["text_config": ["num_attention_heads": 16, "head_dim": 256, "global_head_dim": 512]] }

    @Test("Gemma 4's 512 global head dim is unfused off NAX GPUs and fused on them; a 128 head dim never is")
    func detectsUnfusedHeadDims() {
        #expect(PrefillAttentionScratch.unfused(config: Self.gemma4, nax: false) == PrefillAttentionScratch(heads: 16))
        #expect(PrefillAttentionScratch.unfused(config: Self.gemma4, nax: true) == nil)
        #expect(PrefillAttentionScratch.unfused(config: ["num_attention_heads": 32, "head_dim": 128], nax: false) == nil)
        #expect(PrefillAttentionScratch.unfused(config: ["num_attention_heads": 8, "head_dim": 112], nax: true) == PrefillAttentionScratch(heads: 8))
        #expect(PrefillAttentionScratch.unfused(config: ["head_dim": 512], nax: false) == nil)
        #expect(PrefillAttentionScratch.unfused(config: nil, nax: false) == nil)
    }

    @Test("NAX follows MLX's rule: architecture generation 17 or later, 18 or later for p parts", arguments: [
        ("applegpu_g16s", false), ("applegpu_g16p", false), ("applegpu_g17s", true), ("applegpu_g17g", true),
        ("applegpu_g17p", false), ("applegpu_g18p", true), ("unknown", false),
    ])
    func naxRule(architecture: String, expected: Bool) {
        #expect(PrefillAttentionScratch.naxAvailable(architecture: architecture) == expected)
    }

    @Test("16 heads at 37.5K keys cap the chunk at 111 rows, a short prompt keeps the default chunk, and the cap never goes under 64")
    func stepCap() throws {
        let model = try TinyModel.make(.gemma4)
        let scratch = PrefillAttentionScratch(heads: 16)
        #expect(scratch.stepCap(keys: 37_553) == 111)
        #expect(scratch.stepCap(keys: 4_096) == nil)
        #expect(scratch.stepCap(keys: 1 << 30) == PrefillAttentionScratch.minimumStep)
        let parameters = GenerateParameters()
        #expect(parameters.capped(for: model, promptTokens: 37_553, scratch: scratch).prefill.stepSize == 111)
        #expect(parameters.capped(for: model, promptTokens: 37_553).prefill.stepSize == parameters.prefill.stepSize)
    }

    @Test("On tiny Gemma 4, whose head dims are unfused, prefilling 300 tokens in 64-row chunks gives the last logits of one forward within 1e-3")
    func chunkedPrefillMatchesOneForward() throws {
        let model = try TinyModel.make(.gemma4)
        let tokens = MLXArray((0 ..< 300).map { Int32(($0 * 31 + 7) % 256) })[.newAxis]
        let whole = model(tokens, cache: try model.newCache(parameters: nil))[0..., (-1)..., 0...]
        let cache = try model.newCache(parameters: nil)
        var last = whole
        for start in stride(from: 0, to: 300, by: PrefillAttentionScratch.minimumStep) {
            last = model(tokens[0..., start ..< min(start + PrefillAttentionScratch.minimumStep, 300)], cache: cache)[0..., (-1)..., 0...]
        }
        #expect(PrefillAttentionScratch.unfused(config: ["num_attention_heads": 4, "head_dim": 16, "global_head_dim": 32], nax: true) != nil)
        #expect(abs(last.asType(.float32) - whole.asType(.float32)).max().item(Float.self) < 1e-3)
    }
}
