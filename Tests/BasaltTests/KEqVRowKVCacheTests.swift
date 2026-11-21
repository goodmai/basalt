import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Gemma 4 K=V shared-row KV cache on the gemma4KEqV TinyModel (#645, #228)", .serialized)
struct KEqVRowKVCacheTests {
    static let layout = KEqVRowLayout(headDim: 64, partialRotaryFactor: 0.25, kvHeads: 1, globalLayers: 2)!

    static func rowCaches(_ model: any LanguageModel) throws -> [any KVCache] {
        KEqVRowKVCache.replacingFullAttention(try model.newCache(parameters: nil), model: model, layout: layout)
    }

    static func greedy(_ model: any LanguageModel, cache: [any KVCache], prompt: [Int32], steps: Int) -> ([Int32], [MLXArray]) {
        var out = model(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: cache, state: nil)
        var tokens: [Int32] = []
        var logits: [MLXArray] = []
        for _ in 0 ..< steps {
            let last = out.logits[0, -1].asType(.float32)
            let next = argMax(last, axis: -1).item(Int32.self)
            tokens.append(next)
            logits.append(last)
            out = model(LMInput.Text(tokens: MLXArray([next])[.newAxis]), cache: cache, state: out.state)
        }
        return (tokens, logits)
    }

    @Test("64 greedy tokens equal the stock KVCacheSimple run on three prompts, logits within 1e-4", arguments: [
        [Int32(5), 19, 42, 7, 88, 13], [Int32(200), 3, 3, 150, 9], [Int32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ])
    func parity(prompt: [Int32]) throws {
        let model = try TinyModel.make(.gemma4KEqV)
        let rows = try Self.rowCaches(model)
        #expect(rows.compactMap { $0 as? KEqVRowKVCache }.count == 2)
        let (stockTokens, stockLogits) = Self.greedy(model, cache: try model.newCache(parameters: nil), prompt: prompt, steps: 64)
        let (rowTokens, rowLogits) = Self.greedy(model, cache: rows, prompt: prompt, steps: 64)
        #expect(rowTokens == stockTokens)
        let gap = zip(stockLogits, rowLogits).map { abs($0 - $1).max().item(Float.self) }.max() ?? 0
        #expect(gap <= 1e-4)
        #expect(rows.compactMap { $0 as? KEqVRowKVCache }.allSatisfy { !$0.materializedForCaller })
    }

    @Test("Global-layer bytes are 0.625x the stock cache at 300 and 1000 tokens (same 256-token steps)", arguments: [300, 1000])
    func bytes(tokens: Int) throws {
        let model = try TinyModel.make(.gemma4KEqV)
        let stock = try model.newCache(parameters: nil)
        let rows = try Self.rowCaches(model)
        let prompt = MLXArray((0 ..< tokens).map { Int32(($0 * 37 + 11) % 256) })[.newAxis]
        _ = model(LMInput.Text(tokens: prompt), cache: stock, state: nil)
        _ = model(LMInput.Text(tokens: prompt), cache: rows, state: nil)
        let rowBytes = rows.compactMap { $0 as? KEqVRowKVCache }.map(\.nbytes).reduce(0, +)
        let stockBytes = zip(stock, rows).filter { $0.1 is KEqVRowKVCache }
            .map { $0.0.innerState().map(\.nbytes).reduce(0, +) }.reduce(0, +)
        #expect(stockBytes > 0)
        #expect(Double(rowBytes) / Double(stockBytes) == 0.625)
    }

    @Test("After trimming rejected tokens and decoding the right ones, logits equal a plain prefill of the same tokens")
    func rollback() throws {
        let model = try TinyModel.make(.gemma4KEqV)
        let accepted: [Int32] = [5, 19, 42, 7, 88, 13]
        let rows = try Self.rowCaches(model)
        _ = model(LMInput.Text(tokens: MLXArray(accepted.prefix(3) + [Int32(250), 251, 252])[.newAxis]), cache: rows, state: nil)
        for cache in rows { #expect(cache.trim(3) == 3) }
        var out = model(LMInput.Text(tokens: MLXArray(Array(accepted[3 ..< 5]))[.newAxis]), cache: rows, state: nil)
        out = model(LMInput.Text(tokens: MLXArray([accepted[5]])[.newAxis]), cache: rows, state: out.state)
        let reference = model(LMInput.Text(tokens: MLXArray(accepted)[.newAxis]), cache: try model.newCache(parameters: nil), state: nil)
        let gap = abs(out.logits[0, -1].asType(.float32) - reference.logits[0, -1].asType(.float32)).max().item(Float.self)
        #expect(gap <= 1e-4)
    }

    @Test("The load probe accepts gemma4KEqV and the layout refuses a Gemma 4 without attention_k_eq_v")
    func probeAndRefusal() throws {
        #expect(try KEqVRowKVCache.modelAttendsThroughCache(try TinyModel.make(.gemma4KEqV), layout: Self.layout))
        let config = try JSONSerialization.jsonObject(with: Data(TinyModel.Family.gemma4.config.utf8)) as? [String: Any]
        #expect(config.flatMap(KEqVRowLayout.init(config:)) == nil)
        let kEqV = try JSONSerialization.jsonObject(with: Data(TinyModel.Family.gemma4KEqV.config.utf8)) as? [String: Any]
        #expect(kEqV.flatMap(KEqVRowLayout.init(config:)) == Self.layout)
    }
}
