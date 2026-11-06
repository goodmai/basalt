import MLX
import MLXLMCommon
@testable import BasaltInference

extension KVHeadMask {
    static func classify(scores: [[Float]], ratio: Double) -> KVHeadMask {
        let ranked = scores.enumerated().flatMap { layer, heads in heads.enumerated().map { (layer, $0.offset, $0.element) } }
            .sorted { $0.2 > $1.2 }
        let kept = Int((Double(ranked.count) * ratio).rounded())
        var retrieval = scores.map { _ in [Int]() }
        for (layer, head, _) in ranked.prefix(kept) { retrieval[layer].append(head) }
        return KVHeadMask(retrieval: retrieval.map { $0.sorted() })
    }
}

extension KVHeadClassifier {
    static func ablationScores(model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer, length: Int = 3072) throws -> [[Float]] {
        var total: [[Float]] = []
        for (index, depth) in [0.2, 0.5, 0.8].enumerated() {
            let key = 41_000 + index * 7_919
            var text = ""
            var sentence = 0
            while tokenizer.encode(text: text, addSpecialTokens: false).count < length {
                text += filler[sentence % filler.count] + " "
                sentence += 1
            }
            let body = tokenizer.encode(text: text, addSpecialTokens: false)
            let needle = tokenizer.encode(text: " The pass key is \(key). Remember it: \(key). ", addSpecialTokens: false)
            let question = tokenizer.encode(text: " What is the pass key? The pass key is", addSpecialTokens: false)
            let answer = tokenizer.encode(text: " \(key)", addSpecialTokens: false)
            let at = Int(Double(body.count) * depth)
            let head = Array(body[..<at]) + needle + Array(body[at...])
            let tail = question + answer
            let dense = try model.newCache(parameters: nil)
            eval(model(MLXArray(head.map(Int32.init)).reshaped(1, -1), cache: dense))
            eval(dense.flatMap { $0.innerState() })
            func answerLogProb(_ caches: [any KVCache]) -> Float {
                let logits = model(MLXArray(tail.map(Int32.init)).reshaped(1, -1), cache: caches)[0].asType(.float32)
                let predicting = logits[(question.count - 1) ..< (tail.count - 1)]
                let logp = predicting - logSumExp(predicting, axis: -1, keepDims: true)
                let targets = MLXArray(answer.map(Int32.init))[0..., .newAxis]
                return takeAlong(logp, targets, axis: -1).sum().item(Float.self)
            }
            let baseline = answerLogProb(dense.map { $0.copy() })
            let attention = dense.indices.filter { type(of: dense[$0]) == KVCacheSimple.self }
            var drops: [[Float]] = []
            for position in attention {
                guard let layer = dense[position] as? KVCacheSimple, let kvHeads = layer.state.first?.dim(1) else { continue }
                drops.append((0 ..< kvHeads).map { head in
                    var caches = dense.map { $0.copy() }
                    caches[position] = HeadSplitKVCache(splitting: layer, retrieval: (0 ..< kvHeads).filter { $0 != head })
                    return max(0, baseline - answerLogProb(caches))
                })
            }
            total = total.isEmpty ? drops : zip(total, drops).map { zip($0, $1).map { $0 + $1 } }
        }
        return total
    }
}
