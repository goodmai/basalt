import Foundation
import MLX
import BasaltCore

struct Zonos3VoicePrompt: Sendable, Equatable {
    let codes: [[Int32]]?
    let embedding: [Float]?
}

struct Zonos3SynthesisPlan: Sendable {
    let textIDs: [Int32]
    let voicePrompt: Zonos3VoicePrompt
    let conditionBuckets: [Int]
    let temperature: Float
    let cfgScale: Float
    let maxNewFrames: Int
    let speed: Double
    let seed: UInt64
    let epoch: Int
}

struct Zonos3BuiltPrompt {
    let rows: [[Int32]]
    let conditioning: Zonos3Conditioning
    let twinRows: [[Int32]]?
    let twinConditioning: Zonos3Conditioning?
    let silenceRows: [[Int32]]?
}

enum Zonos3PromptBuilder {
    static func loadVoiceTable(in directory: URL) -> [String: Zonos3VoicePrompt] {
        let voicesURL = directory.appending(component: "voices.safetensors")
        guard FileManager.default.fileExists(atPath: voicesURL.path),
              let arrays = try? MLX.loadArrays(url: voicesURL) else { return [:] }
        var table: [String: Zonos3VoicePrompt] = [:]
        var names = Set<String>()
        for key in arrays.keys {
            if let dot = key.firstIndex(of: ".") { names.insert(String(key[..<dot])) }
        }
        for name in names {
            let codesArr = arrays["\(name).codes"].map { tensor -> [[Int32]] in
                let rows = tensor.dim(0), cols = tensor.dim(1)
                let flat = tensor.asType(.int32).reshaped(-1).asArray(Int32.self)
                return (0 ..< rows).map { r in Array(flat[(r * cols) ..< ((r + 1) * cols)]) }
            }
            let embArr = arrays["\(name).embedding"]?.asType(.float32).reshaped(-1).asArray(Float.self)
            table[name.lowercased()] = Zonos3VoicePrompt(codes: codesArr, embedding: embArr)
        }
        return table
    }

    static func build(plan: Zonos3SynthesisPlan, config: Zonos3MiniConfig) -> Zonos3BuiltPrompt {
        let c = config.lm
        let g = config.generation
        let n = c.nCodebooks
        let pad = Int32(c.audioPadId)
        let textPad = Int32(c.textVocab)
        var rows: [[Int32]] = []
        if let refCodes = plan.voicePrompt.codes, !refCodes.isEmpty {
            let prefixLen = refCodes.count + n - 1
            let sheared = Zonos3DelaySchedule.shearRows(codes: refCodes, pad: pad, length: prefixLen)
            rows += sheared.map { $0 + [textPad] }
        }
        var spkRow: Int? = nil
        var spkEmb: MLXArray? = nil
        if plan.voicePrompt.codes == nil, let emb = plan.voicePrompt.embedding {
            spkRow = rows.count
            spkEmb = MLXArray(emb)
            rows.append(Array(repeating: pad, count: n) + [textPad])
        }
        var condEntries: [(row: Int, bucket: Int)] = []
        for bucket in plan.conditionBuckets {
            condEntries.append((row: rows.count, bucket: bucket))
            rows.append(Array(repeating: pad, count: n) + [textPad])
        }
        for tid in plan.textIDs {
            rows.append(Array(repeating: pad, count: n) + [tid])
        }
        var silenceRows: [[Int32]]? = nil
        let silCount = max(0, g.silencePrefillFrames)
        if silCount > 0, g.silenceFrame.count >= n {
            let silFrames = Array(repeating: Array(g.silenceFrame.prefix(n)), count: silCount)
            let shearedSil = Zonos3DelaySchedule.shearRows(codes: silFrames, pad: pad, length: silCount)
            let silWithText = shearedSil.map { $0 + [textPad] }
            silenceRows = silWithText
            rows += silWithText
        }
        let cond = Zonos3Conditioning(speakerRow: spkRow, speakerEmbedding: spkEmb, conditions: condEntries)
        var twinRows: [[Int32]]? = nil
        var twinCond: Zonos3Conditioning? = nil
        let lead = (plan.voicePrompt.codes.map { $0.isEmpty ? 0 : ($0.count + n - 1) } ?? 0) + (spkRow != nil ? 1 : 0)
        if abs(plan.cfgScale - 1.0) > 1e-4, lead > 0, lead < rows.count {
            twinRows = Array(rows[lead...])
            let unkConds = c.conditionBucketCounts.enumerated().map { idx, bucket in (row: idx, bucket: bucket) }
            twinCond = Zonos3Conditioning(speakerRow: nil, speakerEmbedding: nil, conditions: unkConds)
        }
        return Zonos3BuiltPrompt(
            rows: rows, conditioning: cond, twinRows: twinRows, twinConditioning: twinCond, silenceRows: silenceRows
        )
    }
}

enum Zonos3Sampler {
    static func sampleRow(
        logits: MLXArray,
        history: [[Int32]],
        generated: Int,
        config: Zonos3MiniConfig,
        temperature: Float,
        rng: inout UInt64
    ) -> [Int32] {
        let c = config.lm
        let g = config.generation
        let n = c.nCodebooks, v = c.audioVocab
        let flat = logits.asType(.float32).reshaped(-1).asArray(Float.self)
        let active = (g.repetitionWindow > 0 && g.repetitionPenalty > 1.0) ? min(g.repetitionWindow, generated) : 0
        let recentSlice: ArraySlice<[Int32]> = active > 0 ? history.suffix(active) : []
        var sampled = [Int32]()
        sampled.reserveCapacity(n)
        let invTemp = 1.0 / max(1e-6, temperature)
        for cb in 0 ..< n {
            let seen = collectSeen(recentSlice, codebook: cb, config: c, generation: g)
            var rowLogits = Array(flat[(cb * v) ..< ((cb + 1) * v)])
            if c.audioPadId < v { rowLogits[c.audioPadId] = -.infinity }
            if generated < 4 && c.eoaId < v { rowLogits[c.eoaId] = -.infinity }
            for idx in seen {
                rowLogits[idx] = rowLogits[idx] > 0
                    ? rowLogits[idx] / g.repetitionPenalty
                    : rowLogits[idx] * g.repetitionPenalty
            }
            for i in 0 ..< v where rowLogits[i].isFinite { rowLogits[i] *= invTemp }
            sampled.append(drawTopKTopP(&rowLogits, topK: g.topK, topP: g.topP, rng: &rng))
        }
        return sampled
    }

    private static func collectSeen(
        _ recentSlice: ArraySlice<[Int32]>,
        codebook: Int,
        config: Zonos3MiniConfig.LM,
        generation: Zonos3MiniConfig.Generation
    ) -> Set<Int> {
        guard codebook < generation.repetitionCodebooks else { return [] }
        var seen = Set<Int>()
        for past in recentSlice {
            let id = Int(past[codebook])
            if id >= 0 && id < config.codebookSize { seen.insert(id) }
        }
        return seen
    }

    static func drawTopKTopP(
        _ logits: inout [Float], topK: Int, topP: Float, rng: inout UInt64
    ) -> Int32 {
        let k = topK > 0 ? min(topK, logits.count) : logits.count
        let ordered = logits.indices.filter { logits[$0].isFinite }.sorted { logits[$0] > logits[$1] }
        guard let best = ordered.first else { return 0 }
        let kept = Array(ordered.prefix(max(1, k)))
        let maxL = logits[best]
        var probs = kept.map { exp(Double(logits[$0] - maxL)) }
        let total = probs.reduce(0, +)
        guard total > 0 else { return Int32(best) }
        for i in probs.indices { probs[i] /= total }
        if topP > 0 && topP < 1.0 {
            var cum = 0.0
            for i in probs.indices {
                if cum > Double(topP) && i > 0 { probs[i] = 0 }
                cum += probs[i]
            }
            let renorm = probs.reduce(0, +)
            if renorm > 0 { for i in probs.indices { probs[i] /= renorm } }
        }
        rng ^= rng >> 12
        rng ^= rng << 25
        rng ^= rng >> 27
        let u = Double((rng &* 0x2545_F491_4F6C_DD1D) >> 11) / Double(1 << 53)
        var acc = 0.0
        for (i, p) in probs.enumerated() {
            acc += p
            if u <= acc { return Int32(kept[i]) }
        }
        return Int32(kept.last ?? best)
    }
}
