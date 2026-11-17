import Foundation
import MLX
import BasaltCore

public struct Zonos3MiniConfig: Sendable, Equatable {
    package struct MossStageSpec: Sendable, Codable, Equatable {
        package var inputDimension: Int
        package var dModel: Int
        package var numHeads: Int
        package var dimFeedforward: Int
        package var contextFrames: Int
        package var numLayers: Int = 2
        package var outputDimension: Int? = nil
    }

    public struct LM: Sendable, Equatable {
        public var nLayers: Int = 11
        public var dim: Int = 1_024
        public var headDim: Int = 32
        public var nHeads: Int = 16
        public var nKVHeads: Int = 4
        public var normEps: Float = 1e-5
        public var ropeTheta: Float = 10_000.0
        public var maxSeqlen: Int = 32_768
        public var lossSoftcap: Float = 15.0
        public var gdnPeriod: Int = 4
        public var gdnHeads: Int = 4
        public var gdnHeadDim: Int = 128
        public var gdnKernel: Int = 4
        public var ffnHidden: Int = 2_048
        public var moeHidden: Int = 1_536
        public var moeRoutedDim: Int = 256
        public var nExperts: Int = 64
        public var moeTopK: Int = 2
        public var nCodebooks: Int = 10
        public var codebookSize: Int = 4_096
        public var eoaId: Int = 4_096
        public var audioPadId: Int = 4_097
        public var textVocab: Int = 8_196
        public var textRows: Int = 8_197
        public var speakerEmbeddingDim: Int = 192
        public var referenceMaxFrames: Int = 900
        public var conditionNames: [String] = ["loudness", "fmax", "speaking_rate"]
        public var conditionBucketCounts: [Int] = [10, 10, 30]
        public var conditionEdges: [String: [Float]] = [:]

        public init() {}

        public var audioVocab: Int { codebookSize + 2 }
        public func usesGDN(layer: Int) -> Bool { (layer + 1) % max(1, gdnPeriod) != 0 }
    }

    public struct Codec: Sendable, Equatable {
        public var codebookSize: Int = 4_096
        public var codebookDim: Int = 8
        public var residualCodebooks: Int = 9
        public var latentDim: Int = 1_024
        public var semanticWidth: Int = 768
        public var semanticHidden: Int = 2_048
        public var semanticDepth: Int = 4
        package var stages: [MossStageSpec] = [
            MossStageSpec(inputDimension: 1_024, dModel: 1_280, numHeads: 20, dimFeedforward: 5_120, contextFrames: 150, outputDimension: 1_280),
            MossStageSpec(inputDimension: 640, dModel: 768, numHeads: 12, dimFeedforward: 3_072, contextFrames: 300, outputDimension: 768),
            MossStageSpec(inputDimension: 384, dModel: 768, numHeads: 12, dimFeedforward: 3_072, contextFrames: 480, outputDimension: 768),
            MossStageSpec(inputDimension: 384, dModel: 768, numHeads: 12, dimFeedforward: 3_072, contextFrames: 480, outputDimension: 768),
            MossStageSpec(inputDimension: 384, dModel: 768, numHeads: 12, dimFeedforward: 3_072, contextFrames: 480, outputDimension: nil),
        ]
        public var headDepth: Int = 8
        public var headHidden: Int = 2_048
        public var nFFTs: [Int] = [736, 2_208]
        public var headHop: Int = 184
        public var firKernel: Int = 127
        public var resampleOrig: Int = 736
        public var resampleNew: Int = 735
        public var resampleWidth: Int = 17
        public var hop: Int = 2_940

        public init() {}
    }

    public struct Generation: Sendable, Equatable {
        public var silenceFrame: [Int32] = [1_918, 746, 3_544, 234, 1_267, 2_914, 3_944, 2_849, 326, 1_337]
        public var silencePrefillFrames: Int = 3
        public var silencePrefillMaxFrames: Int = 30
        public var decodeContextFrames: Int = 30
        public var overlapFrames: Int = 4
        public var headroomDB: Float = 3.0
        public var outputLimitDBFS: Float = -1.0
        public var temperature: Float = 0.75
        public var topK: Int = 80
        public var topP: Float = 0.69
        public var minP: Float = 0.0
        public var repetitionPenalty: Float = 2.4
        public var repetitionWindow: Int = 96
        public var repetitionCodebooks: Int = 6

        public init() {}
    }

    public var modelID: String
    public var sampleRate: Int
    public var lm: LM
    public var codec: Codec
    public var generation: Generation

    public init(
        modelID: String = "Zyphra/ZONOS3-Mini-hf",
        sampleRate: Int = Zonos3Catalog.sampleRate,
        lm: LM = LM(),
        codec: Codec = Codec(),
        generation: Generation = Generation()
    ) {
        self.modelID = modelID
        self.sampleRate = max(8_000, sampleRate)
        self.lm = lm
        self.codec = codec
        self.generation = generation
    }

    public static var tiny: Zonos3MiniConfig {
        var lm = LM()
        lm.nLayers = 4
        lm.dim = 32
        lm.headDim = 8
        lm.nHeads = 4
        lm.nKVHeads = 2
        lm.gdnPeriod = 4
        lm.gdnHeads = 2
        lm.gdnHeadDim = 16
        lm.gdnKernel = 4
        lm.ffnHidden = 64
        lm.moeHidden = 48
        lm.moeRoutedDim = 16
        lm.nExperts = 4
        lm.moeTopK = 2
        lm.nCodebooks = 10
        lm.codebookSize = 64
        lm.eoaId = 64
        lm.audioPadId = 65
        lm.textVocab = 256
        lm.textRows = 257
        lm.speakerEmbeddingDim = 32
        lm.referenceMaxFrames = 32

        var codec = Codec()
        codec.codebookSize = 64
        codec.codebookDim = 8
        codec.residualCodebooks = 9
        codec.latentDim = 32
        codec.semanticWidth = 24
        codec.semanticHidden = 48
        codec.semanticDepth = 2
        codec.stages = [
            MossStageSpec(inputDimension: 32, dModel: 32, numHeads: 4, dimFeedforward: 64, contextFrames: 32, numLayers: 1, outputDimension: 32),
            MossStageSpec(inputDimension: 16, dModel: 32, numHeads: 4, dimFeedforward: 64, contextFrames: 64, numLayers: 1, outputDimension: 32),
            MossStageSpec(inputDimension: 16, dModel: 32, numHeads: 4, dimFeedforward: 64, contextFrames: 64, numLayers: 1, outputDimension: 32),
            MossStageSpec(inputDimension: 16, dModel: 32, numHeads: 4, dimFeedforward: 64, contextFrames: 64, numLayers: 1, outputDimension: 32),
            MossStageSpec(inputDimension: 16, dModel: 32, numHeads: 4, dimFeedforward: 64, contextFrames: 64, numLayers: 1, outputDimension: nil),
        ]
        codec.headDepth = 2
        codec.headHidden = 64
        codec.nFFTs = [368, 736]
        codec.headHop = 184
        codec.firKernel = 15
        codec.resampleOrig = 736
        codec.resampleNew = 735
        codec.resampleWidth = 17
        codec.hop = 2_940

        var gen = Generation()
        gen.silenceFrame = (0 ..< 10).map { Int32(($0 * 7 + 3) % 64) }
        gen.topK = 16
        return Zonos3MiniConfig(modelID: "Zyphra/ZONOS3-Mini-Tiny", sampleRate: Zonos3Catalog.sampleRate, lm: lm, codec: codec, generation: gen)
    }
}

extension Zonos3MiniConfig {
    public static func from(directory: URL, fallbackID: String = "Zyphra/ZONOS3-Mini-hf") throws(BasaltError) -> Zonos3MiniConfig {
        _ = try TTSModelFamily.validateSupported(identifier: fallbackID, directory: directory)
        let url = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: url),
              let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return Zonos3MiniConfig(modelID: fallbackID)
        }
        _ = try TTSModelFamily.validateSupported(identifier: fallbackID, config: raw, directory: directory)
        var cfg = Zonos3MiniConfig(
            modelID: raw["model_id"] as? String ?? fallbackID,
            sampleRate: raw["sampling_rate"] as? Int ?? raw["sample_rate"] as? Int ?? Zonos3Catalog.sampleRate
        )
        if let lmDict = raw["lm"] as? [String: Any] { cfg.lm = parseLM(lmDict) }
        if let codecDict = raw["codec"] as? [String: Any] { cfg.codec = parseCodec(codecDict) }
        if let genDict = raw["generation"] as? [String: Any] { cfg.generation = parseGeneration(genDict) }
        return cfg
    }

    private static func parseLM(_ d: [String: Any]) -> LM {
        var lm = LM()
        lm.nLayers = d["n_layers"] as? Int ?? lm.nLayers
        lm.dim = d["dim"] as? Int ?? lm.dim
        lm.headDim = d["head_dim"] as? Int ?? lm.headDim
        lm.nHeads = d["n_heads"] as? Int ?? lm.nHeads
        lm.nKVHeads = d["n_kv_heads"] as? Int ?? lm.nKVHeads
        lm.normEps = (d["norm_eps"] as? NSNumber)?.floatValue ?? lm.normEps
        lm.ropeTheta = (d["rope_theta"] as? NSNumber)?.floatValue ?? lm.ropeTheta
        lm.maxSeqlen = d["max_seqlen"] as? Int ?? lm.maxSeqlen
        lm.lossSoftcap = (d["loss_softcap"] as? NSNumber)?.floatValue ?? lm.lossSoftcap
        lm.gdnPeriod = d["gdn_period"] as? Int ?? lm.gdnPeriod
        lm.gdnHeads = d["gdn_heads"] as? Int ?? lm.gdnHeads
        lm.gdnHeadDim = d["gdn_head_dim"] as? Int ?? lm.gdnHeadDim
        lm.gdnKernel = d["gdn_kernel"] as? Int ?? lm.gdnKernel
        lm.ffnHidden = d["ffn_hidden"] as? Int ?? lm.ffnHidden
        lm.moeHidden = d["moe_hidden"] as? Int ?? lm.moeHidden
        lm.moeRoutedDim = d["moe_routed_dim"] as? Int ?? lm.moeRoutedDim
        lm.nExperts = d["n_experts"] as? Int ?? lm.nExperts
        lm.moeTopK = d["moe_topk"] as? Int ?? lm.moeTopK
        lm.nCodebooks = d["n_codebooks"] as? Int ?? lm.nCodebooks
        lm.codebookSize = d["codebook_size"] as? Int ?? lm.codebookSize
        lm.eoaId = d["eoa_id"] as? Int ?? lm.eoaId
        lm.audioPadId = d["audio_pad_id"] as? Int ?? lm.audioPadId
        lm.textVocab = d["text_vocab"] as? Int ?? lm.textVocab
        lm.textRows = d["text_rows"] as? Int ?? lm.textRows
        lm.speakerEmbeddingDim = d["speaker_embedding_dim"] as? Int ?? lm.speakerEmbeddingDim
        lm.referenceMaxFrames = d["reference_max_frames"] as? Int ?? lm.referenceMaxFrames
        if let names = d["condition_names"] as? [String] { lm.conditionNames = names }
        if let counts = d["condition_bucket_counts"] as? [Int] { lm.conditionBucketCounts = counts }
        if let edges = d["condition_edges"] as? [String: [NSNumber]] {
            lm.conditionEdges = edges.mapValues { $0.map(\.floatValue) }
        }
        return lm
    }

    private static func parseCodec(_ d: [String: Any]) -> Codec {
        var c = Codec()
        c.codebookSize = d["codebook_size"] as? Int ?? c.codebookSize
        c.codebookDim = d["codebook_dim"] as? Int ?? c.codebookDim
        c.residualCodebooks = d["residual_codebooks"] as? Int ?? c.residualCodebooks
        c.latentDim = d["latent_dim"] as? Int ?? c.latentDim
        c.semanticWidth = d["semantic_width"] as? Int ?? c.semanticWidth
        c.semanticHidden = d["semantic_hidden"] as? Int ?? c.semanticHidden
        c.semanticDepth = d["semantic_depth"] as? Int ?? c.semanticDepth
        c.headDepth = d["head_depth"] as? Int ?? c.headDepth
        c.headHidden = d["head_hidden"] as? Int ?? c.headHidden
        if let ffts = d["n_ffts"] as? [Int], !ffts.isEmpty { c.nFFTs = ffts }
        c.headHop = d["head_hop"] as? Int ?? c.headHop
        c.firKernel = d["fir_kernel"] as? Int ?? c.firKernel
        c.resampleOrig = d["resample_orig"] as? Int ?? c.resampleOrig
        c.resampleNew = d["resample_new"] as? Int ?? c.resampleNew
        c.resampleWidth = d["resample_width"] as? Int ?? c.resampleWidth
        c.hop = d["hop"] as? Int ?? c.hop
        if let stageList = d["stages"] as? [[String: Any]], !stageList.isEmpty {
            c.stages = stageList.map { item in
                MossStageSpec(
                    inputDimension: item["input_dimension"] as? Int ?? 384,
                    dModel: item["d_model"] as? Int ?? 768,
                    numHeads: item["num_heads"] as? Int ?? 12,
                    dimFeedforward: item["dim_feedforward"] as? Int ?? 3_072,
                    contextFrames: item["context_frames"] as? Int ?? 480,
                    numLayers: item["num_layers"] as? Int ?? 2,
                    outputDimension: item["output_dimension"] as? Int
                )
            }
        }
        return c
    }

    private static func parseGeneration(_ d: [String: Any]) -> Generation {
        var g = Generation()
        if let sil = d["silence_frame"] as? [Int], !sil.isEmpty { g.silenceFrame = sil.map(Int32.init) }
        g.silencePrefillFrames = d["silence_prefill_frames"] as? Int ?? g.silencePrefillFrames
        g.silencePrefillMaxFrames = d["silence_prefill_max_frames"] as? Int ?? g.silencePrefillMaxFrames
        g.decodeContextFrames = d["decode_context_frames"] as? Int ?? g.decodeContextFrames
        g.overlapFrames = d["overlap_frames"] as? Int ?? g.overlapFrames
        g.headroomDB = (d["headroom_db"] as? NSNumber)?.floatValue ?? g.headroomDB
        g.outputLimitDBFS = (d["output_limit_dbfs"] as? NSNumber)?.floatValue ?? g.outputLimitDBFS
        g.temperature = (d["temperature"] as? NSNumber)?.floatValue ?? g.temperature
        g.topK = d["topk"] as? Int ?? g.topK
        g.topP = (d["top_p"] as? NSNumber)?.floatValue ?? g.topP
        g.minP = (d["min_p"] as? NSNumber)?.floatValue ?? g.minP
        g.repetitionPenalty = (d["repetition_penalty"] as? NSNumber)?.floatValue ?? g.repetitionPenalty
        g.repetitionWindow = d["repetition_window"] as? Int ?? g.repetitionWindow
        g.repetitionCodebooks = d["repetition_codebooks"] as? Int ?? g.repetitionCodebooks
        return g
    }
}

package enum Zonos3DelaySchedule {
    package static func shearRows(codes: [[Int32]], pad: Int32, length: Int) -> [[Int32]] {
        let frames = codes.count
        let n = codes.first?.count ?? 0
        guard length > 0, n > 0 else { return [] }
        var out = Array(repeating: Array(repeating: pad, count: n), count: length)
        for j in 0 ..< n {
            let end = min(length, frames + j)
            if end > j {
                for r in j ..< end {
                    out[r][j] = codes[r - j][j]
                }
            }
        }
        return out
    }

    package static func undelay(buffer: [[Int32]], start: Int, end: Int, n: Int, pad: Int32) -> [[Int32]] {
        guard end > start, n > 0, start < buffer.count else { return [] }
        let sliceEnd = min(end + n - 1, buffer.count)
        let rows = Array(buffer[start ..< sliceEnd])
        var out = Array(repeating: Array(repeating: pad, count: n), count: rows.count)
        for j in 0 ..< n where rows.count > j {
            for r in 0 ..< (rows.count - j) {
                out[r][j] = rows[r + j][j]
            }
        }
        return Array(out.prefix(end - start))
    }

    package static func takeCodes(
        rows: [[Int32]],
        eosFrame: Int?,
        prefillRows: [[Int32]]?,
        lm: Zonos3MiniConfig.LM,
        overlapFrames: Int
    ) -> (codes: [[Int32]], start: Int) {
        let nCodebooks = lm.nCodebooks
        let pad = Int32(lm.audioPadId)
        let eoa = Int32(lm.eoaId)
        let prefill = prefillRows?.count ?? 0
        var buffer: [[Int32]] = prefillRows?.map { Array($0.prefix(nCodebooks)) } ?? []
        buffer.append(contentsOf: rows.map { Array($0.prefix(nCodebooks)) })
        var eos: Int? = nil
        for (i, row) in rows.enumerated() {
            var candidates: [Int] = []
            if let eosFrame { candidates.append(max(0, eosFrame) + prefill) }
            let cols = (0 ..< min(nCodebooks, row.count)).filter { row[$0] == eoa }
            if let maxCol = cols.max() {
                candidates.append(max(0, prefill + i - maxCol))
            }
            if let minCand = candidates.min() {
                eos = eos.map { min($0, minCand) } ?? minCand
            }
        }
        var target = buffer.count - (nCodebooks - 1)
        if let eos { target = min(target, eos) }
        let start = prefill - min(overlapFrames, prefill)
        if target <= prefill { return ([], start) }
        return (undelay(buffer: buffer, start: start, end: target, n: nCodebooks, pad: pad), start)
    }

    package static func outputStage(
        _ audio: [Float],
        headroomDB: Float,
        outputLimitDBFS: Float,
        fadeOutMs: Float = 0,
        sampleRate: Int = Zonos3Catalog.sampleRate
    ) -> [Float] {
        guard !audio.isEmpty else { return [] }
        let gain = Float(pow(10.0, -Double(headroomDB) / 20.0))
        let knee = Float(pow(10.0, Double(outputLimitDBFS) / 20.0))
        let width = max(Float(1e-6), 1.0 - knee)
        var out = audio.map { raw -> Float in
            guard raw.isFinite else { return 0 }
            let x = raw * gain
            let mag = abs(x)
            guard mag > knee else { return x }
            let limited = knee + width * Float(tanh(Double((mag - knee) / width)))
            return x >= 0 ? limited : -limited
        }
        if fadeOutMs > 0 {
            let count = min(out.count, Int((Float(sampleRate) * fadeOutMs / 1_000.0).rounded(.down)))
            if count > 1 {
                let start = out.count - count
                for i in 0 ..< count {
                    let w = Float(0.5 * (1.0 + cos(Double.pi * Double(i) / Double(count - 1))))
                    out[start + i] *= w
                }
            }
        }
        return out
    }
}
