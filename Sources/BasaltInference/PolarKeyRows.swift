import Foundation
import MLX

struct PolarKeyRows {
    static let modulusBits = 3
    static let phaseLevels = 1 << (8 - modulusBits)
    static let modulusLevels = 1 << modulusBits
    static let rayleighCenters: [Float] = (0 ..< modulusLevels).map { i in
        (-2 * Float.log(1 - (Float(i) + 0.5) / Float(modulusLevels))).squareRoot()
    }

    let rotated: Int
    private(set) var codes: MLXArray?
    private(set) var sigma: MLXArray?
    private var rest: ZonedKVCache.QuantizedRows?

    init(rotated: Int, restBits: Int, groupSize: Int, headDim: Int) {
        self.rotated = rotated
        rest = headDim > rotated ? ZonedKVCache.QuantizedRows(bits: restBits, groupSize: groupSize) : nil
    }

    var pairs: Int { rotated / 2 }
    var count: Int { codes?.dim(2) ?? 0 }
    var arrays: [MLXArray] { [codes, sigma].compactMap { $0 } + (rest?.arrays ?? []) }
    private var keyDim: Int {
        rotated + (rest.flatMap { r in r.parts.map { $0.1.dim(-1) * r.groupSize } } ?? 0)
    }

    private func canAppend(_ rows: MLXArray) -> Bool {
        guard rows.ndim == 4, rows.dim(0) > 0, rows.dim(1) > 0, rows.dim(2) > 0,
              rotated.isMultiple(of: 2), pairs > 0 else {
            return false
        }
        let validWidth = if let rest {
            rows.dim(3) > rotated && (rows.dim(3) - rotated).isMultiple(of: rest.groupSize)
        } else {
            rows.dim(3) == rotated
        }
        guard validWidth else { return false }
        if let existing = codes {
            return rows.dim(0) == existing.dim(0) && rows.dim(1) == existing.dim(1) && rows.dim(3) == keyDim
        }
        return true
    }

    mutating func append(_ rows: MLXArray) {
        guard canAppend(rows) else { return }
        let cleanRows = PolarFusedScores.supportedDTypes.contains(rows.dtype)
            ? rows
            : rows.asType(.float32, stream: rows.dtype == .float64 ? .cpu : .default)
        let re = cleanRows[.ellipsis, ..<pairs].asType(.float32)
        let im = cleanRows[.ellipsis, pairs ..< rotated].asType(.float32)
        let r = sqrt(re * re + im * im)
        let scale = sigma ?? maximum(sqrt((r * r).mean(axis: 2, keepDims: true) / 2), MLXArray(Float(1e-6)))
        sigma = scale
        let turn = (atan2(im, re) * (Float(Self.phaseLevels) / (2 * Float.pi))).round()
        let phase = MLX.remainder(turn + Float(Self.phaseLevels), Float(Self.phaseLevels)).asType(.uint8)
        let centers = MLXArray(Self.rayleighCenters)
        let modulus = argMin(abs((r / scale).expandedDimensions(axis: -1) - centers), axis: -1).asType(.uint8)
        let code = phase * UInt8(Self.modulusLevels) + modulus
        codes = codes.map { concatenated([$0, code], axis: 2) } ?? code
        if cleanRows.dim(-1) > rotated { rest?.append(cleanRows[.ellipsis, rotated...]) }
    }

    mutating func trim(_ n: Int) -> Int {
        let removed = min(max(0, n), count)
        guard removed > 0, let codes else { return 0 }
        let kept = count - removed
        self.codes = kept == 0 ? nil : codes[.ellipsis, ..<kept, 0...]
        if kept == 0 { sigma = nil }
        _ = rest?.trim(removed)
        return removed
    }

    func dequantizedRows(dtype: DType) -> MLXArray? {
        guard let codes, let sigma else { return nil }
        if Device.defaultDevice() == .gpu, (dtype == .float64 || sigma.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) { dequantizedRows(dtype: dtype) }
        }
        let phase = (codes >> UInt8(Self.modulusBits)).asType(.float32) * (2 * Float.pi / Float(Self.phaseLevels))
        let r = take(MLXArray(Self.rayleighCenters), (codes % UInt8(Self.modulusLevels)).asType(.int32)) * sigma
        let rotatedPart = concatenated([r * cos(phase), r * sin(phase)], axis: -1).asType(dtype)
        guard let restPart = rest?.dequantizedRows() else { return rotatedPart }
        return concatenated([rotatedPart, restPart.asType(dtype)], axis: -1)
    }

    func scores(_ grouped: MLXArray) -> MLXArray? {
        guard let codes, let sigma,
              grouped.ndim == 5, grouped.dim(0) == codes.dim(0), grouped.dim(1) == codes.dim(1),
              grouped.dim(4) == keyDim else { return nil }
        let origType = grouped.dtype
        guard PolarFusedScores.canUseKernel(grouped: grouped, codes: codes, sigma: sigma, rotated: rotated, keyDim: keyDim) else {
            if Device.defaultDevice() == .gpu, (origType == .float64 || sigma.dtype == .float64) {
                return Device.withDefaultDevice(.cpu) { scores(grouped) }
            }
            let computeType: DType = origType == .float64
                ? .float64
                : (PolarFusedScores.supportedDTypes.contains(origType) ? origType : .float32)
            guard let keys = dequantizedRows(dtype: computeType) else { return nil }
            let query = grouped.dtype == computeType ? grouped : grouped.asType(computeType)
            return matmul(query, expandedDimensions(keys, axis: 2).swappedAxes(-1, -2))
        }
        var total = PolarFusedScores.run(
            groupedRot: grouped[.ellipsis, ..<rotated],
            codes: codes,
            sigma: sigma)
        if grouped.dim(-1) > rotated, let restParts = rest?.parts, let rest {
            let restScores = quantizedMM(
                grouped[.ellipsis, rotated...],
                expandedDimensions(restParts.0, axis: 2),
                scales: expandedDimensions(restParts.1.asType(origType), axis: 2),
                biases: restParts.2.map { expandedDimensions($0.asType(origType), axis: 2) },
                transpose: true,
                groupSize: rest.groupSize,
                bits: rest.bits)
            total = total + restScores
        }
        return total
    }
}

private enum PolarFusedScores {
    static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    static let phaseTrigTable: [Float] = (0 ..< PolarKeyRows.phaseLevels).flatMap { k in
        let angle = Float(k) * (2.0 * Float.pi / Float(PolarKeyRows.phaseLevels))
        return [Foundation.cos(angle), Foundation.sin(angle)]
    }

    static func canUseKernel(
        grouped: MLXArray, codes: MLXArray, sigma: MLXArray, rotated: Int, keyDim: Int
    ) -> Bool {
        guard Device.defaultDevice() == .gpu, supportedDTypes.contains(grouped.dtype) else { return false }
        guard grouped.ndim == 5, codes.ndim == 4, sigma.ndim == 4 else { return false }
        let pairs = rotated / 2
        let (b, hKV, tMid) = (grouped.dim(0), grouped.dim(1), codes.dim(2))
        guard pairs > 0, rotated.isMultiple(of: 2), grouped.dim(4) == keyDim, keyDim >= rotated else { return false }
        return grouped.size > 0 && tMid > 0
            && codes.shape == [b, hKV, tMid, pairs]
            && sigma.shape == [b, hKV, 1, pairs]
    }

    private static func floatTable(_ name: String, _ values: [Float]) -> String {
        let posix = Locale(identifier: "en_US_POSIX")
        let body = values.map { String(format: "%.9ef", locale: posix, Double($0)) }.joined(separator: ",")
        return "constant float \(name)[\(values.count)] = {\(body)};\n"
    }

    static let header = "#include <metal_simdgroup>\n"
        + floatTable("polar_trig", phaseTrigTable)
        + floatTable("polar_centers", PolarKeyRows.rayleighCenters)

    static func run(groupedRot: MLXArray, codes: MLXArray, sigma: MLXArray) -> MLXArray {
        let origType = groupedRot.dtype
        let b = groupedRot.dim(0)
        let hKV = groupedRot.dim(1)
        let gQ = groupedRot.dim(2)
        let lQ = groupedRot.dim(3)
        let rot = groupedRot.dim(4)
        let pairs = rot / 2
        let tMid = codes.dim(2)
        let bhCount = b * hKV
        let qRows = gQ * lQ
        let tasks = bhCount * qRows
        let tGroups = (tMid + 3) / 4
        let sigmaClean = supportedDTypes.contains(sigma.dtype)
            ? sigma : sigma.asType(.float32, stream: sigma.dtype == .float64 ? .cpu : .default)
        let shapes = MLXArray([Int32(bhCount), Int32(qRows), Int32(tMid), Int32(pairs)])
        let out = polarScoresKernel(
            [
                groupedRot.reshaped(tasks, rot),
                codes.reshaped(bhCount, tMid, pairs),
                sigmaClean.reshaped(bhCount, pairs),
                shapes,
            ],
            template: [("T", origType), ("TS", sigmaClean.dtype)],
            grid: (32, tGroups * 4, tasks),
            threadGroup: (32, 4, 1),
            outputShapes: [[tasks, tMid]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(b, hKV, gQ, lQ, tMid)
    }

    static let source = """
        const int QROWS = int(shapes[1]);
        const int TMID = int(shapes[2]);
        const int PAIRS = int(shapes[3]);
        const int ROT = PAIRS * 2;
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int t = int(threadgroup_position_in_grid.y) * 4 + sg;
        const int task = int(threadgroup_position_in_grid.z);
        if (t >= TMID) { return; }
        const int bh = task / QROWS;
        const size_t q_base = size_t(task) * size_t(ROT);
        const size_t c_base = (size_t(bh) * size_t(TMID) + size_t(t)) * size_t(PAIRS);
        const size_t s_base = size_t(bh) * size_t(PAIRS);
        float acc = 0.0f;
        for (int l = lane; l < PAIRS; l += 32) {
            const uint code = uint(codes[c_base + size_t(l)]);
            const uint ph = (code >> 3u) & 31u;
            const uint mo = code & 7u;
            const float r = polar_centers[mo] * float(sigma[s_base + size_t(l)]);
            const float k_re = r * polar_trig[ph * 2u];
            const float k_im = r * polar_trig[ph * 2u + 1u];
            const float q_re = float(q_rot[q_base + size_t(l)]);
            const float q_im = float(q_rot[q_base + size_t(PAIRS + l)]);
            acc = fma(q_im, k_im, fma(q_re, k_re, acc));
        }
        const float total = simd_sum(acc);
        if (lane == 0) {
            out[size_t(task) * size_t(TMID) + size_t(t)] = static_cast<T>(total);
        }
        """
}

private let polarScoresKernel = MLXFast.metalKernel(
    name: "polar_middle_scores_fused",
    inputNames: ["q_rot", "codes", "sigma", "shapes"],
    outputNames: ["out"],
    source: PolarFusedScores.source,
    header: PolarFusedScores.header
)

enum PolarGeometry {
    static let halfSplitRoPE: Set<String> = ["qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_next", "lfm2", "lfm2_moe"]

    static func rotatedDims(config: [String: Any]) -> Int? {
        let text = config["text_config"] as? [String: Any] ?? config
        guard let type = (text["model_type"] ?? config["model_type"]) as? String, halfSplitRoPE.contains(type) else { return nil }
        let parameters = text["rope_parameters"] as? [String: Any]
        let factor = (parameters?["partial_rotary_factor"] ?? text["partial_rotary_factor"]) as? Double ?? 1
        guard let heads = text["num_attention_heads"] as? Int, heads > 0 else { return nil }
        let headDim = text["head_dim"] as? Int ?? ((text["hidden_size"] as? Int ?? 0) / heads)
        let rotated = Int((Double(headDim) * factor).rounded())
        return rotated > 0 && rotated.isMultiple(of: 2) && rotated <= headDim ? rotated : nil
    }
}
