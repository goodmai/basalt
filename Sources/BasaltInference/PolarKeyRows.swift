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

    mutating func append(_ rows: MLXArray) {
        let re = rows[.ellipsis, ..<pairs].asType(.float32)
        let im = rows[.ellipsis, pairs ..< rotated].asType(.float32)
        let r = sqrt(re * re + im * im)
        let scale = sigma ?? maximum(sqrt((r * r).mean(axis: 2, keepDims: true) / 2), MLXArray(Float(1e-6)))
        sigma = scale
        let turn = (atan2(im, re) * (Float(Self.phaseLevels) / (2 * Float.pi))).round()
        let phase = MLX.remainder(turn + Float(Self.phaseLevels), Float(Self.phaseLevels)).asType(.uint8)
        let centers = MLXArray(Self.rayleighCenters)
        let modulus = argMin(abs((r / scale).expandedDimensions(axis: -1) - centers), axis: -1).asType(.uint8)
        let code = phase * UInt8(Self.modulusLevels) + modulus
        codes = codes.map { concatenated([$0, code], axis: 2) } ?? code
        if rows.dim(-1) > rotated { rest?.append(rows[.ellipsis, rotated...]) }
    }

    mutating func trim(_ n: Int) -> Int {
        let removed = min(max(0, n), count)
        guard removed > 0, let codes else { return 0 }
        let kept = count - removed
        self.codes = kept == 0 ? nil : codes[.ellipsis, ..<kept, 0...]
        _ = rest?.trim(removed)
        return removed
    }

    func dequantizedRows(dtype: DType) -> MLXArray? {
        guard let codes, let sigma else { return nil }
        let phase = (codes >> UInt8(Self.modulusBits)).asType(.float32) * (2 * Float.pi / Float(Self.phaseLevels))
        let r = take(MLXArray(Self.rayleighCenters), (codes % UInt8(Self.modulusLevels)).asType(.int32)) * sigma
        let rotatedPart = concatenated([r * cos(phase), r * sin(phase)], axis: -1).asType(dtype)
        guard let restPart = rest?.dequantizedRows() else { return rotatedPart }
        return concatenated([rotatedPart, restPart.asType(dtype)], axis: -1)
    }
}

enum PolarGeometry {
    static let halfSplitRoPE: Set<String> = ["qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "lfm2", "lfm2_moe"]

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
