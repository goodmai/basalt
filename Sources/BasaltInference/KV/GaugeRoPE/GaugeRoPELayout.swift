import Foundation

struct GaugeRoPELayout: Sendable, Equatable {
    static let defaultCalibrationTokens = 512
    static let defaultCarrierRank = 2
    static let qwen35PartialRotaryFactor = 0.25

    let rotaryDims: Int
    let ropeBase: Float
    let calibrationTokens: Int
    let carrierRank: Int
    let codeRank: Int?

    init(
        rotaryDims: Int, ropeBase: Float, calibrationTokens: Int = GaugeRoPELayout.defaultCalibrationTokens,
        carrierRank: Int = GaugeRoPELayout.defaultCarrierRank, codeRank: Int? = nil
    ) {
        self.rotaryDims = rotaryDims
        self.ropeBase = ropeBase
        self.calibrationTokens = max(1, calibrationTokens)
        self.carrierRank = carrierRank
        self.codeRank = codeRank
    }

    init?(config root: [String: Any]) {
        let text = root["text_config"] as? [String: Any] ?? root
        let params = text["rope_parameters"] as? [String: Any] ?? text
        let kind = params["rope_type"] as? String ?? params["type"] as? String ?? "default"
        guard kind == "default", let theta = params["rope_theta"] as? NSNumber, theta.floatValue > 0,
              let heads = text["num_attention_heads"] as? Int, heads > 0,
              let headDim = text["head_dim"] as? Int ?? (text["hidden_size"] as? Int).map({ $0 / heads }), headDim > 0 else { return nil }
        let factor = (params["partial_rotary_factor"] as? NSNumber ?? text["partial_rotary_factor"] as? NSNumber)?.doubleValue
        let rotary = Int(Double(headDim) * (factor ?? Self.qwen35PartialRotaryFactor))
        guard rotary >= 2, rotary.isMultiple(of: 2), rotary <= headDim else { return nil }
        self.init(rotaryDims: rotary, ropeBase: theta.floatValue)
    }
}
