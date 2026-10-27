import Foundation
import Synchronization

public struct KVCacheGeometry: Sendable, Equatable {
    public let elementsPerToken: Double
    public let windowedElementsPerToken: Double
    public let window: Int
    public let fixedStateBytes: Int64
    public let quantizedGroupSize: Int
    public let isFromConfig: Bool

    public static let activationBytes = 2.0
    public static let defaultQuantizedGroupSize = 64
    public static let defaultQuantizedKVStart = 512
    public static let adaptiveKVBitsLadder: [Int?] = [nil, 8, 4]
    static let supportedQuantizedGroupSizes = [32, 64, 128]
    static let fromConfig = Mutex<[String: (modified: Date, geometry: KVCacheGeometry)]>([:])
    static let recurrentStateBytes = 4.0
    static let proxyGBPer4kPerBillion = 0.034
    static let proxyGBPer4kFloor = 0.01
    static let proxyParameterCap = 32.0

    public static func kvBytesPerToken(layers: Int, kvHeads: Int, headDim: Int, bytesPerElement: Double) -> Double {
        2 * Double(layers) * Double(kvHeads) * Double(headDim) * bytesPerElement
    }

    public static func bytesPerElement(kvBits: Int?, groupSize: Int = defaultQuantizedGroupSize) -> Double {
        guard let kvBits else { return activationBytes }
        return Double(kvBits) / 8 + 2 * activationBytes / Double(groupSize)
    }

    public func bytesPerElement(kvBits: Int?) -> Double {
        Self.bytesPerElement(kvBits: kvBits, groupSize: quantizedGroupSize)
    }

    public init(parameterProxy parameters: Double) {
        let gbPer4k = min(Self.proxyParameterCap, parameters) * Self.proxyGBPer4kPerBillion + Self.proxyGBPer4kFloor
        elementsPerToken = gbPer4k * 1e9 / 4096 / Self.activationBytes
        windowedElementsPerToken = 0
        window = 0
        fixedStateBytes = 0
        quantizedGroupSize = Self.defaultQuantizedGroupSize
        isFromConfig = false
    }

    public init?(config root: [String: Any]) {
        let c = root["text_config"] as? [String: Any] ?? root
        guard let layers = c["num_hidden_layers"] as? Int, layers > 0,
              let heads = c["num_attention_heads"] as? Int, heads > 0 else { return nil }
        let kvHeads = c["num_key_value_heads"] as? Int ?? heads
        guard let headDim = c["head_dim"] as? Int ?? (c["hidden_size"] as? Int).map({ $0 / heads }),
              headDim > 0, kvHeads > 0 else { return nil }

        let shared = min(layers, max(0, c["num_kv_shared_layers"] as? Int ?? 0))
        let kinds = Self.layerKinds(c, layers: layers).prefix(layers - shared)
        let full = kinds.filter { $0 == .full }.count
        let windowedLayers = kinds.filter { $0 == .windowed }.count
        let linear = kinds.filter { $0 == .linear }.count
        let global = Self.fullAttentionShape(c, heads: heads, kvHeads: kvHeads, headDim: headDim)

        elementsPerToken = Double(full * global.heads * (global.keyDim + global.valueDim))
        windowedElementsPerToken = Self.kvBytesPerToken(
            layers: windowedLayers, kvHeads: kvHeads, headDim: headDim, bytesPerElement: 1
        )
        window = c["sliding_window"] as? Int ?? 0
        fixedStateBytes = Int64(linear) * Self.linearStateBytes(c)
        let dims = [global.keyDim, global.valueDim]
        quantizedGroupSize = Self.supportedQuantizedGroupSizes
            .filter { g in dims.allSatisfy { $0.isMultiple(of: g) } }
            .min { abs($0 - Self.defaultQuantizedGroupSize) < abs($1 - Self.defaultQuantizedGroupSize) }
            ?? Self.defaultQuantizedGroupSize
        isFromConfig = true
    }

    static func fullAttentionShape(
        _ c: [String: Any], heads: Int, kvHeads: Int, headDim: Int
    ) -> (heads: Int, keyDim: Int, valueDim: Int) {
        if c["kv_lora_rank"] is Int, let nope = c["qk_nope_head_dim"] as? Int, let rope = c["qk_rope_head_dim"] as? Int {
            return (heads, nope + rope, c["v_head_dim"] as? Int ?? nope)
        }
        let dim = c["global_head_dim"] as? Int ?? headDim
        let globalHeads = c["attention_k_eq_v"] as? Bool == true ? c["num_global_key_value_heads"] as? Int : nil
        return (globalHeads ?? kvHeads, dim, dim)
    }

    public static func load(modelDirectory: URL) -> KVCacheGeometry? {
        guard let json = ModelCache.config(at: modelDirectory) else { return nil }
        return KVCacheGeometry(config: json)
    }

    public static func forModel(_ repoIdOrPath: String, parameters: Double) -> KVCacheGeometry {
        guard let dir = ModelCache.resolve(repoId: repoIdOrPath),
              let modified = try? dir.appendingPathComponent("config.json")
                .resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate else {
            return KVCacheGeometry(parameterProxy: parameters)
        }
        let key = dir.standardizedFileURL.path
        if let known = fromConfig.withLock({ $0[key] }), known.modified == modified { return known.geometry }
        guard let read = load(modelDirectory: dir) else { return KVCacheGeometry(parameterProxy: parameters) }
        fromConfig.withLock { $0[key] = (modified, read) }
        return read
    }

    public func bytes(tokens: Int, kvBits: Int? = nil) -> Int64 {
        let t = Double(max(0, tokens))
        let windowed = window > 0 ? min(t, Double(window)) : t
        let growing = elementsPerToken * t * bytesPerElement(kvBits: kvBits)
        return Int64(growing + windowedElementsPerToken * windowed * Self.activationBytes) + fixedStateBytes
    }

    public func conversionPeakBytes(tokens: Int, kvBits: Int?) -> Int64 {
        guard kvBits != nil else { return bytes(tokens: tokens) }
        let quantizedCopy = elementsPerToken * Double(max(0, tokens)) * bytesPerElement(kvBits: kvBits)
        return bytes(tokens: tokens) + Int64(quantizedCopy)
    }

    public func maxPromptTokens(fitting budget: Int64, kvBits: Int?) -> Int {
        let unquantized = maxTokens(fitting: budget)
        guard kvBits != nil else { return unquantized }
        var fits = 0
        var tooMany = unquantized + 1
        while tooMany - fits > 1 {
            let mid = fits + (tooMany - fits) / 2
            if conversionPeakBytes(tokens: mid, kvBits: kvBits) <= budget { fits = mid } else { tooMany = mid }
        }
        return fits
    }

    public func maxTokens(
        fitting budget: Int64, kvBits: Int?, promptTokens: Int, quantizedKVStart: Int = defaultQuantizedKVStart
    ) -> Int {
        let unquantized = maxTokens(fitting: budget)
        guard kvBits != nil, unquantized > quantizedKVStart else { return unquantized }
        let convertedAt = max(promptTokens, quantizedKVStart + 1)
        guard conversionPeakBytes(tokens: convertedAt, kvBits: kvBits) <= budget else { return quantizedKVStart }
        return maxTokens(fitting: budget, kvBits: kvBits)
    }

    public func maxTokens(fitting budget: Int64, kvBits: Int? = nil) -> Int {
        let room = Double(budget - fixedStateBytes)
        guard room > 0 else { return 0 }
        let growing = elementsPerToken * bytesPerElement(kvBits: kvBits)
        let windowed = windowedElementsPerToken * Self.activationBytes
        let tokens: Double
        if window > 0, windowed > 0, room >= (growing + windowed) * Double(window) {
            tokens = growing > 0 ? Double(window) + (room - (growing + windowed) * Double(window)) / growing : .infinity
        } else {
            tokens = growing + windowed > 0 ? room / (growing + windowed) : .infinity
        }
        return Int(min(tokens, Double(Int32.max)))
    }

    enum LayerKind { case full, windowed, linear, cacheless }

    static func layerKinds(_ c: [String: Any], layers: Int) -> [LayerKind] {
        if let types = c["layer_types"] as? [String], types.count == layers {
            return types.map { type in
                if type.contains("sliding") { return .windowed }
                if ["linear", "mamba", "conv"].contains(where: { type.contains($0) }) { return .linear }
                return .full
            }
        }
        if let interval = c["full_attention_interval"] as? Int, interval > 0 {
            return (0..<layers).map { ($0 + 1) % interval == 0 ? .full : .linear }
        }
        if let period = c["attn_layer_period"] as? Int, period > 0 {
            let offset = c["attn_layer_offset"] as? Int ?? 0
            return (0..<layers).map { $0 % period == offset ? .full : .linear }
        }
        if let blocks = c["hybrid_override_pattern"] as? String ?? (c["hybrid_override_pattern"] as? [String])?.joined() {
            let kinds: [LayerKind] = blocks.map { $0 == "*" ? .full : $0 == "M" ? .linear : .cacheless }
            return kinds + Array(repeating: .cacheless, count: max(0, layers - kinds.count))
        }
        if let pattern = c["sliding_window_pattern"] as? Int, pattern > 0, c["sliding_window"] is Int {
            return (0..<layers).map { ($0 + 1) % pattern == 0 ? .full : .windowed }
        }
        return Array(repeating: .full, count: layers)
    }

    static func linearStateBytes(_ c: [String: Any]) -> Int64 {
        guard let keyHeads = c["linear_num_key_heads"] as? Int,
              let valueHeads = c["linear_num_value_heads"] as? Int,
              let keyDim = c["linear_key_head_dim"] as? Int,
              let valueDim = c["linear_value_head_dim"] as? Int else { return 0 }
        let kernel = c["linear_conv_kernel_dim"] as? Int ?? 4
        let convChannels = 2 * keyHeads * keyDim + valueHeads * valueDim
        let conv = Double(max(0, kernel - 1) * convChannels) * activationBytes
        let recurrent = Double(valueHeads * valueDim * keyDim) * recurrentStateBytes
        return Int64(conv + recurrent)
    }
}

extension ModelDefinition {
    public var kvGeometry: KVCacheGeometry {
        KVCacheGeometry.forModel(id, parameters: totalParameters)
    }
}
