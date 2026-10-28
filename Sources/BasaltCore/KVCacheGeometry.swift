import Foundation
import Synchronization

public struct KVPrecision: Sendable, Hashable, CustomStringConvertible {
    public let keyBits: Int
    public let valueBits: Int
    public let rotatesValues: Bool

    public static let q8 = KVPrecision(bits: 8)
    public static let q4 = KVPrecision(bits: 4)
    public static let k8v4 = KVPrecision(keyBits: 8, valueBits: 4)
    public static let k8v4h = KVPrecision(keyBits: 8, valueBits: 4, rotatesValues: true)
    public static let supportedBits = [2, 3, 4, 5, 6, 8]

    public init(keyBits: Int, valueBits: Int, rotatesValues: Bool = false) {
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.rotatesValues = rotatesValues
    }

    public init(bits: Int) {
        self.init(keyBits: bits, valueBits: bits)
    }

    public init?(argument: String) {
        if let named = [Self.k8v4, .k8v4h].first(where: { $0.description == argument }) {
            self = named
        } else if let bits = Int(argument), Self.supportedBits.contains(bits) {
            self.init(bits: bits)
        } else {
            return nil
        }
    }

    public var isUniform: Bool { keyBits == valueBits }

    public static var accepted: String {
        (supportedBits.map(String.init) + [k8v4, k8v4h].map(\.description)).joined(separator: ", ")
    }

    public static func flagConflict(kvBitsSet: Bool, fixed: KVPrecision?, zones: Bool) -> String? {
        if zones, !kvBitsSet {
            return "--kv-zones needs --kv-bits (one of \(accepted), or auto on serve and models check): it says which tokens stay fp16 in a quantized cache"
        }
        if let fixed, !fixed.isUniform, !zones {
            return "--kv-bits \(fixed) needs --kv-zones: 8-bit keys and 4-bit values are a precision of the zoned cache's quantized middle"
        }
        return nil
    }

    public var uniform: KVPrecision { isUniform ? self : KVPrecision(bits: max(keyBits, valueBits)) }

    public var description: String {
        (isUniform ? "\(keyBits)-bit" : "k\(keyBits)v\(valueBits)") + (rotatesValues ? "h" : "")
    }
}

public struct KVCacheGeometry: Sendable, Equatable {
    public let elementsPerToken: Double
    public let keyElementsPerToken: Double
    public let windowedElementsPerToken: Double
    public let window: Int
    public let fixedStateBytes: Int64
    public let quantizedGroupSize: Int
    public let isFromConfig: Bool

    public static let activationBytes = 2.0
    public static let defaultQuantizedGroupSize = 64
    public static let defaultQuantizedKVStart = 512
    public static let adaptiveKVPrecisionLadder: [KVPrecision?] = [nil, .q8, .k8v4, .q4]
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

    public func bytesPerToken(_ precision: KVPrecision?) -> Double {
        guard let precision else { return elementsPerToken * Self.activationBytes }
        let valueElements = elementsPerToken - keyElementsPerToken
        return keyElementsPerToken * Self.bytesPerElement(kvBits: precision.keyBits, groupSize: quantizedGroupSize)
            + valueElements * Self.bytesPerElement(kvBits: precision.valueBits, groupSize: quantizedGroupSize)
    }

    public init(parameterProxy parameters: Double) {
        let gbPer4k = min(Self.proxyParameterCap, parameters) * Self.proxyGBPer4kPerBillion + Self.proxyGBPer4kFloor
        elementsPerToken = gbPer4k * 1e9 / 4096 / Self.activationBytes
        keyElementsPerToken = elementsPerToken / 2
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
        keyElementsPerToken = Double(full * global.heads * global.keyDim)
        windowedElementsPerToken = Self.kvBytesPerToken(
            layers: windowedLayers, kvHeads: kvHeads, headDim: headDim, bytesPerElement: 1
        )
        window = c["sliding_window"] as? Int ?? 0
        fixedStateBytes = Int64(linear) * Self.linearStateBytes(c)
        quantizedGroupSize = Self.quantizedGroupSize(keyDim: global.keyDim, valueDim: global.valueDim)
            ?? Self.defaultQuantizedGroupSize
        isFromConfig = true
    }

    public static func quantizedGroupSize(keyDim: Int, valueDim: Int) -> Int? {
        supportedQuantizedGroupSizes
            .filter { keyDim.isMultiple(of: $0) && valueDim.isMultiple(of: $0) }
            .min { abs($0 - defaultQuantizedGroupSize) < abs($1 - defaultQuantizedGroupSize) }
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

    public func bytes(tokens: Int, kvPrecision: KVPrecision? = nil, zones: KVZones? = nil) -> Int64 {
        let t = Double(max(0, tokens))
        let windowed = window > 0 ? min(t, Double(window)) : t
        let exact = kvPrecision == nil ? t : min(t, Double(zones?.maxFullPrecisionTokens ?? 0))
        let growing = exact * bytesPerToken(nil) + (t - exact) * bytesPerToken(kvPrecision)
        return Int64(growing + windowedElementsPerToken * windowed * Self.activationBytes) + fixedStateBytes
    }

    public func conversionPeakBytes(tokens: Int, kvPrecision: KVPrecision?, zones: KVZones? = nil) -> Int64 {
        guard kvPrecision != nil, zones == nil else { return bytes(tokens: tokens, kvPrecision: kvPrecision, zones: zones) }
        let quantizedCopy = Double(max(0, tokens)) * bytesPerToken(kvPrecision)
        return bytes(tokens: tokens) + Int64(quantizedCopy)
    }

    public func maxPromptTokens(fitting budget: Int64, kvPrecision: KVPrecision?, zones: KVZones? = nil) -> Int {
        let unquantized = maxTokens(fitting: budget)
        guard kvPrecision != nil else { return unquantized }
        if zones != nil { return maxTokens(fitting: budget, kvPrecision: kvPrecision, zones: zones) }
        var fits = 0
        var tooMany = unquantized + 1
        while tooMany - fits > 1 {
            let mid = fits + (tooMany - fits) / 2
            if conversionPeakBytes(tokens: mid, kvPrecision: kvPrecision) <= budget { fits = mid } else { tooMany = mid }
        }
        return fits
    }

    public func maxTokens(
        fitting budget: Int64, kvPrecision: KVPrecision?, promptTokens: Int, quantizedKVStart: Int = defaultQuantizedKVStart,
        zones: KVZones? = nil
    ) -> Int {
        if zones != nil { return maxTokens(fitting: budget, kvPrecision: kvPrecision, zones: zones) }
        let unquantized = maxTokens(fitting: budget)
        guard kvPrecision != nil, unquantized > quantizedKVStart else { return unquantized }
        let convertedAt = max(promptTokens, quantizedKVStart + 1)
        guard conversionPeakBytes(tokens: convertedAt, kvPrecision: kvPrecision) <= budget else { return quantizedKVStart }
        return maxTokens(fitting: budget, kvPrecision: kvPrecision)
    }

    public func maxTokens(fitting budget: Int64, kvPrecision: KVPrecision? = nil, zones: KVZones?) -> Int {
        let uniform = maxTokens(fitting: budget, kvPrecision: kvPrecision)
        guard kvPrecision != nil, let zones, uniform < Int(Int32.max) else { return uniform }
        var fits = min(uniform, maxTokens(fitting: budget))
        var tooMany = uniform + 1
        while tooMany - fits > 1 {
            let mid = fits + (tooMany - fits) / 2
            if bytes(tokens: mid, kvPrecision: kvPrecision, zones: zones) <= budget { fits = mid } else { tooMany = mid }
        }
        return fits
    }

    public func maxTokens(fitting budget: Int64, kvPrecision: KVPrecision? = nil) -> Int {
        let room = Double(budget - fixedStateBytes)
        guard room > 0 else { return 0 }
        let growing = bytesPerToken(kvPrecision)
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

public struct KVZones: Sendable, Equatable {
    public let sinkTokens: Int
    public let windowTokens: Int
    public let migrationStep: Int

    public static let standard = KVZones(sinkTokens: 64, windowTokens: 1024, migrationStep: 256)

    public init(sinkTokens: Int, windowTokens: Int, migrationStep: Int) {
        precondition(sinkTokens >= 0 && windowTokens >= 0 && migrationStep > 0, "KV zones need non-negative sizes and a positive migration step")
        self.sinkTokens = sinkTokens
        self.windowTokens = windowTokens
        self.migrationStep = migrationStep
    }

    public var maxFullPrecisionTokens: Int { sinkTokens + windowTokens + migrationStep - 1 }
}
