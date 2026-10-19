import Foundation

public struct KVCacheGeometry: Sendable, Equatable {
    public let elementsPerToken: Double
    public let windowedElementsPerToken: Double
    public let window: Int
    public let fixedStateBytes: Int64
    public let isFromConfig: Bool

    public static let activationBytes = 2.0
    public static let quantizedGroupSize = 64
    static let recurrentStateBytes = 4.0
    static let proxyGBPer4kPerBillion = 0.034
    static let proxyGBPer4kFloor = 0.01
    static let proxyParameterCap = 32.0

    public static func kvBytesPerToken(layers: Int, kvHeads: Int, headDim: Int, bytesPerElement: Double) -> Double {
        2 * Double(layers) * Double(kvHeads) * Double(headDim) * bytesPerElement
    }

    public static func bytesPerElement(kvBits: Int?) -> Double {
        guard let kvBits else { return activationBytes }
        return Double(kvBits) / 8 + 2 * activationBytes / Double(quantizedGroupSize)
    }

    public init(parameterProxy parameters: Double) {
        let gbPer4k = min(Self.proxyParameterCap, parameters) * Self.proxyGBPer4kPerBillion + Self.proxyGBPer4kFloor
        elementsPerToken = gbPer4k * 1e9 / 4096 / Self.activationBytes
        windowedElementsPerToken = 0
        window = 0
        fixedStateBytes = 0
        isFromConfig = false
    }

    public init?(config root: [String: Any]) {
        let c = root["text_config"] as? [String: Any] ?? root
        guard let layers = c["num_hidden_layers"] as? Int, layers > 0,
              let heads = c["num_attention_heads"] as? Int, heads > 0 else { return nil }
        let kvHeads = c["num_key_value_heads"] as? Int ?? heads
        guard let headDim = c["head_dim"] as? Int ?? (c["hidden_size"] as? Int).map({ $0 / heads }),
              headDim > 0, kvHeads > 0 else { return nil }

        let kinds = Self.layerKinds(c, layers: layers)
        let full = kinds.filter { $0 == .full }.count
        let windowedLayers = kinds.filter { $0 == .windowed }.count
        let linear = kinds.filter { $0 == .linear }.count
        let perLayer = Self.kvBytesPerToken(layers: 1, kvHeads: kvHeads, headDim: headDim, bytesPerElement: 1)

        elementsPerToken = Double(full) * perLayer
        windowedElementsPerToken = Double(windowedLayers) * perLayer
        window = c["sliding_window"] as? Int ?? 0
        fixedStateBytes = Int64(linear) * Self.linearStateBytes(c)
        isFromConfig = true
    }

    public static func load(modelDirectory: URL) -> KVCacheGeometry? {
        guard let data = try? Data(contentsOf: modelDirectory.appendingPathComponent("config.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return KVCacheGeometry(config: json)
    }

    public static func forModel(_ repoIdOrPath: String, parameters: Double) -> KVCacheGeometry {
        ModelCache.resolve(repoId: repoIdOrPath).flatMap(load(modelDirectory:))
            ?? KVCacheGeometry(parameterProxy: parameters)
    }

    public func bytes(tokens: Int, kvBits: Int? = nil) -> Int64 {
        let t = Double(max(0, tokens))
        let windowed = window > 0 ? min(t, Double(window)) : t
        let elements = elementsPerToken * t + windowedElementsPerToken * windowed
        return Int64(elements * Self.bytesPerElement(kvBits: kvBits)) + fixedStateBytes
    }

    public func maxTokens(fitting budget: Int64, kvBits: Int? = nil) -> Int {
        let room = Double(budget - fixedStateBytes)
        guard room > 0 else { return 0 }
        let b = Self.bytesPerElement(kvBits: kvBits)
        let growing = elementsPerToken * b
        let windowed = windowedElementsPerToken * b
        let tokens: Double
        if window > 0, windowed > 0, room >= (growing + windowed) * Double(window) {
            tokens = growing > 0 ? Double(window) + (room - (growing + windowed) * Double(window)) / growing : .infinity
        } else {
            tokens = growing + windowed > 0 ? room / (growing + windowed) : .infinity
        }
        return Int(min(tokens, Double(Int32.max)))
    }

    enum LayerKind { case full, windowed, linear }

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
