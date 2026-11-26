import Foundation

extension KVCacheGeometry {
    static func localLayerKinds(_ c: [String: Any], layers: Int) -> [LayerKind]? {
        if let local = c["local_layer_ids"] as? [Int], c["sliding_window_size"] is Int {
            let windowed = Set(local)
            return (0 ..< layers).map { windowed.contains($0) ? .windowed : .full }
        }
        if let group = c["layer_group_size"] as? Int, group > 0, c["kv_lora_rank"] is Int {
            let full = (layers / group) * group
            return (0 ..< layers).map { ($0 + 1).isMultiple(of: group) || $0 >= full ? .full : .linear }
        }
        return nil
    }

    static func bailingLinearStateBytes(_ c: [String: Any]) -> Int64 {
        guard c["layer_group_size"] is Int, c["kv_lora_rank"] is Int,
              let heads = c["num_attention_heads"] as? Int, heads > 0,
              let headDim = c["head_dim"] as? Int, headDim > 0 else { return 0 }
        let kernel = c["short_conv_kernel_size"] as? Int ?? c["conv_kernel_size"] as? Int ?? 4
        let conv = Double(3 * max(0, kernel - 1) * heads * headDim) * activationBytes
        let recurrent = Double(heads * headDim * headDim) * recurrentStateBytes
        return Int64(conv + recurrent)
    }

    static func visits(_ kinds: [LayerKind], _ c: [String: Any]) -> [LayerKind] {
        guard let loop = c["loop_config"] as? [String: Any], let start = loop["core_start"] as? Int, let end = loop["core_end"] as? Int,
              let passes = loop["num_passes"] as? Int, (2 ... 64).contains(passes), start >= 0, start < end, end <= kinds.count else { return kinds }
        return Array(kinds[..<start]) + Array(repeating: kinds[start ..< end], count: passes).flatMap { $0 } + Array(kinds[end...])
    }

    static func shortConvStateBytes(_ c: [String: Any], visits: Int, kvHeads: Int, headDim: Int) -> Int64 {
        guard let kernel = c["sconv_kernel_size"] as? Int, kernel > 1, let hidden = c["hidden_size"] as? Int else { return 0 }
        let channels = 2 * kvHeads * headDim + 2 * hidden
        return Int64(visits * (kernel - 1) * channels) * Int64(activationBytes)
    }
    init(recurrentFixedStateBytes bytes: Int64) {
        elementsPerToken = 0
        keyElementsPerToken = 0
        windowedElementsPerToken = 0
        window = 0
        fixedStateBytes = bytes
        quantizedGroupSize = Self.defaultQuantizedGroupSize
        isFromConfig = true
    }

    static func xlstmGeometry(_ c: [String: Any]) -> KVCacheGeometry? {
        guard (c["model_type"] as? String)?.lowercased() == "xlstm",
              let dims = xlstmDimensions(c) else { return nil }
        let slstmCount = Set((c["slstm_at"] as? [Int] ?? []).filter { (0 ..< dims.layers).contains($0) }).count
        let mlstmCount = dims.layers - slstmCount
        let qkHead = Int64(dims.qkDim / dims.heads)
        let vHead = Int64(dims.vDim / dims.heads)
        let mlstmLayer = Int64(dims.heads) * (qkHead * vHead + qkHead + 1) * Int64(recurrentStateBytes)
        let slstmLayer = Int64(4 * dims.hidden) * Int64(recurrentStateBytes)
        return KVCacheGeometry(recurrentFixedStateBytes: Int64(mlstmCount) * mlstmLayer + Int64(slstmCount) * slstmLayer)
    }

    private static func xlstmDimensions(_ c: [String: Any]) -> (layers: Int, hidden: Int, heads: Int, qkDim: Int, vDim: Int)? {
        guard let layers = intField(c, "num_blocks", "num_hidden_layers"),
              let hidden = intField(c, "embedding_dim", "hidden_size"),
              let heads = intField(c, "num_heads", "num_attention_heads") else { return nil }
        let qkFactor = doubleField(c["qk_dim_factor"]) ?? 0.5
        let vFactor = doubleField(c["v_dim_factor"]) ?? 1.0
        guard qkFactor.isFinite && qkFactor > 0 && vFactor.isFinite && vFactor > 0 else { return nil }
        let (qkDim, vDim) = (Int(Double(hidden) * qkFactor), Int(Double(hidden) * vFactor))
        guard qkDim >= heads && vDim >= heads && qkDim.isMultiple(of: heads) && vDim.isMultiple(of: heads) else { return nil }
        return (layers, hidden, heads, qkDim, vDim)
    }

    private static func intField(_ c: [String: Any], _ first: String, _ second: String) -> Int? {
        guard let value = (c[first] as? Int) ?? (c[second] as? Int), (1 ... Self.maxDimension).contains(value) else { return nil }
        return value
    }

    private static func doubleField(_ raw: Any?) -> Double? {
        if let value = raw as? Double { return value }
        if let value = raw as? Int { return Double(value) }
        return (raw as? NSNumber)?.doubleValue
    }

    static func recurrentModelGeometry(_ c: [String: Any]) -> KVCacheGeometry? {
        guard let modelType = (c["model_type"] as? String)?.lowercased() else { return nil }
        let isMamba3 = Mamba3Configuration.supportedModelTypes.contains(modelType)
        let isGDN2 = GatedDeltaNet2Configuration.supportedModelTypes.contains(modelType)
        guard isMamba3 || isGDN2,
              JSONSerialization.isValidJSONObject(c),
              let data = try? JSONSerialization.data(withJSONObject: c)
        else { return nil }
        if isMamba3 {
            guard let cfg = try? JSONDecoder().decode(Mamba3Configuration.self, from: data).validated() else { return nil }
            return KVCacheGeometry(recurrentFixedStateBytes: cfg.recurrentStateFootprint().bytes.rawValue)
        }
        guard let cfg = try? JSONDecoder().decode(GatedDeltaNet2Configuration.self, from: data).validated() else { return nil }
            let sliding = cfg.slidingAttentionLayerIndices.count
            let full = max(0, cfg.attentionLayerCount - sliding)
            let kvDim = cfg.numKeyValueHeads * cfg.attentionHeadDim
            let groupSize = quantizedGroupSize(keyDim: cfg.attentionHeadDim, valueDim: cfg.attentionHeadDim) ?? defaultQuantizedGroupSize
            return KVCacheGeometry(
                elementsPerToken: Double(full * 2 * kvDim),
                keyElementsPerToken: Double(full * kvDim),
                windowedElementsPerToken: kvBytesPerToken(
                    layers: sliding,
                    kvHeads: cfg.numKeyValueHeads,
                    headDim: cfg.attentionHeadDim,
                    bytesPerElement: 1
                ),
                window: cfg.slidingWindow ?? 0,
                fixedStateBytes: cfg.recurrentStateFootprint(activationBytes: Int(activationBytes)).bytes.rawValue,
                quantizedGroupSize: groupSize
            )
    }
}
