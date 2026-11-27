import Foundation

public struct RecurrentStateFootprint: Sendable, Codable, Equatable {
    public let recurrentLayers: Int
    public let bytesPerLayer: ByteCount
    public let totalBytes: ByteCount

    public var bytes: ByteCount { totalBytes }

    public init(recurrentLayers: Int, bytesPerLayer: ByteCount) {
        self.recurrentLayers = max(0, recurrentLayers)
        self.bytesPerLayer = max(.zero, bytesPerLayer)
        self.totalBytes = ByteCount(Int64(self.recurrentLayers) * self.bytesPerLayer.rawValue)
    }
}

public struct Mamba3Configuration: Codable, Sendable, Equatable {
    public static let modelType = "mamba3"
    public static let supportedModelTypes = ["mamba3", "mamba_3", "mamba-3"]

    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var expand: Int
    public var dInner: Int
    public var dState: Int
    public var headDim: Int
    public var nGroups: Int
    public var mimoRank: Int
    public var chunkSize: Int
    public var heavyTail: Bool
    public var ropeFraction: Float
    public var aFloor: Float
    public var isOutprojNorm: Bool
    public var useGatedMLP: Bool
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case expand
        case dInner = "d_inner"
        case dState = "d_state"
        case headDim = "head_dim"
        case nGroups = "n_groups"
        case mimoRank = "mimo_rank"
        case chunkSize = "chunk_size"
        case heavyTail = "heavy_tail"
        case ropeFraction = "rope_fraction"
        case aFloor = "a_floor"
        case isOutprojNorm = "is_outproj_norm"
        case useGatedMLP = "use_gated_mlp"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    private enum AliasKeys: String, CodingKey {
        case dModel = "d_model"
        case nLayer = "n_layer"
        case innerSize = "inner_size"
        case stateSize = "state_size"
        case ssmStateSize = "ssm_state_size"
        case headdim
        case mambaHeadDim = "mamba_head_dim"
        case ngroups
        case numGroups = "num_groups"
        case isMimo = "is_mimo"
        case aFloorUpper = "A_floor"
        case tieEmbeddings = "tie_embeddings"
        case dIntermediate = "d_intermediate"
        case useMLP = "use_mlp"
        case normEps = "norm_eps"
        case layerNormEpsilon = "layer_norm_epsilon"
        case ssmCfg = "ssm_cfg"
    }

    private struct SSMSubConfig: Decodable {
        let expand: Int?
        let dInner: Int?
        let dState: Int?
        let headDim: Int?
        let headdim: Int?
        let nGroups: Int?
        let ngroups: Int?
        let mimoRank: Int?
        let isMimo: Bool?
        let chunkSize: Int?
        let heavyTail: Bool?
        let ropeFraction: Float?
        let aFloor: Float?
        let aFloorUpper: Float?
        let isOutprojNorm: Bool?

        enum CodingKeys: String, CodingKey {
            case expand
            case dInner = "d_inner"
            case dState = "d_state"
            case headDim = "head_dim"
            case headdim
            case nGroups = "n_groups"
            case ngroups
            case mimoRank = "mimo_rank"
            case isMimo = "is_mimo"
            case chunkSize = "chunk_size"
            case heavyTail = "heavy_tail"
            case ropeFraction = "rope_fraction"
            case aFloor = "a_floor"
            case aFloorUpper = "A_floor"
            case isOutprojNorm = "is_outproj_norm"
        }
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let a = try decoder.container(keyedBy: AliasKeys.self)
        let ssm = try a.decodeIfPresent(SSMSubConfig.self, forKey: .ssmCfg)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? Self.modelType
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try Self.decodeRequiredInt(c, .hiddenSize, a, .dModel)
        numHiddenLayers = try Self.decodeRequiredInt(c, .numHiddenLayers, a, .nLayer)
        expand = try c.decodeIfPresent(Int.self, forKey: .expand) ?? ssm?.expand ?? 2
        dInner = try c.decodeIfPresent(Int.self, forKey: .dInner)
            ?? a.decodeIfPresent(Int.self, forKey: .innerSize)
            ?? ssm?.dInner
            ?? (hiddenSize * expand)
        dState = try c.decodeIfPresent(Int.self, forKey: .dState)
            ?? a.decodeIfPresent(Int.self, forKey: .stateSize)
            ?? a.decodeIfPresent(Int.self, forKey: .ssmStateSize)
            ?? ssm?.dState
            ?? 64
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? a.decodeIfPresent(Int.self, forKey: .headdim)
            ?? a.decodeIfPresent(Int.self, forKey: .mambaHeadDim)
            ?? ssm?.headDim
            ?? ssm?.headdim
            ?? 64
        nGroups = try c.decodeIfPresent(Int.self, forKey: .nGroups)
            ?? a.decodeIfPresent(Int.self, forKey: .ngroups)
            ?? a.decodeIfPresent(Int.self, forKey: .numGroups)
            ?? ssm?.nGroups
            ?? ssm?.ngroups
            ?? 1
        let rawMimoRank = try c.decodeIfPresent(Int.self, forKey: .mimoRank) ?? ssm?.mimoRank ?? 1
        let explicitMimo = try a.decodeIfPresent(Bool.self, forKey: .isMimo) ?? ssm?.isMimo
        mimoRank = (explicitMimo == false) ? 1 : rawMimoRank
        chunkSize = try c.decodeIfPresent(Int.self, forKey: .chunkSize) ?? ssm?.chunkSize ?? 64
        heavyTail = try c.decodeIfPresent(Bool.self, forKey: .heavyTail) ?? ssm?.heavyTail ?? true
        ropeFraction = try c.decodeIfPresent(Float.self, forKey: .ropeFraction) ?? ssm?.ropeFraction ?? 1.0
        aFloor = try c.decodeIfPresent(Float.self, forKey: .aFloor)
            ?? a.decodeIfPresent(Float.self, forKey: .aFloorUpper)
            ?? ssm?.aFloor ?? ssm?.aFloorUpper ?? 1e-4
        isOutprojNorm = try c.decodeIfPresent(Bool.self, forKey: .isOutprojNorm) ?? ssm?.isOutprojNorm ?? false
        let explicitIntermediate = try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
        let mambaIntermediate = try a.decodeIfPresent(Int.self, forKey: .dIntermediate)
        let roundedMambaIntermediate = mambaIntermediate.map { $0 > 0 ? (($0 + 127) / 128) * 128 : $0 }
        let decodedIntermediate = explicitIntermediate ?? roundedMambaIntermediate
        let explicitGatedMLP = try c.decodeIfPresent(Bool.self, forKey: .useGatedMLP)
            ?? a.decodeIfPresent(Bool.self, forKey: .useMLP)
        useGatedMLP = explicitGatedMLP ?? ((decodedIntermediate ?? 0) > 0)
        intermediateSize = decodedIntermediate ?? (useGatedMLP ? max(hiddenSize * 2, dInner) : 0)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps)
            ?? a.decodeIfPresent(Float.self, forKey: .normEps)
            ?? a.decodeIfPresent(Float.self, forKey: .layerNormEpsilon)
            ?? 1e-5
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? a.decodeIfPresent(Bool.self, forKey: .tieEmbeddings) ?? false
    }

    private static func decodeRequiredInt(
        _ primary: KeyedDecodingContainer<CodingKeys>,
        _ primaryKey: CodingKeys,
        _ alias: KeyedDecodingContainer<AliasKeys>,
        _ aliasKey: AliasKeys
    ) throws -> Int {
        if let value = try primary.decodeIfPresent(Int.self, forKey: primaryKey) {
            return value
        }
        if let fallback = try alias.decodeIfPresent(Int.self, forKey: aliasKey) {
            return fallback
        }
        return try primary.decode(Int.self, forKey: primaryKey)
    }
}

extension Mamba3Configuration {
    public var numHeads: Int {
        headDim > 0 ? dInner / headDim : 0
    }

    public var isMIMO: Bool {
        mimoRank > 1
    }

    public var ropeDim: Int {
        let clamped = min(max(Double(ropeFraction), 0.0), 1.0)
        return min(dState, 2 * (Int((Double(dState) * clamped).rounded(.down)) / 2))
    }

    public var numRopeAngles: Int {
        ropeDim / 2
    }

    public var inProjDim: Int {
        2 * dInner + 2 * nGroups * mimoRank * dState + 3 * numHeads + numRopeAngles
    }

    public var recurrentStateBytes: ByteCount {
        recurrentStateFootprint().bytes
    }

    public func recurrentStateFootprint(batchSize: Int = 1) -> RecurrentStateFootprint {
        let b = max(0, batchSize)
        let angleElements = numHeads * numRopeAngles
        let ssmElements = numHeads * headDim * dState
        let keyElements = numHeads * mimoRank * dState
        let valueElements = numHeads * mimoRank * headDim
        let bytesPerLayer = Int64(b) * 4 * Int64(angleElements + ssmElements + keyElements + valueElements)
        return RecurrentStateFootprint(recurrentLayers: numHiddenLayers, bytesPerLayer: ByteCount(bytesPerLayer))
    }

    public func validated() throws(BasaltError) -> Mamba3Configuration {
        guard vocabSize > 0, hiddenSize > 0, numHiddenLayers > 0, expand > 0, dInner > 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "mamba3 requires positive vocab_size, hidden_size, num_hidden_layers, expand and d_inner"
            )
        }
        guard dState > 0, dState.isMultiple(of: 2), headDim > 0, dInner.isMultiple(of: headDim) else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "mamba3 requires even positive d_state (\(dState)) and head_dim (\(headDim)) dividing d_inner (\(dInner))"
            )
        }
        guard nGroups > 0, numHeads > 0, numHeads.isMultiple(of: nGroups), mimoRank >= 1, chunkSize >= 1 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "mamba3 requires n_groups (\(nGroups)) dividing num_heads (\(numHeads)), mimo_rank >= 1 and chunk_size >= 1"
            )
        }
        guard rmsNormEps > 0, !useGatedMLP || intermediateSize > 0,
              ropeFraction.isFinite, ropeFraction >= 0, ropeFraction <= 1,
              aFloor.isFinite, aFloor >= 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "mamba3 requires positive rms_norm_eps, valid rope_fraction in [0, 1], non-negative a_floor, and positive intermediate_size when use_gated_mlp is enabled"
            )
        }
        return self
    }
}
