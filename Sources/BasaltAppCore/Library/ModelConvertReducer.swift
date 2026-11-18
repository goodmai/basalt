import BasaltCore
import Foundation

public enum ConversionTargetFamily: String, Sendable, CaseIterable, Identifiable {
    case macaq
    case macaqMoE
    case flatQuant
    case lirQuant
    case affine
    case mxfp4
    case nf4
    case qtip
    case qtipBudget
    case gsqRCO
    case dense

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .macaq: "MACaQ (Hadamard + Flip)"
        case .macaqMoE: "MACaQ MoE (Router-Weighted)"
        case .flatQuant: "FlatQuant (Kronecker P1⊗P2)"
        case .lirQuant: "LIRQuant (Givens Subspace)"
        case .affine: "Standard Affine"
        case .mxfp4: "OCP MXFP4"
        case .nf4: "NF4 NormalFloat"
        case .qtip: "QTIP Trellis"
        case .qtipBudget: "QTIP Mixed Budget"
        case .gsqRCO: "GSQ-RCO Mixed"
        case .dense: "Dense (BF16 / FP16)"
        }
    }
}

public struct ConversionDraft: Sendable, Equatable {
    public var sourceId: String
    public var sourceBytes: Int64
    public var family: ConversionTargetFamily = .macaq
    public var bits: Int = 4
    public var groupSize: Int = 64
    public var useMXFP4InMACaQ: Bool = false
    public var hadamardBlock: Int = 64
    public var lirSubspaceDim: Int = 128
    public var moeBudgetBPW: Double = 2.8
    public var qtipRate: Double = 2.5
    public var budgetBPW: Double = 3.0
    public var denseFP16: Bool = false
    public var recipe: String = "sinq"
    public var pvTune: Bool = true
    public var pvPreset: PVTunePreset = .balanced
    public var pvTeacher: String = ""
    public var outputId: String
    public var plan: ModelConversionPlan?

    public init(sourceId: String, sourceBytes: Int64) {
        self.sourceId = sourceId
        self.sourceBytes = sourceBytes
        let baseName = sourceId.split(separator: "/").last.map(String.init) ?? "model"
        self.outputId = "local/\(baseName)-MACaQ-4bit-PV"
    }

    public var resolvedTarget: QuantizationTarget {
        switch family {
        case .macaq:
            useMXFP4InMACaQ ? .macaqMXFP4 : .macaqAffine(bits: min(max(bits, 2), 4), groupSize: groupSize)
        case .macaqMoE:
            .macaqMoEBudget(bpw: moeBudgetBPW)
        case .flatQuant:
            .flatQuant(bits: [2, 3, 4, 8].contains(bits) ? bits : 4, groupSize: groupSize)
        case .lirQuant:
            .lirQuant(
                bits: [2, 3, 4, 8].contains(bits) ? bits : 4,
                groupSize: groupSize,
                subspaceDim: [32, 64, 128, 256].contains(lirSubspaceDim) ? lirSubspaceDim : 128)
        case .affine:
            .affine(bits: bits, groupSize: groupSize)
        case .mxfp4:
            .mxfp4
        case .nf4:
            .nf4(groupSize: groupSize)
        case .qtip:
            .qtip(bpw: qtipRate)
        case .qtipBudget:
            .qtipBudget(bpw: budgetBPW)
        case .gsqRCO:
            .gsqRCO(bpw: budgetBPW)
        case .dense:
            denseFP16 ? .denseFP16 : .denseBF16
        }
    }

    public mutating func refreshSuggestedOutputId() {
        let rawBase = sourceId.split(separator: "/").last.map(String.init) ?? "model"
        let stripped = rawBase
            .replacingOccurrences(of: "-4bit", with: "")
            .replacingOccurrences(of: "-8bit", with: "")
            .replacingOccurrences(of: "-bf16", with: "")
        let suffix = (pvTune && resolvedTarget.supportsPVTune) ? "-PV" : ""
        outputId = "local/\(stripped)-\(resolvedTarget.suggestedTag)\(suffix)"
    }

    public func toRequest() -> ModelConversionRequest {
        let target = resolvedTarget
        let effectiveRecipe = family == .macaqMoE ? "moe-rw" : recipe
        var req = ModelConversionRequest(
            source: sourceId,
            outputId: outputId.trimmingCharacters(in: .whitespacesAndNewlines),
            target: target,
            recipe: effectiveRecipe,
            pvTune: pvTune && target.supportsPVTune,
            stream: true)
        req.hadamardBlock = (family == .macaq || family == .macaqMoE) ? hadamardBlock : nil
        req.pvPreset = pvPreset
        req.pvTeacher = pvTeacher.isEmpty ? nil : pvTeacher
        return req
    }
}

public struct ConversionJobState: Sendable, Equatable {
    public var sourceId: String
    public var outputId: String
    public var targetSpec: String
    public var phase: String = "starting"
    public var completed: Int = 0
    public var total: Int = 4
    public var message: String = "Preparing conversion…"
    public var valKLBefore: Double?
    public var valKLCurrent: Double?
    public var active: Bool = true
    public var error: String?

    public init(sourceId: String, outputId: String, targetSpec: String) {
        self.sourceId = sourceId
        self.outputId = outputId
        self.targetSpec = targetSpec
    }

    public var fraction: Double {
        guard total > 0 else { return 0 }
        return min(1.0, max(0.0, Double(completed) / Double(total)))
    }
}

extension LibraryReducer {
    static func reduceConversion(_ state: inout LibraryState, _ action: LibraryAction) -> [LibraryEffect] {
        switch action {
        case .openConvert, .updateConvertDraft, .closeConvert, .conversionPlanned, .startConversion:
            return reduceDraftConversion(&state, action)
        case .conversionProgressed, .conversionCompleted, .conversionFailed, .cancelConversion:
            return reduceActiveConversion(&state, action)
        default:
            return []
        }
    }

    private static func reduceDraftConversion(_ state: inout LibraryState, _ action: LibraryAction) -> [LibraryEffect] {
        switch action {
        case .openConvert(let model):
            let draft = ConversionDraft(sourceId: model.id, sourceBytes: model.sizeBytes)
            state.convertDraft = draft
            let target = draft.resolvedTarget
            return [.planConversion(source: draft.sourceId, target: target, budgetGB: nil, pvTune: draft.pvTune && target.supportsPVTune)]
        case .updateConvertDraft(let next):
            let prev = state.convertDraft
            state.convertDraft = next
            guard prev?.resolvedTarget != next.resolvedTarget || prev?.pvTune != next.pvTune else { return [] }
            let target = next.resolvedTarget
            return [.planConversion(source: next.sourceId, target: target, budgetGB: nil, pvTune: next.pvTune && target.supportsPVTune)]
        case .closeConvert:
            state.convertDraft = nil
            return []
        case .conversionPlanned(let plan):
            if state.convertDraft?.sourceId == plan.sourceId { state.convertDraft?.plan = plan }
            return []
        case .startConversion:
            return startConversionFromDraft(&state)
        default:
            return []
        }
    }

    private static func startConversionFromDraft(_ state: inout LibraryState) -> [LibraryEffect] {
        guard let draft = state.convertDraft else { return [] }
        let req = draft.toRequest()
        guard isRepoID(req.outputId) else {
            state.problem = "\(req.outputId) must be an org/name repo id such as local/model-4bit"
            return []
        }
        guard state.conversions.values.allSatisfy({ !$0.active }) else {
            state.problem = "Another conversion is currently running"
            return []
        }
        state.conversions[req.outputId] = ConversionJobState(
            sourceId: req.source, outputId: req.outputId, targetSpec: req.target.specString)
        state.convertDraft = nil
        state.problem = nil
        return [.convert(req)]
    }

    private static func reduceActiveConversion(_ state: inout LibraryState, _ action: LibraryAction) -> [LibraryEffect] {
        switch action {
        case .conversionProgressed(let id, let prog):
            state.conversions[id]?.phase = prog.phase
            state.conversions[id]?.completed = prog.completed
            state.conversions[id]?.total = prog.total
            state.conversions[id]?.message = prog.message
            if let kl0 = prog.valKLBefore { state.conversions[id]?.valKLBefore = kl0 }
            if let kl1 = prog.valKLCurrent { state.conversions[id]?.valKLCurrent = kl1 }
            return []
        case .conversionCompleted(let id, _):
            state.conversions[id] = nil
            return [.list, .fit(id)]
        case .conversionFailed(let id, let msg):
            state.conversions[id]?.active = false
            state.conversions[id]?.error = msg
            return []
        case .cancelConversion(let id):
            state.conversions[id] = nil
            return [.cancelConversion(id)]
        default:
            return []
        }
    }
}
