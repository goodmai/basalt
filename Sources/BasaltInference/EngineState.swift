import MLXLMCommon
import BasaltCore

enum ModelEpochTag {}
typealias ModelEpoch = Tagged<ModelEpochTag, Int>

struct ModelFacts: Sendable {
    var profile: any ModelProfile = GenericProfile()
    var samplingDefaults = SamplingDefaults()
    var controlTokens = ControlTokenGuard.none
    var fingerprint: String?
    var description = ""
    var hasMTPDrafter = false
    var isHybrid: Bool?
    var hasRecurrentState: Bool?
    var routesExperts: Bool?
    var cacheLayout: [String]?
    var supportsMultiTokenForward: Bool?
    var attendsThroughZones = false
    var prefillsInChunks = false
    var valueRotationUnavailable = false
    var kvHeadMask: KVHeadMask?
    var questApplies = false
    var capabilities = ModelCapabilities.unrestricted
    var deltas: [String: AppliedDelta] = [:]
    var cacheFence = 0
}

struct LoadedModel: Sendable {
    let container: ModelContainer
    let epoch: ModelEpoch
    var facts: ModelFacts
}

enum EngineEvent {
    case loadStarted(path: String)
    case loadFinished(ModelEpoch, ModelContainer, ModelFacts)
    case loadFailed(ModelEpoch, BasaltError)
    case learned(ModelEpoch, (inout ModelFacts) -> Void)
    case unloaded
}

enum EngineState: Sendable {
    case unloaded(ModelEpoch)
    case loading(epoch: ModelEpoch, path: String)
    case ready(LoadedModel)
    case failed(epoch: ModelEpoch, error: BasaltError)

    var epoch: ModelEpoch {
        switch self {
        case .unloaded(let epoch), .loading(let epoch, _), .failed(let epoch, _): epoch
        case .ready(let model): model.epoch
        }
    }

    var loaded: LoadedModel? {
        if case .ready(let model) = self { model } else { nil }
    }

    func checkInvariants(after previous: EngineState) -> [String] {
        epoch < previous.epoch ? ["the epoch went back from \(previous.epoch) to \(epoch)"] : []
    }

    static func reduce(_ state: EngineState, _ event: EngineEvent) -> EngineState {
        switch event {
        case .loadStarted(let path):
            return .loading(epoch: state.epoch + 1, path: path)
        case .loadFinished(let epoch, let container, let facts):
            guard case .loading(epoch, _) = state else { return state }
            return .ready(LoadedModel(container: container, epoch: epoch, facts: facts))
        case .loadFailed(let epoch, let error):
            guard case .loading(epoch, _) = state else { return state }
            return .failed(epoch: epoch, error: error)
        case .learned(let epoch, let learn):
            guard case .ready(var model) = state, model.epoch == epoch else { return state }
            learn(&model.facts)
            return .ready(model)
        case .unloaded:
            return .unloaded(state.epoch + 1)
        }
    }
}

extension MLXInferenceEngine {
    var container: ModelContainer? { state.loaded?.container }
    var modelEpoch: ModelEpoch { state.epoch }
    var modelFacts: ModelFacts { state.loaded?.facts ?? ModelFacts() }
    var profile: any ModelProfile { modelFacts.profile }
    var samplingDefaults: SamplingDefaults { modelFacts.samplingDefaults }
    var controlTokens: ControlTokenGuard { modelFacts.controlTokens }
    var modelFingerprint: String? { modelFacts.fingerprint }
    var modelDescription: String { modelFacts.description }
    var hasMTPDrafter: Bool { modelFacts.hasMTPDrafter }
    var modelIsHybrid: Bool? { modelFacts.isHybrid }
    var modelHasRecurrentState: Bool? { modelFacts.hasRecurrentState }
    var modelAttendsThroughZones: Bool { modelFacts.attendsThroughZones }
    var modelPrefillsInChunks: Bool { modelFacts.prefillsInChunks }
    var valueRotationUnavailable: Bool { modelFacts.valueRotationUnavailable }
    var kvHeadMask: KVHeadMask? { modelFacts.kvHeadMask }
    var questApplies: Bool { modelFacts.questApplies }
}
