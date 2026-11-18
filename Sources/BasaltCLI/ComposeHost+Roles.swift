import Foundation
import BasaltCore
import BasaltInference
import BasaltServer
import BasaltSpeech

extension ComposeHost {
    func loadDecider(_ name: String, source: String, model: ComposeSpec.Model) async throws -> ComposeHeld {
        let onDemand = model.keep == .onDemand
        await events.emit(onDemand ? "registered" : "loading", model: name, source)
        guard onDemand else {
            let loaded = try await services.decider(source)
            await events.emit("ready", model: name, source)
            return .decision(loaded)
        }
        let path = try await services.resolve(source)
        let ane = CoreAIDecisionEngine.isBundle(URL(fileURLWithPath: path))
        let decider = services.decider
        return .onDemandDecision(OnDemandDecision(
            modelID: source, path: path, idle: .seconds(model.idleUnload ?? ComposeSpec.defaultIdleUnload),
            runsBesideDecode: ane, loader: { try await decider(source) }))
    }

    func loadSpeaker(_ name: String, source: String, model: ComposeSpec.Model) async throws -> ComposeHeld {
        let onDemand = model.keep == .onDemand
        let path = try await services.resolve(source)
        await events.emit(onDemand ? "registered" : "loading", model: name, source)
        guard onDemand else {
            let loaded = try await services.speaker(source, path)
            await events.emit("ready", model: name, source)
            return .speaker(loaded)
        }
        let directory = URL(fileURLWithPath: path)
        let inspected: (candidate: SpeechCandidate, voices: [VoiceSpec])
        if FileManager.default.fileExists(atPath: directory.path) {
            inspected = try SpeechSynthesizerLoader.inspectOnDemand(identifier: source, directory: directory)
        } else {
            let family = try TTSModelFamily.validateSupported(identifier: source)
            inspected = (
                SpeechCandidate(id: source, model: source, compute: .mlx, isAvailable: true, estimatedTTFAMs: family.estimatedTTFAMs),
                []
            )
        }
        let speaker = services.speaker
        let descriptor = OnDemandSpeechSynthesizer.Descriptor(
            modelID: source,
            path: path,
            candidate: inspected.candidate,
            declaredVoices: inspected.voices,
            idle: .seconds(model.idleUnload ?? ComposeSpec.defaultIdleUnload)
        )
        return .onDemandSpeaker(
            OnDemandSpeechSynthesizer(descriptor: descriptor, loader: { try await speaker(source, path).synthesizer })
        )
    }

    func speechSynthesizer(for view: ComposeSpec, held: [String: ComposeHeld]) -> (any SpeechSynthesizer)? {
        guard let speak = view.speak, !speak.chain.isEmpty else { return nil }
        let tiers: [RoutedSpeechSynthesizer.Tier] = speak.chain.map { entry in
            if entry == ComposeSpec.appleSpeech {
                return RoutedSpeechSynthesizer.Tier(candidate: .appleSpeech, synthesizer: AppleSpeechSynthesizer())
            }
            switch held[entry] {
            case .speaker(let loaded):
                var candidate = loaded.candidate
                candidate.id = entry
                return RoutedSpeechSynthesizer.Tier(candidate: candidate, synthesizer: loaded.synthesizer)
            case .onDemandSpeaker(let onDemand):
                var candidate = onDemand.descriptor.candidate
                candidate.id = entry
                return RoutedSpeechSynthesizer.Tier(candidate: candidate, synthesizer: onDemand)
            default:
                let fallbackModel = view.local(entry) ?? entry
                let unavailable = SpeechCandidate(id: entry, model: fallbackModel, compute: .mlx, isAvailable: false)
                return RoutedSpeechSynthesizer.Tier(candidate: unavailable, synthesizer: nil)
            }
        }
        let gate = chat.gate
        let gpuBusyProbe: @Sendable () async -> Bool = {
            if await gate.inFlight > 0 { return true }
            return await gate.queued > 0
        }
        return RoutedSpeechSynthesizer(
            tiers: tiers,
            onGPUBusy: speak.onGPUBusy ?? .fallbackANE,
            maxTTFAMs: speak.maxTTFAMs,
            defaults: speak,
            gpuBusyProbe: gpuBusyProbe,
            memoryPressureProbe: services.memoryPressure
        )
    }
    func switchChat(_ diff: ComposeDiff, _ next: ComposeSpec) async throws {
        guard diff.chat != nil else { return }
        try await chat.loadModel(path: try await services.resolve(next.local(next.answer) ?? next.answer))
    }

    func ensureSelfDecider(_ spec: ComposeSpec) async throws {
        guard selfDecider == nil, spec.views.contains(where: { $0.spec.cascade.contains(spec.answer) }) else { return }
        selfDecider = try await services.decider("self")
    }

    func localDeciders(_ view: ComposeSpec, of spec: ComposeSpec, held: [String: ComposeHeld]) -> [String: any DecisionEngine] {
        view.cascade.reduce(into: [:]) { deciders, name in
            if name == spec.answer, let selfDecider { deciders[name] = selfDecider }
            if case .decision(let engine)? = held[name] { deciders[name] = engine }
            if case .onDemandDecision(let engine)? = held[name] { deciders[name] = engine }
        }
    }

    func restoreChat(_ path: String) async {
        let (loaded, current) = (await chat.isLoaded, await chat.modelPath)
        guard !loaded || current != path else { return }
        do {
            try await chat.loadModel(path: path)
        } catch {
            failures[spec.answer] = error.localizedDescription
        }
    }
}
