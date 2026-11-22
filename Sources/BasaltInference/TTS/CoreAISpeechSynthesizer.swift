import Foundation
import CoreAI
import CoreML
import BasaltCore

actor CoreMLSpeechRunner {
    private let model: MLModel
    private let temporaryCompiledURL: URL?

    init(compiledURL: URL, cleanupOnDeinit: Bool = false) throws(BasaltError) {
        self.temporaryCompiledURL = cleanupOnDeinit ? compiledURL : nil
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        do {
            model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } catch {
            if cleanupOnDeinit { try? FileManager.default.removeItem(at: compiledURL) }
            throw .modelConfigurationInvalid(path: compiledURL.path, reason: error.localizedDescription)
        }
    }

    deinit {
        if let url = temporaryCompiledURL {
            try? FileManager.default.removeItem(at: url)
        }
    }

    func synthesize(tokens: [Int32], length: Int, voiceIndex: Int, speed: Float) throws(BasaltError) -> [Float] {
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(shapedArray: MLShapedArray<Int32>(scalars: tokens, shape: [1, length])),
                "voice_id": MLFeatureValue(shapedArray: MLShapedArray<Int32>(scalars: [Int32(voiceIndex)], shape: [1])),
                "speed": MLFeatureValue(shapedArray: MLShapedArray<Float>(scalars: [speed], shape: [1])),
            ])
            let output = try model.prediction(from: provider)
            for key in ["audio", "waveform", "output"] {
                if let f32 = output.featureValue(for: key)?.shapedArrayValue(of: Float.self) {
                    return f32.withUnsafeShapedBufferPointer { ptr, _, _ in Array(ptr) }
                }
                if let f16 = output.featureValue(for: key)?.shapedArrayValue(of: Float16.self) {
                    return f16.withUnsafeShapedBufferPointer { ptr, _, _ in ptr.map(Float.init) }
                }
            }
            throw BasaltError.inferenceError("speech: CoreML model returned no audio or waveform feature")
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("speech: CoreML ANE synthesis failed: \(error.localizedDescription)")
        }
    }

    func decodeMel(flatMel: [Float], frames: Int, bins: Int) throws(BasaltError) -> [Float] {
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(shapedArray: MLShapedArray<Float>(scalars: flatMel, shape: [1, bins, frames])),
            ])
            let output = try model.prediction(from: provider)
            for key in ["audio", "waveform", "output"] {
                if let f32 = output.featureValue(for: key)?.shapedArrayValue(of: Float.self) {
                    return f32.withUnsafeShapedBufferPointer { ptr, _, _ in Array(ptr) }
                }
                if let f16 = output.featureValue(for: key)?.shapedArrayValue(of: Float16.self) {
                    return f16.withUnsafeShapedBufferPointer { ptr, _, _ in ptr.map(Float.init) }
                }
            }
            throw BasaltError.inferenceError("speech: CoreML vocoder returned no audio feature")
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("speech: CoreML ANE vocoder failed: \(error.localizedDescription)")
        }
    }
}

public struct CoreAISpeechSynthesizer: SpeechSynthesizer {
    public static let maxInFlight = 2
    public static let defaultEstimatedTTFAMs = 45

    private enum Runtime: Sendable {
        case coreAI([String: InferenceFunction])
        case coreML(CoreMLSpeechRunner)
        case syntheticHarmonic
    }

    public let bundle: CoreAISpeechBundle
    let admission = InFlightLimit(maxInFlight)
    private let runtime: Runtime

    public static func isBundle(_ directory: URL) -> Bool {
        CoreAISpeechBundle.isBundle(directory)
    }

    public static func validateBundle(_ directory: URL) throws(BasaltError) {
        _ = try CoreAISpeechBundle(directory: directory)
    }

    public static func makeFixture(
        modelID: String = CoreAISpeechBundle.defaultModelID,
        sampleRate: Int = SpeechRequest.defaultSampleRate
    ) throws(BasaltError) -> CoreAISpeechSynthesizer {
        let bundle = try CoreAISpeechBundle(
            manifest: [
                "model_id": modelID,
                "asset": "kokoro_synthetic",
                "sample_rate": sampleRate,
                "compute": SpeechComputeBackend.ane.rawValue,
                "voices": TTSVoiceCatalog.kokoroVoiceIDs,
                "synthetic": true,
            ],
            path: "memory://\(modelID)"
        )
        return CoreAISpeechSynthesizer(bundle: bundle, runtime: .syntheticHarmonic)
    }

    private init(bundle: CoreAISpeechBundle, runtime: Runtime) {
        self.bundle = bundle
        self.runtime = runtime
    }

    public init(directory: URL) async throws(BasaltError) {
        let parsed = try CoreAISpeechBundle(directory: directory)
        bundle = parsed
        let assetURL = parsed.resolvedAssetURL(in: directory)
        switch parsed.assetKind {
        case .coreAI:
            runtime = .coreAI(try await Self.loadCoreAIFunctions(bundle: parsed, assetURL: assetURL, basePath: directory.path))
        case .coreMLCompiled:
            runtime = .coreML(try CoreMLSpeechRunner(compiledURL: assetURL))
        case .coreMLPackage:
            let compiled: URL
            do {
                compiled = try await MLModel.compileModel(at: assetURL)
            } catch {
                throw .modelConfigurationInvalid(path: directory.path, reason: error.localizedDescription)
            }
            runtime = .coreML(try CoreMLSpeechRunner(compiledURL: compiled, cleanupOnDeinit: true))
        case .synthetic:
            runtime = .syntheticHarmonic
        }
    }

    public var voices: [VoiceSpec] {
        get async { bundle.voiceSpecs }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(
            selected: SpeechCandidate(
                id: bundle.modelID,
                model: bundle.modelID,
                compute: bundle.compute,
                isAvailable: true,
                estimatedTTFAMs: Self.defaultEstimatedTTFAMs
            )
        )
    }

    public func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        try request.validate()
        guard let voice = bundle.voice(for: request.voice, language: request.language) else {
            throw .invalidRequestStructure(details: "speech: voice '\(request.voice)' is not declared in \(bundle.modelID)")
        }
        guard await admission.enter() else {
            throw .serverBusy(retryAfterSeconds: 1)
        }
        let synth = self
        return AsyncThrowingStream { continuation in
            let task = Task {
                defer {
                    await withTaskCancellationShield {
                        await synth.admission.leave()
                    }
                }
                do {
                    try Task.checkCancellation()
                    let samples = try await synth.runUtterance(request: request, voiceIndex: voice.index)
                    let frameSize = max(1_200, synth.bundle.sampleRate / 10)
                    var cursor = 0
                    while cursor < samples.count {
                        try Task.checkCancellation()
                        let end = min(samples.count, cursor + frameSize)
                        continuation.yield(
                            AudioChunk(
                                samples: Array(samples[cursor ..< end]),
                                sampleRate: synth.bundle.sampleRate,
                                isFinal: end == samples.count
                            )
                        )
                        cursor = end
                    }
                    if samples.isEmpty {
                        continuation.yield(AudioChunk(samples: [], sampleRate: synth.bundle.sampleRate, isFinal: true))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

extension CoreAISpeechSynthesizer {
    private static func loadCoreAIFunctions(
        bundle: CoreAISpeechBundle,
        assetURL: URL,
        basePath: String
    ) async throws(BasaltError) -> [String: InferenceFunction] {
        do {
            let model = try await AIModel.specialize(
                contentsOf: assetURL,
                options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
            )
            var loadedFunctions: [String: InferenceFunction] = [:]
            for entry in bundle.functions {
                guard let loaded = try model.loadFunction(named: entry.name) else {
                    throw BasaltError.modelConfigurationInvalid(path: basePath, reason: "\(bundle.asset).aimodel has no function \(entry.name)")
                }
                loadedFunctions[entry.name] = loaded
            }
            return loadedFunctions
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelConfigurationInvalid(path: basePath, reason: error.localizedDescription)
        }
    }

    private func runUtterance(request: SpeechRequest, voiceIndex: Int) async throws(BasaltError) -> [Float] {
        let tokenCount = min(max(1, request.input.count), bundle.maxPhonemeTokens)
        let bucket = bundle.function(for: tokenCount)
        let tokens = bundle.encodeTokens(request.input, length: bucket.length)
        switch runtime {
        case .coreAI(let functions):
            guard let function = functions[bucket.name] else {
                throw .inferenceError("speech: Core AI bundle has no loaded function \(bucket.name)")
            }
            return try await runCoreAIFunction(function, name: bucket.name, tokens: tokens, length: bucket.length, voiceIndex: voiceIndex, speed: Float(request.speed))
        case .coreML(let runner):
            return try await runner.synthesize(tokens: tokens, length: bucket.length, voiceIndex: voiceIndex, speed: Float(request.speed))
        case .syntheticHarmonic:
            return Self.synthesizeHarmonicFormants(
                tokens: Array(tokens.prefix(tokenCount)),
                voiceIndex: voiceIndex,
                speed: request.speed,
                sampleRate: bundle.sampleRate
            )
        }
    }

    private func runCoreAIFunction(
        _ function: InferenceFunction,
        name: String,
        tokens: [Int32],
        length: Int,
        voiceIndex: Int,
        speed: Float
    ) async throws(BasaltError) -> [Float] {
        let inputs = Self.makeCoreAIInputs(tokens: tokens, length: length, voiceIndex: voiceIndex, speed: speed)
        do {
            var outputs = try await function.run(inputs: inputs)
            guard let audio = (outputs.remove("audio") ?? outputs.remove("waveform"))?.ndArray else {
                throw BasaltError.inferenceError("speech: Core AI function \(name) returned no audio tensor")
            }
            return try Self.extractWaveform(audio)
        } catch let error as BasaltError {
            throw error
        } catch is CancellationError {
            throw .inferenceError("speech synthesis was cancelled")
        } catch {
            throw .inferenceError("speech: Core AI synthesis failed: \(error.localizedDescription)")
        }
    }

    private static func makeCoreAIInputs(tokens: [Int32], length: Int, voiceIndex: Int, speed: Float) -> [String: NDArray] {
        var ids = NDArray(shape: [1, length], scalarType: .int32)
        var idView = ids.mutableView(as: Int32.self)
        idView.copyElements(from: tokens)

        var voice = NDArray(shape: [1], scalarType: .int32)
        var voiceView = voice.mutableView(as: Int32.self)
        voiceView.copyElements(from: [Int32(voiceIndex)])

        var rate = NDArray(shape: [1], scalarType: .float32)
        var rateView = rate.mutableView(as: Float.self)
        rateView.copyElements(from: [speed])

        return ["input_ids": ids, "voice_id": voice, "speed": rate]
    }

    static func extractWaveform(_ array: NDArray) throws(BasaltError) -> [Float] {
        if let float32 = array.view(as: Float.self).contiguousElements {
            return (0 ..< float32.count).map { float32[$0] }
        }
        if let float16 = array.view(as: Float16.self).contiguousElements {
            return (0 ..< float16.count).map { Float(float16[$0]) }
        }
        throw .inferenceError("speech: Core AI waveform output must be contiguous Float32 or Float16")
    }

    static func synthesizeHarmonicFormants(
        tokens: [Int32],
        voiceIndex: Int,
        speed: Double,
        sampleRate: Int
    ) -> [Float] {
        let samplesPerToken = max(120, Int((Double(sampleRate) * 0.035 / max(0.25, speed)).rounded()))
        let total = max(samplesPerToken, tokens.count * samplesPerToken)
        let baseF0 = 135.0 + Double(voiceIndex % 16) * 8.5
        var phase = 0.0
        var out: [Float] = []
        out.reserveCapacity(total)
        for i in 0 ..< total {
            let tok = tokens.isEmpty ? Int32(1) : tokens[min(tokens.count - 1, i / samplesPerToken)]
            let local = Double(i % samplesPerToken) / Double(samplesPerToken)
            let env = sin(.pi * local)
            let f0 = baseF0 + Double(tok % 29) * 3.2
            phase += 2.0 * .pi * f0 / Double(sampleRate)
            let h1 = 0.42 * sin(phase)
            let h2 = 0.24 * sin(2.0 * phase + 0.3)
            let h3 = 0.14 * sin(3.0 * phase + 0.7)
            out.append(Float(env * (h1 + h2 + h3)))
        }
        return out
    }
}

public struct ANEAcousticVocoder: Sendable {
    private enum Backend: Sendable {
        case coreAI(InferenceFunction)
        case coreML(CoreMLSpeechRunner)
        case harmonicFilterbank
    }

    public let sampleRate: Int
    public let hopLength: Int
    private let backend: Backend

    public static func makeSyntheticFixture(
        sampleRate: Int = SpeechRequest.defaultSampleRate,
        hopLength: Int = 240
    ) -> ANEAcousticVocoder {
        ANEAcousticVocoder(sampleRate: sampleRate, hopLength: max(16, hopLength), backend: .harmonicFilterbank)
    }

    public init(
        directory: URL,
        sampleRate: Int = SpeechRequest.defaultSampleRate,
        hopLength: Int = 240
    ) async throws(BasaltError) {
        self.sampleRate = sampleRate
        self.hopLength = max(16, hopLength)
        let aiURL = directory.appending(component: "vocoder.aimodel")
        let mlcURL = directory.appending(component: "vocoder.mlmodelc")
        if FileManager.default.fileExists(atPath: aiURL.path) {
            do {
                let model = try await AIModel.specialize(
                    contentsOf: aiURL,
                    options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
                )
                guard let fn = try model.loadFunction(named: "vocode") ?? model.loadFunction(named: "main") else {
                    throw BasaltError.modelConfigurationInvalid(path: directory.path, reason: "vocoder.aimodel has no vocode or main function")
                }
                backend = .coreAI(fn)
            } catch let error as BasaltError {
                throw error
            } catch {
                throw .modelConfigurationInvalid(path: directory.path, reason: error.localizedDescription)
            }
        } else if FileManager.default.fileExists(atPath: mlcURL.path) {
            backend = .coreML(try CoreMLSpeechRunner(compiledURL: mlcURL))
        } else {
            throw .modelConfigurationInvalid(path: directory.path, reason: "missing vocoder.aimodel or vocoder.mlmodelc")
        }
    }

    private init(sampleRate: Int, hopLength: Int, backend: Backend) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.backend = backend
    }

    public func decodeMel(_ melFrames: [[Float]]) async throws(BasaltError) -> [Float] {
        guard let bins = melFrames.first?.count, bins > 0 else { return [] }
        let frames = melFrames.count
        switch backend {
        case .coreAI(let function):
            var array = NDArray(shape: [1, bins, frames], scalarType: .float32)
            var flat: [Float] = []
            flat.reserveCapacity(bins * frames)
            for b in 0 ..< bins {
                for t in 0 ..< frames { flat.append(melFrames[t][b]) }
            }
            var view = array.mutableView(as: Float.self)
            view.copyElements(from: flat)
            do {
                var outputs = try await function.run(inputs: ["mel": array])
                guard let out = (outputs.remove("audio") ?? outputs.remove("waveform"))?.ndArray else {
                    throw BasaltError.inferenceError("speech: Core AI vocoder returned no audio tensor")
                }
                return try CoreAISpeechSynthesizer.extractWaveform(out)
            } catch let error as BasaltError {
                throw error
            } catch {
                throw .inferenceError("speech: Core AI vocoder failed: \(error.localizedDescription)")
            }
        case .coreML(let runner):
            var flat: [Float] = []
            flat.reserveCapacity(bins * frames)
            for b in 0 ..< bins {
                for t in 0 ..< frames { flat.append(melFrames[t][b]) }
            }
            return try await runner.decodeMel(flatMel: flat, frames: frames, bins: bins)
        case .harmonicFilterbank:
            return Self.synthesizeFromMel(melFrames, bins: bins, hopLength: hopLength, sampleRate: sampleRate)
        }
    }

    private static func synthesizeFromMel(_ melFrames: [[Float]], bins: Int, hopLength: Int, sampleRate: Int) -> [Float] {
        let total = melFrames.count * hopLength
        var out = Array(repeating: Float(0), count: total)
        let harmonics = min(6, bins)
        for (frameIdx, mel) in melFrames.enumerated() {
            let energy = mel.reduce(Float(0)) { $0 + exp(min(4, max(-6, $1))) } / Float(max(1, bins))
            let baseFreq = 150.0 + Double(abs(mel[0]).truncatingRemainder(dividingBy: 1.0)) * 90.0
            let offset = frameIdx * hopLength
            for n in 0 ..< hopLength {
                let t = Double(offset + n) / Double(sampleRate)
                var sample = 0.0
                for h in 0 ..< harmonics {
                    let weight = Double(exp(min(3, max(-5, mel[h])))) / Double(h + 1)
                    sample += weight * sin(2.0 * .pi * baseFreq * Double(h + 1) * t)
                }
                out[offset + n] = Float(tanh(Double(energy) * sample * 0.35))
            }
        }
        return TTSWaveformMetrics.normalizeSoft(out)
    }
}
