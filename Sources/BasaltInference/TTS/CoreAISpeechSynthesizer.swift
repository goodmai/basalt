import Foundation
import CoreAI
import BasaltCore

public struct CoreAISpeechSynthesizer: SpeechSynthesizer {
    public static let maxInFlight = 2

    public let bundle: CoreAISpeechBundle
    let admission = InFlightLimit(maxInFlight)
    private let functions: [String: InferenceFunction]

    public static func isBundle(_ directory: URL) -> Bool {
        CoreAISpeechBundle.isBundle(directory)
    }

    public static func validateBundle(_ directory: URL) throws(BasaltError) {
        _ = try CoreAISpeechBundle(directory: directory)
    }

    public init(directory: URL) async throws(BasaltError) {
        bundle = try CoreAISpeechBundle(directory: directory)
        let path = directory.path
        do {
            let model = try await AIModel.specialize(
                contentsOf: directory.appending(component: "\(bundle.asset).aimodel"),
                options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
            )
            var loadedFunctions: [String: InferenceFunction] = [:]
            for entry in bundle.functions {
                guard let loaded = try model.loadFunction(named: entry.name) else {
                    throw BasaltError.modelConfigurationInvalid(path: path, reason: "\(bundle.asset).aimodel has no function \(entry.name)")
                }
                loadedFunctions[entry.name] = loaded
            }
            functions = loadedFunctions
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelConfigurationInvalid(path: path, reason: error.localizedDescription)
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
                estimatedTTFAMs: 35
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
                    continuation.yield(AudioChunk(samples: samples, sampleRate: synth.bundle.sampleRate, isFinal: true))
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

    private func runUtterance(request: SpeechRequest, voiceIndex: Int) async throws(BasaltError) -> [Float] {
        let tokenCount = min(max(1, request.input.count), bundle.maxPhonemeTokens)
        let bucket = bundle.function(for: tokenCount)
        guard let function = functions[bucket.name] else {
            throw .inferenceError("speech: Core AI bundle has no loaded function \(bucket.name)")
        }
        let inputs = makeInputs(text: request.input, length: bucket.length, voiceIndex: voiceIndex, speed: Float(request.speed))
        do {
            var outputs = try await function.run(inputs: inputs)
            guard let audio = (outputs.remove("audio") ?? outputs.remove("waveform"))?.ndArray else {
                throw BasaltError.inferenceError("speech: Core AI function \(bucket.name) returned no audio tensor")
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

    private func makeInputs(text: String, length: Int, voiceIndex: Int, speed: Float) -> [String: NDArray] {
        var ids = NDArray(shape: [1, length], scalarType: .int32)
        var idView = ids.mutableView(as: Int32.self)
        idView.copyElements(from: bundle.encodeTokens(text, length: length))

        var voice = NDArray(shape: [1], scalarType: .int32)
        var voiceView = voice.mutableView(as: Int32.self)
        voiceView.copyElements(from: [Int32(voiceIndex)])

        var rate = NDArray(shape: [1], scalarType: .float32)
        var rateView = rate.mutableView(as: Float.self)
        rateView.copyElements(from: [speed])

        return ["input_ids": ids, "voice_id": voice, "speed": rate]
    }

    private static func extractWaveform(_ array: NDArray) throws(BasaltError) -> [Float] {
        if let float32 = array.view(as: Float.self).contiguousElements {
            return (0 ..< float32.count).map { float32[$0] }
        }
        if let float16 = array.view(as: Float16.self).contiguousElements {
            return (0 ..< float16.count).map { Float(float16[$0]) }
        }
        throw .inferenceError("speech: Core AI waveform output must be contiguous Float32 or Float16")
    }
}
