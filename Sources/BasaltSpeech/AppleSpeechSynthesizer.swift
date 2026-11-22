import Foundation
import AVFoundation
import Synchronization
import BasaltCore

public struct AppleSpeechSynthesizer: SpeechSynthesizer {
    public let sampleRate: Int

    public init(sampleRate: Int = SpeechRequest.defaultSampleRate) {
        self.sampleRate = max(8_000, sampleRate)
    }

    public var voices: [VoiceSpec] {
        get async {
            AVSpeechSynthesisVoice.speechVoices().map { voice in
                VoiceSpec(
                    id: voice.identifier,
                    name: voice.name,
                    language: voice.language,
                    model: ComposeSpec.appleSpeech,
                    compute: .appleSpeech
                )
            }
        }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(selected: .appleSpeech)
    }

    public func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        try request.validate()
        guard let voice = Self.resolveVoice(for: request) else {
            throw .invalidRequestStructure(details: "apple-speech found no system voice for '\(request.voice)'")
        }
        let targetRate = sampleRate
        let session = await SynthesisSession()
        return AsyncThrowingStream { continuation in
            let sink = StreamSink(continuation: continuation, targetRate: targetRate)
            let startup = Task { @MainActor in
                guard !Task.isCancelled, !sink.isClosed else { return }
                session.start(request: request, voice: voice, sink: sink)
            }
            continuation.onTermination = { _ in
                startup.cancel()
                sink.cancel()
                Task { @MainActor in
                    session.stop()
                }
            }
        }
    }

    static func resolveVoice(for request: SpeechRequest) -> AVSpeechSynthesisVoice? {
        if let exact = AVSpeechSynthesisVoice(identifier: request.voice) {
            return exact
        }
        let allVoices = AVSpeechSynthesisVoice.speechVoices()
        if let byName = allVoices.first(where: { $0.name.caseInsensitiveCompare(request.voice) == .orderedSame }) {
            return byName
        }
        if let language = request.language, let byLanguage = AVSpeechSynthesisVoice(language: language) {
            return byLanguage
        }
        return AVSpeechSynthesisVoice(language: "en-US") ?? allVoices.first
    }

    static func utteranceRate(for speed: Double) -> Float {
        let raw = Float(speed) * AVSpeechUtteranceDefaultSpeechRate
        return min(AVSpeechUtteranceMaximumSpeechRate, max(AVSpeechUtteranceMinimumSpeechRate, raw))
    }
}

private final class StreamSink: Sendable {
    private let continuation: AsyncThrowingStream<AudioChunk, any Error>.Continuation
    let targetRate: Int
    private let closed = Mutex(false)

    init(continuation: AsyncThrowingStream<AudioChunk, any Error>.Continuation, targetRate: Int) {
        self.continuation = continuation
        self.targetRate = targetRate
    }

    static func makeCallback(_ sink: StreamSink, onDone: @escaping @Sendable () -> Void) -> AVSpeechSynthesizer.BufferCallback {
        { buffer in
            guard let pcm = buffer as? AVAudioPCMBuffer else { return }
            if pcm.frameLength == 0 {
                sink.finishOnce()
                onDone()
                return
            }
            sink.yieldPCM(pcm)
        }
    }

    func yieldPCM(_ pcm: AVAudioPCMBuffer) {
        guard !closed.withLock({ $0 }) else { return }
        do {
            let samples = try Self.extractSamples(from: pcm, targetRate: targetRate)
            if !samples.isEmpty {
                continuation.yield(AudioChunk(samples: samples, sampleRate: targetRate, isFinal: false))
            }
        } catch {
            failOnce(error)
        }
    }

    func finishOnce() {
        let shouldFinish = closed.withLock { state in
            guard !state else { return false }
            state = true
            return true
        }
        guard shouldFinish else { return }
        continuation.yield(AudioChunk(samples: [], sampleRate: targetRate, isFinal: true))
        continuation.finish()
    }

    func failOnce(_ error: any Error) {
        let shouldFail = closed.withLock { state in
            guard !state else { return false }
            state = true
            return true
        }
        guard shouldFail else { return }
        continuation.finish(throwing: error)
    }

    var isClosed: Bool {
        closed.withLock { $0 }
    }

    func cancel() {
        let shouldClose = closed.withLock { state in
            guard !state else { return false }
            state = true
            return true
        }
        if shouldClose {
            continuation.finish()
        }
    }

    private static func extractSamples(from source: AVAudioPCMBuffer, targetRate: Int) throws(BasaltError) -> [Float] {
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(targetRate), channels: 1, interleaved: false) else {
            throw .inferenceError("apple-speech could not construct target PCM format at \(targetRate) Hz")
        }
        if source.format.commonFormat == .pcmFormatFloat32,
           source.format.channelCount == 1,
           abs(source.format.sampleRate - target.sampleRate) < 1,
           let channel = source.floatChannelData?[0] {
            return Array(UnsafeBufferPointer(start: channel, count: Int(source.frameLength)))
        }
        let capacity = AVAudioFrameCount(Double(source.frameLength) * target.sampleRate / source.format.sampleRate + 512)
        guard let converter = AVAudioConverter(from: source.format, to: target),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: max(capacity, 512)) else {
            throw .inferenceError("apple-speech could not convert audio from \(Int(source.format.sampleRate)) Hz")
        }
        converter.downmix = true
        var fed = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, outStatus in
            if fed {
                outStatus.pointee = .endOfStream
                return nil
            }
            fed = true
            outStatus.pointee = .haveData
            return source
        }
        guard status != .error, conversionError == nil, let channel = output.floatChannelData?[0] else {
            throw .inferenceError("apple-speech resampling failed")
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
    }
}

private final class FinishDelegate: NSObject, AVSpeechSynthesizerDelegate, Sendable {
    private let onComplete: @Sendable () -> Void

    init(onComplete: @escaping @Sendable () -> Void) {
        self.onComplete = onComplete
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        onComplete()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        onComplete()
    }
}

@MainActor
private final class SynthesisSession {
    private var synthesizer: AVSpeechSynthesizer?
    private var delegate: FinishDelegate?

    init() {}

    func start(request: SpeechRequest, voice: AVSpeechSynthesisVoice, sink: StreamSink) {
        guard !sink.isClosed else { return }
        let synth = AVSpeechSynthesizer()
        let cleanup: @Sendable () -> Void = { [weak self] in
            sink.finishOnce()
            Task { @MainActor in
                self?.stop()
            }
        }
        let finishDelegate = FinishDelegate(onComplete: cleanup)
        synth.delegate = finishDelegate
        synthesizer = synth
        delegate = finishDelegate
        let utterance = AVSpeechUtterance(string: request.input)
        utterance.voice = voice
        utterance.rate = AppleSpeechSynthesizer.utteranceRate(for: request.speed)
        synth.write(utterance, toBufferCallback: StreamSink.makeCallback(sink, onDone: cleanup))
    }

    func stop() {
        if synthesizer?.isSpeaking == true {
            synthesizer?.stopSpeaking(at: .immediate)
        }
        synthesizer?.delegate = nil
        synthesizer = nil
        delegate = nil
    }
}
