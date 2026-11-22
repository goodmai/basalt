import AVFoundation
import BasaltAppCore
import BasaltCore
import Foundation
import Observation
import Synchronization

public enum AudioCaptureError: Error, LocalizedError, Sendable, Equatable {
    case inputUnavailable
    case alreadyRecording
    case notRecording
    case playbackFailed

    public var errorDescription: String? {
        switch self {
        case .inputUnavailable: "No audio input device is available."
        case .alreadyRecording: "Audio recording is already in progress."
        case .notRecording: "No active audio recording."
        case .playbackFailed: "Audio playback could not start."
        }
    }
}

private final class PlaybackDelegate: NSObject, AVAudioPlayerDelegate, Sendable {
    private let onFinish: @Sendable (Bool) -> Void

    nonisolated init(onFinish: @escaping @Sendable (Bool) -> Void) {
        self.onFinish = onFinish
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onFinish(flag)
    }

    nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: (any Error)?) {
        onFinish(false)
    }
}

private final class CaptureSink: Sendable {
    let state: Mutex<(rate: Double, samples: [Float])>

    init(rate: Double) {
        state = Mutex((rate: rate, samples: []))
    }
}

public actor AudioCaptureActor {
    public static let targetSampleRate = 16_000

    private var engine: AVAudioEngine?
    private var captured: CaptureSink?
    private var player: AVAudioPlayer?
    private var delegate: PlaybackDelegate?
    private var playbackContinuation: AsyncStream<Bool>.Continuation?
    private var playbackEpoch: UInt64 = 0

    public init() {}

    public var isRecording: Bool { engine != nil }

    public var isPlaying: Bool { player?.isPlaying == true }

    public func startRecording() throws {
        guard engine == nil else { throw AudioCaptureError.alreadyRecording }
        stopPlayback()
        let audioEngine = AVAudioEngine()
        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw AudioCaptureError.inputUnavailable
        }
        let sink = CaptureSink(rate: format.sampleRate)
        try input.installAudioTap(onBus: 0, bufferSize: 2048, format: format, tapProvider: Self.makeTap(into: sink))
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw error
        }
        (self.engine, self.captured) = (audioEngine, sink)
    }

    public func stopRecording() throws -> Data {
        guard let audioEngine = engine, let sink = captured else {
            throw AudioCaptureError.notRecording
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        (self.engine, self.captured) = (nil, nil)
        let recorded = sink.state.withLock { $0 }
        return Self.encodeMono16kWAV(samples: recorded.samples, sourceRate: recorded.rate)
    }

    public func play(_ wavData: Data) async throws {
        stopPlayback()
        playbackEpoch &+= 1
        let epoch = playbackEpoch
        let audioPlayer = try AVAudioPlayer(data: wavData)
        let (stream, continuation) = AsyncStream<Bool>.makeStream()
        let finishDelegate = PlaybackDelegate { flag in
            continuation.yield(flag)
            continuation.finish()
        }
        audioPlayer.delegate = finishDelegate
        audioPlayer.prepareToPlay()
        guard audioPlayer.play() else {
            throw AudioCaptureError.playbackFailed
        }
        (player, delegate, playbackContinuation) = (audioPlayer, finishDelegate, continuation)
        defer { cleanupPlayback(epoch: epoch) }
        var succeeded = false
        await withTaskCancellationHandler {
            for await flag in stream {
                succeeded = flag
                break
            }
        } onCancel: {
            continuation.finish()
        }
        if !Task.isCancelled && !succeeded {
            throw AudioCaptureError.playbackFailed
        }
    }

    public func stopPlayback() {
        playbackEpoch &+= 1
        playbackContinuation?.finish()
        player?.stop()
        (player, delegate, playbackContinuation) = (nil, nil, nil)
    }

    private func cleanupPlayback(epoch: UInt64) {
        guard epoch == playbackEpoch else { return }
        player?.stop()
        playbackContinuation?.finish()
        (player, delegate, playbackContinuation) = (nil, nil, nil)
    }

    public nonisolated static func encodeMono16kWAV(samples: [Float], sourceRate: Double) -> Data {
        let resampled = resample(samples: samples, from: sourceRate, to: Double(targetSampleRate))
        return WAVEncoder.encode(samples: resampled, sampleRate: targetSampleRate)
    }

    public nonisolated static func resample(samples: [Float], from sourceRate: Double, to targetRate: Double) -> [Float] {
        guard !samples.isEmpty, sourceRate > 0, targetRate > 0 else { return [] }
        if abs(sourceRate - targetRate) < 1.0 { return samples }
        let ratio = sourceRate / targetRate
        let outCount = max(1, Int((Double(samples.count) / ratio).rounded(.down)))
        var output = [Float]()
        output.reserveCapacity(outCount)
        for index in 0 ..< outCount {
            let position = Double(index) * ratio
            let lower = min(Int(position), samples.count - 1)
            let upper = min(lower + 1, samples.count - 1)
            let fraction = Float(position - Double(lower))
            output.append(samples[lower] + (samples[upper] - samples[lower]) * fraction)
        }
        return output
    }

    private nonisolated static func makeTap(
        into sink: CaptureSink
    ) -> @Sendable (AVReadOnlyAudioPCMBuffer, AVAudioTime) -> Void {
        { buffer, _ in
            let frames = buffer.frameLength
            let channels = Int(buffer.format.channelCount)
            guard frames > 0, channels > 0 else { return }
            var mono = [Float](repeating: 0, count: frames)
            let invChannels = 1.0 / Float(channels)
            for ch in 0 ..< channels {
                switch buffer.channelData(ch) {
                case .float(let span):
                    let count = min(frames, span.count)
                    for i in 0 ..< count { mono[i] += span[i] * invChannels }
                case .int16(let span):
                    let count = min(frames, span.count)
                    let scale = invChannels / Float(Int16.max)
                    for i in 0 ..< count { mono[i] += Float(span[i]) * scale }
                case .int32(let span):
                    let count = min(frames, span.count)
                    let scale = invChannels / Float(Int32.max)
                    for i in 0 ..< count { mono[i] += Float(span[i]) * scale }
                @unknown default:
                    break
                }
            }
            let rate = buffer.format.sampleRate
            sink.state.withLock { state in
                if rate > 0 { state.rate = rate }
                state.samples.append(contentsOf: mono)
            }
        }
    }
}

public typealias VoiceStore = Store<VoiceSessionState, VoiceAction, VoiceEffect>

@Observable @MainActor
public final class AudioSessionController {
    public let store: VoiceStore
    @ObservationIgnored public let capture: AudioCaptureActor
    @ObservationIgnored private let callbackBox = TranscriptCallbackBox()

    public var onSubmitTranscript: @MainActor @Sendable (String) -> Void {
        get { callbackBox.get() }
        set { callbackBox.set(newValue) }
    }

    public init(
        engine: @escaping @Sendable () async -> EngineClient?,
        capture: AudioCaptureActor = AudioCaptureActor()
    ) {
        self.capture = capture
        let box = callbackBox
        let environment = VoiceEnvironment(
            engine: engine,
            startCapture: { try await capture.startRecording() },
            stopCapture: { try await capture.stopRecording() },
            playAudio: { data in try await capture.play(data) },
            stopPlayback: { await capture.stopPlayback() },
            submitTranscript: { text in
                await box.call(text)
            }
        )
        store = Store(VoiceSessionState(), reduce: VoiceReducer.reduce, handle: VoiceEffects.handler(environment))
    }
}

private final class TranscriptCallbackBox: Sendable {
    private let state = Mutex<@MainActor @Sendable (String) -> Void>({ _ in })

    func get() -> @MainActor @Sendable (String) -> Void {
        state.withLock { $0 }
    }

    func set(_ callback: @escaping @MainActor @Sendable (String) -> Void) {
        state.withLock { $0 = callback }
    }

    @MainActor
    func call(_ text: String) {
        let callback = state.withLock { $0 }
        callback(text)
    }
}
