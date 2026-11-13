import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit
import Synchronization

public struct VideoConfig: Sendable, Equatable {
    public var fps: Int
    public var scale: Int
    public var showsCursor: Bool
    public var codec: AVVideoCodecType

    public init(
        fps: Int = 60,
        scale: Int = 2,
        showsCursor: Bool = false,
        codec: AVVideoCodecType = .hevc
    ) {
        self.fps = fps
        self.scale = scale
        self.showsCursor = showsCursor
        self.codec = codec
    }
}

public struct RecordedVideo: Sendable, Codable, Equatable {
    public var url: URL
    public var durationSeconds: Double
    public var byteSize: Int64
    public var width: Int
    public var height: Int

    public init(url: URL, durationSeconds: Double, byteSize: Int64, width: Int, height: Int) {
        self.url = url
        self.durationSeconds = durationSeconds
        self.byteSize = byteSize
        self.width = width
        self.height = height
    }
}

public struct VideoRecordingLease: ~Copyable, Sendable {
    let sessionID: UUID
    private var active = true

    init(sessionID: UUID) {
        self.sessionID = sessionID
    }

    consuming func takeID() -> UUID {
        active = false
        return sessionID
    }

    deinit {
        precondition(!active, "VideoRecordingLease dropped without calling VideoRecorderActor.finish")
    }
}

private final class RecordingCompletionDelegate: NSObject, SCRecordingOutputDelegate, Sendable {
    private enum State: Sendable {
        case idle
        case waiting(CheckedContinuation<Void, any Error>)
        case finished(Result<Void, any Error>)
    }

    private let state = Mutex<State>(.idle)

    func recordingOutputDidFinishRecording(_ recordingOutput: SCRecordingOutput) {
        complete(.success(()))
    }

    func recordingOutput(_ recordingOutput: SCRecordingOutput, didFailWithError error: any Error) {
        complete(.failure(error))
    }

    private func complete(_ result: Result<Void, any Error>) {
        let continuation = state.withLock { current -> CheckedContinuation<Void, any Error>? in
            switch current {
            case .idle:
                current = .finished(result)
                return nil
            case .waiting(let pending):
                current = .finished(result)
                return pending
            case .finished:
                return nil
            }
        }
        continuation?.resume(with: result)
    }

    func waitUntilFinished() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            let immediate = state.withLock { current -> Result<Void, any Error>? in
                switch current {
                case .idle:
                    current = .waiting(continuation)
                    return nil
                case .waiting:
                    return .success(())
                case .finished(let outcome):
                    return outcome
                }
            }
            if let immediate {
                continuation.resume(with: immediate)
            }
        }
    }
}

public actor VideoRecorderActor {
    private struct ActiveSession {
        let stream: SCStream
        let output: SCRecordingOutput
        let delegate: RecordingCompletionDelegate
        let url: URL
        let startedAt: ContinuousClock.Instant
        let width: Int
        let height: Int
    }

    private var sessions: [UUID: ActiveSession] = [:]

    public init() {}

    public func start(
        windowID: CGWindowID,
        outputURL: URL,
        config: VideoConfig = VideoConfig()
    ) async throws(VisualTestError) -> VideoRecordingLease {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        } catch {
            throw .screenCaptureUnavailable(error.localizedDescription)
        }
        guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
            throw .windowNotFound(title: "\(windowID)")
        }
        try? FileManager.default.removeItem(at: outputURL)
        try? FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let filter = SCContentFilter(desktopIndependentWindow: window)
        let streamConfig = SCStreamConfiguration()
        let width = max(2, Int(window.frame.width) * max(1, config.scale))
        let height = max(2, Int(window.frame.height) * max(1, config.scale))
        streamConfig.width = width
        streamConfig.height = height
        streamConfig.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(max(1, config.fps)))
        streamConfig.showsCursor = config.showsCursor

        let recordingConfig = SCRecordingOutputConfiguration()
        recordingConfig.outputURL = outputURL
        recordingConfig.outputFileType = .mp4
        recordingConfig.videoCodecType = config.codec

        let delegate = RecordingCompletionDelegate()
        let recordingOutput = SCRecordingOutput(configuration: recordingConfig, delegate: delegate)
        let stream = SCStream(filter: filter, configuration: streamConfig, delegate: nil)
        do {
            try stream.addRecordingOutput(recordingOutput)
            try await stream.startCapture()
        } catch {
            throw .recordingFailed(error.localizedDescription)
        }

        let id = UUID()
        sessions[id] = ActiveSession(
            stream: stream, output: recordingOutput, delegate: delegate, url: outputURL,
            startedAt: ContinuousClock.now, width: width, height: height
        )
        return VideoRecordingLease(sessionID: id)
    }

    public func finish(_ lease: consuming VideoRecordingLease) async throws(VisualTestError) -> RecordedVideo {
        let id = lease.takeID()
        guard let session = sessions.removeValue(forKey: id) else {
            throw .recordingFailed("unknown recording session")
        }
        return try await withTaskCancellationShield { () async throws(VisualTestError) -> RecordedVideo in
            do {
                try await session.stream.stopCapture()
                try session.stream.removeRecordingOutput(session.output)
                try await session.delegate.waitUntilFinished()
            } catch {
                throw .recordingFailed(error.localizedDescription)
            }
            let elapsed = ContinuousClock.now - session.startedAt
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
            let bytes = (try? FileManager.default.attributesOfItem(atPath: session.url.path)[.size] as? Int64) ?? 0
            return RecordedVideo(
                url: session.url, durationSeconds: seconds,
                byteSize: bytes, width: session.width, height: session.height
            )
        }
    }
}
