@preconcurrency import AVFoundation
import BasaltCore
import CoreImage
import Foundation
import MLXLMCommon
import MLXVLM

public struct VideoSamplingConfiguration: Sendable, Equatable {
    public var maxFrames: Int
    public var minFrames: Int
    public var targetFPS: Double
    public var maximumDimension: CGFloat?
    public var timeToleranceSeconds: Double

    public init(
        maxFrames: Int = 16,
        minFrames: Int = 1,
        targetFPS: Double = 1.0,
        maximumDimension: CGFloat? = nil,
        timeToleranceSeconds: Double = 0.05
    ) {
        self.maxFrames = maxFrames
        self.minFrames = minFrames
        self.targetFPS = targetFPS
        self.maximumDimension = maximumDimension
        self.timeToleranceSeconds = timeToleranceSeconds
    }

    public func validated() throws(BasaltError) -> VideoSamplingConfiguration {
        guard maxFrames >= 1, maxFrames <= 512 else {
            throw .invalidRequestStructure(details: "maxFrames must be in 1...512, got \(maxFrames)")
        }
        guard minFrames >= 1, minFrames <= maxFrames else {
            throw .invalidRequestStructure(details: "minFrames (\(minFrames)) must be in 1...\(maxFrames)")
        }
        guard targetFPS.isFinite, targetFPS > 0 else {
            throw .invalidRequestStructure(details: "targetFPS must be finite and > 0")
        }
        guard timeToleranceSeconds.isFinite, timeToleranceSeconds >= 0 else {
            throw .invalidRequestStructure(details: "timeToleranceSeconds must be finite and >= 0")
        }
        if let maximumDimension, (!maximumDimension.isFinite || maximumDimension < 1) {
            throw .invalidRequestStructure(details: "maximumDimension must be finite and >= 1")
        }
        return self
    }
}

public struct SampledFrameInput: Sendable {
    public let image: CIImage
    public let timestamp: CMTime

    public init(image: CIImage, timestamp: CMTime) {
        self.image = image
        self.timestamp = timestamp
    }
}

public struct SampledVideoFrames: Sendable {
    public let frames: [CIImage]
    public let timestamps: [CMTime]
    public let duration: CMTime

    public init(frames: [CIImage], timestamps: [CMTime], duration: CMTime) {
        self.frames = frames
        self.timestamps = timestamps
        self.duration = duration
    }

    public func asUserInputVideo() -> UserInput.Video {
        let videoFrames = zip(frames, timestamps).map { image, timestamp in
            UserInput.VideoFrame(image: .ciImage(image), timeStamp: timestamp)
        }
        return .frames(videoFrames)
    }

    public func asUserInputImages(labelPrefix: String? = nil) -> [UserInput.Image] {
        frames.enumerated().map { index, image in
            let label = labelPrefix.map { "\($0)-\(index)" }
            return .ciImage(image, label: label)
        }
    }
}

public enum VideoFrameSampler {
    public static let supportedExtensions: Set<String> = ["mp4", "mov", "m4v"]
    public static let remoteRefusal = "remote video URLs are not fetched; pass a local .mp4, .mov, or .m4v file URL"

    public static func resolveVideoURL(_ rawURL: String) throws(BasaltError) -> URL {
        let trimmed = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw .invalidRequestStructure(details: "video_url must not be empty")
        }
        let scheme = trimmed.prefix(while: { $0 != ":" }).lowercased()
        if scheme == "http" || scheme == "https" {
            throw .invalidRequestStructure(details: remoteRefusal)
        }
        let resolved: URL
        if scheme == "file", let parsed = URL(string: trimmed), parsed.isFileURL {
            resolved = parsed
        } else if trimmed.contains("://") {
            throw .invalidRequestStructure(details: "unsupported video_url scheme '\(scheme)'")
        } else {
            resolved = URL(fileURLWithPath: trimmed)
        }
        let ext = resolved.pathExtension.lowercased()
        guard supportedExtensions.contains(ext) else {
            throw .invalidRequestStructure(
                details: "video extension '.\(ext)' is not supported; expected .mp4, .mov, or .m4v"
            )
        }
        return resolved
    }

    public static func uniformTimestamps(
        duration: CMTime,
        configuration: VideoSamplingConfiguration = .init(),
        availableFrameCount: Int? = nil
    ) throws(BasaltError) -> [CMTime] {
        let config = try configuration.validated()
        guard duration.isNumeric, !duration.isIndefinite, duration.seconds >= 0 else {
            throw .invalidRequestStructure(details: "video duration must be finite and non-negative")
        }
        let timescale: CMTimeScale = duration.timescale > 0 ? duration.timescale : 600
        let scaled = CMTimeConvertScale(duration, timescale: timescale, method: .default)
        let totalTicks = max(Int64(0), scaled.value)
        let rawRequested = totalTicks == 0
            ? config.minFrames
            : Int((duration.seconds * config.targetFPS).rounded())
        var count = min(max(rawRequested, config.minFrames), config.maxFrames)
        if let availableFrameCount, availableFrameCount > 0 {
            count = min(count, availableFrameCount)
        }
        guard count > 1, totalTicks > 0 else {
            return [CMTime(value: 0, timescale: timescale)]
        }
        let step = Double(totalTicks) / Double(count - 1)
        return (0 ..< count).map { index in
            let tick = min(totalTicks, max(0, Int64((Double(index) * step).rounded())))
            return CMTime(value: tick, timescale: timescale)
        }
    }

    public static func sampleMemoryFrames(
        _ sourceFrames: [SampledFrameInput],
        configuration: VideoSamplingConfiguration = .init()
    ) throws(BasaltError) -> SampledVideoFrames {
        let config = try configuration.validated()
        guard !sourceFrames.isEmpty else {
            throw .invalidRequestStructure(details: "cannot sample an empty video frame buffer")
        }
        let ordered = sourceFrames.sorted { CMTimeCompare($0.timestamp, $1.timestamp) < 0 }
        let start = ordered[0].timestamp
        let end = ordered[ordered.count - 1].timestamp
        let span = CMTimeSubtract(end, start)
        let relativeTargets = try uniformTimestamps(
            duration: span.seconds >= 0 ? span : .zero,
            configuration: config,
            availableFrameCount: ordered.count
        )
        var pickedFrames: [CIImage] = []
        var pickedTimes: [CMTime] = []
        var cursor = 0
        for target in relativeTargets {
            let absoluteTarget = CMTimeAdd(start, target)
            var chosen: Int?
            while cursor < ordered.count {
                if CMTimeCompare(ordered[cursor].timestamp, absoluteTarget) > 0 { break }
                chosen = cursor
                cursor += 1
            }
            let selected = ordered[chosen ?? min(cursor, ordered.count - 1)]
            pickedFrames.append(selected.image)
            pickedTimes.append(selected.timestamp)
            if pickedFrames.count >= config.maxFrames { break }
        }
        return SampledVideoFrames(frames: pickedFrames, timestamps: pickedTimes, duration: span)
    }
}

extension VideoFrameSampler {
    public static func sampleFrames(
        from rawVideoURL: String,
        configuration: VideoSamplingConfiguration = .init()
    ) async throws(BasaltError) -> SampledVideoFrames {
        let url = try resolveVideoURL(rawVideoURL)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw .invalidRequestStructure(details: "video file does not exist at \(url.path)")
        }
        return try await sampleFrames(from: url, configuration: configuration)
    }

    public static func sampleFrames(
        from url: URL,
        configuration: VideoSamplingConfiguration = .init()
    ) async throws(BasaltError) -> SampledVideoFrames {
        try await sampleFrames(from: AVURLAsset(url: url), configuration: configuration)
    }

    public static func sampleFrames(
        from asset: AVAsset,
        configuration: VideoSamplingConfiguration = .init()
    ) async throws(BasaltError) -> SampledVideoFrames {
        let config = try configuration.validated()
        let duration = try await validateAndLoadDuration(asset)
        let requestedTimes = try uniformTimestamps(duration: duration, configuration: config)
        let generator = makeGenerator(for: asset, configuration: config)
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        var ciImages: [CIImage] = []
        var actualTimestamps: [CMTime] = []
        ciImages.reserveCapacity(requestedTimes.count)
        actualTimestamps.reserveCapacity(requestedTimes.count)
        for await result in generator.images(for: requestedTimes) {
            if Task.isCancelled {
                throw .cancelled
            }
            if case .success(_, let cgImage, let actualTime) = result {
                ciImages.append(CIImage(cgImage: cgImage, options: [.colorSpace: sRGB]))
                actualTimestamps.append(actualTime)
                if ciImages.count >= config.maxFrames { break }
            }
        }
        guard !ciImages.isEmpty else {
            throw .invalidRequestStructure(details: "failed to decode any video frames from asset")
        }
        return SampledVideoFrames(frames: ciImages, timestamps: actualTimestamps, duration: duration)
    }

    private static func validateAndLoadDuration(_ asset: AVAsset) async throws(BasaltError) -> CMTime {
        do {
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let track = tracks.first else {
                throw BasaltError.invalidRequestStructure(details: "video asset contains no video track")
            }
            let decodable = try await track.load(.isDecodable)
            guard decodable else {
                throw BasaltError.invalidRequestStructure(details: "video track is not decodable")
            }
            return try await asset.load(.duration)
        } catch let basaltError as BasaltError {
            throw basaltError
        } catch is CancellationError {
            throw .cancelled
        } catch {
            throw .invalidRequestStructure(details: "failed to inspect video asset: \(error.localizedDescription)")
        }
    }

    private static func makeGenerator(
        for asset: AVAsset,
        configuration: VideoSamplingConfiguration
    ) -> AVAssetImageGenerator {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        let tolerance = CMTime(seconds: configuration.timeToleranceSeconds, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance
        if let maxDim = configuration.maximumDimension {
            generator.maximumSize = CGSize(width: maxDim, height: maxDim)
        }
        return generator
    }
}
