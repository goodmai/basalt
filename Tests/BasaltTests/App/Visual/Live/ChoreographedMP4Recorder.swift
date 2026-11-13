import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation

public enum ChoreographedMP4Writer {
    public static func writeMP4(
        frames: [RenderedBitmap],
        fps: Int = 60,
        to outputURL: URL
    ) async throws(VisualTestError) -> RecordedVideo {
        guard let first = frames.first else {
            throw .recordingFailed("no frames to encode")
        }
        let width = first.width - (first.width % 2)
        let height = first.height - (first.height % 2)
        let (writer, receiver) = try startWriter(outputURL: outputURL, width: width, height: height)

        let timescale = CMTimeScale(max(1, fps))
        for (index, frame) in frames.enumerated() {
            let buffer = try makePixelBuffer(from: frame.cgImage, width: width, height: height)
            let readOnly = CVReadOnlyPixelBuffer(unsafeBuffer: buffer)
            let presentation = CMTime(value: CMTimeValue(index), timescale: timescale)
            do {
                try await receiver.append(readOnly, with: presentation)
            } catch {
                throw .recordingFailed(error.localizedDescription)
            }
        }

        receiver.finish()
        await withTaskCancellationShield {
            await writer.finishWriting()
        }
        guard writer.status == .completed else {
            throw .recordingFailed(writer.error?.localizedDescription ?? "AVAssetWriter failed with status \(writer.status.rawValue)")
        }
        let byteSize = (try? FileManager.default.attributesOfItem(atPath: outputURL.path)[.size] as? Int64) ?? 0
        let durationSeconds = Double(frames.count) / Double(max(1, fps))
        return RecordedVideo(
            url: outputURL, durationSeconds: durationSeconds,
            byteSize: byteSize, width: width, height: height
        )
    }

    private static func startWriter(
        outputURL: URL,
        width: Int,
        height: Int
    ) throws(VisualTestError) -> (AVAssetWriter, AVAssetWriterInput.PixelBufferReceiver) {
        try? FileManager.default.removeItem(at: outputURL)
        do {
            try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        } catch {
            throw .ioFailed(error.localizedDescription)
        }
        let writer: AVAssetWriter
        do {
            writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        } catch {
            throw .recordingFailed(error.localizedDescription)
        }
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let receiver = writer.inputPixelBufferReceiver(for: input, pixelBufferAttributes: nil)
        do {
            try writer.start()
        } catch {
            throw .recordingFailed(error.localizedDescription)
        }
        writer.startSession(atSourceTime: .zero)
        return (writer, receiver)
    }

    private static func makePixelBuffer(
        from image: CGImage,
        width: Int,
        height: Int
    ) throws(VisualTestError) -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any],
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attributes as CFDictionary, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw .recordingFailed("CVPixelBuffer allocation failed (\(status))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let context = CGContext(
                  data: base, width: width, height: height,
                  bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                  space: space, bitmapInfo: bitmapInfo
              ) else {
            throw .recordingFailed("CGContext over CVPixelBuffer failed")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
