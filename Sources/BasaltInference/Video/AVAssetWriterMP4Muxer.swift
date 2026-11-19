import AVFoundation
import AudioToolbox
import BasaltCore
import CoreMedia
import CoreVideo
import Foundation

package struct AVVideoFrameBuffer: Sendable, Equatable {
    package var width: Int
    package var height: Int
    package var bgraBytes: Data

    package init(width: Int, height: Int, bgraBytes: Data) throws(BasaltError) {
        let expected = width * height * 4
        guard width > 0, height > 0, bgraBytes.count == expected else {
            throw .invalidRequestStructure(
                details: "video muxer: frame buffer \(width)x\(height) requires \(expected) BGRA bytes, got \(bgraBytes.count)"
            )
        }
        self.width = width
        self.height = height
        self.bgraBytes = bgraBytes
    }

    package static func fromRGBFloats(_ rgb: Span<Float>, width: Int, height: Int) throws(BasaltError) -> AVVideoFrameBuffer {
        let pixelCount = width * height
        guard width > 0, height > 0, rgb.count >= pixelCount * 3 else {
            throw .invalidRequestStructure(details: "video muxer: insufficient RGB float values for \(width)x\(height)")
        }
        var out = Data(count: pixelCount * 4)
        out.withUnsafeMutableBytes { rawBuf in
            guard let dst = rawBuf.bindMemory(to: UInt8.self).baseAddress else { return }
            for idx in 0 ..< pixelCount {
                let r = Self.quantizeByte(rgb[idx * 3])
                let g = Self.quantizeByte(rgb[idx * 3 + 1])
                let b = Self.quantizeByte(rgb[idx * 3 + 2])
                let base = idx * 4
                dst[base] = b
                dst[base + 1] = g
                dst[base + 2] = r
                dst[base + 3] = 255
            }
        }
        return try AVVideoFrameBuffer(width: width, height: height, bgraBytes: out)
    }

    private static func quantizeByte(_ value: Float) -> UInt8 {
        guard value.isFinite else { return 0 }
        let clamped = min(1.0, max(0.0, value))
        return UInt8((clamped * 255.0).rounded())
    }
}

package struct AVAudioTrackPayload: Sendable, Equatable {
    package var samplesInt16Interleaved: [Int16]
    package var sampleRate: Int
    package var channels: Int

    package init(samplesInt16Interleaved: [Int16], sampleRate: Int = 44_100, channels: Int = 2) throws(BasaltError) {
        guard (1 ... 2).contains(channels), sampleRate > 0 else {
            throw .invalidRequestStructure(details: "video muxer: invalid audio format \(sampleRate) Hz / \(channels) ch")
        }
        guard !samplesInt16Interleaved.isEmpty, samplesInt16Interleaved.count.isMultiple(of: channels) else {
            throw .invalidRequestStructure(details: "video muxer: interleaved PCM count must be non-empty and divisible by channels")
        }
        self.samplesInt16Interleaved = samplesInt16Interleaved
        self.sampleRate = sampleRate
        self.channels = channels
    }

    package var frameCount: Int {
        samplesInt16Interleaved.count / max(1, channels)
    }

    package static func fromFloatSamples(
        _ pcm: Span<Float>,
        sampleRate: Int,
        channels: Int
    ) throws(BasaltError) -> AVAudioTrackPayload {
        guard (1 ... 2).contains(channels), pcm.count >= channels else {
            throw .invalidRequestStructure(details: "video muxer: audio float span is empty")
        }
        let alignedCount = (pcm.count / channels) * channels
        var quantized = [Int16](repeating: 0, count: alignedCount)
        for idx in 0 ..< alignedCount {
            let v = pcm[idx].isFinite ? min(1.0, max(-1.0, pcm[idx])) : 0.0
            quantized[idx] = Int16((v * 32_767.0).rounded())
        }
        return try AVAudioTrackPayload(samplesInt16Interleaved: quantized, sampleRate: sampleRate, channels: channels)
    }
}

package struct AVMuxingSpec: Sendable, Equatable {
    package var containerFormat: VideoContainerFormat
    package var videoCodec: VideoCodecKind
    package var audioCodec: VideoAudioCodecKind
    package var width: Int
    package var height: Int
    package var fps: Int

    package init(
        containerFormat: VideoContainerFormat = .mp4,
        videoCodec: VideoCodecKind = .h264,
        audioCodec: VideoAudioCodecKind = .aac,
        dimensions: VideoDimensions,
        fps: Int = 24
    ) {
        self.containerFormat = containerFormat
        self.videoCodec = videoCodec
        self.audioCodec = audioCodec
        self.width = dimensions.width
        self.height = dimensions.height
        self.fps = max(1, fps)
    }
}

package enum AVAssetWriterMP4Muxer {
    package static func muxToData(
        frames: [AVVideoFrameBuffer],
        audio: AVAudioTrackPayload?,
        spec: AVMuxingSpec
    ) async throws(BasaltError) -> Data {
        let tempURL = FileManager.default.temporaryDirectory
            .appending(path: "basalt-vid-\(UUID().uuidString).\(spec.containerFormat.fileExtension)")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try await muxToFile(frames: frames, audio: audio, spec: spec, outputURL: tempURL)
        do {
            return try Data(contentsOf: tempURL)
        } catch {
            throw .modelInferenceError(details: "video muxer: failed to read finished container: \(error.localizedDescription)")
        }
    }

    package static func muxToFile(
        frames: [AVVideoFrameBuffer],
        audio: AVAudioTrackPayload?,
        spec: AVMuxingSpec,
        outputURL: URL
    ) async throws(BasaltError) {
        guard !frames.isEmpty else {
            throw .invalidRequestStructure(details: "video muxer: at least 1 video frame is required")
        }
        try? FileManager.default.removeItem(at: outputURL)
        let fileType: AVFileType = spec.containerFormat == .mov ? .mov : .mp4
        let writer: AVAssetWriter
        do {
            writer = try AVAssetWriter(outputURL: outputURL, fileType: fileType)
        } catch {
            throw .modelInferenceError(details: "video muxer: cannot create AVAssetWriter: \(error.localizedDescription)")
        }
        writer.shouldOptimizeForNetworkUse = true
        try await writeTracks(writer: writer, frames: frames, audio: audio, spec: spec, outputURL: outputURL)
    }

    private static func writeTracks(
        writer: AVAssetWriter,
        frames: [AVVideoFrameBuffer],
        audio: AVAudioTrackPayload?,
        spec: AVMuxingSpec,
        outputURL: URL
    ) async throws(BasaltError) {
        let videoInput = makeVideoInput(spec: spec)
        guard writer.canAdd(videoInput) else {
            throw .modelInferenceError(details: "video muxer: container rejected \(spec.videoCodec.rawValue) video track")
        }
        let pixelAttrs = CVPixelBufferCreationAttributes(
            pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA),
            size: CVImageSize(width: spec.width, height: spec.height)
        )
        let videoReceiver = writer.inputPixelBufferReceiver(for: videoInput, pixelBufferAttributes: pixelAttrs)
        let audioReceiver = try attachAudioReceiverIfNeeded(writer: writer, audio: audio, spec: spec)

        do {
            try writer.start()
            writer.startSession(atSourceTime: .zero)
            try await appendAllVideoFrames(frames: frames, receiver: videoReceiver, attrs: pixelAttrs, fps: spec.fps)
            videoReceiver.finish()
            if let audio, let audioReceiver {
                let sampleBuffer = try makePCMSampleBuffer(audio: audio)
                try await audioReceiver.append(sampleBuffer)
                audioReceiver.finish()
            }
            await writer.finishWriting()
        } catch is CancellationError {
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: outputURL)
            throw .cancelled
        } catch let basaltErr as BasaltError {
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: outputURL)
            throw basaltErr
        } catch {
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: outputURL)
            throw .modelInferenceError(details: "video muxer: \(error.localizedDescription)")
        }
        guard writer.status == .completed else {
            let reason = writer.error?.localizedDescription ?? "writer status \(writer.status.rawValue)"
            try? FileManager.default.removeItem(at: outputURL)
            throw .modelInferenceError(details: "video muxer failed: \(reason)")
        }
    }
}

extension AVAssetWriterMP4Muxer {
    fileprivate static func makeVideoInput(spec: AVMuxingSpec) -> AVAssetWriterInput {
        let codecType: AVVideoCodecType = spec.videoCodec == .hevc ? .hevc : .h264
        let bitRate = max(256_000, spec.width * spec.height * spec.fps / 2)
        let settings: [String: Any] = [
            AVVideoCodecKey: codecType,
            AVVideoWidthKey: spec.width,
            AVVideoHeightKey: spec.height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitRate,
                AVVideoExpectedSourceFrameRateKey: spec.fps,
            ],
        ]
        return AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    }

    fileprivate static func appendAllVideoFrames(
        frames: [AVVideoFrameBuffer],
        receiver: AVAssetWriterInput.PixelBufferReceiver,
        attrs: CVPixelBufferCreationAttributes,
        fps: Int
    ) async throws {
        for (index, frame) in frames.enumerated() {
            try Task.checkCancellation()
            var mutableBuf: CVMutablePixelBuffer
            if let pool = receiver.pixelBufferPool {
                mutableBuf = try pool.makeMutablePixelBuffer()
            } else {
                mutableBuf = try CVMutablePixelBuffer(attrs)
            }
            copyBGRAFrame(frame, into: &mutableBuf)
            let pts = CMTime(value: Int64(index), timescale: CMTimeScale(fps))
            try await receiver.append(CVReadOnlyPixelBuffer(mutableBuf), with: pts)
        }
    }

    private static func copyBGRAFrame(_ frame: AVVideoFrameBuffer, into buffer: inout CVMutablePixelBuffer) {
        let srcStride = frame.width * 4
        let height = frame.height
        frame.bgraBytes.withUnsafeBytes { srcRaw in
            guard let srcBase = srcRaw.baseAddress else { return }
            buffer.accessUnsafeMutableRawPlaneBytes { planes in
                guard let plane = planes.first, let dstBase = plane.bytes.baseAddress else { return }
                let dstStride = plane.properties.bytesPerRow
                let copyWidth = min(srcStride, dstStride)
                for row in 0 ..< height {
                    guard (row + 1) * dstStride <= plane.bytes.count else { break }
                    memcpy(
                        dstBase.advanced(by: row * dstStride),
                        srcBase.advanced(by: row * srcStride),
                        copyWidth
                    )
                }
            }
        }
    }
}

extension AVAssetWriterMP4Muxer {
    fileprivate static func attachAudioReceiverIfNeeded(
        writer: AVAssetWriter,
        audio: AVAudioTrackPayload?,
        spec: AVMuxingSpec
    ) throws(BasaltError) -> AVAssetWriterInput.SampleBufferReceiver? {
        guard let audio else { return nil }
        let preferred = makeAudioOutputSettings(audio: audio, codec: spec.audioCodec)
        let input = AVAssetWriterInput(mediaType: .audio, outputSettings: preferred)
        if writer.canAdd(input) {
            return writer.inputReceiver(for: input)
        }
        let fallback = makeAudioOutputSettings(audio: audio, codec: .aac)
        let aacInput = AVAssetWriterInput(mediaType: .audio, outputSettings: fallback)
        guard writer.canAdd(aacInput) else {
            throw .modelInferenceError(details: "video muxer: container rejected audio output settings")
        }
        return writer.inputReceiver(for: aacInput)
    }

    private static func makeAudioOutputSettings(
        audio: AVAudioTrackPayload,
        codec: VideoAudioCodecKind
    ) -> [String: Any] {
        switch codec {
        case .aac:
            return [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: audio.sampleRate,
                AVNumberOfChannelsKey: audio.channels,
                AVEncoderBitRateKey: audio.channels == 2 ? 128_000 : 64_000,
            ]
        case .linearPCM:
            return [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: audio.sampleRate,
                AVNumberOfChannelsKey: audio.channels,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ]
        }
    }

    fileprivate static func makePCMSampleBuffer(
        audio: AVAudioTrackPayload
    ) throws(BasaltError) -> CMReadySampleBuffer<CMSampleBuffer.DynamicContent> {
        let formatDesc = try makeAudioFormatDescription(audio: audio)
        let blockBuffer = try makeAudioBlockBuffer(audio: audio)
        var sampleBuffer: CMSampleBuffer?
        let sbStatus = CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            allocator: kCFAllocatorDefault,
            dataBuffer: blockBuffer,
            formatDescription: formatDesc,
            sampleCount: audio.frameCount,
            presentationTimeStamp: .zero,
            packetDescriptions: nil,
            sampleBufferOut: &sampleBuffer
        )
        guard sbStatus == noErr, let sampleBuffer else {
            throw .modelInferenceError(details: "video muxer: CMAudioSampleBufferCreateReady failed (\(sbStatus))")
        }
        return CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: sampleBuffer)
    }

    private static func makeAudioFormatDescription(audio: AVAudioTrackPayload) throws(BasaltError) -> CMAudioFormatDescription {
        let bytesPerFrame = UInt32(audio.channels * 2)
        var asbd = AudioStreamBasicDescription(
            mSampleRate: Float64(audio.sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
            mBytesPerPacket: bytesPerFrame,
            mFramesPerPacket: 1,
            mBytesPerFrame: bytesPerFrame,
            mChannelsPerFrame: UInt32(audio.channels),
            mBitsPerChannel: 16,
            mReserved: 0
        )
        var formatDesc: CMAudioFormatDescription?
        let status = CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            asbd: &asbd,
            layoutSize: 0,
            layout: nil,
            magicCookieSize: 0,
            magicCookie: nil,
            extensions: nil,
            formatDescriptionOut: &formatDesc
        )
        guard status == noErr, let formatDesc else {
            throw .modelInferenceError(details: "video muxer: CMAudioFormatDescriptionCreate failed (\(status))")
        }
        return formatDesc
    }

    private static func makeAudioBlockBuffer(audio: AVAudioTrackPayload) throws(BasaltError) -> CMBlockBuffer {
        let byteLength = audio.samplesInt16Interleaved.count * 2
        var blockBuffer: CMBlockBuffer?
        let blkStatus = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: nil,
            blockLength: byteLength,
            blockAllocator: kCFAllocatorDefault,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: byteLength,
            flags: 0,
            blockBufferOut: &blockBuffer
        )
        guard blkStatus == kCMBlockBufferNoErr, let blockBuffer else {
            throw .modelInferenceError(details: "video muxer: CMBlockBufferCreateWithMemoryBlock failed (\(blkStatus))")
        }
        let copyStatus = audio.samplesInt16Interleaved.withUnsafeBytes { raw -> OSStatus in
            guard let base = raw.baseAddress else { return kCMBlockBufferBadPointerParameterErr }
            return CMBlockBufferReplaceDataBytes(
                with: base,
                blockBuffer: blockBuffer,
                offsetIntoDestination: 0,
                dataLength: byteLength
            )
        }
        guard copyStatus == kCMBlockBufferNoErr else {
            throw .modelInferenceError(details: "video muxer: CMBlockBufferReplaceDataBytes failed (\(copyStatus))")
        }
        return blockBuffer
    }
}
