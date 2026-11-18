import Foundation
import AVFoundation

extension AudioClip {
    static var format: AVAudioFormat {
        AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
    }

    public init(encoded data: Data, format fileExtension: String) throws(BasaltError) {
        let fileExtension = try Self.checkedExtension(fileExtension)
        guard !data.isEmpty else { throw .invalidRequestStructure(details: "the audio file is empty") }
        guard data.count <= Self.maxBytes else {
            throw .invalidRequestStructure(details: "an audio file of \(data.count) bytes is over the limit of 25 MB")
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-audio-\(UUID().uuidString).\(fileExtension)")
        guard FileManager.default.createFile(atPath: url.path, contents: data, attributes: [.posixPermissions: 0o600]) else {
            throw .inferenceError("cannot stage the uploaded audio in the temporary directory")
        }
        defer { try? FileManager.default.removeItem(at: url) }
        try self.init(samples: try Self.decode(url, as: fileExtension))
    }

    static func decode(_ url: URL, as fileExtension: String) throws(BasaltError) -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw .invalidRequestStructure(details: "the upload is not decodable \(fileExtension) audio")
        }
        let source = file.processingFormat
        guard file.length > 0, source.sampleRate.isFinite, source.sampleRate > 0 else {
            throw .invalidRequestStructure(details: "the \(fileExtension) audio header has invalid frame count or sample rate")
        }
        let seconds = Double(file.length) / source.sampleRate
        guard seconds.isFinite, seconds <= maxSeconds else { throw tooLong(seconds) }
        var readError: (any Error)?
        let output = try convert(from: source, frames: file.length) { frames in
            guard file.framePosition < file.length, let chunk = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: frames) else { return nil }
            do { try file.read(into: chunk, frameCount: frames) } catch { readError = error }
            return chunk.frameLength > 0 ? chunk : nil
        }
        guard readError == nil else { throw .invalidRequestStructure(details: "the \(fileExtension) audio is truncated or corrupt") }
        let count = Int(output.frameLength)
        return [Float](capacity: count) { out in
            guard let channel = output.floatChannelData?[0] else { return }
            for index in 0 ..< count { out.append(channel[index]) }
        }
    }

    public func buffer(in target: AVAudioFormat) throws(BasaltError) -> AVAudioPCMBuffer {
        guard let source = AVAudioPCMBuffer(pcmFormat: Self.format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw .inferenceError("cannot allocate an audio buffer of \(samples.count) frames")
        }
        source.frameLength = source.frameCapacity
        if let channel = source.floatChannelData?[0] {
            for index in samples.indices { channel[index] = samples[index] }
        }
        var pending: AVAudioPCMBuffer? = source
        return try Self.convert(from: Self.format, to: target, frames: AVAudioFramePosition(samples.count)) { _ in
            defer { pending = nil }
            return pending
        }
    }

    static func convert(
        from source: AVAudioFormat, to target: AVAudioFormat = format, frames: AVAudioFramePosition,
        next: (AVAudioFrameCount) -> AVAudioPCMBuffer?
    ) throws(BasaltError) -> AVAudioPCMBuffer {
        guard frames > 0, source.sampleRate.isFinite, source.sampleRate > 0, target.sampleRate.isFinite, target.sampleRate > 0 else {
            throw .invalidRequestStructure(details: "audio cannot be resampled due to invalid format parameters")
        }
        let capacity = Double(frames) * target.sampleRate / source.sampleRate + 4_096
        guard capacity.isFinite, capacity >= 1, capacity <= Double(AVAudioFrameCount.max),
              let converter = AVAudioConverter(from: source, to: target),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(capacity)) else {
            throw .invalidRequestStructure(details: "audio of \(source.channelCount) channels cannot be resampled")
        }
        converter.downmix = true
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { packets, status in
            guard let chunk = next(min(packets, 65_536)) else {
                status.pointee = .endOfStream
                return nil
            }
            status.pointee = .haveData
            return chunk
        }
        guard status != .error, conversionError == nil else {
            throw .invalidRequestStructure(details: "the audio could not be resampled to \(Int(target.sampleRate)) Hz")
        }
        return output
    }
}
