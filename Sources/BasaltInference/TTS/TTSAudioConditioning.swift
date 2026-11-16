import Foundation
import MLX
import MLXNN
import BasaltCore

package enum TTSWaveformMetrics {
    package static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sumSq: Double = 0
        for sample in samples {
            guard sample.isFinite else { return 0 }
            sumSq += Double(sample) * Double(sample)
        }
        return Float((sumSq / Double(samples.count)).squareRoot())
    }

    package static func allFinite(_ samples: [Float]) -> Bool {
        samples.allSatisfy(\.isFinite)
    }

    package static func normalizeSoft(_ samples: [Float], peakTarget: Float = 0.8) -> [Float] {
        guard !samples.isEmpty else { return [] }
        var peak: Float = 0
        for s in samples where s.isFinite {
            peak = max(peak, abs(s))
        }
        guard peak > 1e-6 else { return samples }
        let gain = min(4.0, peakTarget / peak)
        return samples.map { s in
            guard s.isFinite else { return 0 }
            return Float(tanh(Double(s * gain)))
        }
    }

    package static func resampleLinear(_ samples: [Float], from sourceRate: Int, to targetRate: Int) -> [Float] {
        guard !samples.isEmpty, sourceRate > 0, targetRate > 0, sourceRate != targetRate else {
            return samples
        }
        let outCount = max(1, Int((Double(samples.count) * Double(targetRate) / Double(sourceRate)).rounded()))
        if samples.count == 1 { return Array(repeating: samples[0], count: outCount) }
        let ratio = Double(samples.count - 1) / Double(max(1, outCount - 1))
        return (0 ..< outCount).map { idx in
            let pos = Double(idx) * ratio
            let lo = min(samples.count - 1, Int(pos))
            let hi = min(samples.count - 1, lo + 1)
            let frac = Float(pos - Double(lo))
            return samples[lo] * (1 - frac) + samples[hi] * frac
        }
    }
}

package enum ReferenceAudioConditioner {
    package static let maxReferenceBytes = 16 * 1024 * 1024

    package static func loadWaveform(
        at path: String,
        targetSampleRate: Int = SpeechRequest.defaultSampleRate,
        maxSeconds: Double = 10.0
    ) throws(BasaltError) -> [Float] {
        try SpeechRequest.validateReferenceAudio(path)
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw .invalidRequestStructure(details: "speech: reference_audio file does not exist at '\(path)'")
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw .invalidRequestStructure(details: "speech: could not read reference_audio: \(error.localizedDescription)")
        }
        guard !data.isEmpty, data.count <= maxReferenceBytes else {
            throw .invalidRequestStructure(details: "speech: reference_audio must be non-empty and at most 16 MB")
        }
        let ext = url.pathExtension.lowercased()
        let rawSamples: [Float]
        let rawRate: Int
        if ext == "wav" {
            let decoded = try WAVEncoder.decode(data)
            (rawSamples, rawRate) = (decoded.samples, decoded.sampleRate)
        } else if ext == "pcm" {
            let chunk = try AudioChunk.decodePCM16LE(data, sampleRate: targetSampleRate)
            (rawSamples, rawRate) = (chunk.samples, chunk.sampleRate)
        } else {
            rawSamples = decodeByteEnvelope(data)
            rawRate = targetSampleRate
        }
        let resampled = TTSWaveformMetrics.resampleLinear(rawSamples, from: rawRate, to: targetSampleRate)
        let maxSamples = max(1, Int((maxSeconds * Double(targetSampleRate)).rounded()))
        return Array(resampled.prefix(maxSamples))
    }

    package static func speakerEmbedding(
        from path: String?,
        voiceSeed: Int,
        dimension: Int,
        maxSeconds: Double = 3.0
    ) throws(BasaltError) -> [Float] {
        let dim = max(1, dimension)
        guard let path else {
            return presetEmbedding(seed: voiceSeed, dimension: dim)
        }
        let waveform = try loadWaveform(at: path, maxSeconds: maxSeconds)
        return acousticEmbedding(from: waveform, voiceSeed: voiceSeed, dimension: dim)
    }

    private static func presetEmbedding(seed: Int, dimension: Int) -> [Float] {
        let base = Double(seed + 1)
        var vec = (0 ..< dimension).map { idx in
            Float(sin(base * 0.73 + Double(idx + 1) * 0.41) + 0.5 * cos(base * 1.19 - Double(idx) * 0.27))
        }
        normalizeL2(&vec)
        return vec
    }

    private static func acousticEmbedding(from samples: [Float], voiceSeed: Int, dimension: Int) -> [Float] {
        guard !samples.isEmpty else { return presetEmbedding(seed: voiceSeed, dimension: dimension) }
        let bands = min(16, max(4, dimension / 2))
        let strideCount = max(1, samples.count / bands)
        var stats: [Double] = []
        stats.reserveCapacity(bands * 2)
        for band in 0 ..< bands {
            let start = min(samples.count - 1, band * strideCount)
            let end = min(samples.count, start + strideCount)
            var energy = 0.0
            var crossings = 0.0
            if end > start {
                for i in start ..< end {
                    let v = Double(samples[i])
                    energy += v * v
                    if i > start, (samples[i] >= 0) != (samples[i - 1] >= 0) {
                        crossings += 1
                    }
                }
                let len = Double(end - start)
                energy = (energy / len).squareRoot()
                crossings /= len
            }
            stats.append(energy)
            stats.append(crossings * 12.0)
        }
        var out = (0 ..< dimension).map { idx -> Float in
            var acc = 0.0
            for (k, val) in stats.enumerated() {
                acc += val * cos(Double((idx + 1) * (k + 1)) * 0.37 + Double(voiceSeed) * 0.11)
            }
            return Float(acc)
        }
        normalizeL2(&out)
        return out
    }

    private static func decodeByteEnvelope(_ data: Data) -> [Float] {
        data.prefix(48_000).map { (Float($0) - 128.0) / 128.0 }
    }

    private static func normalizeL2(_ vector: inout [Float]) {
        var normSq: Double = 0
        for v in vector { normSq += Double(v) * Double(v) }
        let inv = normSq > 1e-12 ? Float(1.0 / normSq.squareRoot()) : 1.0
        for i in vector.indices { vector[i] *= inv }
    }
}

package enum ISTFTNetSynthesizer {
    package static func hannWindow(length: Int) -> [Float] {
        guard length > 1 else { return [1.0] }
        return (0 ..< length).map { n in
            Float(0.5 * (1.0 - cos(2.0 * .pi * Double(n) / Double(length))))
        }
    }

    package static func mlxInverseSTFT(
        magnitude: MLXArray,
        phase: MLXArray,
        nFFT: Int,
        hopLength: Int
    ) -> MLXArray {
        let frames = magnitude.dim(0)
        guard frames > 0, nFFT >= 2, hopLength >= 1 else {
            return MLXArray.zeros([0], dtype: .float32)
        }
        let bins = nFFT / 2 + 1
        let mag = magnitude.asType(.float32)
        let ph = phase.asType(.float32)
        let real = mag * cos(ph)
        var imag = mag * sin(ph)
        if bins >= 2 {
            let interiorMask = MLXArray((0 ..< bins).map { idx in
                (idx == 0 || (nFFT.isMultiple(of: 2) && idx == bins - 1)) ? Float(0) : Float(1)
            }).reshaped(1, bins)
            imag = imag * interiorMask
        }
        let complexSpec = real.asType(.complex64) + MLXArray(real: 0, imaginary: 1) * imag.asType(.complex64)
        let timeFrames = MLXFFT.irfft(complexSpec, n: nFFT, axis: -1)
        let windowValues = hannWindow(length: nFFT)
        let window = MLXArray(windowValues).reshaped(1, nFFT)
        let windowed = timeFrames * window
        let winSq = broadcast(window * window, to: [frames, nFFT])
        let outLen = (frames - 1) * hopLength + nFFT
        let frameOffsets = MLXArray(Int32(0) ..< Int32(frames)).expandedDimensions(axis: 1) * Int32(hopLength)
        let sampleOffsets = MLXArray(Int32(0) ..< Int32(nFFT)).expandedDimensions(axis: 0)
        let flatIndices = (frameOffsets + sampleOffsets).reshaped(-1)
        let numer = MLXArray.zeros([outLen], dtype: .float32).at[flatIndices].add(windowed.reshaped(-1))
        let denom = MLXArray.zeros([outLen], dtype: .float32).at[flatIndices].add(winSq.reshaped(-1))
        return numer / maximum(denom, MLXArray(Float(1e-4)))
    }

    package static func cpuInverseSTFT(
        magnitude: [[Float]],
        phase: [[Float]],
        nFFT: Int,
        hopLength: Int
    ) -> [Float] {
        let frames = min(magnitude.count, phase.count)
        guard frames > 0, nFFT >= 2, hopLength >= 1 else { return [] }
        let bins = nFFT / 2 + 1
        let window = hannWindow(length: nFFT)
        let outLen = (frames - 1) * hopLength + nFFT
        var numer = Array(repeating: Double(0), count: outLen)
        var denom = Array(repeating: Double(0), count: outLen)
        let invN = 1.0 / Double(nFFT)
        for t in 0 ..< frames {
            let magRow = magnitude[t]
            let phRow = phase[t]
            let rowBins = min(bins, min(magRow.count, phRow.count))
            for n in 0 ..< nFFT {
                var sum = 0.0
                if rowBins > 0 {
                    sum += Double(magRow[0] * cos(phRow[0]))
                }
                let nyquistIndex = nFFT.isMultiple(of: 2) ? (bins - 1) : -1
                for k in 1 ..< rowBins {
                    let re = Double(magRow[k]) * cos(Double(phRow[k]))
                    if k == nyquistIndex {
                        sum += (n.isMultiple(of: 2) ? 1.0 : -1.0) * re
                    } else {
                        let im = Double(magRow[k]) * sin(Double(phRow[k]))
                        let angle = 2.0 * .pi * Double(k * n) * invN
                        sum += 2.0 * (re * cos(angle) - im * sin(angle))
                    }
                }
                let sample = sum * invN
                let w = Double(window[n])
                let outIdx = t * hopLength + n
                numer[outIdx] += sample * w
                denom[outIdx] += w * w
            }
        }
        return zip(numer, denom).map { num, den in
            Float(num / max(1e-4, den))
        }
    }

    package static func loadSafetensorsIfPresent(in directory: URL, into module: Module) throws(BasaltError) {
        let files = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { return }
        let existing = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        var matched: [(String, MLXArray)] = []
        do {
            for file in files {
                let loaded = try MLX.loadArrays(url: file)
                for (key, tensor) in loaded {
                    if let target = existing[key], target.shape == tensor.shape {
                        matched.append((key, tensor.asType(target.dtype)))
                    }
                }
            }
            if !matched.isEmpty {
                module.update(parameters: ModuleParameters.unflattened(matched))
            }
            eval(module)
        } catch {
            throw .weightsCorrupted(path: directory.path, reason: error.localizedDescription)
        }
    }
}
