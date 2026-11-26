import Foundation
import MLX

enum WhisperAudio {
    static let sampleRate = 16_000
    static let fftSize = 400
    static let hopLength = 160

    static func melFilters(count: Int) -> MLXArray {
        let bins = fftSize / 2 + 1
        let nyquist = Double(sampleRate) / 2
        let top = hertzToMel(nyquist)
        let edges = (0 ..< count + 2).map { melToHertz(top * Double($0) / Double(count + 1)) }
        var weights = [Float](repeating: 0, count: count * bins)
        for filter in 0 ..< count {
            let (low, center, high) = (edges[filter], edges[filter + 1], edges[filter + 2])
            let norm = 2 / (high - low)
            for bin in 0 ..< bins {
                let frequency = nyquist * Double(bin) / Double(bins - 1)
                let rising = (frequency - low) / (center - low)
                let falling = (high - frequency) / (high - center)
                weights[filter * bins + bin] = Float(max(0, min(rising, falling)) * norm)
            }
        }
        return MLXArray(weights, [count, bins])
    }

    static func hertzToMel(_ hertz: Double) -> Double {
        hertz < 1_000 ? hertz * 3 / 200 : 15 + Foundation.log(hertz / 1_000) / logStep
    }

    static func melToHertz(_ mel: Double) -> Double {
        mel < 15 ? mel * 200 / 3 : 1_000 * Foundation.exp(logStep * (mel - 15))
    }

    static var logStep: Double { Foundation.log(6.4) / 27 }

    static func hann() -> MLXArray {
        MLXArray((0 ..< fftSize).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fftSize))) })
    }

    static func reflectPadded(_ samples: ArraySlice<Float>, frames: Int) -> [Float] {
        let length = frames * hopLength
        let pad = fftSize / 2
        let window = Array(samples.prefix(length)) + [Float](repeating: 0, count: max(0, length - samples.count))
        let left = (1 ... pad).reversed().map { window[min($0, length - 1)] }
        let right = (1 ... pad).map { window[max(length - 1 - $0, 0)] }
        return left + window + right
    }

    static func logMel(_ samples: ArraySlice<Float>, filters: MLXArray, frames: Int) -> MLXArray {
        let padded = MLXArray(reflectPadded(samples, frames: frames))
        let starts = MLXArray(Int32(0) ..< Int32(frames)).expandedDimensions(axis: 1) * Int32(hopLength)
        let offsets = MLXArray(Int32(0) ..< Int32(fftSize)).expandedDimensions(axis: 0)
        let spectrum = MLXFFT.rfft(padded[starts + offsets] * hann(), axis: -1)
        let mel = matmul(abs(spectrum).square(), filters.transposed())
        let logSpectrum = log10(maximum(mel, Float(1e-10)))
        return (maximum(logSpectrum, logSpectrum.max() - Float(8)) + Float(4)) / Float(4)
    }
}
