import Foundation
import MLX

enum ParakeetAudio {
    static let logGuard = 1 / Float(1 << 24)
    static let deviationFloor: Float = 1e-5

    static func window(length: Int, fftSize: Int) -> MLXArray {
        let left = (fftSize - length) / 2
        let denominator = Double(max(length - 1, 1))
        return MLXArray((0 ..< fftSize).map { index -> Float in
            let n = index - left
            guard (0 ..< length).contains(n) else { return 0 }
            return Float(0.5 - 0.5 * cos(2 * Double.pi * Double(n) / denominator))
        })
    }

    static func frameCount(samples: Int, preprocessor: ParakeetConfiguration.Preprocessor) -> Int {
        (samples + 2 * (preprocessor.fftSize / 2) - preprocessor.fftSize) / preprocessor.hopLength + 1
    }

    static func logMel(
        _ samples: ArraySlice<Float>, preprocessor: ParakeetConfiguration.Preprocessor, filters: MLXArray
    ) -> MLXArray {
        let fftSize = preprocessor.fftSize
        let signal = Array(samples)
        let emphasized = signal.indices.map { $0 == 0 ? signal[0] : signal[$0] - preprocessor.preemphasis * signal[$0 - 1] }
        let half = [Float](repeating: 0, count: fftSize / 2)
        let padded = MLXArray(half + emphasized + half)
        let frames = frameCount(samples: signal.count, preprocessor: preprocessor)
        let starts = MLXArray(Int32(0) ..< Int32(frames)).expandedDimensions(axis: 1) * Int32(preprocessor.hopLength)
        let offsets = MLXArray(Int32(0) ..< Int32(fftSize)).expandedDimensions(axis: 0)
        let windowed = padded[starts + offsets] * window(length: preprocessor.windowLength, fftSize: fftSize)
        let power = abs(MLXFFT.rfft(windowed, axis: -1)).square()
        let mel = log(matmul(power, filters.transposed()) + logGuard)
        return normalized(mel, perFeature: preprocessor.normalizesPerFeature)
    }

    static func normalized(_ mel: MLXArray, perFeature: Bool) -> MLXArray {
        let axes = perFeature ? [0] : [0, 1]
        let count = axes.reduce(1) { $0 * mel.dim($1) }
        let mean = mel.mean(axes: axes, keepDims: true)
        let deviation = sqrt((mel - mean).square().sum(axes: axes, keepDims: true) / Float(max(count - 1, 1))) + deviationFloor
        return (mel - mean) / deviation
    }
}
