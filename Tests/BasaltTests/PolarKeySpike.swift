import Foundation
import MLX

let twoPi = 2 * Float.pi
@testable import MLXLMCommon
@testable import BasaltInference

enum PolarKeySpike {
    struct Rope {
        let rotated: Int
        let theta: Float

        var pairs: Int { rotated / 2 }
        func omega(_ m: Int) -> Float { pow(theta, -2 * Float(m) / Float(rotated)) }
        func slow(_ m: Int) -> Bool { omega(m) * 32768 < 2 * .pi }
    }

    struct Scores {
        var logitError: Float
        var kl: Float
        var overlap: Float
    }

    static func affine(_ keys: MLXArray, bits: Int) -> MLXArray {
        let q = quantized(keys, groupSize: 64, bits: bits)
        return dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: bits)
    }

    static func turbo(_ keys: MLXArray, bits: Int) -> MLXArray {
        let codec = MSECodec(dim: keys.dim(-1), bits: bits)
        return codec.decode(codec.encode(keys.asType(.float32))).asType(keys.dtype)
    }

    static func split(_ keys: MLXArray, rope: Rope) -> (re: MLXArray, im: MLXArray, rest: MLXArray?) {
        let half = rope.pairs
        let rest = keys.dim(-1) > rope.rotated ? keys[.ellipsis, rope.rotated...] : nil
        return (keys[.ellipsis, ..<half].asType(.float32), keys[.ellipsis, half ..< rope.rotated].asType(.float32), rest)
    }

    static func lloydMax(_ r: MLXArray, levels: Int, iterations: Int = 8) -> MLXArray {
        let sortedR = sorted(r, axis: -1)
        let count = r.dim(-1)
        var centers = concatenated((0 ..< levels).map { sortedR[.ellipsis, ((2 * $0 + 1) * count / (2 * levels))].expandedDimensions(axis: -1) }, axis: -1)
        for _ in 0 ..< iterations {
            let assign = argMin(abs(r.expandedDimensions(axis: -1) - centers.expandedDimensions(axis: -2)), axis: -1)
            let onehot = (assign.expandedDimensions(axis: -1) .== MLXArray(0 ..< levels).reshaped(1, 1, 1, levels)).asType(.float32)
            let sums = (onehot * r.expandedDimensions(axis: -1)).sum(axis: -2)
            let counts = onehot.sum(axis: -2)
            centers = MLX.where(counts .> 0, sums / maximum(counts, 1), centers)
            eval(centers)
        }
        return centers
    }

    static func polar(_ keys: MLXArray, rope: Rope, phaseBits: Int, modulusBits: Int, restBits: Int) -> MLXArray {
        let (re, im, rest) = split(keys, rope: rope)
        let r = sqrt(re * re + im * im)
        let phi = atan2(im, re)
        let steps = Float(1 << phaseBits)
        let codes: MLXArray = round(phi * (steps / twoPi))
        let phiHat: MLXArray = codes * (twoPi / steps)
        let pairsFirst = r.squeezed(axis: 0).transposed(0, 2, 1)
        let centers = lloydMax(pairsFirst, levels: 1 << modulusBits)
        let index = argMin(abs(pairsFirst.expandedDimensions(axis: -1) - centers.expandedDimensions(axis: -2)), axis: -1)
        let rHat = takeAlong(centers, index, axis: -1).transposed(0, 2, 1).expandedDimensions(axis: 0)
        let rotated = concatenated([rHat * cos(phiHat), rHat * sin(phiHat)], axis: -1).asType(keys.dtype)
        return rest.map { concatenated([rotated, affine($0, bits: restBits)], axis: -1) } ?? rotated
    }

    static func scores(queries: [(position: Int, query: MLXArray)], keys: MLXArray, approx: MLXArray) -> Scores {
        var (errors, kls, overlaps) = ([Float](), [Float](), [Float]())
        let groups = (queries.first?.query.dim(1) ?? keys.dim(1)) / keys.dim(1)
        let expand = { (k: MLXArray) in repeated(k.asType(.float32), count: groups, axis: 1) }
        let (full, rough) = (expand(keys), expand(approx))
        let scale = 1 / Float(keys.dim(-1)).squareRoot()
        for (position, query) in queries {
            let visible = min(position, keys.dim(2))
            let exact = matmul(query, full[.ellipsis, ..<visible, 0...].swappedAxes(-1, -2)) * scale
            let approximate = matmul(query, rough[.ellipsis, ..<visible, 0...].swappedAxes(-1, -2)) * scale
            errors.append((sqrt((approximate - exact).square().sum()) / sqrt(exact.square().sum())).item(Float.self))
            let p = softmax(exact, axis: -1)
            let logq: MLXArray = approximate - logSumExp(approximate, axis: -1, keepDims: true)
            kls.append((p * (log(maximum(p, 1e-30)) - logq)).sum(axis: -1).mean().item(Float.self))
            let k = min(32, visible)
            let top = argSort(-exact, axis: -1)[.ellipsis, ..<k]
            let topApprox = argSort(-approximate, axis: -1)[.ellipsis, ..<k]
            let shared = (top.expandedDimensions(axis: -1) .== topApprox.expandedDimensions(axis: -2)).any(axis: -1).asType(.float32)
            overlaps.append(shared.mean().item(Float.self))
        }
        let mean = { (values: [Float]) in values.reduce(0, +) / Float(max(values.count, 1)) }
        return Scores(logitError: mean(errors), kl: mean(kls), overlap: mean(overlaps))
    }

    static func pairStatistics(_ keys: MLXArray, rope: Rope, phaseBits: Int) -> (slow: [Float], fast: [Float], entropySlow: Float, entropyFast: Float, ksSlow: Float, ksFast: Float) {
        let (re, im, _) = split(keys, rope: rope)
        let r = sqrt(re * re + im * im)
        let safe: MLXArray = maximum(r, MLXArray(Float(1e-12)))
        let meanCos: MLXArray = (re / safe).mean(axis: 2)
        let meanSin: MLXArray = (im / safe).mean(axis: 2)
        let resultantArray: MLXArray = sqrt(meanCos.square() + meanSin.square())
        let resultant = resultantArray.mean(axes: [0, 1]).asArray(Float.self)
        let bins = 1 << phaseBits
        let angle: MLXArray = atan2(im, re) + Float.pi
        let phase: MLXArray = floor(angle * (Float(bins) / twoPi)).asType(.int32)
        let binIndex: MLXArray = MLXArray(0 ..< Int32(bins)).reshaped(1, 1, 1, 1, bins)
        let histogram: MLXArray = (phase.expandedDimensions(axis: -1) .== binIndex).asType(.float32).mean(axis: 2)
        let plogp: MLXArray = histogram * log2(maximum(histogram, MLXArray(Float(1e-12))))
        let entropyArray: MLXArray = -plogp.sum(axis: -1) / Float(phaseBits)
        let entropy = entropyArray.mean(axes: [0, 1]).asArray(Float.self)
        let sigma2: MLXArray = (r * r).mean(axis: 2, keepDims: true) / 2
        let sortedR: MLXArray = sorted(r, axis: 2)
        let n = Float(r.dim(2))
        let empirical: MLXArray = MLXArray(1 ... r.dim(2)).asType(.float32).reshaped(1, 1, -1, 1) / n
        let rayleigh: MLXArray = 1 - exp(-sortedR * sortedR / (2 * sigma2))
        let ksArray: MLXArray = abs(empirical - rayleigh).max(axis: 2)
        let ks = ksArray.mean(axes: [0, 1]).asArray(Float.self)
        let slowPairs = (0 ..< rope.pairs).filter(rope.slow)
        let fastPairs = (0 ..< rope.pairs).filter { !rope.slow($0) }
        let average = { (values: [Float], pairs: [Int]) in pairs.isEmpty ? Float.nan : pairs.map { values[$0] }.reduce(0, +) / Float(pairs.count) }
        return (slowPairs.map { resultant[$0] }, fastPairs.map { resultant[$0] }, average(entropy, slowPairs), average(entropy, fastPairs), average(ks, slowPairs), average(ks, fastPairs))
    }

    static func rotate(_ keys: MLXArray, rope: Rope, by delta: Float) -> MLXArray {
        let (re, im, rest) = split(keys, rope: rope)
        let angle = MLXArray((0 ..< rope.pairs).map { rope.omega($0) * delta })
        let (c, s) = (cos(angle), sin(angle))
        let rotated = concatenated([re * c - im * s, re * s + im * c], axis: -1).asType(keys.dtype)
        return rest.map { concatenated([rotated, $0], axis: -1) } ?? rotated
    }

    static func shiftedPolarPhaseError(_ keys: MLXArray, rope: Rope, phaseBits: Int, delta: Int) -> (maxError: Float, rmsError: Float) {
        let (re, im, _) = split(keys, rope: rope)
        let steps = Float(1 << phaseBits)
        let phi = atan2(im, re)
        let code: MLXArray = round(phi * (steps / twoPi))
        let shift = MLXArray((0 ..< rope.pairs).map { (steps * rope.omega($0) * Float(delta) / twoPi).rounded() })
        let phiShifted: MLXArray = (code + shift) * (twoPi / steps)
        let truth = phi + MLXArray((0 ..< rope.pairs).map { rope.omega($0) * Float(delta) })
        let error = abs(atan2(sin(phiShifted - truth), cos(phiShifted - truth)))
        return (error.max().item(Float.self), sqrt(error.square().mean()).item(Float.self))
    }
}
