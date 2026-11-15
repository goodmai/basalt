import CryptoKit
import Foundation

public struct CalibrationSample: Sendable, Codable, Equatable {
    public var stateHash, questionID: String
    public var kind: SystemOneKind
    public var rawScores, teacherProbabilities: [Double]
    public var goldIndex: Int?
    public init(stateHash: String, questionID: String, kind: SystemOneKind, rawScores: [Double], teacherProbabilities: [Double] = [], goldIndex: Int? = nil) {
        (self.stateHash, self.questionID, self.kind, self.rawScores, self.teacherProbabilities, self.goldIndex) = (stateHash, questionID, kind, rawScores, teacherProbabilities, goldIndex)
    }
}

public struct CalibrationReliabilityBin: Sendable, Codable, Equatable {
    public var lower, upper: Double
    public var count: Int
    public var meanConfidence, meanAccuracy: Double
    public init(lower: Double, upper: Double, count: Int, meanConfidence: Double, meanAccuracy: Double) {
        (self.lower, self.upper, self.count, self.meanConfidence, self.meanAccuracy) = (lower, upper, count, meanConfidence, meanAccuracy)
    }
}

public struct CalibrationFit: Sendable, Codable, Equatable {
    public typealias Bin = CalibrationReliabilityBin
    public var readout: SystemOneReadout
    public var sampleCount: Int
    public var klBefore, klAfter, eceBefore, eceAfter, brierBefore, brierAfter: Double
    public var binsBefore, binsAfter: [CalibrationReliabilityBin]
    public init(readout: SystemOneReadout, sampleCount: Int, before: SystemOneCalibrationCore.Metrics, after: SystemOneCalibrationCore.Metrics) {
        (self.readout, self.sampleCount, self.klBefore, self.klAfter) = (readout, sampleCount, before.kl, after.kl)
        (self.eceBefore, self.eceAfter, self.brierBefore, self.brierAfter, self.binsBefore, self.binsAfter) = (before.ece, after.ece, before.brier, after.brier, before.bins, after.bins)
    }
}

public enum SystemOneCalibrationCore {
    public typealias CalibrationBin = CalibrationReliabilityBin
    public struct Metrics: Sendable, Codable, Equatable {
        public var kl, ece, brier: Double
        public var bins: [CalibrationReliabilityBin]
        public init(kl: Double, ece: Double, brier: Double, bins: [CalibrationReliabilityBin]) { (self.kl, self.ece, self.brier, self.bins) = (kl, ece, brier, bins) }
    }
    public static let temperatureBounds = 0.05 ... 20.0, biasBounds = -10.0 ... 10.0, binCount = 10

    public static func probabilities(for sample: CalibrationSample, readout: SystemOneReadout) -> [Double] { probabilities(kind: sample.kind, rawScores: sample.rawScores, readout: readout) }

    public static func probabilities(kind: SystemOneKind, rawScores: [Double], readout: SystemOneReadout) -> [Double] {
        guard kind == .noul else { return softmax(rawScores, temperature: readout.temperature) }
        let logits = rawScores.count == 1 ? [rawScores[0], 0.0] : Array(rawScores.prefix(2))
        guard logits.count == 2 else { return [0.5, 0.5] }
        let pYes = sigmoid(noulLogit(logits, baseTemperature: readout.temperature) / readout.noulTemperature + readout.noulBias)
        return [pYes, 1.0 - pYes]
    }

    public static func targetProbabilities(for sample: CalibrationSample) -> [Double] {
        let count = sample.kind == .noul ? 2 : sample.rawScores.count
        guard count >= 2 else { return [] }
        if sample.kind == .noul, sample.teacherProbabilities.count == 1, sample.teacherProbabilities[0].isFinite {
            let yes = min(max(sample.teacherProbabilities[0], 0), 1)
            return [yes, 1.0 - yes]
        }
        if sample.teacherProbabilities.count == count {
            let clean = sample.teacherProbabilities.map { $0.isFinite ? max($0, 0) : 0 }
            let sum = clean.reduce(0, +)
            if sum > 0 { return clean.map { $0 / sum } }
        }
        if let gold = sample.goldIndex, (0 ..< count).contains(gold) { return (0 ..< count).map { $0 == gold ? 1.0 : 0.0 } }
        return Array(repeating: 1.0 / Double(count), count: count)
    }

    public static func fit(_ samples: [CalibrationSample], holdoutFraction: Double = 0) -> CalibrationFit {
        let valid = samples.filter { $0.rawScores.count >= ($0.kind == .noul ? 1 : 2) && $0.rawScores.allSatisfy(\.isFinite) }
        let (train, eval) = split(valid, holdoutFraction: holdoutFraction)
        let (multi, noul) = (train.filter { $0.kind != .noul }, train.filter { $0.kind == .noul })
        let temperature = multi.isEmpty ? 1.0 : fitTemperature(multi)
        let (noulTemperature, noulBias) = noul.isEmpty ? (1.0, 0.0) : fitNoul(noul, baseTemperature: temperature)
        let readout = SystemOneReadout(temperature: temperature, noulTemperature: noulTemperature, noulBias: noulBias)
        return CalibrationFit(readout: readout, sampleCount: eval.count, before: evaluate(eval, readout: .uncalibrated), after: evaluate(eval, readout: readout))
    }

    public static func evaluate(_ samples: [CalibrationSample], readout: SystemOneReadout) -> Metrics {
        let valid = samples.filter { $0.rawScores.count >= ($0.kind == .noul ? 1 : 2) && $0.rawScores.allSatisfy(\.isFinite) }
        guard !valid.isEmpty else {
            return Metrics(kl: 0, ece: 0, brier: 0, bins: (0 ..< binCount).map { CalibrationReliabilityBin(lower: Double($0) / Double(binCount), upper: Double($0 + 1) / Double(binCount), count: 0, meanConfidence: 0, meanAccuracy: 0) })
        }
        var (klSum, brierSum) = (0.0, 0.0)
        var (counts, confSums, accSums) = (Array(repeating: 0, count: binCount), Array(repeating: 0.0, count: binCount), Array(repeating: 0.0, count: binCount))
        for sample in valid {
            let (p, q) = (probabilities(for: sample, readout: readout), targetProbabilities(for: sample))
            for k in p.indices where q[k] > 0 { klSum += q[k] * log(q[k] / max(p[k], 1e-15)) }
            let pred = p.indices.max(by: { p[$0] < p[$1] }) ?? 0
            let (conf, hasGold) = (min(max(p[pred], 0), 1), sample.goldIndex.map(p.indices.contains) == true)
            let acc = hasGold ? (pred == sample.goldIndex ? 1.0 : 0.0) : q[pred]
            for k in p.indices { let d = p[k] - (hasGold ? (k == sample.goldIndex ? 1.0 : 0.0) : q[k]); brierSum += 0.5 * d * d }
            let idx = min(binCount - 1, max(0, Int(conf * Double(binCount))))
            (counts[idx], confSums[idx], accSums[idx]) = (counts[idx] + 1, confSums[idx] + conf, accSums[idx] + acc)
        }
        let n = Double(valid.count)
        let bins = (0 ..< binCount).map { i in CalibrationReliabilityBin(lower: Double(i) / Double(binCount), upper: Double(i + 1) / Double(binCount), count: counts[i], meanConfidence: counts[i] > 0 ? confSums[i] / Double(counts[i]) : 0, meanAccuracy: counts[i] > 0 ? accSums[i] / Double(counts[i]) : 0) }
        return Metrics(kl: klSum / n, ece: bins.reduce(0.0) { $0 + (Double($1.count) / n) * abs($1.meanAccuracy - $1.meanConfidence) }, brier: brierSum / n, bins: bins)
    }

    public static func stateHash(for state: String) -> String { hexSHA256(Data(state.utf8)) }
    public static func cacheKey(teacherModel: String, stateHash: String, questionSpec: String) -> String { hexSHA256(Data("\(teacherModel)\0\(stateHash)\0\(questionSpec)".utf8)) }

    public static func cacheKey(teacherModel: String, stateHash: String, question: SystemOneQuestion) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let spec = (try? encoder.encode(question)).map { String(decoding: $0, as: UTF8.self) } ?? "\(question.type.rawValue):\(question.instructions)"
        return cacheKey(teacherModel: teacherModel, stateHash: stateHash, questionSpec: spec)
    }

    public static func encodeJSONL(_ samples: [CalibrationSample]) throws(BasaltError) -> Data {
        guard !samples.isEmpty else { return Data() }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var lines: [String] = []
        for sample in samples {
            guard let data = try? encoder.encode(sample) else { throw .invalidRequestStructure(details: "calibration: sample \(sample.questionID) could not be encoded") }
            lines.append(String(decoding: data, as: UTF8.self))
        }
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    public static func decodeJSONL(_ data: Data) throws(BasaltError) -> [CalibrationSample] {
        let decoder = JSONDecoder()
        var samples: [CalibrationSample] = []
        for raw in String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            guard let sample = try? decoder.decode(CalibrationSample.self, from: Data(line.utf8)) else { throw .invalidRequestStructure(details: "calibration: malformed JSONL sample") }
            samples.append(sample)
        }
        return samples
    }
}

extension SystemOneCalibrationCore {
    static func softmax(_ scores: [Double], temperature: Double) -> [Double] {
        guard !scores.isEmpty else { return [] }
        let z = scores.map { $0 / temperature }
        let top = z.max() ?? 0
        let weights = z.map { exp($0 - top) }
        let total = weights.reduce(0, +)
        return total > 0 ? weights.map { $0 / total } : Array(repeating: 1.0 / Double(scores.count), count: scores.count)
    }

    static func noulLogit(_ logits: [Double], baseTemperature: Double) -> Double {
        let yes = min(max(softmax(logits, temperature: baseTemperature).first ?? 0.5, 1e-4), 1 - 1e-4)
        return log(yes / (1.0 - yes))
    }

    static func sigmoid(_ x: Double) -> Double { 1.0 / (1.0 + exp(-x)) }
    static func hexSHA256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    static func split(_ samples: [CalibrationSample], holdoutFraction: Double) -> (train: [CalibrationSample], eval: [CalibrationSample]) {
        guard holdoutFraction > 0, holdoutFraction < 1, samples.count >= 5 else { return (samples, samples) }
        var (train, eval) = ([CalibrationSample](), [CalibrationSample]())
        for stratum in [samples.filter { $0.kind != .noul }, samples.filter { $0.kind == .noul }] where !stratum.isEmpty {
            guard stratum.count >= 2 else { train.append(contentsOf: stratum); continue }
            let hold = min(stratum.count - 1, max(1, Int((Double(stratum.count) * holdoutFraction).rounded())))
            let (stride, count) = (max(2, stratum.count / hold), stratum.count)
            var taken = 0
            for (i, item) in stratum.enumerated() {
                if taken < hold, (i + 1) % stride == 0 || count - i == hold - taken { eval.append(item); taken += 1 } else { train.append(item) }
            }
        }
        return train.isEmpty || eval.isEmpty ? (samples, samples) : (train, eval)
    }

    static func fitTemperature(_ samples: [CalibrationSample]) -> Double {
        let pairs = samples.map { ($0.rawScores, targetProbabilities(for: $0)) }
        func loss(_ beta: Double) -> Double {
            pairs.reduce(0.0) { acc, pair in
                let top = pair.0.max() ?? 0
                return acc + log(pair.0.reduce(0.0) { $0 + exp(beta * ($1 - top)) }) - beta * zip(pair.1, pair.0).reduce(0.0) { $0 + $1.0 * ($1.1 - top) }
            } / Double(pairs.count)
        }
        let phi = 0.6180339887498949
        var (lo, hi) = (temperatureBounds.lowerBound, temperatureBounds.upperBound)
        var (x1, x2) = (hi - phi * (hi - lo), lo + phi * (hi - lo))
        var (f1, f2) = (loss(x1), loss(x2))
        for _ in 0 ..< 44 {
            if f1 <= f2 { (hi, x2, f2) = (x2, x1, f1); x1 = hi - phi * (hi - lo); f1 = loss(x1) } else { (lo, x1, f1) = (x1, x2, f2); x2 = lo + phi * (hi - lo); f2 = loss(x2) }
        }
        var beta = 0.5 * (lo + hi)
        for _ in 0 ..< 12 {
            var (grad, hess) = (0.0, 0.0)
            for (z, q) in pairs {
                let p = softmax(z, temperature: 1.0 / beta)
                let mu = zip(p, z).reduce(0.0) { $0 + $1.0 * $1.1 }
                grad += mu - zip(q, z).reduce(0.0) { $0 + $1.0 * $1.1 }
                hess += zip(p, z).reduce(0.0) { $0 + $1.0 * ($1.1 - mu) * ($1.1 - mu) }
            }
            guard hess > 1e-12, let next = [1.0, 0.5, 0.25, 0.125].map({ min(max(beta - $0 * grad / hess, temperatureBounds.lowerBound), temperatureBounds.upperBound) }).first(where: { loss($0) < loss(beta) - 1e-15 }) else { break }
            beta = next
        }
        if loss(1.0) <= loss(beta) { beta = 1.0 }
        return min(max(1.0 / beta, temperatureBounds.lowerBound), temperatureBounds.upperBound)
    }

    static func fitNoul(_ samples: [CalibrationSample], baseTemperature: Double) -> (Double, Double) {
        let pairs: [(u: Double, q: Double)] = samples.map { s in
            let logits = s.rawScores.count == 1 ? [s.rawScores[0], 0.0] : Array(s.rawScores.prefix(2))
            return (noulLogit(logits, baseTemperature: baseTemperature), targetProbabilities(for: s).first ?? 0.5)
        }
        var (alpha, bias) = (1.0, 0.0)
        for _ in 0 ..< 60 {
            let s = noulStats(pairs, alpha: alpha, bias: bias)
            let freeA = !((alpha <= temperatureBounds.lowerBound && s.gA >= 0) || (alpha >= temperatureBounds.upperBound && s.gA <= 0))
            let freeB = !((bias <= biasBounds.lowerBound && s.gB >= 0) || (bias >= biasBounds.upperBound && s.gB <= 0))
            if !freeA && !freeB { break }
            let (dA, dB) = noulDirection(s, freeA: freeA, freeB: freeB)
            var (step, moved) = (1.0, false)
            for _ in 0 ..< 32 {
                let (nextA, nextB) = (min(max(alpha + step * dA, temperatureBounds.lowerBound), temperatureBounds.upperBound), min(max(bias + step * dB, biasBounds.lowerBound), biasBounds.upperBound))
                if noulStats(pairs, alpha: nextA, bias: nextB).loss < s.loss - 1e-15 { (alpha, bias, moved) = (nextA, nextB, true); break }
                step *= 0.5
            }
            if !moved { break }
        }
        return (min(max(1.0 / alpha, temperatureBounds.lowerBound), temperatureBounds.upperBound), min(max(bias, biasBounds.lowerBound), biasBounds.upperBound))
    }

    private struct NoulStats { let loss, gA, gB, hAA, hAB, hBB: Double }
    private static func noulStats(_ pairs: [(u: Double, q: Double)], alpha: Double, bias: Double) -> NoulStats {
        var (loss, gA, gB, hAA, hAB, hBB) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for (u, q) in pairs {
            let (x, p) = (alpha * u + bias, sigmoid(alpha * u + bias))
            let (r, w) = (p - q, p * (1.0 - p))
            loss += max(x, 0) + log(1.0 + exp(-abs(x))) - q * x
            (gA, gB, hAA, hAB, hBB) = (gA + r * u, gB + r, hAA + w * u * u, hAB + w * u, hBB + w)
        }
        let n = Double(pairs.count)
        return NoulStats(loss: loss / n, gA: gA / n, gB: gB / n, hAA: hAA / n, hAB: hAB / n, hBB: hBB / n)
    }

    private static func noulDirection(_ s: NoulStats, freeA: Bool, freeB: Bool) -> (Double, Double) {
        var (dA, dB) = (0.0, 0.0)
        if freeA && freeB {
            let (a, b, d) = (s.hAA + 1e-10, s.hAB, s.hBB + 1e-10)
            let det = a * d - b * b
            (dA, dB) = det > 1e-18 ? (-(d * s.gA - b * s.gB) / det, -(-b * s.gA + a * s.gB) / det) : (-s.gA, -s.gB)
        } else if freeA { dA = -s.gA / (s.hAA + 1e-10) } else if freeB { dB = -s.gB / (s.hBB + 1e-10) }
        if (freeA ? s.gA * dA : 0) + (freeB ? s.gB * dB : 0) >= 0 { (dA, dB) = (freeA ? -s.gA : 0, freeB ? -s.gB : 0) }
        return (dA, dB)
    }
}
