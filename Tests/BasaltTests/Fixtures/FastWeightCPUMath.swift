import Foundation
import MLX
import MLXNN
import Testing

struct CPUSlidingAttentionInput {
    let q: [[[Float]]]
    let k: [[[Float]]]
    let v: [[[Float]]]
    let kPersistent: [[[Float]]]
    let vPersistent: [[[Float]]]
    let windowSize: Int
}

enum FastWeightCPUMath {
    static func flatWeights(_ model: Module) -> [String: [Float]] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, $0.1.asType(.float32).asArray(Float.self))
        })
    }

    static func embedding(_ tokens: [Int32], weight: [Float], hiddenSize: Int) -> [[Float]] {
        tokens.map { tok in
            let base = Int(tok) * hiddenSize
            return Array(weight[base ..< (base + hiddenSize)])
        }
    }

    static func linear(_ x: [[Float]], weight: [Float], bias: [Float]? = nil, outDim: Int) -> [[Float]] {
        let inDim = x.first?.count ?? 0
        return x.map { row in
            (0 ..< outDim).map { o in
                var acc = bias?[o] ?? 0
                let offset = o * inDim
                for i in 0 ..< inDim {
                    acc += row[i] * weight[offset + i]
                }
                return acc
            }
        }
    }

    static func rmsNorm(_ x: [[Float]], weight: [Float], eps: Float) -> [[Float]] {
        x.map { row in
            let d = Float(row.count)
            let meanSq = row.reduce(Float(0)) { $0 + $1 * $1 } / d
            let inv = Float(1.0) / sqrt(meanSq + eps)
            return zip(row, weight).map { $0 * inv * $1 }
        }
    }

    static func layerNorm(_ x: [[Float]], weight: [Float], bias: [Float], eps: Float) -> [[Float]] {
        x.map { row in
            layerNormVec(row, weight: weight, bias: bias, eps: eps)
        }
    }

    static func layerNormVec(_ row: [Float], weight: [Float], bias: [Float], eps: Float) -> [Float] {
        let d = Float(row.count)
        let mu = row.reduce(Float(0), +) / d
        let variance = row.reduce(Float(0)) { $0 + ($1 - mu) * ($1 - mu) } / d
        let inv = Float(1.0) / sqrt(variance + eps)
        return (0 ..< row.count).map { i in (row[i] - mu) * inv * weight[i] + bias[i] }
    }

    static func lnFusedL2BackwardVec(
        x: [Float],
        target: [Float],
        weight: [Float],
        bias: [Float],
        eps: Float
    ) -> [Float] {
        let d = Float(x.count)
        let mu = x.reduce(Float(0), +) / d
        let variance = x.reduce(Float(0)) { $0 + ($1 - mu) * ($1 - mu) } / d
        let inv = Float(1.0) / sqrt(variance + eps)
        let xHat = x.map { ($0 - mu) * inv }
        let gXHat = (0 ..< x.count).map { i in (xHat[i] * weight[i] + bias[i] - target[i]) * weight[i] }
        let c1 = gXHat.reduce(Float(0), +) / d
        let c2 = zip(gXHat, xHat).reduce(Float(0)) { $0 + $1.0 * $1.1 } / d
        return (0 ..< x.count).map { i in inv * (gXHat[i] - c1 - xHat[i] * c2) }
    }

    static func depthwiseConv1d(
        _ x: [[Float]],
        weight: [Float],
        bias: [Float]?,
        kernel: Int,
        applySiLU: Bool
    ) -> [[Float]] {
        let steps = x.count
        let channels = x.first?.count ?? 0
        return (0 ..< steps).map { t in
            (0 ..< channels).map { c in
                var acc = bias?[c] ?? 0
                for k in 0 ..< kernel {
                    let src = t + k - (kernel - 1)
                    if src >= 0 {
                        acc += x[src][c] * weight[c * kernel + k]
                    }
                }
                return applySiLU ? siluScalar(acc) : acc
            }
        }
    }

    static func splitHeads(_ x: [[Float]], heads: Int, headDim: Int) -> [[[Float]]] {
        (0 ..< heads).map { h in
            let start = h * headDim
            return x.map { Array($0[start ..< (start + headDim)]) }
        }
    }

    static func mergeHeads(_ x: [[[Float]]]) -> [[Float]] {
        let heads = x.count
        let steps = x.first?.count ?? 0
        return (0 ..< steps).map { t in
            var row: [Float] = []
            for h in 0 ..< heads { row.append(contentsOf: x[h][t]) }
            return row
        }
    }

    static func applyRoPE(_ x: [[[Float]]], theta: Float) -> [[[Float]]] {
        let d = x.first?.first?.count ?? 0
        let half = d / 2
        let invFreq = (0 ..< half).map { j in
            pow(Double(theta), -Double(2 * j) / Double(d))
        }
        return x.map { headSeq in
            headSeq.enumerated().map { t, vec in
                var out = vec
                for j in 0 ..< half {
                    let angle = Double(t) * invFreq[j]
                    let c = Float(cos(angle))
                    let s = Float(sin(angle))
                    let x0 = vec[j]
                    let x1 = vec[j + half]
                    out[j] = x0 * c - x1 * s
                    out[j + half] = x0 * s + x1 * c
                }
                return out
            }
        }
    }

    static func swiGLU(
        _ x: [[Float]],
        gateW: [Float],
        upW: [Float],
        downW: [Float],
        interDim: Int
    ) -> [[Float]] {
        let outDim = x.first?.count ?? 0
        let gate = linear(x, weight: gateW, outDim: interDim)
        let up = linear(x, weight: upW, outDim: interDim)
        let hidden = zip(gate, up).map { gRow, uRow in
            zip(gRow, uRow).map { siluScalar($0) * $1 }
        }
        return linear(hidden, weight: downW, outDim: outDim)
    }

    static func slidingAttention(_ input: CPUSlidingAttentionInput) -> [[Float]] {
        let heads = input.q.count
        let kvHeads = input.k.count
        let group = heads / kvHeads
        let steps = input.q.first?.count ?? 0
        let d = input.q.first?.first?.count ?? 0
        let scale = Float(1.0) / Float(d).squareRoot()
        let perHead: [[[Float]]] = (0 ..< heads).map { h in
            let kvH = h / group
            let kAll = input.kPersistent[kvH] + input.k[kvH]
            let vAll = input.vPersistent[kvH] + input.v[kvH]
            let pCount = input.kPersistent[kvH].count
            return (0 ..< steps).map { t in
                attendRow(
                    query: input.q[h][t],
                    keys: kAll,
                    values: vAll,
                    visible: Array(0 ..< pCount) + Array((pCount + Swift.max(0, t - input.windowSize + 1)) ... (pCount + t)),
                    scale: scale
                )
            }
        }
        return mergeHeads(perHead)
    }

    private static func attendRow(
        query: [Float],
        keys: [[Float]],
        values: [[Float]],
        visible: [Int],
        scale: Float
    ) -> [Float] {
        let scores = visible.map { idx in
            zip(query, keys[idx]).reduce(Float(0)) { $0 + $1.0 * $1.1 } * scale
        }
        let maxScore = scores.max() ?? 0
        let exps = scores.map { exp($0 - maxScore) }
        let sumExp = exps.reduce(Float(0), +)
        var out = Array(repeating: Float(0), count: query.count)
        for (pos, idx) in visible.enumerated() {
            let w = exps[pos] / sumExp
            for c in 0 ..< out.count { out[c] += w * values[idx][c] }
        }
        return out
    }
}

extension FastWeightCPUMath {
    static func add(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
        zip(a, b).map { zip($0, $1).map(+) }
    }

    static func sigmoidScalar(_ x: Float) -> Float {
        Float(1.0) / (Float(1.0) + exp(-x))
    }

    static func siluScalar(_ x: Float) -> Float {
        x * sigmoidScalar(x)
    }

    static func siluDerivScalar(_ x: Float) -> Float {
        let s = sigmoidScalar(x)
        return s * (Float(1.0) + x * (Float(1.0) - s))
    }

    static func geluTanhScalar(_ x: Float) -> Float {
        let coeff: Float = 0.7978845608028654
        let inner = coeff * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + tanh(inner))
    }

    static func geluTanhDerivScalar(_ x: Float) -> Float {
        let coeff: Float = 0.7978845608028654
        let x2 = x * x
        let inner = coeff * (x + 0.044715 * x2 * x)
        let t = tanh(inner)
        let dInner = coeff * (1.0 + 3.0 * 0.044715 * x2)
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dInner
    }

    static func l2NormVec(_ v: [Float], eps: Float = 1e-6) -> [Float] {
        let denom = sqrt(v.reduce(Float(0)) { $0 + $1 * $1 } + eps * eps)
        return v.map { $0 / denom }
    }

    static func maxAbsDiff(_ array: MLXArray, _ reference: [[Float]]) -> Float {
        let actual = array.asType(.float32).asArray(Float.self)
        let flat = reference.flatMap { $0 }
        guard actual.count == flat.count else { return .infinity }
        return zip(actual, flat).reduce(Float(0)) { Swift.max($0, abs($1.0 - $1.1)) }
    }

    static func argMaxRows(_ rows: [[Float]]) -> [Int32] {
        rows.map { row in
            var bestIdx = 0
            var bestVal = row[0]
            for i in 1 ..< row.count where row[i] > bestVal {
                bestVal = row[i]
                bestIdx = i
            }
            return Int32(bestIdx)
        }
    }

    static func nextTokenNLL(logits: MLXArray, tokens: [Int32], positions: Range<Int>) -> Float {
        let z = logits[0].asType(.float32)
        let logProbs = z - logSumExp(z, axis: -1, keepDims: true)
        var total: Float = 0
        for pos in positions {
            let target = Int(tokens[pos + 1])
            total -= logProbs[pos, target].item(Float.self)
        }
        return total / Float(positions.count)
    }

    static func transitionMatrix(factors: MLXArray) -> MLXArray {
        let ff = factors.asType(.float32)
        let length = ff.dim(-1)
        let prefixShape = Array(ff.shape.dropLast())
        guard length > 0 else { return MLXArray.zeros(prefixShape + [0, 0], dtype: .float32) }
        var rows: [MLXArray] = []
        rows.reserveCapacity(length)
        var current = MLXArray.zeros(prefixShape + [length], dtype: .float32)
        let ones = MLXArray.ones(prefixShape + [1], dtype: .float32)
        for step in 0 ..< length {
            let scale = ff[.ellipsis, step ..< (step + 1)]
            let left = step > 0 ? current[.ellipsis, 0 ..< step] * scale : nil
            let rightCount = length - step - 1
            let right = rightCount > 0 ? MLXArray.zeros(prefixShape + [rightCount], dtype: .float32) : nil
            let parts = [left, ones, right].compactMap { $0 }
            current = concatenated(parts, axis: -1)
            rows.append(current)
        }
        return MLX.stacked(rows, axis: -2)
    }
}
