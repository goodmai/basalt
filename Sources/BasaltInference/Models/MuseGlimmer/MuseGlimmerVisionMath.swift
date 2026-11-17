import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import BasaltCore

package enum MuseGlimmerVisionMath {
    static func cuSeqlens(_ grid: [THW]) -> [Int] {
        var result = [0]
        for frame in grid {
            for _ in 0 ..< frame.t {
                result.append(result[result.count - 1] + frame.h * frame.w)
            }
        }
        return result
    }

    static func windowIndex(_ grid: [THW], windowSize: Int) -> (MLXArray?, [Int]) {
        var indices = [Int32]()
        var cumulative = [0]
        var offset = 0
        for frame in grid {
            appendFrameWindows(frame, windowSize: windowSize, offset: offset, indices: &indices, cumulative: &cumulative)
            offset += frame.product
        }
        var boundaries = [cumulative[0]]
        for value in cumulative.dropFirst() where value != boundaries[boundaries.count - 1] {
            boundaries.append(value)
        }
        let isIdentity = indices.enumerated().allSatisfy { Int32($0.offset) == $0.element }
        return (isIdentity ? nil : MLXArray(indices), boundaries)
    }

    private struct WindowBlock {
        let t: Int
        let h: Int
        let w: Int
        let windowSize: Int
        let offset: Int
    }

    private static func appendFrameWindows(
        _ frame: THW, windowSize: Int, offset: Int, indices: inout [Int32], cumulative: inout [Int]
    ) {
        let (t, h, w) = (frame.t, frame.h, frame.w)
        let nH = (h + (windowSize - h % windowSize) % windowSize) / windowSize
        let nW = (w + (windowSize - w % windowSize) % windowSize) / windowSize
        for time in 0 ..< t {
            for blockH in 0 ..< nH {
                for blockW in 0 ..< nW {
                    let block = WindowBlock(t: time, h: blockH, w: blockW, windowSize: windowSize, offset: offset)
                    let count = appendBlock(frame, block, into: &indices)
                    cumulative.append(cumulative[cumulative.count - 1] + count)
                }
            }
        }
    }

    private static func appendBlock(_ frame: THW, _ block: WindowBlock, into indices: inout [Int32]) -> Int {
        var length = 0
        for row in 0 ..< block.windowSize where block.h * block.windowSize + row < frame.h {
            let y = block.h * block.windowSize + row
            for col in 0 ..< block.windowSize where block.w * block.windowSize + col < frame.w {
                let x = block.w * block.windowSize + col
                indices.append(Int32(block.offset + block.t * frame.h * frame.w + y * frame.w + x))
                length += 1
            }
        }
        return length
    }

    static func positionIds(_ grid: [THW]) -> MLXArray {
        var values = [Int32]()
        for frame in grid {
            var frameValues = [Int32]()
            for row in 0 ..< frame.h {
                for col in 0 ..< frame.w {
                    frameValues.append(Int32(col + 1))
                    frameValues.append(Int32(row + 1))
                }
            }
            for _ in 0 ..< frame.t { values.append(contentsOf: frameValues) }
        }
        return MLXArray(values, [grid.reduce(0) { $0 + $1.product }, 2])
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    static func fusedSDPA(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
        let d = q.dim(-1)
        guard let target = [64, 80, 128].first(where: { d <= $0 }), target != d else {
            return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        }
        let widths = Array(repeating: IntOrPair(0), count: q.ndim - 1) + [IntOrPair((0, target - d))]
        let output = MLXFast.scaledDotProductAttention(
            queries: padded(q, widths: widths),
            keys: padded(k, widths: widths),
            values: padded(v, widths: widths),
            scale: scale,
            mask: .none
        )
        return output[.ellipsis, ..<d]
    }

    static func axisInterpolation(_ count: Int, side: Int) -> ([Int], [Int], [Float], [Bool], [Bool]) {
        var low = [Int](repeating: 0, count: count)
        var high = [Int](repeating: 0, count: count)
        var delta = [Float](repeating: 0, count: count)
        var validLow = [Bool](repeating: false, count: count)
        var validHigh = [Bool](repeating: false, count: count)
        for i in 0 ..< count {
            let g = (Float(i) + 0.5) * (Float(side) / Float(count)) - 0.5
            let floored = Int(g.rounded(.down))
            delta[i] = g - Float(floored)
            validLow[i] = floored >= 0 && floored < side
            validHigh[i] = floored + 1 >= 0 && floored + 1 < side
            low[i] = min(max(floored, 0), side - 1)
            high[i] = min(max(floored + 1, 0), side - 1)
        }
        return (low, high, delta, validLow, validHigh)
    }

    static func interpolatedPositions(grid: [THW], side: Int, lookup: (MLXArray) -> MLXArray) -> MLXArray {
        let total = grid.reduce(0) { $0 + $1.product }
        var indices = [Int32](repeating: 0, count: 4 * total)
        var weights = [Float](repeating: 0, count: 4 * total)
        var cursor = 0
        for frame in grid {
            let (h0, h1, dh, vH0, vH1) = axisInterpolation(frame.h, side: side)
            let (w0, w1, dw, vW0, vW1) = axisInterpolation(frame.w, side: side)
            let frameCount = frame.h * frame.w
            for i in 0 ..< frame.h {
                for j in 0 ..< frame.w {
                    let corners = [
                        (h0[i], w0[j], (1 - dh[i]) * (1 - dw[j]), vH0[i] && vW0[j]),
                        (h0[i], w1[j], (1 - dh[i]) * dw[j], vH0[i] && vW1[j]),
                        (h1[i], w0[j], dh[i] * (1 - dw[j]), vH1[i] && vW0[j]),
                        (h1[i], w1[j], dh[i] * dw[j], vH1[i] && vW1[j]),
                    ]
                    for (corner, entry) in corners.enumerated() {
                        for time in 0 ..< frame.t {
                            let pos = cursor + time * frameCount + i * frame.w + j
                            indices[corner * total + pos] = Int32(entry.0 * side + entry.1)
                            weights[corner * total + pos] = entry.3 ? entry.2 : 0
                        }
                    }
                }
            }
            cursor += frame.product
        }
        let table = lookup(MLXArray(indices, [4, total]))
        return (table * expandedDimensions(MLXArray(weights, [4, total]), axis: -1)).sum(axis: 0)
    }

    static func pixelShuffle(_ hiddenStates: MLXArray, grid: [THW], mergeSize merge: Int) -> MLXArray {
        let dim = hiddenStates.dim(-1)
        var outputs = [MLXArray]()
        var offset = 0
        for frame in grid {
            let chunk = hiddenStates[offset ..< (offset + frame.product)]
                .reshaped(frame.t, frame.h / merge, merge, frame.w / merge, merge, dim)
                .transposed(0, 1, 3, 5, 2, 4)
            outputs.append(chunk.reshaped(-1, dim * merge * merge))
            offset += frame.product
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }
}
