import MLX

enum NF4Codec {
    static let groupSizes: Set<Int> = [32, 64]

    static let codebook: [Float] = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ]

    static func quantized(_ weight: MLXArray, groupSize: Int) -> (weight: MLXArray, scales: MLXArray) {
        let columns = weight.dim(-1)
        let groups = weight.asType(.float32).reshaped(-1, groupSize)
        let scales = abs(groups).max(axis: -1, keepDims: true).asType(.float16)
        let scale = scales.asType(.float32)
        let normalized = groups / which(scale .> 0, scale, 1)
        var codes = MLXArray.zeros(normalized.shape, dtype: .uint32)
        for (low, high) in zip(codebook, codebook.dropFirst()) {
            codes = codes + (normalized .> ((low + high) / 2)).asType(.uint32)
        }
        let shifts = MLXArray((0 ..< 8).map { UInt32(1) << UInt32(4 * $0) })
        let packed = (codes.reshaped(-1, 8) * shifts).sum(axis: -1).asType(.uint32)
        let leading = Array(weight.shape.dropLast())
        return (packed.reshaped(leading + [columns / 8]), scales.reshaped(leading + [columns / groupSize]))
    }

    static func dequantized(_ weight: MLXArray, scales: MLXArray, groupSize: Int) -> MLXArray {
        let shifts = MLXArray((0 ..< 8).map { UInt32(4 * $0) })
        let codes = (weight.expandedDimensions(axis: -1) >> shifts) & UInt32(15)
        let values = MLXArray(codebook).take(codes.reshaped(-1))
        let grouped = values.reshaped(-1, groupSize) * scales.reshaped(-1, 1).asType(.float32)
        return grouped.reshaped(Array(weight.shape.dropLast()) + [weight.dim(-1) * 8]).asType(scales.dtype)
    }

    static func bytes(elements: Int, groupSize: Int) -> Int { elements / 2 + elements / groupSize * 2 }
}
