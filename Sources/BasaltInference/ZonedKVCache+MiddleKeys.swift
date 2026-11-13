import BasaltCore
import MLX

enum MiddleKeys {
    case affine(ZonedKVCache.QuantizedRows)
    case polar(PolarKeyRows)

    init(precision: KVPrecision, groupSize: Int, headDim: Int) {
        if let rotated = precision.polarRotated, rotated > 0, rotated <= headDim {
            self = .polar(PolarKeyRows(rotated: rotated, restBits: precision.keyBits, groupSize: groupSize, headDim: headDim))
        } else {
            self = .affine(ZonedKVCache.QuantizedRows(bits: precision.keyBits, groupSize: groupSize))
        }
    }

    var count: Int {
        switch self {
        case .affine(let rows): rows.count
        case .polar(let rows): rows.count
        }
    }

    var arrays: [MLXArray] {
        switch self {
        case .affine(let rows): rows.arrays
        case .polar(let rows): rows.arrays
        }
    }

    mutating func append(_ keys: MLXArray) {
        switch self {
        case .affine(var rows):
            rows.append(keys)
            self = .affine(rows)
        case .polar(var rows):
            rows.append(keys)
            self = .polar(rows)
        }
    }

    mutating func trim(_ n: Int) -> Int {
        switch self {
        case .affine(var rows):
            defer { self = .affine(rows) }
            return rows.trim(n)
        case .polar(var rows):
            defer { self = .polar(rows) }
            return rows.trim(n)
        }
    }

    func dequantizedRows(dtype: DType) -> MLXArray? {
        switch self {
        case .affine(let rows): rows.dequantizedRows()
        case .polar(let rows): rows.dequantizedRows(dtype: dtype)
        }
    }

    func scores(_ grouped: MLXArray) -> MLXArray? {
        switch self {
        case .affine(let rows):
            guard let parts = rows.parts else { return nil }
            return quantizedMM(
                grouped, expandedDimensions(parts.0, axis: 2), scales: expandedDimensions(parts.1, axis: 2),
                biases: parts.2.map { expandedDimensions($0, axis: 2) }, transpose: true, groupSize: rows.groupSize, bits: rows.bits)
        case .polar(let rows):
            guard let keys = rows.dequantizedRows(dtype: grouped.dtype) else { return nil }
            return matmul(grouped, expandedDimensions(keys, axis: 2).swappedAxes(-1, -2))
        }
    }
}
