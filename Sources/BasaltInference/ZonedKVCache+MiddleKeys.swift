import BasaltCore
import MLX

enum MiddleKeys {
    case affine(ZonedKVCache.QuantizedRows)
    case polar(PolarKeyRows)
    case rq(CentroidRQKVRows)

    init(precision: KVPrecision, groupSize: Int, headDim: Int) {
        let polarRows = Self.makePolar(precision: precision, groupSize: groupSize, headDim: headDim)
        if let clusters = precision.centroidClusters, CentroidRQKVRows.supportedClusters.contains(clusters) {
            self = .rq(polarRows.map { CentroidRQKVRows(clusters: clusters, polar: $0) }
                ?? CentroidRQKVRows(clusters: clusters, bits: precision.keyBits, groupSize: groupSize))
            return
        }
        if let polarRows {
            self = .polar(polarRows)
            return
        }
        self = .affine(ZonedKVCache.QuantizedRows(bits: precision.keyBits, groupSize: groupSize))
    }

    private static func makePolar(precision: KVPrecision, groupSize: Int, headDim: Int) -> PolarKeyRows? {
        guard let rotated = precision.polarRotated, rotated > 0, rotated <= headDim else { return nil }
        let restDim = headDim - rotated
        let restGroup = restDim == 0 ? groupSize : KVCacheGeometry.quantizedGroupSize(keyDim: restDim, valueDim: restDim)
        return restGroup.map { PolarKeyRows(rotated: rotated, restBits: precision.keyBits, groupSize: $0, headDim: headDim) }
    }

    var count: Int {
        switch self {
        case .affine(let rows): rows.count
        case .polar(let rows): rows.count
        case .rq(let rows): rows.count
        }
    }

    var arrays: [MLXArray] {
        switch self {
        case .affine(let rows): rows.arrays
        case .polar(let rows): rows.arrays
        case .rq(let rows): rows.arrays
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
        case .rq(var rows):
            rows.append(keys)
            self = .rq(rows)
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
        case .rq(var rows):
            defer { self = .rq(rows) }
            return rows.trim(n)
        }
    }

    func dequantizedRows(dtype: DType) -> MLXArray? {
        switch self {
        case .affine(let rows): rows.dequantizedRows()
        case .polar(let rows): rows.dequantizedRows(dtype: dtype)
        case .rq(let rows): rows.dequantizedRows(dtype: dtype)
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
            return rows.scores(grouped)
        case .rq(let rows):
            return rows.scores(grouped)
        }
    }
}
