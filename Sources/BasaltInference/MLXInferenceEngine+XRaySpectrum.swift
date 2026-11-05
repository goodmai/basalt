import Foundation
import MLX
import BasaltCore

extension WeightSpectrum {
    static let sideCap = 8192
    static let gramChunkRows = 4096

    static func measure(rows: Int, columns: Int, block: (Range<Int>) -> MLXArray) throws(BasaltError) -> WeightSpectrum {
        let side = min(rows, columns)
        guard side <= sideCap else {
            throw .invalidRequestStructure(details: "spectrum: the matrix is \(rows) × \(columns); the smaller side \(side) is above the cap \(sideCap)")
        }
        let start = ContinuousClock.now
        let eigenvalues: [Float]
        do {
            eigenvalues = try withError {
                let gram = Self.gram(rows: rows, columns: columns, block: block)
                eval(gram)
                let values = MLXLinalg.eigvalsh(gram, stream: .cpu)
                eval(values)
                return values.asArray(Float.self)
            }
        } catch {
            throw .invalidRequestStructure(details: "spectrum: eigvalsh failed: \(error)")
        }
        guard var spectrum = WeightSpectrum(eigenvalues: eigenvalues.map(Double.init), rows: rows, columns: columns) else {
            throw .invalidRequestStructure(details: "spectrum: the Gram matrix is zero or not finite")
        }
        spectrum.seconds = (ContinuousClock.now - start) / .seconds(1)
        return spectrum
    }

    static func gram(rows: Int, columns: Int, block: (Range<Int>) -> MLXArray) -> MLXArray {
        guard rows >= columns else {
            let dense = block(0 ..< rows).asType(.float32)
            return matmul(dense, dense.T)
        }
        var gram = MLXArray.zeros([columns, columns], dtype: .float32)
        for start in stride(from: 0, to: rows, by: gramChunkRows) {
            let chunk = block(start ..< min(rows, start + gramChunkRows)).asType(.float32)
            gram = gram + matmul(chunk.T, chunk)
            eval(gram)
        }
        return gram
    }
}

extension MLXInferenceEngine {
    static func spectrum(_ view: WeightView, ranges: [Range<Int>]) throws(BasaltError) -> WeightSpectrum {
        switch ranges.count {
        case 2:
            return try WeightSpectrum.measure(rows: ranges[0].count, columns: ranges[1].count) { block in
                view.rows(block.lowerBound + ranges[0].lowerBound ..< block.upperBound + ranges[0].lowerBound)[0 ..< block.count, ranges[1]]
            }
        case 3 where ranges[0].count == 1:
            let matrix = view.rows(ranges[0])[0, ranges[1], ranges[2]]
            return try WeightSpectrum.measure(rows: ranges[1].count, columns: ranges[2].count) { matrix[$0] }
        default:
            throw .invalidRequestStructure(
                details: "spectrum: \(view.name) has shape \(view.extent); it needs a matrix, or a slice that picks one entry of the first axis of a 3-D stack")
        }
    }
}
