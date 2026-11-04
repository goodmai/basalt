import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

struct WeightView {
    let name: String
    let array: MLXArray
    let quantization: (layer: any Quantized, scales: MLXArray, biases: MLXArray?, globalScale: MLXArray?)?

    static func all(in model: Module) -> [WeightView] {
        let layers = Dictionary(
            model.namedModules().compactMap { path, module in (module as? any Quantized).map { (path, $0) } },
            uniquingKeysWith: { first, _ in first })
        let parameters = model.parameters().flattened()
        let byName = Dictionary(parameters, uniquingKeysWith: { first, _ in first })
        return parameters.map { name, array in
            let path = String(name.dropLast(".weight".count))
            guard name.hasSuffix(".weight"), let layer = layers[path], let scales = byName[path + ".scales"] else {
                return WeightView(name: name, array: array, quantization: nil)
            }
            return WeightView(
                name: name, array: array,
                quantization: (layer, scales, byName[path + ".biases"], byName[path + ".global_scale"]))
        }
    }

    var shape: [Int] {
        guard let layer = quantization?.layer, let last = array.shape.indices.last else { return array.shape }
        var shape = array.shape
        shape[last] = shape[last] * 32 / layer.bits
        return shape
    }

    var extent: [Int] { shape.isEmpty ? [1] : shape }

    var info: TensorInfo {
        let dtype = quantization.map { "\($0.layer.bits)-bit \($0.layer.mode.rawValue) g\($0.layer.groupSize)" } ?? "\(array.dtype)"
        return TensorInfo(name: name, shape: shape, dtype: dtype, quantized: quantization != nil)
    }

    func rows(_ range: Range<Int>) -> MLXArray {
        guard let quantization else { return (array.ndim == 0 ? array.reshaped(1) : array)[range] }
        return dequantized(
            array[range], scales: quantization.scales[range], biases: quantization.biases?[range],
            groupSize: quantization.layer.groupSize, bits: quantization.layer.bits, mode: quantization.layer.mode,
            globalScale: quantization.globalScale)
    }
}

extension MLXInferenceEngine {
    static let tensorValuesCap = 4096

    public func tensors(matching filter: String?) async throws(BasaltError) -> [TensorInfo] {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        return await container.performEvaluating(nonSendable: ()) { context, _ in
            WeightView.all(in: context.model).filter { view in filter.map { $0.isEmpty || view.name.contains($0) } ?? true }.map(\.info)
        }
    }

    public func tensor(_ request: TensorRequest) async throws(BasaltError) -> TensorReport {
        guard request.maxValues >= 0 else {
            throw .invalidRequestStructure(details: "maxValues must not be negative, got \(request.maxValues)")
        }
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        let result = await container.performEvaluating(nonSendable: ()) { context, _ in
            Result { () throws(BasaltError) -> TensorReport in try MLXInferenceEngine.report(request, model: context.model) }
        }
        return try result.get()
    }

    static func report(_ request: TensorRequest, model: Module) throws(BasaltError) -> TensorReport {
        let views = WeightView.all(in: model)
        guard let view = views.first(where: { $0.name == request.name }) else {
            let matches = closeMatches(request.name, in: views.map(\.name)).joined(separator: ", ")
            throw .invalidRequestStructure(details: "no tensor named \(request.name); close matches: \(matches)")
        }
        let ranges = try validated(request.slice, extent: view.extent, name: view.name)
        let measured = TensorStatistics.measure(
            rows: view.rows, count: view.extent[0], rowElements: view.extent.dropFirst().reduce(1, *))
        let statistics = measured.statistics
        return TensorReport(
            info: view.info, mean: Float(statistics.mean), std: statistics.std, min: statistics.min, max: statistics.max,
            l2: statistics.l2, percentiles: measured.percentiles,
            values: values(view, ranges: ranges, limit: min(request.maxValues, tensorValuesCap)))
    }

    static func validated(_ slice: [Range<Int>]?, extent: [Int], name: String) throws(BasaltError) -> [Range<Int>] {
        let slice = slice ?? []
        guard slice.count <= extent.count else {
            throw .invalidRequestStructure(details: "tensor \(name) has shape \(extent); a slice of \(slice.count) ranges does not fit it")
        }
        for (axis, (range, size)) in zip(slice, extent).enumerated() where range.lowerBound < 0 || range.upperBound > size {
            throw .invalidRequestStructure(details: "slice \(range) is outside axis \(axis) of tensor \(name), which has \(size) entries")
        }
        return slice + extent.dropFirst(slice.count).map { 0 ..< $0 }
    }

    static func values(_ view: WeightView, ranges: [Range<Int>], limit: Int) -> [Float] {
        let inner = ranges.dropFirst().reduce(1) { $0 * $1.count }
        guard limit > 0, inner > 0, let first = ranges.first, !first.isEmpty else { return [] }
        let rows = first.lowerBound ..< min(first.upperBound, first.lowerBound + (limit + inner - 1) / inner)
        var indices: [any MLXArrayIndex] = [0 ..< rows.count]
        indices.append(contentsOf: ranges.dropFirst().map { $0 as any MLXArrayIndex })
        let flat = view.rows(rows)[indices].asType(.float32).flattened()
        return flat[0 ..< min(limit, flat.size)].asArray(Float.self)
    }

    static func closeMatches(_ name: String, in names: [String], count: Int = 3) -> [String] {
        names.map { (editDistance(name, $0), $0) }.sorted { $0 < $1 }.prefix(count).map(\.1)
    }

    static func editDistance(_ lhs: String, _ rhs: String) -> Int {
        let rhs = Array(rhs.utf8)
        var previous = Array(0 ... rhs.count)
        for (row, left) in lhs.utf8.enumerated() {
            var current = [row + 1]
            for (column, right) in rhs.enumerated() {
                current.append(min(previous[column + 1] + 1, current[column] + 1, previous[column] + (left == right ? 0 : 1)))
            }
            previous = current
        }
        return previous[rhs.count]
    }
}
