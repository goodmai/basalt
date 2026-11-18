import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

struct WeightView {
    enum Spec {
        case standard(layer: any Quantized, scales: MLXArray, biases: MLXArray?, globalScale: MLXArray?)
        case nf4(groupSize: Int, scales: MLXArray)

        var bits: Int {
            switch self {
            case .standard(let layer, _, _, _): layer.bits
            case .nf4: 4
            }
        }

        var dtypeLabel: String {
            switch self {
            case .standard(let layer, _, _, _): "\(layer.bits)-bit \(layer.mode.rawValue) g\(layer.groupSize)"
            case .nf4(let groupSize, _): "4-bit nf4 g\(groupSize)"
            }
        }
    }

    let name: String
    let array: MLXArray
    let quantization: Spec?

    static func all(in model: Module) -> [WeightView] {
        var layers: [String: any Quantized] = [:]
        var nf4GroupSizes: [String: Int] = [:]
        for (path, module) in model.namedModules() {
            if let q = module as? any Quantized {
                layers[path] = q
            } else if let nf4 = module as? NF4Linear {
                nf4GroupSizes[path] = nf4.groupSize
            } else if let nf4s = module as? NF4SwitchLinear {
                nf4GroupSizes[path] = nf4s.groupSize
            }
        }
        let parameters = model.parameters().flattened()
        let byName = Dictionary(parameters, uniquingKeysWith: { first, _ in first })
        return parameters.map { name, array in
            let path = String(name.dropLast(".weight".count))
            guard name.hasSuffix(".weight"), let scales = byName[path + ".scales"] else {
                return WeightView(name: name, array: array, quantization: nil)
            }
            if let layer = layers[path] {
                return WeightView(
                    name: name, array: array,
                    quantization: .standard(layer: layer, scales: scales, biases: byName[path + ".biases"], globalScale: byName[path + ".global_scale"]))
            }
            if let groupSize = nf4GroupSizes[path] {
                return WeightView(name: name, array: array, quantization: .nf4(groupSize: groupSize, scales: scales))
            }
            return WeightView(name: name, array: array, quantization: nil)
        }
    }

    var shape: [Int] {
        guard let quantization, let last = array.shape.indices.last else { return array.shape }
        var shape = array.shape
        shape[last] = (shape[last] * 32) / quantization.bits
        return shape
    }

    var extent: [Int] { shape.isEmpty ? [1] : shape }

    var info: TensorInfo {
        let dtype = quantization?.dtypeLabel ?? "\(array.dtype)"
        return TensorInfo(name: name, shape: shape, dtype: dtype, quantized: quantization != nil)
    }

    func rows(_ range: Range<Int>) -> MLXArray {
        guard let quantization else { return (array.ndim == 0 ? array.reshaped(1) : array)[range] }
        switch quantization {
        case .standard(let layer, let scales, let biases, let globalScale):
            return dequantized(
                array[range], scales: scales[range], biases: biases?[range],
                groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode,
                globalScale: globalScale)
        case .nf4(let groupSize, let scales):
            return NF4Metal.dequantized(array[range], scales: scales[range], groupSize: groupSize)
        }
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
        switch result {
        case .success(let report): return report
        case .failure(let error): throw error
        }
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
        var report = TensorReport(
            info: view.info, mean: Float(statistics.mean), std: statistics.std, min: statistics.min, max: statistics.max,
            l2: statistics.l2)
        report.percentiles = measured.percentiles
        report.values = values(view, ranges: ranges, limit: min(request.maxValues, tensorValuesCap))
        if request.spectrum {
            var measuredSpectrum = try spectrum(view, ranges: ranges)
            if let oftDefect = oftOrthogonalityDefect(for: view.name, in: model) {
                measuredSpectrum.orthogonalityDefect = oftDefect
            }
            report.spectrum = measuredSpectrum
        }
        return report
    }

    static func oftOrthogonalityDefect(for tensorName: String, in model: Module) -> Float? {
        let suffixes = [".weight", ".oft_r"]
        guard let suffix = suffixes.first(where: { tensorName.hasSuffix($0) }) else { return nil }
        let modulePath = String(tensorName.dropLast(suffix.count))
        guard let layer = model.namedModules().first(where: { $0.0 == modulePath })?.1 as? any OFTAdapterLayer else {
            return nil
        }
        return layer.maxOrthogonalityDefect()
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
