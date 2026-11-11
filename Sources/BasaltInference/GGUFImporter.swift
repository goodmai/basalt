import BasaltCore
import Foundation
import MLX

package struct GGUFImporter: Sendable {
    package enum Grid: String, Sendable, CaseIterable {
        case native, affine
    }

    package struct Summary: Sendable, Equatable {
        package var tensors = 0
        package var bytes = 0
        package var bits: [Int: Int] = [:]
        package var types: [String: Int] = [:]
        package var skipped: [String] = []
    }

    package let file: GGUFFile
    package let layout: GGUFQwen35Layout
    package let grid: Grid
    package let groupSize: Int
    package let minimumBits: Int

    package init(file: GGUFFile, layout: GGUFQwen35Layout, grid: Grid = .native, groupSize: Int = 64, minimumBits: Int = 2) {
        (self.file, self.layout, self.grid, self.groupSize, self.minimumBits) = (file, layout, grid, groupSize, minimumBits)
    }

    package static func layout(template: URL) throws -> GGUFQwen35Layout {
        let data = try Data(contentsOf: template.appending(component: "config.json"))
        let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let text = config["text_config"] as? [String: Any] ?? config
        guard let keys = text["linear_num_key_heads"] as? Int, let values = text["linear_num_value_heads"] as? Int,
              let keyDim = text["linear_key_head_dim"] as? Int, let valueDim = text["linear_value_head_dim"] as? Int else {
            throw BasaltError.modelConfigurationInvalid(path: template.path, reason: "the template is not a qwen3_5 checkpoint with linear attention")
        }
        return GGUFQwen35Layout(keyHeads: keys, valueHeads: values, keyHeadDim: keyDim, valueHeadDim: valueDim, prefix: config["text_config"] == nil ? "model." : "language_model.model.")
    }

    package func run(output: URL, template: URL) throws -> Summary {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        var writer = ShardWriter(directory: output, maxBytes: 4 << 30, metadata: ["format": "mlx"])
        var (summary, layers, natives) = (Summary(), [String: Int](), [String: GGUFNativeLoader.Entry]())
        for tensor in file.tensors {
            guard let plan = layout.plan(tensor) else {
                summary.skipped.append(tensor.name)
                continue
            }
            let base = String(plan.target.dropLast(".weight".count))
            let arrays: [String: MLXArray]
            if grid == .native, plan.bits != nil, tensor.type.block.elements > 1 {
                arrays = packed(tensor, plan, base: base)
                natives[base] = GGUFNativeLoader.Entry(type: tensor.type.name, inputOrder: arrays[base + ".input_order"] == nil ? nil : true)
                summary.types[tensor.type.name, default: 0] += 1
            } else {
                let bits = grid == .native ? nil : plan.bits.map { max($0, minimumBits) }
                arrays = convert(tensor, plan, bits: bits)
                if let bits {
                    layers[base] = bits
                    summary.bits[bits, default: 0] += 1
                }
            }
            summary.tensors += 1
            summary.bytes += arrays.values.reduce(0) { $0 + $1.nbytes }
            try writer.add(arrays)
        }
        _ = try writer.finish()
        try writeConfig(template: template, output: output, layers: layers, natives: natives)
        return summary
    }

    func packed(_ tensor: GGUFTensor, _ plan: GGUFTensorPlan, base: String) -> [String: MLXArray] {
        let rows = tensor.count / tensor.rowLength
        var weight = MLXArray([UInt8](file.payload(tensor)), [rows, tensor.byteCount / rows])
        var arrays: [String: MLXArray] = [:]
        switch plan.transform {
        case .vHeadRows(let from): weight = Self.regroupRows(weight, from: from, order: layout.tiledToGrouped(rows == layout.valueHeads ? 1 : layout.valueHeadDim))
        case .vHeadColumns: arrays[base + ".input_order"] = MLXArray(Self.inverse(layout.tiledToGrouped(layout.valueHeadDim)).map(Int32.init))
        case .none, .logOfNegated, .convChannels: break
        }
        arrays[plan.target] = weight
        return arrays
    }

    func convert(_ tensor: GGUFTensor, _ plan: GGUFTensorPlan, bits: Int?, dense: DType = .bfloat16) -> [String: MLXArray] {
        let rows = tensor.count / tensor.rowLength
        var weight = MLXArray(file.floats(tensor), tensor.dims.count == 1 ? [tensor.count] : [rows, tensor.rowLength])
        switch plan.transform {
        case .none: break
        case .logOfNegated: weight = Self.regroupRows(log(-weight), from: 0, order: layout.tiledToGrouped(1))
        case .vHeadRows(let from): weight = Self.regroupRows(weight, from: from, order: layout.tiledToGrouped(tensor.dims.count == 1 || rows == layout.valueHeads ? 1 : layout.valueHeadDim))
        case .vHeadColumns: weight = weight.take(MLXArray(layout.tiledToGrouped(layout.valueHeadDim).map(Int32.init)), axis: 1)
        case .convChannels(let from): weight = Self.regroupRows(weight, from: from, order: layout.tiledToGrouped(layout.valueHeadDim)).expandedDimensions(axis: -1)
        }
        guard let bits else { return [plan.target: weight.asType(dense)] }
        let (wq, scales, biases) = quantized(weight.asType(.bfloat16), groupSize: groupSize, bits: bits)
        let base = String(plan.target.dropLast(".weight".count))
        var arrays = [plan.target: wq, base + ".scales": scales]
        if let biases { arrays[base + ".biases"] = biases }
        return arrays
    }

    static func regroupRows(_ weight: MLXArray, from: Int, order: [Int]) -> MLXArray {
        let head = weight[..<from]
        let tail = weight[from...].take(MLXArray(order.map(Int32.init)), axis: 0)
        return from == 0 ? tail : concatenated([head, tail], axis: 0)
    }

    static func inverse(_ order: [Int]) -> [Int] {
        var inverse = [Int](repeating: 0, count: order.count)
        for (index, value) in order.enumerated() { inverse[value] = index }
        return inverse
    }

    func writeConfig(template: URL, output: URL, layers: [String: Int], natives: [String: GGUFNativeLoader.Entry]) throws {
        var config = try JSONSerialization.jsonObject(with: Data(contentsOf: template.appending(component: "config.json"))) as? [String: Any] ?? [:]
        (config["quantization"], config["quantization_config"]) = (nil, nil)
        if !layers.isEmpty {
            let common = Dictionary(layers.values.map { ($0, 1) }, uniquingKeysWith: +).max { $0.value < $1.value }?.key ?? 4
            var block: [String: Any] = ["bits": common, "group_size": groupSize, "mode": "affine"]
            for (layer, bits) in layers where bits != common { block[layer] = ["bits": bits, "group_size": groupSize, "mode": "affine"] }
            (config["quantization"], config["quantization_config"]) = (block, block)
        }
        if !natives.isEmpty {
            config["gguf_quantization"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(natives))
        }
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys]).write(to: output.appending(component: "config.json"))
        for name in try FileManager.default.contentsOfDirectory(atPath: template.path) where !name.hasPrefix(".") {
            let lowered = name.lowercased()
            guard lowered != "config.json", !lowered.hasSuffix(".safetensors"), !lowered.hasSuffix(".safetensors.index.json"), lowered != "readme.md" else { continue }
            try FileManager.default.copyItem(at: template.appending(component: name).resolvingSymlinksInPath(), to: output.appending(component: name))
        }
    }
}
