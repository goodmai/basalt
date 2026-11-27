import BasaltCore
import Foundation
import MLX
import MLXLMCommon

extension WeightConverter {
    static let logger = BasaltLogger(module: "WeightConverter")

    var requestedGroupSize: Int? {
        switch precision {
        case .affine(_, let groupSize): groupSize
        case .mixed(let mixed): mixed.groupSize
        default: nil
        }
    }

    static func fittingGroup(_ key: String, _ tensor: MLXArray, packed: PackedWeight?, sourceQuantized: Bool, requested: Int) -> Int? {
        let candidates = packed == nil ? [requested, 64, 32].filter { $0 <= requested } : [requested]
        return candidates.first { quantizable(key, tensor, packed: packed, sourceQuantized: sourceQuantized, groupSize: $0) }
    }

    func noteGroupChoice(_ key: String, _ tensor: MLXArray, packed: PackedWeight?, sourceQuantized: Bool, into summary: inout Summary) {
        guard let group = requestedGroupSize, packed == nil, !sourceQuantized, Self.isLinearLike(key, tensor),
              tensor.dim(-1) >= 32, !tensor.dim(-1).isMultiple(of: group) else { return }
        if let target = targetQuantization(key, tensor, packed: packed, sourceQuantized: sourceQuantized) {
            summary.groupFallbacks[String(key.dropLast(".weight".count))] = target
        } else {
            summary.groupMisfits[key] = GroupMisfit(inputs: tensor.dim(-1), bytes: tensor.nbytes)
        }
    }

    func finishGroupChoices(_ summary: Summary, output: URL) throws {
        if let group = requestedGroupSize, let message = Self.groupMisfitWarning(summary.groupMisfits, group: group) {
            Self.logger.warn(message)
        }
        guard !summary.groupFallbacks.isEmpty else { return }
        let url = output.appending(component: "config.json")
        guard var config = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else { return }
        for section in ["quantization", "quantization_config"] {
            guard var block = config[section] as? [String: Any] else { continue }
            for (layer, target) in summary.groupFallbacks {
                for name in MixedBits.loaderNames(layer) { block[name] = ["bits": target.bits, "group_size": target.groupSize, "mode": "affine"] }
            }
            config[section] = block
        }
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys]).write(to: url)
    }

    static func groupMisfitWarning(_ misfits: [String: GroupMisfit], group: Int) -> String? {
        guard !misfits.isEmpty else { return nil }
        let bytes = misfits.values.reduce(0) { $0 + $1.bytes }
        let largest = misfits.sorted { $0.value.bytes > $1.value.bytes }.prefix(5)
            .map { "\($0.key) (input \($0.value.inputs), \(String(format: "%.2f", Double($0.value.bytes) / 1e9)) GB)" }
        return "\(misfits.count) linear tensor(s), \(String(format: "%.2f", Double(bytes) / 1e9)) GB, stayed in their source dtype because no group size of \(Set([group, 64, 32].filter { $0 <= group }).sorted().map(String.init).joined(separator: ", ")) divides their input dimension: \(largest.joined(separator: ", "))"
    }
}

package struct GroupMisfit: Sendable, Equatable {
    package let inputs: Int
    package let bytes: Int
}
