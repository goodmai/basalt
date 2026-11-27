import BasaltCore
import Foundation
import MLX

extension WeightConverter {
    static let logger = BasaltLogger(module: "WeightConverter")

    var requestedGroupSize: Int? {
        switch precision {
        case .affine(_, let groupSize): groupSize
        case .mixed(let mixed): mixed.groupSize
        default: nil
        }
    }

    func noteGroupMisfit(_ key: String, _ tensor: MLXArray, packed: PackedWeight?, sourceQuantized: Bool, into summary: inout Summary) {
        guard let group = requestedGroupSize, packed == nil, !sourceQuantized, Self.isLinearLike(key, tensor),
              !tensor.dim(-1).isMultiple(of: group) else { return }
        summary.groupMisfits[key] = GroupMisfit(inputs: tensor.dim(-1), bytes: tensor.nbytes)
    }

    func warnGroupMisfits(_ summary: Summary) {
        guard let group = requestedGroupSize, let message = Self.groupMisfitWarning(summary.groupMisfits, group: group) else { return }
        Self.logger.warn(message)
    }

    static func groupMisfitWarning(_ misfits: [String: GroupMisfit], group: Int) -> String? {
        guard !misfits.isEmpty else { return nil }
        let bytes = misfits.values.reduce(0) { $0 + $1.bytes }
        let largest = misfits.sorted { $0.value.bytes > $1.value.bytes }.prefix(5)
            .map { "\($0.key) (input \($0.value.inputs), \(String(format: "%.2f", Double($0.value.bytes) / 1e9)) GB)" }
        let fits = [128, 64, 32].first { size in size < group && misfits.values.allSatisfy { $0.inputs.isMultiple(of: size) } }
        let advice = fits.map { "group size \($0) quantizes all of them" } ?? "no group size of 32, 64 or 128 divides all of them"
        return "\(misfits.count) linear tensor(s), \(String(format: "%.2f", Double(bytes) / 1e9)) GB, stayed in their source dtype because their input dimension is not divisible by group size \(group): \(largest.joined(separator: ", ")); \(advice)"
    }
}

package struct GroupMisfit: Sendable, Equatable {
    package let inputs: Int
    package let bytes: Int
}
