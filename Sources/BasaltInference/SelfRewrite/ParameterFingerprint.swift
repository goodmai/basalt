import CryptoKit
import Foundation
import MLX
import MLXNN
import BasaltCore

public enum WeightCopyError: Error, Sendable, Equatable, CustomStringConvertible {
    case unsupportedModel(String)
    case sharedHandleMutationBlocked(tensor: String)
    case originalFingerprintMismatch([FingerprintMismatch])
    case invalidOperation(String)
    case budgetExceeded(required: Int64, budget: Int64)
    case copyReleased

    public var description: String {
        switch self {
        case .unsupportedModel(let typeName):
            return "WeightCopy cannot instantiate a fresh skeleton for \(typeName)"
        case .sharedHandleMutationBlocked(let tensor):
            return "blocked in-place parameter update on shared array handle '\(tensor)'"
        case .originalFingerprintMismatch(let mismatches):
            let detail = mismatches.prefix(5).map { "\($0.name): \($0.reason)" }.joined(separator: "; ")
            return "original model parameters mutated (\(mismatches.count) mismatch(es)): \(detail)"
        case .invalidOperation(let reason):
            return "edit operation failed: \(reason)"
        case .budgetExceeded(let required, let budget):
            return "weight copy extra bytes \(required) exceed budget \(budget)"
        case .copyReleased:
            return "weight copy has already been released"
        }
    }
}

public struct ParameterFingerprint: Sendable, Equatable {
    public let digests: [String: ParameterDigest]
    public let aggregateHash: String

    public init(digests: [String: ParameterDigest]) {
        self.digests = digests
        var hasher = SHA256()
        for name in digests.keys.sorted() {
            guard let entry = digests[name] else { continue }
            hasher.update(data: Data("\(name)|\(entry.dtype)|\(entry.shape)|\(entry.contentHash)\n".utf8))
        }
        self.aggregateHash = Self.hex(hasher.finalize())
    }

    public static func capture(of model: Module) -> ParameterFingerprint {
        let flat = model.parameters().flattened()
        eval(flat.map(\.1))
        var digests: [String: ParameterDigest] = [:]
        digests.reserveCapacity(flat.count)
        for (name, array) in flat {
            digests[name] = digest(name: name, array: array)
        }
        return ParameterFingerprint(digests: digests)
    }

    public static func digest(name: String, array: MLXArray) -> ParameterDigest {
        eval(array)
        let raw = array.asData(access: .noCopyIfContiguous)
        let contentHash: String = withExtendedLifetime(array) {
            hex(SHA256.hash(data: raw.data))
        }
        return ParameterDigest(
            name: name,
            arrayIdentity: UInt(bitPattern: ObjectIdentifier(array)),
            dtype: "\(array.dtype)",
            shape: array.shape,
            byteCount: array.nbytes,
            contentHash: contentHash
        )
    }

    public func verify(against model: Module, only keys: Set<String>? = nil) -> [FingerprintMismatch] {
        let current = Dictionary(model.parameters().flattened(), uniquingKeysWith: { first, _ in first })
        eval(Array(current.values))
        var mismatches: [FingerprintMismatch] = []

        if keys == nil {
            for extra in current.keys.sorted() where digests[extra] == nil {
                mismatches.append(FingerprintMismatch(name: extra, reason: "unexpected new parameter"))
            }
        }

        let targetNames = keys.map { $0.sorted() } ?? digests.keys.sorted()
        for name in targetNames {
            guard let expected = digests[name] else { continue }
            guard let array = current[name] else {
                mismatches.append(FingerprintMismatch(name: name, reason: "parameter missing"))
                continue
            }
            let actualIdentity = UInt(bitPattern: ObjectIdentifier(array))
            if actualIdentity != expected.arrayIdentity {
                mismatches.append(FingerprintMismatch(name: name, reason: "array handle identity changed"))
                continue
            }
            let actual = Self.digest(name: name, array: array)
            if actual.shape != expected.shape || actual.dtype != expected.dtype {
                mismatches.append(FingerprintMismatch(name: name, reason: "shape or dtype changed"))
            } else if actual.contentHash != expected.contentHash {
                mismatches.append(FingerprintMismatch(name: name, reason: "content hash changed"))
            }
        }
        return mismatches
    }

    public func requireUnchanged(against model: Module, only keys: Set<String>? = nil) throws(WeightCopyError) {
        let mismatches = verify(against: model, only: keys)
        if !mismatches.isEmpty {
            throw .originalFingerprintMismatch(mismatches)
        }
    }

    private static func hex(_ digest: some Sequence<UInt8>) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
