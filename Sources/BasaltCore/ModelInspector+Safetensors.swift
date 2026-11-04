import Foundation

extension ModelInspector {
    package struct Tensor: Equatable, Sendable {
        package let dtype: String
        package let shape: [Int]
        package let end: Int
    }

    package struct SafetensorsHeader: Sendable {
        package let tensors: [String: Tensor]
        package let metadata: [String: String]
    }

    package struct ShardProblem: Error {
        package let message: String

        package static let malformedHeader = ShardProblem(message: "Invalid or corrupted safetensors header")
    }

    package static func readSafetensors(at url: URL, fileSize: Int64) -> Result<SafetensorsHeader, ShardProblem> {
        guard fileSize >= 8, let handle = try? FileHandle(forReadingFrom: url) else {
            return .failure(.init(message: "Unreadable safetensors file"))
        }
        defer { try? handle.close() }

        guard let lenData = try? handle.read(upToCount: 8),
              let headerLen = safetensorsHeaderLength(lenData, fileSize: fileSize),
              let headerData = try? handle.read(upToCount: headerLen),
              headerData.count == headerLen else {
            return .failure(.malformedHeader)
        }
        return parseSafetensorsHeader(headerData, fileSize: fileSize)
    }

    package static func safetensorsHeaderLength(_ prefix: Data, fileSize: Int64) -> Int? {
        guard prefix.count == 8, fileSize >= 8 else { return nil }
        let headerLen = RawSpan(elements: Array(prefix).span).load(fromByteOffset: 0, as: UInt64.self).littleEndian
        guard headerLen > 0, headerLen <= 100_000_000, headerLen <= UInt64(fileSize - 8) else { return nil }
        return Int(headerLen)
    }

    package static func parseSafetensorsHeader(_ headerData: Data, fileSize: Int64) -> Result<SafetensorsHeader, ShardProblem> {
        guard let json = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            return .failure(.malformedHeader)
        }

        var tensors: [String: Tensor] = [:]
        var spans: [(name: String, begin: Int, end: Int)] = []
        for (name, value) in json where name != "__metadata__" {
            guard let t = value as? [String: Any], let dtype = t["dtype"] as? String,
                  let shape = t["shape"] as? [Int], let offsets = t["data_offsets"] as? [Int],
                  offsets.count == 2, 0 <= offsets[0], offsets[0] <= offsets[1] else {
                return .failure(.init(message: "Malformed tensor entry '\(name)' in safetensors header"))
            }
            tensors[name] = Tensor(dtype: dtype, shape: shape, end: offsets[1])
            spans.append((name, offsets[0], offsets[1]))
        }
        guard !tensors.isEmpty else {
            return .failure(.init(message: "Safetensors header lists no tensors"))
        }
        var cursor = 0
        for span in spans.sorted(by: { ($0.begin, $0.end) < ($1.begin, $1.end) }) {
            guard span.begin == cursor else {
                return .failure(.init(message: "Tensor data of '\(span.name)' starts at \(span.begin), expected \(cursor) (gap or overlap)"))
            }
            cursor = span.end
        }
        let expected = 8 + Int64(headerData.count) + Int64(cursor)
        guard expected == fileSize else {
            return .failure(.init(message: "Truncated or oversized shard (\(fileSize) bytes, header expects \(expected))"))
        }
        return .success(SafetensorsHeader(tensors: tensors, metadata: json["__metadata__"] as? [String: String] ?? [:]))
    }
}
