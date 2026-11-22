import Foundation
import MLX
import BasaltCore

struct ShardCompanions {
    private let shardOf: [String: URL]
    private var cached: (url: URL, arrays: [String: MLXArray])?

    init(shards: [URL]) {
        var index: [String: URL] = [:]
        for shard in shards {
            let size = (try? FileManager.default.attributesOfItem(atPath: shard.path)[.size] as? Int64) ?? 0
            guard case .success(let header) = ModelInspector.readSafetensors(at: shard, fileSize: size) else { continue }
            for key in header.tensors.keys { index[key] = shard }
        }
        shardOf = index
    }

    mutating func tensor(_ key: String, in current: [String: MLXArray], shard: URL) throws -> MLXArray? {
        if let local = current[key] { return local }
        guard let other = shardOf[key], other != shard else { return nil }
        if cached?.url != other { cached = (other, try loadArrays(url: other)) }
        return cached?.arrays[key]
    }

    mutating func scope(_ keys: [String], over current: [String: MLXArray], shard: URL) throws -> [String: MLXArray] {
        var scope = current
        for key in keys where scope[key] == nil {
            if let found = try tensor(key, in: current, shard: shard) { scope[key] = found }
        }
        return scope
    }
}
