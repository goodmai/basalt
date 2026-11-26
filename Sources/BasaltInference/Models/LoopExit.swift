import Foundation
import MLX
import MLXLMCommon

enum LoopExit {
    static let environmentKey = "BASALT_LOOP_EXIT"

    static var environmentThreshold: Float? {
        ProcessInfo.processInfo.environment[environmentKey].flatMap(Float.init).flatMap { $0 > 0 && $0 < 1 ? $0 : nil }
    }

    static func copyLastPosition(from source: KVCache, to target: KVCache) -> [MLXArray] {
        let state = source.state
        guard state.count == 2 else { return [] }
        let (keys, values) = target.update(keys: state[0][0..., 0..., (-1)..., 0...], values: state[1][0..., 0..., (-1)..., 0...])
        return [keys, values]
    }
}
