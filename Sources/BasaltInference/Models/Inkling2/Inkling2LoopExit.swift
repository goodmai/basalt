import Foundation
import MLX
import MLXLMCommon

enum Inkling2LoopExit {
    static func copyable(_ cache: [KVCache]) -> Bool {
        cache.allSatisfy { $0 is KVCacheSimple || $0 is Inkling2WindowKVCache || $0 is ConvolutionStatesCache }
    }

    static func relativeChange(from previous: MLXArray, to hidden: MLXArray) -> Float {
        let (a, b) = (previous.asType(.float32), hidden.asType(.float32))
        let change = sqrt(sum(square(b - a), axis: -1)) / maximum(sqrt(sum(square(b), axis: -1)), MLXArray(Float(1e-6)))
        return change.max().item(Float.self)
    }

    static func fillSkippedPasses(_ cache: [KVCache], loop: Inkling2Configuration.Loop, lastPass: Int) {
        let core = loop.coreEnd - loop.coreStart
        for skipped in (lastPass + 1) ..< loop.numPasses {
            for layer in 0 ..< core {
                let source = loop.coreStart + lastPass * core + layer
                let target = loop.coreStart + skipped * core + layer
                LoopExit.copyLastPosition(from: cache[2 * source], to: cache[2 * target])
                if let from = cache[2 * source + 1] as? ConvolutionStatesCache, let to = cache[2 * target + 1] as? ConvolutionStatesCache {
                    to.commit(states: from.state, windows: nil, count: 1)
                }
            }
        }
    }
}
