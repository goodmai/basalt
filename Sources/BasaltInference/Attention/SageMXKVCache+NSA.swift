import Foundation
import MLX
import MLXLMCommon

extension SageMXKVCache {
    static func canRunNSA(keys: MLXArray, topBlocks: Int, indexedBlocks: Int) -> Bool {
        keys.dim(2) >= Self.minTokensForKernel && topBlocks > 0 && indexedBlocks > topBlocks + 2
    }

    func indexNSACentroids(keys: MLXArray, values: MLXArray) {
        let complete = offset / Self.blockTokens
        guard complete > indexedBlocks else { return }
        let newBlocks = complete - indexedBlocks
        let start = indexedBlocks * Self.blockTokens
        let end = complete * Self.blockTokens
        let kSlice = keys[.ellipsis, start ..< end, 0...]
            .asType(.float32)
            .reshaped(keys.dim(0), keys.dim(1), newBlocks, Self.blockTokens, keys.dim(3))
        let vSlice = values[.ellipsis, start ..< end, 0...]
            .asType(.float32)
            .reshaped(values.dim(0), values.dim(1), newBlocks, Self.blockTokens, values.dim(3))
        let kMean = mean(kSlice, axis: 3)
        let vMean = mean(vSlice, axis: 3)
        centroids = centroids.map { concatenated([$0, kMean], axis: 2) } ?? kMean
        valueCentroids = valueCentroids.map { concatenated([$0, vMean], axis: 2) } ?? vMean
        indexedBlocks = complete
    }

    func nsaAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        config: BlockSparseRoutingMetal.Config
    ) -> MLXArray {
        let (kComp, vComp) = nsaCentroidsWithTail(keys: keys, values: values)
        let qOffset = max(0, keys.dim(2) - queries.dim(2))
        let precomputed = BlockSparseRoutingMetal.PrecomputedCentroids(
            keys: kComp,
            values: vComp,
            blockSize: Self.blockTokens,
            qOffset: qOffset)
        return BlockSparseRoutingMetal.nsaThreeBranchAttentionWithCentroids(
            q: queries,
            k: keys,
            v: values,
            centroids: precomputed,
            config: config)
    }

    func nsaCentroidsWithTail(keys: MLXArray, values: MLXArray) -> (keys: MLXArray, values: MLXArray) {
        let fullEnd = indexedBlocks * Self.blockTokens
        guard let kFull = centroids, let vFull = valueCentroids else {
            let kAll = BlockSparseRoutingMetal.blockCentroids(
                keys.asType(.float32).reshaped(keys.dim(0) * keys.dim(1), keys.dim(2), keys.dim(3)),
                blockSize: Self.blockTokens)
            let vAll = BlockSparseRoutingMetal.blockCentroids(
                values.asType(.float32).reshaped(values.dim(0) * values.dim(1), values.dim(2), values.dim(3)),
                blockSize: Self.blockTokens)
            return (kAll, vAll)
        }
        guard offset > fullEnd else {
            return (kFull, vFull)
        }
        let kTail = mean(keys[.ellipsis, fullEnd ..< offset, 0...].asType(.float32), axis: 2, keepDims: true)
        let vTail = mean(values[.ellipsis, fullEnd ..< offset, 0...].asType(.float32), axis: 2, keepDims: true)
        return (concatenated([kFull, kTail], axis: 2), concatenated([vFull, vTail], axis: 2))
    }
}
