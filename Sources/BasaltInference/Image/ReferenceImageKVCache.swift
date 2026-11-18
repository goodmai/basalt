import CryptoKit
import Foundation
import MLX
import MLXNN
import BasaltCore

package struct ReferenceKVCacheKey: Sendable, Hashable {
    package let modelID: String
    package let width: Int
    package let height: Int
    package let hiddenSize: Int
    package let patchSize: Int
    package let referencesDigestHex: String

    package init(
        modelID: String,
        width: Int,
        height: Int,
        hiddenSize: Int,
        patchSize: Int,
        referenceImages: [Data]
    ) {
        self.modelID = modelID
        self.width = width
        self.height = height
        self.hiddenSize = hiddenSize
        self.patchSize = patchSize
        var hasher = SHA256()
        hasher.update(data: Data("\(referenceImages.count):".utf8))
        for img in referenceImages {
            var countLE = UInt64(img.count).littleEndian
            withUnsafeBytes(of: &countLE) { hasher.update(bufferPointer: $0) }
            hasher.update(data: img)
        }
        let digest = hasher.finalize()
        self.referencesDigestHex = digest.map { String(format: "%02x", $0) }.joined()
    }
}

package struct CachedReferenceLayerKV {
    package let keys: MLXArray
    package let values: MLXArray

    package init(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
    }
}

package struct CachedReferenceSnapshot {
    package let key: ReferenceKVCacheKey
    package let primaryImage: MLXArray
    package let primaryLatent: MLXArray
    package let refTokens: MLXArray
    package let dualLayerKV: [CachedReferenceLayerKV]
    package let singleLayerKV: [CachedReferenceLayerKV]
}

package struct ReferenceImageKVCache {
    package let capacity: Int
    private var entries: [ReferenceKVCacheKey: CachedReferenceSnapshot] = [:]
    private var order: [ReferenceKVCacheKey] = []
    package private(set) var hits: Int = 0
    package private(set) var misses: Int = 0

    package init(capacity: Int = 8) {
        self.capacity = max(1, capacity)
    }

    package var count: Int { entries.count }

    package mutating func lookup(_ key: ReferenceKVCacheKey) -> CachedReferenceSnapshot? {
        guard let found = entries[key] else {
            misses += 1
            return nil
        }
        hits += 1
        if let idx = order.firstIndex(of: key) {
            order.remove(at: idx)
        }
        order.append(key)
        return found
    }

    package mutating func insert(_ snapshot: CachedReferenceSnapshot) {
        let key = snapshot.key
        if let idx = order.firstIndex(of: key) {
            order.remove(at: idx)
        } else if order.count >= capacity, let oldest = order.first {
            order.removeFirst()
            entries.removeValue(forKey: oldest)
        }
        entries[key] = snapshot
        order.append(key)
    }

    package mutating func clear() {
        entries.removeAll()
        order.removeAll()
    }
}

enum ReferenceKVProjector {
    static func projectDualBlocks(
        refTokens: MLXArray,
        blocks: [JointDiTBlock],
        headDim: Int
    ) -> (updatedRef: MLXArray, kvPerLayer: [CachedReferenceLayerKV]) {
        var cur = refTokens
        let refCondVec = mean(cur, axis: 1)
        let nRef = cur.dim(1)
        let rope = RoPE2D.frequencies(gridH: 1, gridW: max(1, nRef), headDim: headDim)
        let stepCtx = DiTStepContext(condVec: refCondVec, rope: rope, timestep: 0.0, guidance: 1.0)
        var layerKVs: [CachedReferenceLayerKV] = []
        layerKVs.reserveCapacity(blocks.count)
        for block in blocks {
            let (nextRef, kv) = block.projectReferenceTower(refImg: cur, stepCtx: stepCtx)
            cur = nextRef
            layerKVs.append(kv)
        }
        return (cur, layerKVs)
    }

    static func projectSingleBlocks(
        refTokens: MLXArray,
        blocks: [SingleStreamS3DiTBlock],
        headDim: Int
    ) -> [CachedReferenceLayerKV] {
        var cur = refTokens
        let refCondVec = mean(cur, axis: 1)
        let nRef = cur.dim(1)
        let rope = RoPE2D.frequencies(gridH: 1, gridW: max(1, nRef), headDim: headDim)
        let stepCtx = DiTStepContext(condVec: refCondVec, rope: rope, timestep: 0.0, guidance: 1.0)
        var layerKVs: [CachedReferenceLayerKV] = []
        layerKVs.reserveCapacity(blocks.count)
        for block in blocks {
            let (nextRef, kv) = block.projectReferenceTower(refTokens: cur, stepCtx: stepCtx)
            cur = nextRef
            layerKVs.append(kv)
        }
        return layerKVs
    }

    static func buildFluxSnapshot(
        key: ReferenceKVCacheKey,
        refImages: [MLXArray],
        module: FluxKleinDiTModule
    ) -> CachedReferenceSnapshot {
        let refLatents = refImages.map { module.vae.encode($0) }
        let refTokens = encodeRefPatchTokens(refLatents: refLatents, refPatchIn: module.refPatchIn, hiddenSize: module.config.hiddenSize)
        let (midRef, dualKV) = projectDualBlocks(
            refTokens: refTokens,
            blocks: module.dualBlocks,
            headDim: module.config.headDim
        )
        let singleKV = projectSingleBlocks(
            refTokens: midRef,
            blocks: module.singleBlocks,
            headDim: module.config.headDim
        )
        var toEval: [MLXArray] = [refLatents[0], refTokens]
        for item in dualKV { toEval.append(contentsOf: [item.keys, item.values]) }
        for item in singleKV { toEval.append(contentsOf: [item.keys, item.values]) }
        eval(toEval)
        return CachedReferenceSnapshot(
            key: key,
            primaryImage: refImages[0],
            primaryLatent: refLatents[0],
            refTokens: refTokens,
            dualLayerKV: dualKV,
            singleLayerKV: singleKV
        )
    }

    static func buildS3Snapshot(
        key: ReferenceKVCacheKey,
        refImages: [MLXArray],
        module: S3DiTModule
    ) -> CachedReferenceSnapshot {
        let refLatents = refImages.map { module.vae.encode($0) }
        let refTokens = encodeRefPatchTokens(refLatents: refLatents, refPatchIn: module.refPatchIn, hiddenSize: module.config.hiddenSize)
        let singleKV = projectSingleBlocks(
            refTokens: refTokens,
            blocks: module.blocks,
            headDim: module.config.headDim
        )
        var toEval: [MLXArray] = [refLatents[0], refTokens]
        for item in singleKV { toEval.append(contentsOf: [item.keys, item.values]) }
        eval(toEval)
        return CachedReferenceSnapshot(
            key: key,
            primaryImage: refImages[0],
            primaryLatent: refLatents[0],
            refTokens: refTokens,
            dualLayerKV: [],
            singleLayerKV: singleKV
        )
    }

    static func encodeRefPatchTokens(
        refLatents: [MLXArray],
        refPatchIn: Conv2d,
        hiddenSize: Int
    ) -> MLXArray {
        concatenated(
            refLatents.map { rLat in
                let p = refPatchIn(rLat)
                return p.reshaped(1, p.dim(1) * p.dim(2), hiddenSize)
            },
            axis: 1
        )
    }
}
