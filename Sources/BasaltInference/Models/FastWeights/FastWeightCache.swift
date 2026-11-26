import BasaltCore
import Foundation
import MLX
import MLXLMCommon

public final class FastWeightLayerCache: KVCache {
    public let chunkSize: Int
    public let slotCount: Int
    public private(set) var slots: [MLXArray?]
    public private(set) var offset: Int = 0
    public private(set) var stepInChunk: Int = 0
    private var restoredPresence: [Bool]?

    public init(chunkSize: Int, slotCount: Int) {
        self.chunkSize = Swift.max(1, chunkSize)
        self.slotCount = Swift.max(0, slotCount)
        self.slots = Array(repeating: nil, count: self.slotCount)
    }

    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { false }

    public func isTrimmable(after positions: Int) -> Bool { false }

    public static func detach(_ array: MLXArray) -> MLXArray {
        stopGradient(contiguous(array))
    }

    public func slot(_ index: Int) -> MLXArray? {
        index >= 0 && index < slots.count ? slots[index] : nil
    }

    public func commit(_ index: Int, _ value: MLXArray?) {
        guard index >= 0 && index < slots.count else { return }
        slots[index] = value.map(Self.detach)
    }

    public func innerState() -> [MLXArray] {
        slots.compactMap { $0 }
    }

    public var state: [MLXArray] {
        get { innerState() }
        set { restoreSlots(from: newValue) }
    }

    public var metaState: [String] {
        get {
            let presence = slots.map { $0 == nil ? "0" : "1" }.joined()
            return [String(offset), String(stepInChunk), String(chunkSize), String(slotCount), presence]
        }
        set { restoreMeta(newValue) }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    public func advance(tokens: Int, nextStepInChunk: Int) {
        offset += Swift.max(0, tokens)
        stepInChunk = ((nextStepInChunk % chunkSize) + chunkSize) % chunkSize
    }

    public func reset() {
        slots = Array(repeating: nil, count: slotCount)
        offset = 0
        stepInChunk = 0
        restoredPresence = nil
    }

    public func copy() -> any KVCache {
        let copied = FastWeightLayerCache(chunkSize: chunkSize, slotCount: slotCount)
        copied.offset = offset
        copied.stepInChunk = stepInChunk
        copied.slots = slots.map { $0.map(Self.detach) }
        return copied
    }

    private func restoreSlots(from values: [MLXArray]) {
        if values.isEmpty {
            slots = Array(repeating: nil, count: slotCount)
            return
        }
        if let mask = restoredPresence, mask.count == slotCount, mask.filter({ $0 }).count == values.count {
            var next: [MLXArray?] = Array(repeating: nil, count: slotCount)
            var cursor = 0
            for idx in 0 ..< slotCount where mask[idx] {
                next[idx] = Self.detach(values[cursor])
                cursor += 1
            }
            slots = next
            return
        }
        var next: [MLXArray?] = Array(repeating: nil, count: slotCount)
        for idx in 0 ..< Swift.min(slotCount, values.count) {
            next[idx] = Self.detach(values[idx])
        }
        slots = next
    }

    private func restoreMeta(_ values: [String]) {
        if values.indices.contains(0), let storedOffset = Int(values[0]) {
            offset = Swift.max(0, storedOffset)
        }
        if values.indices.contains(1), let storedStep = Int(values[1]) {
            stepInChunk = ((storedStep % chunkSize) + chunkSize) % chunkSize
        }
        if values.indices.contains(4), values[4].count == slotCount {
            let mask = values[4].map { $0 == "1" }
            restoredPresence = mask
            let existing = innerState()
            if !existing.isEmpty, mask.filter({ $0 }).count == existing.count {
                restoreSlots(from: existing)
            }
        }
    }
}
