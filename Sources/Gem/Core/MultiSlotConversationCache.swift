import Foundation

/// Manages multiple conversation KV cache slots with parking support and prefix hash tree matching.
///
/// Inactive conversations are "parked" and hold at most 1/8 of the KV buffer (preserving the tool prefix).
/// On a new turn (such as intermediate tool calls or user follow-ups), only the delta prompt is prefilled.
public actor MultiSlotConversationCache {

    public struct ConversationSlot: Sendable, Identifiable {
        public let id: String
        public var prompt: String
        public var toolPrefix: String?
        public var prefixHash: Int
        public var estimatedTokens: Int
        public var isParked: Bool
        public var lastAccessTime: Date
        public var turnCount: Int

        public init(
            id: String,
            prompt: String = "",
            toolPrefix: String? = nil,
            estimatedTokens: Int = 0,
            isParked: Bool = false,
            lastAccessTime: Date = Date(),
            turnCount: Int = 0
        ) {
            self.id = id
            self.prompt = prompt
            self.toolPrefix = toolPrefix
            self.prefixHash = (toolPrefix ?? prompt).hashValue
            self.estimatedTokens = estimatedTokens
            self.isParked = isParked
            self.lastAccessTime = lastAccessTime
            self.turnCount = turnCount
        }
    }

    public struct PrefillDelta: Sendable {
        public let cachedPrefixLength: Int
        public let cachedTokensEstimate: Int
        public let deltaPrompt: String
        public let isIncremental: Bool
        public let slotId: String

        public init(
            cachedPrefixLength: Int,
            cachedTokensEstimate: Int,
            deltaPrompt: String,
            isIncremental: Bool,
            slotId: String
        ) {
            self.cachedPrefixLength = cachedPrefixLength
            self.cachedTokensEstimate = cachedTokensEstimate
            self.deltaPrompt = deltaPrompt
            self.isIncremental = isIncremental
            self.slotId = slotId
        }
    }

    public let maxKVBufferTokens: Int
    public let maxSlots: Int
    private var slots: [String: ConversationSlot] = [:]

    /// Maximum tokens an inactive (parked) conversation is permitted to hold (at most 1/8 of KV buffer).
    public var parkingBudgetTokens: Int {
        max(64, maxKVBufferTokens / 8)
    }

    public init(maxKVBufferTokens: Int = 8192, maxSlots: Int = 16) {
        self.maxKVBufferTokens = maxKVBufferTokens
        self.maxSlots = maxSlots
    }

    // MARK: — Slot Access & Lifecycle

    public func getOrCreateSlot(id: String, toolPrefix: String? = nil) -> ConversationSlot {
        if var existing = slots[id] {
            existing.lastAccessTime = Date()
            if existing.isParked {
                existing.isParked = false
            }
            if let tp = toolPrefix, existing.toolPrefix == nil {
                existing.toolPrefix = tp
                existing.prefixHash = tp.hashValue
            }
            slots[id] = existing
            return existing
        }

        // Enforce slot eviction if limit reached
        if slots.count >= maxSlots {
            evictOldestParkedSlot()
        }

        let newSlot = ConversationSlot(
            id: id,
            toolPrefix: toolPrefix,
            isParked: false,
            lastAccessTime: Date()
        )
        slots[id] = newSlot
        return newSlot
    }

    /// Parks an inactive slot: truncates prompt to fit within 1/8 KV buffer while preserving toolPrefix.
    public func parkSlot(id: String) {
        guard var slot = slots[id] else { return }
        slot.isParked = true

        let limitChars = parkingBudgetTokens * 4
        if slot.prompt.count > limitChars {
            let separator = "\n...\n"
            if let toolPrefix = slot.toolPrefix, !toolPrefix.isEmpty {
                // Preserve tool prefix at the head
                let availableForContent = max(0, limitChars - separator.count)
                let prefixLen = min(toolPrefix.count, availableForContent)
                let remaining = availableForContent - prefixLen
                let tail = remaining > 0 ? String(slot.prompt.suffix(remaining)) : ""
                slot.prompt = String(toolPrefix.prefix(prefixLen)) + separator + tail
            } else {
                // Keep head prefix up to limit
                slot.prompt = String(slot.prompt.prefix(limitChars))
            }
            slot.estimatedTokens = min(parkingBudgetTokens, slot.prompt.count / 4)
        }
        slots[id] = slot
    }

    /// Automatically parks all slots except the currently active conversation.
    public func parkInactiveSlots(except activeId: String) {
        for (id, _) in slots where id != activeId {
            parkSlot(id: id)
        }
    }

    // MARK: — Prefix Matching & Incremental Prefill Delta

    /// Computes the delta prompt to prefill by finding the longest matching prefix with the cached slot.
    public func computePrefillDelta(slotId: String, newPrompt: String) -> PrefillDelta {
        guard let slot = slots[slotId], !slot.prompt.isEmpty else {
            return PrefillDelta(
                cachedPrefixLength: 0,
                cachedTokensEstimate: 0,
                deltaPrompt: newPrompt,
                isIncremental: false,
                slotId: slotId
            )
        }

        // Find common prefix between cached prompt and new prompt
        let commonPrefix = longestCommonPrefix(slot.prompt, newPrompt)
        
        // Only treat as incremental cache hit if common prefix has significant length (>= 32 chars)
        if commonPrefix.count >= 32 {
            let delta = String(newPrompt[commonPrefix.endIndex...])
            let savedTokens = commonPrefix.count / 4
            return PrefillDelta(
                cachedPrefixLength: commonPrefix.count,
                cachedTokensEstimate: savedTokens,
                deltaPrompt: delta,
                isIncremental: true,
                slotId: slotId
            )
        }

        return PrefillDelta(
            cachedPrefixLength: 0,
            cachedTokensEstimate: 0,
            deltaPrompt: newPrompt,
            isIncremental: false,
            slotId: slotId
        )
    }

    /// Records completed turn in the slot.
    public func recordTurn(slotId: String, fullPrompt: String, toolPrefix: String? = nil, completion: String = "") {
        var slot = getOrCreateSlot(id: slotId, toolPrefix: toolPrefix)
        slot.prompt = fullPrompt + (completion.isEmpty ? "" : "\n" + completion)
        slot.estimatedTokens = slot.prompt.count / 4
        slot.turnCount += 1
        slot.lastAccessTime = Date()
        slot.isParked = false
        slots[slotId] = slot
    }

    public func slot(for id: String) -> ConversationSlot? {
        slots[id]
    }

    public func allSlots() -> [ConversationSlot] {
        Array(slots.values)
    }

    public func activeSlotCount() -> Int {
        slots.values.filter { !$0.isParked }.count
    }

    public func parkedSlotCount() -> Int {
        slots.values.filter { $0.isParked }.count
    }

    public func clear() {
        slots.removeAll()
    }

    // MARK: — Helpers

    private func longestCommonPrefix(_ a: String, _ b: String) -> Substring {
        var idxA = a.startIndex
        var idxB = b.startIndex
        while idxA < a.endIndex && idxB < b.endIndex && a[idxA] == b[idxB] {
            idxA = a.index(after: idxA)
            idxB = b.index(after: idxB)
        }
        return a[..<idxA]
    }

    private func evictOldestParkedSlot() {
        let sorted = slots.values.sorted { $0.lastAccessTime < $1.lastAccessTime }
        if let oldestParked = sorted.first(where: { $0.isParked }) {
            slots.removeValue(forKey: oldestParked.id)
        } else if let oldest = sorted.first {
            slots.removeValue(forKey: oldest.id)
        }
    }
}
