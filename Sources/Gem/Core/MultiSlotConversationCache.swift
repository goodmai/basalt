import Foundation

/// Ledger of conversation slots for prefix KV-cache reuse: prefix hashing, parking at 1/8 of the
/// buffer (tool prefix preserved) and the delta a new turn would need to prefill.
///
/// Not consulted by the engine yet: every request is still prefilled in full. Wiring a slot to the
/// engine's `[KVCache]` (trim to the common prefix, prefill only the tail) is #56 — until then
/// nothing may report tokens from this ledger as cached.
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
        public let cachedPrefix: String
        public let cachedPrefixLength: Int
        public let cachedTokensEstimate: Int
        public let deltaPrompt: String
        public let isIncremental: Bool
        public let slotId: String

        public var newPromptSuffix: String { deltaPrompt }
        public var cachedPromptTokens: Int { cachedTokensEstimate }

        public init(
            cachedPrefix: String = "",
            cachedPrefixLength: Int,
            cachedTokensEstimate: Int,
            deltaPrompt: String,
            isIncremental: Bool,
            slotId: String
        ) {
            self.cachedPrefix = cachedPrefix
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
            evictOldestSlot(except: id)
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

    /// Parks an inactive slot: truncates prompt to fit within 1/8 KV buffer on clean token boundaries while preserving prefix.
    public func parkSlot(id: String) {
        guard var slot = slots[id] else { return }
        slot.isParked = true

        let limitChars = parkingBudgetTokens * 4
        if slot.prompt.count > limitChars {
            // Keep head prefix up to limit on a clean line/space boundary (no synthetic separators that invalidate KV matching)
            let rawSub = slot.prompt.prefix(limitChars)
            if let lastBoundary = rawSub.lastIndex(where: { $0.isNewline || $0.isWhitespace }) {
                slot.prompt = String(slot.prompt[..<lastBoundary])
            } else {
                slot.prompt = String(rawSub)
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
        var commonPrefix = longestCommonPrefix(slot.prompt, newPrompt)
        
        // Ensure boundary does not cut mid-word/mid-token
        if commonPrefix.count >= 32 {
            if commonPrefix.endIndex < newPrompt.endIndex && commonPrefix.endIndex < slot.prompt.endIndex {
                let nextCharNew = newPrompt[commonPrefix.endIndex]
                let lastChar = newPrompt[newPrompt.index(before: commonPrefix.endIndex)]
                if !nextCharNew.isWhitespace && !nextCharNew.isPunctuation && !lastChar.isWhitespace {
                    if let lastBoundary = commonPrefix.lastIndex(where: { $0.isWhitespace || $0.isNewline }) {
                        commonPrefix = commonPrefix[...lastBoundary]
                    }
                }
            }
        }

        // Only treat as incremental cache hit if common prefix has significant length (>= 32 chars)
        if commonPrefix.count >= 32 {
            let delta = String(newPrompt[commonPrefix.endIndex...])
            let savedTokens = commonPrefix.count / 4
            return PrefillDelta(
                cachedPrefix: String(commonPrefix),
                cachedPrefixLength: commonPrefix.count,
                cachedTokensEstimate: savedTokens,
                deltaPrompt: delta,
                isIncremental: true,
                slotId: slotId
            )
        }

        return PrefillDelta(
            cachedPrefix: "",
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

    private func evictOldestSlot(except activeId: String? = nil) {
        let candidates = slots.values.filter { $0.id != activeId }
        let sorted = candidates.sorted { $0.lastAccessTime < $1.lastAccessTime }
        if let oldestParked = sorted.first(where: { $0.isParked }) {
            slots.removeValue(forKey: oldestParked.id)
        } else if let oldest = sorted.first {
            slots.removeValue(forKey: oldest.id)
        }
    }
}
