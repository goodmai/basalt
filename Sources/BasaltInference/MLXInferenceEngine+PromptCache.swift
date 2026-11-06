import Foundation
import MLX
import MLXLMCommon
import BasaltCore

struct AnchorSearch: Sendable {
    let prompt: [Int]
    let reused: Int
    let boundary: Int
    let limit: Int
    private(set) var found: [Int] = []

    var hasRoom: Bool { boundary - reused >= 2 * HybridPrefixSnapshot.minAnchorSpacing }
    var wantsMore: Bool { found.count < limit }

    mutating func admit(_ rendered: [Int]) -> Bool {
        let spacing = HybridPrefixSnapshot.minAnchorSpacing
        let offset = zip(prompt, rendered).prefix { $0 == $1 }.count
        guard offset - reused >= spacing else { return false }
        if boundary - offset >= spacing, found.allSatisfy({ abs($0 - offset) >= spacing }) {
            found.append(offset)
        }
        return true
    }
}

extension MLXInferenceEngine {
    static let modelSwitchPersistDeadline = Duration.seconds(10)

    func readPromptCaches<R: Sendable>(_ read: @Sendable (PromptCacheTree) throws -> R) rethrows -> R {
        try read(promptCaches)
    }

    func cacheIsHybrid(_ container: ModelContainer, epoch: ModelEpoch) async -> Bool {
        if let known = modelIsHybrid { return known }
        let kinds = (try? await container.perform { context -> (hybrid: Bool, recurrent: Bool) in
            let cache = try context.model.newCache(parameters: nil)
            return (PromptCache(cache).isHybrid, cache.contains { $0 is MambaCache })
        }) ?? (hybrid: false, recurrent: false)
        record(.learned(epoch) { ($0.isHybrid, $0.hasRecurrentState) = (kinds.hybrid, kinds.recurrent) })
        return kinds.hybrid
    }

    func snapshotBoundary(
        _ count: Int, container: ModelContainer, epoch: ModelEpoch, template: String?,
        _ makeInput: @Sendable (String) -> UserInput
    ) async -> Int {
        guard await cacheIsHybrid(container, epoch: epoch) else { return count - 1 }
        if let template, let suffix = templateSuffixes[template] {
            return count - suffix
        }
        let suffix = await HybridPrefixSnapshot.templateSuffixLength(container, makeInput)
        if let template, epoch == modelEpoch {
            if templateSuffixes.count >= 64 {
                templateSuffixes.removeAll()
            }
            templateSuffixes[template] = suffix
        }
        return count - suffix
    }

    func anchorOffsets(
        _ prompt: RenderedPrompt, _ search: AnchorSearch, container: ModelContainer, tools: [ToolSpec]?
    ) async -> [Int] {
        var search = search
        guard search.hasRoom else { return [] }
        for prefix in HybridPrefixSnapshot.anchorPrefixes(prompt.anchorTurns) where search.wantsMore {
            guard let rendered = try? await container.prepare(
                input: UserInput(
                    chat: Self.chat(prefix, lastText: prefix.last?.text ?? "", images: prompt.images), tools: tools,
                    additionalContext: prompt.context))
            else { continue }
            guard search.admit(rendered.text.tokens.asArray(Int.self)) else { break }
        }
        return search.found
    }

    func restoreFromDisk(_ namespace: String, prompt: [Int], container: ModelContainer, epoch: ModelEpoch) async {
        guard promptCacheDisk != nil, modelFingerprint != nil,
              promptCaches.storedPrefix(of: prompt, in: namespace) < prompt.count - 1,
              let layout = await cacheLayout(container, epoch: epoch),
              epoch == modelEpoch, let fingerprint = modelFingerprint,
              let hit = promptCacheDisk?.longestPrefix(
                fingerprint: fingerprint, namespace: namespace, of: Array(prompt.dropLast()),
                longerThan: promptCaches.storedPrefix(of: prompt, in: namespace), maxBytes: promptCaches.byteBudget)
        else { return }
        let started = ContinuousClock.now
        do {
            guard let cache = try promptCacheDisk?.load(hit.entry, layout: layout) else { return }
            promptCaches.store(namespace, tokens: Array(hit.entry.tokens.prefix(hit.usable)), cache: cache)
            let seconds = (ContinuousClock.now - started).inSeconds
            logger.info(String(
                format: "Restored %d of %d stored prompt tokens from disk: %.1f MB in %.3f s (%.2f GB/s)",
                hit.usable, hit.entry.tokens.count, Double(hit.entry.bytes) / 1e6, seconds,
                Double(hit.entry.bytes) / 1e9 / max(seconds, 1e-9)))
        } catch {
            let removed = promptCacheDisk?.discardIfBroken(hit.entry, after: error) == true
            logger.warn("Prompt-cache file \(hit.entry.name) could not be restored (\(removed ? "removed" : "kept, the error may be transient")): \(error)")
        }
    }

    public func persistPromptCache(within deadline: Duration) async {
        guard let writeRate = promptCacheDisk?.assumedWriteBytesPerSecond, let fingerprint = modelFingerprint else { return }
        let started = ContinuousClock.now
        var written: Set<String> = []
        var skipped = 0
        for path in promptCaches.storedPathsByRecency() {
            guard let lease = promptCaches.lease(path.namespace, for: path.tokens + [0]) else { continue }
            let tokens = Array(path.tokens.prefix(lease.reusedTokens))
            guard !tokens.isEmpty else { continue }
            let alreadyOnDisk = promptCacheDisk?.has(fingerprint: fingerprint, namespace: path.namespace, tokens: tokens) == true
            let expectedWrite = alreadyOnDisk ? 0 : Double(PromptCacheDiskStore.estimatedBytes(of: lease.cache, tokens: tokens.count))
                / writeRate
            guard ContinuousClock.now - started + .seconds(expectedWrite) <= deadline else {
                skipped += 1
                continue
            }
            if let name = promptCacheDisk?.store(
                fingerprint: fingerprint, model: modelDescription, namespace: path.namespace,
                tokens: tokens, cache: lease.cache, keeping: written) {
                written.insert(name)
            }
        }
        logger.info("Persisted \(written.count) prompt-cache paths in \(String(format: "%.2f", (ContinuousClock.now - started).inSeconds)) s"
            + (skipped > 0 ? "; \(skipped) paths skipped that could not be written within the \(deadline) deadline" : ""))
    }
}
