import Foundation
import MLX
import MLXLMCommon

final class PromptCache: @unchecked Sendable {
    let layers: [KVCache]

    init(_ layers: [KVCache]) {
        self.layers = layers
    }

    var tokenCount: Int {
        layers.map(\.offset).min() ?? 0
    }

    var byteCount: Int {
        layers.reduce(0) { total, layer in
            total + layer.innerState().reduce(0) { $0 + $1.nbytes }
        }
    }

    var isSharable: Bool {
        !layers.isEmpty && layers.allSatisfy { type(of: $0) == KVCacheSimple.self }
    }
}

enum PromptCacheBudget {
    static func automatic(workingSet: Int, weights: Int) -> Int {
        max(0, workingSet - weights) / 2
    }

    static func gpuWorkingSet() -> Int {
        GPU.maxRecommendedWorkingSetBytes() ?? Int(ProcessInfo.processInfo.physicalMemory) / 4 * 3
    }
}

final class PromptCacheNode: @unchecked Sendable {
    var tokens: [Int]
    var layers: [[MLXArray]]
    var children: [Int: PromptCacheNode] = [:]
    weak var parent: PromptCacheNode?
    var lastUse: Int
    var pins = 0

    init(tokens: [Int], layers: [[MLXArray]], lastUse: Int) {
        self.tokens = tokens
        self.layers = layers
        self.lastUse = lastUse
    }

    var bytes: Int {
        layers.joined().reduce(0) { $0 + $1.nbytes }
    }
}

struct PromptCacheLease: Sendable {
    let cache: PromptCache
    let reusedTokens: Int
    let pinned: PromptCacheNode
}

struct PromptCacheTree: Sendable {
    typealias Step = (node: PromptCacheNode, used: Int)

    let byteBudget: Int
    private(set) var roots: [String: PromptCacheNode] = [:]
    private(set) var byteCount = 0
    private var clock = 0

    init(byteBudget: Int) {
        self.byteBudget = byteBudget
    }

    func storedPrefix(of tokens: [Int], in namespace: String) -> Int {
        roots[namespace].map { Self.walk(from: $0, along: tokens).reduce(0) { $0 + $1.used } } ?? 0
    }

    mutating func lease(_ namespace: String, for prompt: [Int]) -> PromptCacheLease? {
        guard let root = roots[namespace] else { return nil }
        let path = Self.walk(from: root, along: Array(prompt.dropLast()))
        guard let deepest = path.last?.node else { return nil }
        touch(deepest)
        deepest.pins += 1
        let layers = deepest.layers.indices.map { layer -> KVCache in
            let cache = KVCacheSimple()
            cache.state = [0, 1].map { part in
                concatenated(path.map { $0.node.layers[layer][part][.ellipsis, ..<$0.used, 0...] }, axis: 2)
            }
            eval(cache.innerState())
            return cache
        }
        return PromptCacheLease(
            cache: PromptCache(layers), reusedTokens: path.reduce(0) { $0 + $1.used }, pinned: deepest)
    }

    mutating func release(_ lease: PromptCacheLease) {
        lease.pinned.pins -= 1
    }

    mutating func store(_ namespace: String, tokens: [Int], cache: PromptCache) {
        guard byteBudget > 0, !tokens.isEmpty, cache.isSharable, cache.tokenCount >= tokens.count else { return }
        let root = roots[namespace] ?? PromptCacheNode(tokens: [], layers: [], lastUse: 0)
        roots[namespace] = root
        let path = Self.walk(from: root, along: tokens)
        let matched = path.reduce(0) { $0 + $1.used }
        guard matched < tokens.count else {
            touch(path.last?.node ?? root)
            return
        }
        var parent = root
        for step in path {
            parent = step.used < step.node.tokens.count ? split(step.node, at: step.used) : step.node
        }
        let leaf = PromptCacheNode(
            tokens: Array(tokens[matched...]),
            layers: Self.slice(cache.layers.map(\.state), matched..<tokens.count),
            lastUse: 0)
        attach(leaf, to: parent)
        byteCount += leaf.bytes
        touch(leaf)
        evict()
    }

    mutating func removeAll() {
        roots.removeAll()
        byteCount = 0
    }

    func leaves() -> [PromptCacheNode] {
        var found: [PromptCacheNode] = []
        var pending = Array(roots.values)
        while let node = pending.popLast() {
            if node.children.isEmpty, !node.tokens.isEmpty {
                found.append(node)
            }
            pending.append(contentsOf: node.children.values)
        }
        return found
    }

    static func walk(from root: PromptCacheNode, along tokens: [Int]) -> [Step] {
        var path: [Step] = []
        var node = root
        var position = 0
        while position < tokens.count, let child = node.children[tokens[position]] {
            let used = zip(child.tokens, tokens[position...]).prefix { $0 == $1 }.count
            path.append((child, used))
            position += used
            guard used == child.tokens.count else { break }
            node = child
        }
        return path
    }

    private static func slice(_ layers: [[MLXArray]], _ range: Range<Int>) -> [[MLXArray]] {
        layers.map { layer in
            let sliced = layer.map { contiguous($0[.ellipsis, range, 0...]) }
            eval(sliced)
            return sliced
        }
    }

    private mutating func touch(_ node: PromptCacheNode) {
        clock += 1
        var current: PromptCacheNode? = node
        while let visited = current {
            visited.lastUse = clock
            current = visited.parent
        }
    }

    private func attach(_ child: PromptCacheNode, to parent: PromptCacheNode) {
        parent.children[child.tokens[0]] = child
        child.parent = parent
    }

    private mutating func split(_ node: PromptCacheNode, at count: Int) -> PromptCacheNode {
        let before = node.bytes
        let head = PromptCacheNode(
            tokens: Array(node.tokens[..<count]), layers: Self.slice(node.layers, 0..<count), lastUse: node.lastUse)
        node.layers = Self.slice(node.layers, count..<node.tokens.count)
        node.tokens = Array(node.tokens[count...])
        if let parent = node.parent {
            attach(head, to: parent)
        }
        attach(node, to: head)
        byteCount += head.bytes + node.bytes - before
        return head
    }

    private mutating func evict() {
        while byteCount > byteBudget,
              let victim = leaves().filter({ $0.pins == 0 }).min(by: { $0.lastUse < $1.lastUse }) {
            let bytesPerToken = max(1, victim.bytes / victim.tokens.count)
            let keep = victim.tokens.count - (byteCount - byteBudget + bytesPerToken - 1) / bytesPerToken
            let before = victim.bytes
            if keep > 0 {
                victim.layers = Self.slice(victim.layers, 0..<keep)
                victim.tokens = Array(victim.tokens.prefix(keep))
                byteCount += victim.bytes - before
            } else {
                victim.parent?.children[victim.tokens[0]] = nil
                byteCount -= before
            }
        }
    }
}
