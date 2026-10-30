import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

enum DiskFixture {
    static func directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("basalt-prompt-cache-\(UUID().uuidString)")
    }

    static func store(_ directory: URL, megabytes: Int64 = 64, days: Int = 7) -> PromptCacheDiskStore {
        PromptCacheDiskStore(directory: directory, byteQuota: megabytes << 20, retentionDays: days)
    }

    static func files(_ directory: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" }
    }

    static func permissions(_ url: URL) -> Int? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? NSNumber)?.intValue
    }

    static func engine(
        _ model: any LanguageModel, id: String, disk: PromptCacheDiskStore?, fingerprint: String = "fp-1"
    ) async -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 26, promptCacheDisk: disk)
        await engine.attach(
            ModelContainer(context: ModelContext(
                configuration: ModelConfiguration(id: id), model: model,
                processor: ByteInputProcessor(), tokenizer: ByteTokenizer())),
            fingerprint: disk == nil ? nil : fingerprint, modelDescription: id)
        return engine
    }

    static func dense(_ disk: PromptCacheDiskStore?, fingerprint: String = "fp-1") async throws -> MLXInferenceEngine {
        await engine(try TinyModel.make(), id: "tiny-llama", disk: disk, fingerprint: fingerprint)
    }

    static func hybrid(_ disk: PromptCacheDiskStore?, fingerprint: String = "fp-1") async throws -> MLXInferenceEngine {
        await engine(try TinyHybrid.make(), id: "tiny-qwen35", disk: disk, fingerprint: fingerprint)
    }

    static func prefilled(_ model: any LanguageModel, _ tokens: [Int]) throws -> PromptCache {
        let cache = PromptCache(try model.newCache(parameters: nil))
        eval(model(TinyHybrid.row(tokens), cache: cache.layers))
        return cache
    }

    static func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype && arrayEqual(a, b).item(Bool.self)
    }

    static func nextLogits(_ model: any LanguageModel, _ token: Int, _ cache: PromptCache) -> MLXArray {
        let logits = model(LMInput.Text(tokens: TinyHybrid.row([token])), cache: cache.layers, state: cache.state).logits
        eval(logits)
        return logits
    }
}

@Suite("Prompt cache persisted to disk (#174)", .serialized)
struct PromptCacheDiskTests {

    @Test("Dense KVCacheSimple path saved and loaded gives bit-identical KV and next-token logits to the in-memory lease")
    func denseRoundTripIsBitExact() throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try TinyModel.make()
        let prompt = TinyModel.tokens(Conversation.firstTurn)
        let prefix = Array(prompt.dropLast())
        var tree = PromptCacheTree(byteBudget: 1 << 26)
        tree.store("ns", tokens: prefix, cache: try DiskFixture.prefilled(model, prefix))
        let memory = try tree.required("ns", for: prompt)
        #expect(memory.reusedTokens == prefix.count)

        var writer = DiskFixture.store(directory)
        #expect(writer.store(fingerprint: "fp", model: "tiny", namespace: "ns", tokens: prefix, cache: memory.cache) != nil)

        var reader = DiskFixture.store(directory)
        let layout = try model.newCache(parameters: nil).map(PromptCacheDiskStore.layoutName)
        let entryFound = reader.longestPrefix(fingerprint: "fp", namespace: "ns", of: prefix, longerThan: 0, maxBytes: 1 << 26)?.entry
        let entry = try #require(entryFound)
        #expect(entry.tokens == prefix)
        let loaded = try reader.load(entry, layout: layout)
        #expect(loaded.layers.count == memory.cache.layers.count)
        for (disk, inMemory) in zip(loaded.layers, memory.cache.layers) {
            #expect(type(of: disk) == KVCacheSimple.self)
            #expect(disk.offset == inMemory.offset)
            for (a, b) in zip(disk.state, inMemory.state) {
                #expect(DiskFixture.identical(a, b))
            }
        }
        let fromDisk = DiskFixture.nextLogits(model, prompt.last!, loaded)
        let fromMemory = DiskFixture.nextLogits(model, prompt.last!, memory.cache)
        #expect(DiskFixture.identical(fromDisk, fromMemory))
        tree.release(memory)
    }

    @Test("Hybrid snapshot path (recurrent layers and attention KV) saved and loaded gives bit-identical next-token logits to the in-memory lease")
    func hybridRoundTripIsBitExact() throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try TinyHybrid.make()
        let prompt = TinyModel.tokens(Conversation.firstTurn)
        let kept = try TinyHybrid.snapshotted(model, prompt)
        var tree = PromptCacheTree(byteBudget: 1 << 26)
        tree.store("ns", tokens: Array(prompt.dropLast()), cache: kept)
        let memory = try tree.required("ns", for: prompt)
        #expect(memory.cache.isHybrid)
        let prefix = Array(prompt.prefix(memory.reusedTokens))

        var writer = DiskFixture.store(directory)
        #expect(writer.store(fingerprint: "fp", model: "tiny", namespace: "ns", tokens: prefix, cache: memory.cache) != nil)
        var reader = DiskFixture.store(directory)
        let layout = try model.newCache(parameters: nil).map(PromptCacheDiskStore.layoutName)
        let entryFound = reader.longestPrefix(fingerprint: "fp", namespace: "ns", of: prefix, longerThan: 0, maxBytes: 1 << 26)?.entry
        let entry = try #require(entryFound)
        let loaded = try reader.load(entry, layout: layout)
        #expect(loaded.isSnapshot(of: prefix.count))
        #expect(loaded.layers.contains { $0 is ArraysCache })

        let fromDisk = DiskFixture.nextLogits(model, prompt[prefix.count], loaded)
        let fromMemory = DiskFixture.nextLogits(model, prompt[prefix.count], memory.cache)
        #expect(DiskFixture.identical(fromDisk, fromMemory))
        tree.release(memory)
    }

    @Test("A server restart with the disk tier resumes the conversation from disk with the output of an uninterrupted run",
          arguments: ["dense", "hybrid"])
    func restartResumesFromDisk(kind: String) async throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        func make(_ disk: PromptCacheDiskStore?) async throws -> MLXInferenceEngine {
            kind == "dense" ? try await DiskFixture.dense(disk) : try await DiskFixture.hybrid(disk)
        }
        let before = try await make(DiskFixture.store(directory))
        let first = try await before.generate(request: greedy(Conversation.firstTurn, key: "shared"))
        await before.persistPromptCache(within: .seconds(30))
        #expect(DiskFixture.files(directory).count == 1)

        let resumed = try await make(DiskFixture.store(directory))
        let warm = try await resumed.generate(request: greedy(Conversation.afterToolCall, key: "shared"))
        let cold = try await make(nil).generate(request: greedy(Conversation.afterToolCall, key: "shared"))
        #expect(first.cachedPromptTokens == 0)
        #expect(cold.cachedPromptTokens == 0)
        #expect(warm.cachedPromptTokens > 0)
        #expect(warm.generatedText == cold.generatedText)
    }

    @Test("basalt chat --resume with the disk tier reuses the KV its previous run persisted on its first request")
    func chatResumeHitsDisk() async throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let before = try await DiskFixture.dense(DiskFixture.store(directory))
        let first = try await before.generate(
            request: ChatCommand.request(Conversation.firstTurn, history: [], maxTokens: 6, sessionId: "chat-r"))
        await before.persistPromptCache(within: .seconds(30))

        let history = [ChatMessageInput(role: .user, text: Conversation.firstTurn), ChatMessageInput(role: .assistant, text: first.generatedText)]
        let resumed = try await DiskFixture.dense(DiskFixture.store(directory))
        let reply = try await resumed.generate(
            request: ChatCommand.request("and the tests", history: history, maxTokens: 6, sessionId: "chat-r"))
        #expect(reply.cachedPromptTokens >= TinyModel.tokens(Conversation.firstTurn).count - 1)
        let other = try await DiskFixture.dense(DiskFixture.store(directory))
        let unrelated = try await other.generate(
            request: ChatCommand.request("and the tests", history: history, maxTokens: 6, sessionId: "chat-other"))
        #expect(unrelated.cachedPromptTokens == 0)
    }

    @Test("A file written for another model fingerprint is a silent miss and stays for its own model")
    func otherFingerprintMisses() async throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let before = try await DiskFixture.dense(DiskFixture.store(directory), fingerprint: "weights-a")
        _ = try await before.generate(request: greedy(Conversation.firstTurn, key: "shared"))
        await before.persistPromptCache(within: .seconds(30))

        let changed = try await DiskFixture.dense(DiskFixture.store(directory), fingerprint: "weights-b")
        let reply = try await changed.generate(request: greedy(Conversation.afterToolCall, key: "shared"))
        let cold = try await DiskFixture.dense(nil).generate(request: greedy(Conversation.afterToolCall, key: "shared"))
        #expect(reply.cachedPromptTokens == 0)
        #expect(reply.generatedText == cold.generatedText)
        #expect(DiskFixture.files(directory).count == 1)
    }

    @Test("A truncated file is removed and the request is served by a normal prefill")
    func truncatedFileFallsBack() async throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let before = try await DiskFixture.dense(DiskFixture.store(directory))
        _ = try await before.generate(request: greedy(Conversation.firstTurn, key: "shared"))
        await before.persistPromptCache(within: .seconds(30))
        let file = try #require(DiskFixture.files(directory).first)
        let size = try #require(try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? Int)
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(size / 2))
        try handle.close()

        let resumed = try await DiskFixture.dense(DiskFixture.store(directory))
        let reply = try await resumed.generate(request: greedy(Conversation.afterToolCall, key: "shared"))
        let cold = try await DiskFixture.dense(nil).generate(request: greedy(Conversation.afterToolCall, key: "shared"))
        #expect(reply.cachedPromptTokens == 0)
        #expect(reply.generatedText == cold.generatedText)
        #expect(DiskFixture.files(directory).isEmpty)
    }

    @Test("A file whose layers do not match the model's cache layout is removed and the request is served by a normal prefill")
    func wrongLayoutFallsBack() async throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let dense = try await DiskFixture.dense(DiskFixture.store(directory), fingerprint: "same")
        _ = try await dense.generate(request: greedy(Conversation.firstTurn, key: "shared"))
        await dense.persistPromptCache(within: .seconds(30))
        #expect(DiskFixture.files(directory).count == 1)

        let hybrid = try await DiskFixture.hybrid(DiskFixture.store(directory), fingerprint: "same")
        let reply = try await hybrid.generate(request: greedy(Conversation.afterToolCall, key: "shared"))
        let cold = try await DiskFixture.hybrid(nil).generate(request: greedy(Conversation.afterToolCall, key: "shared"))
        #expect(reply.cachedPromptTokens == 0)
        #expect(reply.generatedText == cold.generatedText)
        #expect(DiskFixture.files(directory).isEmpty)
    }

    @Test("A file of another cache-format version is kept but never loaded, a store file with a corrupt header is removed, and other files in the directory are never touched")
    func foreignFilesAreLeftAlone() throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var store = PromptCacheDiskStore(directory: directory, byteQuota: 0, retentionDays: 0)
        try store.prepareDirectory()
        let tokens = [1, 2, 3]
        let otherFormat = directory.appendingPathComponent(PromptCacheDiskStore.fileName(fingerprint: "fp", namespace: "ns", tokens: tokens))
        let cache = try DiskFixture.prefilled(try TinyModel.make(), tokens)
        try savePromptCache(
            url: otherFormat, cache: cache.layers,
            metadata: ["basalt_format": "0", "basalt_fingerprint": "fp", "basalt_namespace": "ns", "basalt_tokens": "1,2,3"])
        let corrupt = directory.appendingPathComponent(PromptCacheDiskStore.fileName(fingerprint: "fp", namespace: "ns", tokens: [9]))
        try Data(repeating: 0xFF, count: 64).write(to: corrupt)
        let weights = directory.appendingPathComponent("model.safetensors")
        try save(arrays: ["x": MLXArray([1, 2, 3])], url: weights)
        let notes = directory.appendingPathComponent("notes.txt")
        try Data("keep".utf8).write(to: notes)

        #expect(store.longestPrefix(fingerprint: "fp", namespace: "ns", of: tokens + [4], longerThan: 0, maxBytes: 1 << 26) == nil)
        #expect(FileManager.default.fileExists(atPath: otherFormat.path))
        #expect(!FileManager.default.fileExists(atPath: corrupt.path))
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -86_400)], ofItemAtPath: weights.path)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -86_400)], ofItemAtPath: notes.path)
        try store.prune()
        #expect(!FileManager.default.fileExists(atPath: otherFormat.path))
        #expect(FileManager.default.fileExists(atPath: weights.path))
        #expect(FileManager.default.fileExists(atPath: notes.path))
    }

    @Test("The model fingerprint changes with the weight shards, the config and the chat template, and only with them")
    func fingerprintTracksModelFiles() throws {
        let model = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: model) }
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        func write(_ name: String, _ text: String) throws {
            try Data(text.utf8).write(to: model.appendingPathComponent(name))
        }
        try write("config.json", "{}")
        try write("chat_template.jinja", "{{ messages }}")
        try write("model.safetensors", "1234")
        let original = PromptCacheDiskStore.fingerprint(modelDirectory: model)
        try write("README.md", "unrelated")
        #expect(PromptCacheDiskStore.fingerprint(modelDirectory: model) == original)
        try write("chat_template.jinja", "{{ messages }}!")
        let retemplated = PromptCacheDiskStore.fingerprint(modelDirectory: model)
        #expect(retemplated != original)
        try write("model.safetensors", "12345")
        let reweighted = PromptCacheDiskStore.fingerprint(modelDirectory: model)
        #expect(reweighted != retemplated)
        try write("config.json", "{\"a\": 1}")
        #expect(PromptCacheDiskStore.fingerprint(modelDirectory: model) != reweighted)
    }

    @Test("A stored file is 0600 in a 0700 directory, and no temporary file is left or ever visible to a lookup")
    func writesAreAtomicAndPrivate() throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o755])
        let interrupted = directory.appendingPathComponent(".tmp-interrupted.safetensors")
        try Data(repeating: 7, count: 100).write(to: interrupted)
        let tokens = [5, 6, 7, 8]
        var store = DiskFixture.store(directory)
        let name = try #require(store.store(
            fingerprint: "fp", model: "tiny", namespace: "ns", tokens: tokens, cache: try DiskFixture.prefilled(try TinyModel.make(), tokens)))
        #expect(DiskFixture.permissions(directory.appendingPathComponent(name)) == 0o600)
        #expect(DiskFixture.permissions(directory) == 0o700)
        #expect(store.entries().map(\.name) == [name])
        #expect(FileManager.default.fileExists(atPath: interrupted.path))
        #expect(store.temporaryFiles().map(\.lastPathComponent) == [interrupted.lastPathComponent])
        try store.prune()
        #expect(store.temporaryFiles().isEmpty)
        #expect(store.entries().map(\.name) == [name])
    }

    @Test("Writes stay within the quota by evicting the least recently used file, never one this flush already wrote, and skip a path larger than the quota")
    func quotaEvictsLeastRecentlyUsed() throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try TinyModel.make()
        let cache = try DiskFixture.prefilled(model, Array(0..<200))
        let fileBytes = PromptCacheDiskStore.estimatedBytes(of: cache, tokens: 200)
        var store = PromptCacheDiskStore(directory: directory, byteQuota: fileBytes * 2, retentionDays: 7)
        let aFound = store.store(fingerprint: "fp", model: "m", namespace: "a", tokens: Array(0..<200), cache: cache)
        let a = try #require(aFound)
        let bFound = store.store(fingerprint: "fp", model: "m", namespace: "b", tokens: Array(0..<200), cache: cache)
        let b = try #require(bFound)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -60)], ofItemAtPath: directory.appendingPathComponent(b).path)
        let cFound = store.store(fingerprint: "fp", model: "m", namespace: "c", tokens: Array(0..<200), cache: cache)
        let c = try #require(cFound)
        #expect(Set(store.entries().map(\.name)) == [a, c])
        #expect(store.store(fingerprint: "fp", model: "m", namespace: "d", tokens: Array(0..<200), cache: cache, keeping: [a, c]) == nil)
        #expect(Set(store.entries().map(\.name)) == [a, c])
        let total = DiskFixture.files(directory).reduce(Int64(0)) { sum, url in
            sum + Int64((try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0)
        }
        #expect(total <= store.byteQuota)

        var tiny = PromptCacheDiskStore(directory: DiskFixture.directory(), byteQuota: fileBytes / 2, retentionDays: 7)
        #expect(tiny.store(fingerprint: "fp", model: "m", namespace: "a", tokens: Array(0..<200), cache: cache) == nil)
        #expect(!FileManager.default.fileExists(atPath: tiny.directory.path))
    }

    @Test("A dense file is loaded for any common prefix that covers at least half of it, a hybrid file only for its whole snapshot point")
    func lookupServesWorthwhilePrefixes() throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let stored = Array(0..<100)
        var store = DiskFixture.store(directory)
        let denseFound = store.store(fingerprint: "fp", model: "m", namespace: "dense", tokens: stored, cache: try DiskFixture.prefilled(try TinyModel.make(), stored))
        let dense = try #require(denseFound)
        let hybridCache = try TinyHybrid.snapshotted(try TinyHybrid.make(), stored + [100])
        let hybridFound = store.store(fingerprint: "fp", model: "m", namespace: "hybrid", tokens: stored, cache: hybridCache)
        _ = try #require(hybridFound)
        func usable(_ namespace: String, _ prompt: [Int], stored memory: Int = 0) -> Int? {
            store.longestPrefix(fingerprint: "fp", namespace: namespace, of: prompt, longerThan: memory, maxBytes: 1 << 26)?.usable
        }
        #expect(usable("dense", Array(0..<60) + [999]) == 60)
        #expect(usable("dense", Array(0..<10) + [999]) == nil)
        #expect(usable("dense", stored + [7]) == 100)
        #expect(usable("dense", stored + [7], stored: 100) == nil)
        #expect(usable("hybrid", Array(0..<99) + [999]) == nil)
        #expect(usable("hybrid", stored + [7]) == 100)
        #expect(store.entries().first { $0.name == dense }?.servesAnyPrefix == true)
    }

    @Test("prune drops files past their retention, then the least recently used beyond the cap")
    func pruneAppliesRetentionAndCap() throws {
        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = try DiskFixture.prefilled(try TinyModel.make(), [1, 2, 3])
        var store = DiskFixture.store(directory, days: 7)
        let oldFound = store.store(fingerprint: "fp", model: "m", namespace: "old", tokens: [1, 2, 3], cache: cache)
        let old = try #require(oldFound)
        let coldFound = store.store(fingerprint: "fp", model: "m", namespace: "cold", tokens: [1, 2, 3], cache: cache)
        let cold = try #require(coldFound)
        let hotFound = store.store(fingerprint: "fp", model: "m", namespace: "hot", tokens: [1, 2, 3], cache: cache)
        let hot = try #require(hotFound)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -8 * 86_400)], ofItemAtPath: directory.appendingPathComponent(old).path)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -86_400)], ofItemAtPath: directory.appendingPathComponent(cold).path)
        #expect(try store.prune().files == 1)
        #expect(Set(store.entries().map(\.name)) == [cold, hot])
        let size = try #require(store.entries().first?.bytes)
        let capped = PromptCacheDiskStore(directory: directory, byteQuota: size, retentionDays: 7)
        #expect(try capped.prune().files == 1)
        #expect(store.entries().map(\.name) == [hot])
    }

    @Test("Without --prompt-cache-disk-mb no store exists, so nothing is written and no directory is created")
    func diskTierIsOptIn() async throws {
        let directory = DiskFixture.directory()
        let options = try ServeCommand.parse(["--prompt-cache-dir", directory.path]).promptCacheDisk
        #expect(options.megabytes == 0)
        #expect(try options.store() == nil)
        let enabled = try ServeCommand.parse(["--prompt-cache-dir", directory.path, "--prompt-cache-disk-mb", "1"]).promptCacheDisk
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        let store = try #require(try enabled.store())
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(store.byteQuota == 1 << 20)
        #expect(DiskFixture.permissions(directory) == 0o700)
    }

    @Test("The disk tier is refused with --session-log meta or off, for serve and chat", arguments: ["meta", "off"])
    func refusedWithoutFullSessionLog(mode: String) throws {
        #expect(throws: (any Error).self) {
            _ = try ServeCommand.parse(["--prompt-cache-disk-mb", "100", "--session-log", mode])
        }
        #expect(throws: (any Error).self) {
            _ = try ChatCommand.parse(["--prompt-cache-disk-mb", "100", "--session-log", mode])
        }
        #expect(throws: Never.self) { _ = try ServeCommand.parse(["--session-log", mode]) }
        #expect(throws: Never.self) { _ = try ServeCommand.parse(["--prompt-cache-disk-mb", "100", "--session-log", "full"]) }
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--prompt-cache-disk-mb", "100", "--prompt-cache-mb", "0"]) }
    }
}
