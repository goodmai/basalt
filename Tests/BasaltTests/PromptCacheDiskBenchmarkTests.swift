import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Prompt cache disk tier on real models: resumed-session TTFT against a full prefill (#174)", .serialized)
struct PromptCacheDiskBenchmarkTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_DISK_MODELS"] ?? "").split(separator: ",").map(String.init)
    }

    static var lengths: [Int] {
        (env["BASALT_E2E_DISK_TOKENS"] ?? "4096,16384").split(separator: ",").compactMap { Int($0) }
    }

    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    struct Timed {
        let ttft: Double
        let text: String
        let promptTokens: Int
        let cachedTokens: Int
    }

    static func timed(_ engine: MLXInferenceEngine, _ request: GenerationRequest) async throws -> Timed {
        let clock = ContinuousClock()
        let start = clock.now
        let stream = try await engine.generateStream(request: request)
        var first: Duration?
        var text = ""
        var response: GenerationResponse?
        for await chunk in stream {
            switch chunk {
            case .text(let piece):
                first = first ?? clock.now - start
                text += piece
            case .reasoning:
                first = first ?? clock.now - start
            case .metadata(let metadata):
                response = metadata
            case .toolCall:
                first = first ?? clock.now - start
            }
        }
        let seconds = first.map { Double($0.components.seconds) + Double($0.components.attoseconds) / 1e18 } ?? .nan
        return Timed(
            ttft: seconds, text: text, promptTokens: response?.promptTokens ?? 0, cachedTokens: response?.cachedPromptTokens ?? 0)
    }

    static func document(tokens wanted: Int, engine: MLXInferenceEngine) async throws -> String {
        let sources = root.appendingPathComponent("Sources")
        let files = (FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? [])
            .filter { $0.pathExtension == "swift" }.sorted { $0.path < $1.path }
        var text = ""
        for file in files {
            text += "// \(file.lastPathComponent)\n" + (try String(contentsOf: file, encoding: .utf8)) + "\n"
            if let count = await engine.promptTokenCount(for: GenerationRequest(prompt: text)), count >= wanted {
                return String(text.prefix(text.count * wanted / count))
            }
        }
        throw BasaltError.invalidRequestStructure(details: "the repository sources hold fewer than \(wanted) tokens")
    }

    static func request(_ messages: [ChatMessageInput]) -> GenerationRequest {
        GenerationRequest(
            prompt: "", maxTokens: 24, temperature: 0, enableThinking: false, cacheKey: "shared", messages: messages)
    }

    static func uncache(_ file: URL) throws {
        let copy = file.deletingLastPathComponent().appendingPathComponent(".tmp-uncached-" + file.lastPathComponent)
        let source = open(file.path, O_RDONLY)
        let target = open(copy.path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        defer {
            close(source)
            close(target)
        }
        guard source >= 0, target >= 0, fcntl(source, F_NOCACHE, 1) == 0, fcntl(target, F_NOCACHE, 1) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        var buffer = [UInt8](repeating: 0, count: 8 << 20)
        while true {
            let count = read(source, &buffer, buffer.count)
            guard count > 0 else {
                guard count == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                break
            }
            guard write(target, buffer, count) == count else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        }
        fsync(target)
        guard rename(copy.path, file.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    }

    static func files(_ directory: URL) -> [(url: URL, bytes: Int)] {
        ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? [])
            .filter { $0.pathExtension == "safetensors" && !$0.lastPathComponent.hasPrefix(".") }
            .map { ($0, (try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
    }

    static func measure(repo: String, path: String, length: Int) async throws {
        let log = try await ModelRunLog("prompt-cache-disk", modelPath: path)
        defer { log.releaseGPU() }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-disk-bench-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = PromptCacheDiskStore(directory: directory, byteQuota: 32 << 30, retentionDays: 1)
        log.record("params", "tokens \(length) quota 32 GB dir \(directory.path)")

        var writer: MLXInferenceEngine? = MLXInferenceEngine(promptCacheDisk: store)
        try await log.step("load-writer") { try await writer!.load(modelPath: path) }
        let text = try await log.step("document") { try await document(tokens: length, engine: writer!) }
        let firstTurn = [ChatMessageInput(role: .user, text: text + "\n\nIn one sentence, what does this code do?")]
        let first = try await log.step("turn1-full-prefill") { try await timed(writer!, request(firstTurn)) }
        log.record("turn1", String(format: "prompt %d cached %d ttft %.3f s", first.promptTokens, first.cachedTokens, first.ttft))
        let secondTurn = firstTurn + [
            ChatMessageInput(role: .assistant, text: first.text), ChatMessageInput(role: .user, text: "Name two risks in it."),
        ]

        let persistStart = ContinuousClock.now
        await writer!.persistPromptCache(within: .seconds(600))
        let persistSeconds = (ContinuousClock.now - persistStart).components
        let written = files(directory)
        log.record("persist", String(
            format: "%d files %.1f MB in %.3f s", written.count, Double(written.reduce(0) { $0 + $1.bytes }) / 1e6,
            Double(persistSeconds.seconds) + Double(persistSeconds.attoseconds) / 1e18))
        guard !written.isEmpty else {
            log.record("FAIL persist", "no file written")
            throw CancellationError()
        }

        let uninterrupted = try await log.step("turn2-uninterrupted") { try await timed(writer!, request(secondTurn)) }
        log.record("turn2-uninterrupted", String(
            format: "prompt %d cached %d ttft %.3f s", uninterrupted.promptTokens, uninterrupted.cachedTokens, uninterrupted.ttft))
        writer = nil
        Memory.clearCache()
        try await log.step("uncache") { for file in written { try uncache(file.url) } }

        var resumed: MLXInferenceEngine? = MLXInferenceEngine(promptCacheDisk: store)
        try await log.step("load-resumed") { try await resumed!.load(modelPath: path) }
        let resume = try await log.step("turn2-resumed-from-disk") { try await timed(resumed!, request(secondTurn)) }
        log.record("turn2-resumed", String(
            format: "prompt %d cached %d ttft %.3f s output %@", resume.promptTokens, resume.cachedTokens, resume.ttft,
            resume.text == uninterrupted.text ? "identical to uninterrupted" : "DIFFERS from uninterrupted"))
        resumed = nil
        Memory.clearCache()

        var cold: MLXInferenceEngine? = MLXInferenceEngine(promptCacheBytes: 0)
        try await log.step("load-no-cache") { try await cold!.load(modelPath: path) }
        let full = try await log.step("turn2-full-prefill") { try await timed(cold!, request(secondTurn)) }
        log.record("turn2-full", String(format: "prompt %d cached %d ttft %.3f s", full.promptTokens, full.cachedTokens, full.ttft))
        cold = nil
        Memory.clearCache()

        try await log.step("uncache-read") { for file in files(directory) { try uncache(file.url) } }
        let largest = try #require(files(directory).max { $0.bytes < $1.bytes })
        let readStart = ContinuousClock.now
        let snapshot = try loadPromptCacheSnapshot(url: largest.url)
        eval(snapshot.cache.flatMap(\.state))
        let read = (ContinuousClock.now - readStart).components
        let readSeconds = Double(read.seconds) + Double(read.attoseconds) / 1e18
        log.record("cold-read", String(format: "%.1f MB in %.3f s, %.2f GB/s", Double(largest.bytes) / 1e6, readSeconds, Double(largest.bytes) / 1e9 / readSeconds))
        log.record("result", String(
            format: "tokens %d resume/full ttft %.3f (%.3f / %.3f s) target <= 0.2 %@ cached %d output %@ %@",
            full.promptTokens, resume.ttft / full.ttft, resume.ttft, full.ttft, resume.ttft <= 0.2 * full.ttft ? "met" : "MISSED",
            resume.cachedTokens, resume.text == uninterrupted.text ? "identical" : "DIFFERS", log.host()))
        #expect(resume.cachedTokens > 0)
        #expect(resume.text == uninterrupted.text)
    }

    @Test("Resume a session from the disk tier after a restart and compare its TTFT with a full prefill, its output with the uninterrupted session",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_DISK_MODELS to comma-separated cached repo ids (BASALT_E2E_DISK_TOKENS, default 4096,16384)"))
    func resumeFromDiskAgainstFullPrefill() async throws {
        var failed: [String] = []
        for repo in Self.models {
            guard let directory = ModelCache.resolve(repoId: repo) else {
                failed.append("\(repo): not in the local cache")
                continue
            }
            for length in Self.lengths {
                do {
                    try await Self.measure(repo: repo, path: directory.path, length: length)
                } catch {
                    failed.append("\(repo) \(length): \(error)")
                }
            }
        }
        #expect(failed.isEmpty, "\(failed)")
    }
}
