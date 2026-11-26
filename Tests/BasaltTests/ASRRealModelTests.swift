import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Speech recognition on real weights: transcript, word error rate and real-time factor per clip (#818)", .serialized)
struct ASRRealModelTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var models: [String] { (env["BASALT_E2E_ASR_MODELS"] ?? "").split(separator: ",").map(String.init) }
    static var clips: [(path: String, text: String)] {
        let paths = (env["BASALT_E2E_ASR_AUDIO"] ?? "").split(separator: ",").map(String.init)
        let texts = (env["BASALT_E2E_ASR_TEXT"] ?? "").split(separator: "|").map(String.init)
        return Array(zip(paths, texts))
    }
    static var enabled: Bool { env["BASALT_E2E"] == "1" && !models.isEmpty && !clips.isEmpty }

    static func words(_ text: String) -> [String] {
        text.lowercased().unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? Character($0) : " " }
            .split(separator: " ").map { String($0) }
    }

    static func wordErrors(_ hypothesis: [String], _ reference: [String]) -> Int {
        var row = Array(0 ... hypothesis.count)
        for (i, word) in reference.enumerated() {
            var next = [i + 1]
            for (j, guess) in hypothesis.enumerated() {
                next.append(min(row[j + 1] + 1, next[j] + 1, row[j] + (word == guess ? 0 : 1)))
            }
            row = next
        }
        return row[hypothesis.count]
    }

    @Test("Word error rate stays low across the clips: each transcript, errors, words and real-time factor are logged",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_ASR_MODELS=<repo,…>, BASALT_E2E_ASR_AUDIO=<file,…> and BASALT_E2E_ASR_TEXT=<text|…>"), arguments: models)
    func transcribes(entry: String) async throws {
        let path = GoldenLogitsRealModelTests.path(entry)
        let log = try await ModelRunLog("asr", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 0))
        try await log.step("load") { try await engine.load(modelPath: path) }
        var (errors, total) = (0, 0)
        for clip in Self.clips {
            let url = URL(fileURLWithPath: clip.path)
            let audio = try AudioClip(encoded: try Data(contentsOf: url), format: url.pathExtension)
            let start = ContinuousClock.now
            let result = try await engine.transcribe(try TranscriptionRequest(audio: audio, language: "en"))
            let seconds = (ContinuousClock.now - start).inSeconds
            let (hypothesis, reference) = (Self.words(result.text), Self.words(clip.text))
            let wrong = Self.wordErrors(hypothesis, reference)
            (errors, total) = (errors + wrong, total + reference.count)
            log.record("result", "clip=\(url.lastPathComponent) errors=\(wrong)/\(reference.count) rtf=\(String(format: "%.3f", seconds / audio.seconds)) text=\(result.text)")
        }
        log.record("summary", "WER=\(String(format: "%.3f", Double(errors) / Double(max(total, 1)))) words=\(total)")
        #expect(Double(errors) / Double(max(total, 1)) < 0.25)
    }
}
