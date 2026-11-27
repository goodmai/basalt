import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Whisper checkpoints from mlx-community load without manual files: the tokenizer comes from openai/whisper-*, npz and tokenizer-less folders are refused clearly (#851)")
struct WhisperCheckpointTests {
    static func folder(config: [String: Any], files: [String]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        for file in files { try Data("x".utf8).write(to: directory.appendingPathComponent(file)) }
        return directory
    }

    @Test("The tokenizer repository follows the vocabulary size: English-only, multilingual and v3; other models and sizes get none",
          arguments: [
              (["model_type": "whisper", "n_vocab": 51864], "openai/whisper-tiny.en"),
              (["model_type": "whisper", "n_vocab": 51865], "openai/whisper-tiny"),
              (["model_type": "whisper", "vocab_size": 51866], "openai/whisper-large-v3-turbo"),
              (["model_type": "whisper", "n_vocab": 50000], nil),
              (["model_type": "qwen3", "n_vocab": 51865], nil),
          ] as [([String: any Sendable], String?)])
    func repositoryByVocabulary(config: [String: any Sendable], expected: String?) {
        #expect(WhisperTokenizerSource.repository(config: config) == expected)
    }

    @Test("A downloaded Whisper folder without tokenizer.json gets the one from the matching openai repository; a folder that has one is left alone")
    func fetchesMissingTokenizer() async throws {
        let host = StubURLProtocol.host { request, _ in
            request.url?.path == "/openai/whisper-large-v3-turbo/resolve/main/tokenizer.json"
                ? (200, [:], Data(#"{"model":{"type":"BPE"}}"#.utf8)) : (404, [:], Data())
        }
        let hub = HuggingFaceHub(configuration: StubURLProtocol.configuration, endpoint: "https://\(host)")
        let missing = try Self.folder(config: ["model_type": "whisper", "n_vocab": 51866], files: ["weights.safetensors"])
        let present = try Self.folder(config: ["model_type": "whisper", "n_vocab": 51866], files: ["weights.safetensors", "tokenizer.json"])
        defer { try? FileManager.default.removeItem(at: missing); try? FileManager.default.removeItem(at: present) }
        _ = try await hub.addingCompanionFiles(to: missing)
        _ = try await hub.addingCompanionFiles(to: present)
        #expect(try String(contentsOf: missing.appendingPathComponent("tokenizer.json"), encoding: .utf8) == #"{"model":{"type":"BPE"}}"#)
        #expect(try String(contentsOf: present.appendingPathComponent("tokenizer.json"), encoding: .utf8) == "x")
    }

    @Test("Load refuses an npz-only checkpoint and a Whisper folder without tokenizer.json, naming the cause; complete folders pass")
    func refusesIncompleteCheckpoints() throws {
        let npz = try Self.folder(config: ["model_type": "whisper", "n_vocab": 51865], files: ["weights.npz", "tokenizer.json"])
        let noTokenizer = try Self.folder(config: ["model_type": "whisper", "n_vocab": 51866], files: ["weights.safetensors"])
        let complete = try Self.folder(config: ["model_type": "whisper", "n_vocab": 51866], files: ["weights.safetensors", "tokenizer.json"])
        let other = try Self.folder(config: ["model_type": "qwen3"], files: ["model.safetensors"])
        defer { for url in [npz, noTokenizer, complete, other] { try? FileManager.default.removeItem(at: url) } }
        #expect { try ModelCapabilities.refuseIncompleteCheckpoint(at: npz, modelType: "whisper", path: "npz") } throws: { "\($0)".contains(".npz archive") }
        #expect { try ModelCapabilities.refuseIncompleteCheckpoint(at: noTokenizer, modelType: "whisper", path: "w") } throws: { "\($0)".contains("openai/whisper-large-v3-turbo") }
        try ModelCapabilities.refuseIncompleteCheckpoint(at: complete, modelType: "whisper", path: "c")
        try ModelCapabilities.refuseIncompleteCheckpoint(at: other, modelType: "qwen3", path: "o")
    }
}
