import Foundation

public enum WhisperTokenizerSource {
    static let repositories = [
        51864: "openai/whisper-tiny.en",
        51865: "openai/whisper-tiny",
        51866: "openai/whisper-large-v3-turbo",
    ]

    public static func repository(config: [String: Any]?) -> String? {
        guard config?["model_type"] as? String == "whisper",
              let vocabulary = (config?["n_vocab"] ?? config?["vocab_size"]) as? Int else { return nil }
        return repositories[vocabulary]
    }
}

extension HuggingFaceHub {
    func addingCompanionFiles(to directory: URL) async throws -> URL {
        let tokenizer = directory.appendingPathComponent("tokenizer.json")
        guard let source = WhisperTokenizerSource.repository(config: ModelCache.config(at: directory)),
              !FileManager.default.fileExists(atPath: tokenizer.path) else { return directory }
        try await fetch(repoId: source, path: "tokenizer.json").data.write(to: tokenizer, options: .atomic)
        return directory
    }
}
