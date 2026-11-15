import Foundation

public struct CoreAISpeechBundle: Sendable, Codable, Equatable {
    public static let manifest = "basalt_speech.json"
    public static let defaultModelID = "Kokoro-82M-ANE"

    public struct Function: Sendable, Codable, Equatable {
        public var name: String
        public var length: Int

        public init(name: String, length: Int) {
            (self.name, self.length) = (name, length)
        }
    }

    public struct VoiceEntry: Sendable, Codable, Equatable {
        public var id: String
        public var name: String
        public var language: String
        public var index: Int

        public init(id: String, name: String, language: String, index: Int) {
            (self.id, self.name, self.language, self.index) = (id, name, language, index)
        }
    }

    public var modelID: String
    public var asset: String
    public var sampleRate: Int
    public var voices: [VoiceEntry]
    public var maxPhonemeTokens: Int
    public var compute: SpeechComputeBackend
    public var functions: [Function]
    public var vocab: [String: Int]
    public var padToken: Int

    public static func isBundle(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appending(component: manifest).path)
    }

    public init(directory: URL) throws(BasaltError) {
        let manifestURL = directory.appending(component: Self.manifest)
        guard let data = try? Data(contentsOf: manifestURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw .modelConfigurationInvalid(path: directory.path, reason: "\(Self.manifest) is missing or not valid JSON")
        }
        let body = (root["basalt_speech"] as? [String: Any]) ?? root
        try self.init(manifest: body, path: directory.path)
        let modelPath = directory.appending(component: "\(asset).aimodel").path
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw .modelConfigurationInvalid(path: directory.path, reason: "the Core AI speech bundle has \(Self.manifest) but no \(asset).aimodel")
        }
    }

    public init(manifest: [String: Any], path: String) throws(BasaltError) {
        guard let asset = manifest["asset"] as? String, !asset.isEmpty else {
            throw .modelConfigurationInvalid(path: path, reason: "basalt_speech.json needs a non-empty asset name")
        }
        let sampleRate = manifest["sample_rate"] as? Int ?? manifest["sampleRate"] as? Int ?? SpeechRequest.defaultSampleRate
        let maxTokens = manifest["max_phoneme_tokens"] as? Int ?? manifest["maxPhonemeTokens"] as? Int ?? 512
        guard sampleRate > 0, maxTokens > 0 else {
            throw .modelConfigurationInvalid(path: path, reason: "sampleRate and maxPhonemeTokens must be positive")
        }
        let rawCompute = manifest["compute"] as? String ?? SpeechComputeBackend.ane.rawValue
        guard let compute = SpeechComputeBackend(rawValue: rawCompute) else {
            throw .modelConfigurationInvalid(path: path, reason: "compute must be mlx, ane or apple_speech")
        }
        let parsedVoices = Self.parseVoices(manifest["voices"])
        guard !parsedVoices.isEmpty else {
            throw .modelConfigurationInvalid(path: path, reason: "basalt_speech.json must declare at least one voice")
        }
        let parsedFunctions = Self.parseFunctions(manifest["functions"], fallbackLength: maxTokens)
        guard !parsedFunctions.isEmpty, maxTokens <= parsedFunctions[parsedFunctions.count - 1].length else {
            throw .modelConfigurationInvalid(path: path, reason: "functions must have positive lengths covering maxPhonemeTokens")
        }
        modelID = manifest["model_id"] as? String ?? manifest["modelID"] as? String ?? Self.defaultModelID
        self.asset = asset
        self.sampleRate = sampleRate
        voices = parsedVoices
        maxPhonemeTokens = maxTokens
        self.compute = compute
        functions = parsedFunctions
        vocab = manifest["vocab"] as? [String: Int] ?? [:]
        padToken = manifest["pad_token"] as? Int ?? manifest["padToken"] as? Int ?? 0
    }

    public var voiceSpecs: [VoiceSpec] {
        voices.map { VoiceSpec(id: $0.id, name: $0.name, language: $0.language, model: modelID, compute: compute) }
    }

    public func voice(for requested: String, language: String?) -> VoiceEntry? {
        if let exact = voices.first(where: { $0.id == requested || $0.name.caseInsensitiveCompare(requested) == .orderedSame }) {
            return exact
        }
        if requested == "alloy" || requested == "default" {
            if let language, let matched = voices.first(where: { $0.language.lowercased().hasPrefix(language.lowercased()) }) {
                return matched
            }
            return voices.first
        }
        return nil
    }

    public func function(for count: Int) -> Function {
        functions.first { $0.length >= count } ?? functions[functions.count - 1]
    }

    public func encodeTokens(_ text: String, length: Int) -> [Int32] {
        var ids: [Int] = []
        ids.reserveCapacity(min(text.count, maxPhonemeTokens))
        for char in text {
            if ids.count >= maxPhonemeTokens { break }
            if let mapped = vocab[String(char)] {
                ids.append(mapped)
            } else if let scalar = char.unicodeScalars.first {
                ids.append(Int(scalar.value % 255) + 1)
            }
        }
        let clipped = Array(ids.prefix(length))
        let padded = clipped + Array(repeating: padToken, count: max(0, length - clipped.count))
        return padded.map(Int32.init)
    }

    private static func parseVoices(_ raw: Any?) -> [VoiceEntry] {
        if let names = raw as? [String] {
            return names.enumerated().map { idx, id in
                VoiceEntry(id: id, name: id, language: id.hasPrefix("bf_") ? "en-GB" : "en-US", index: idx)
            }
        }
        guard let items = raw as? [[String: Any]] else { return [] }
        return items.enumerated().compactMap { idx, entry in
            guard let id = entry["id"] as? String, !id.isEmpty else { return nil }
            let name = entry["name"] as? String ?? id
            let language = entry["language"] as? String ?? "en-US"
            let index = entry["index"] as? Int ?? idx
            return VoiceEntry(id: id, name: name, language: language, index: index)
        }
    }

    private static func parseFunctions(_ raw: Any?, fallbackLength: Int) -> [Function] {
        guard let items = raw as? [[String: Any]], !items.isEmpty else {
            return [Function(name: "main", length: fallbackLength)]
        }
        return items.compactMap { item -> Function? in
            guard let name = item["function"] as? String ?? item["name"] as? String,
                  let length = item["length"] as? Int, !name.isEmpty, length > 0 else { return nil }
            return Function(name: name, length: length)
        }.sorted { $0.length < $1.length }
    }
}
