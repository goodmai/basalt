import Foundation

public struct CoreAISpeechBundle: Sendable, Codable, Equatable {
    public static let manifest = "basalt_speech.json"
    public static let kokoroANEManifest = "kokoro_ane.json"
    public static let defaultModelID = "Kokoro-82M-ANE"

    public enum AssetKind: String, Sendable, Codable, Equatable {
        case coreAI = "core_ai"
        case coreMLCompiled = "coreml_compiled"
        case coreMLPackage = "coreml_package"
        case synthetic
    }

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
    public var assetKind: AssetKind

    public static func isBundle(_ directory: URL) -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: directory.appending(component: manifest).path)
            || fm.fileExists(atPath: directory.appending(component: kokoroANEManifest).path) {
            return true
        }
        let ext = directory.pathExtension.lowercased()
        return (ext == "mlmodelc" || ext == "mlpackage") && directory.lastPathComponent.lowercased().contains("kokoro")
    }

    public init(directory: URL) throws(BasaltError) {
        if let directKind = Self.directCoreMLAssetKind(directory) {
            let name = directory.deletingPathExtension().lastPathComponent
            try self.init(
                manifest: ["model_id": name, "asset": directory.lastPathComponent, "voices": TTSVoiceCatalog.kokoroVoiceIDs],
                path: directory.path
            )
            assetKind = directKind
            return
        }
        let (manifestURL, usedKokoroManifest) = Self.resolveManifestURL(in: directory)
        guard let data = try? Data(contentsOf: manifestURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw .modelConfigurationInvalid(path: directory.path, reason: "\(manifestURL.lastPathComponent) is missing or not valid JSON")
        }
        var body = (root["basalt_speech"] as? [String: Any]) ?? root
        if usedKokoroManifest, body["voices"] == nil {
            body["voices"] = TTSVoiceCatalog.kokoroVoiceIDs
        }
        if usedKokoroManifest, body["asset"] == nil {
            body["asset"] = "kokoro"
        }
        try self.init(manifest: body, path: directory.path)
        assetKind = try Self.resolveAssetKind(asset: asset, manifest: body, in: directory)
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
        assetKind = (manifest["synthetic"] as? Bool == true) ? .synthetic : .coreAI
    }
}

extension CoreAISpeechBundle {

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
        functions.first { $0.length >= count } ?? functions.last ?? Function(name: "main", length: max(1, count))
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
        return padded.map { Int32(clamping: $0) }
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

    public func resolvedAssetURL(in directory: URL) -> URL {
        switch assetKind {
        case .coreAI:
            return asset.hasSuffix(".aimodel") ? directory.appending(component: asset) : directory.appending(component: "\(asset).aimodel")
        case .coreMLCompiled:
            if directory.pathExtension.lowercased() == "mlmodelc" { return directory }
            return asset.hasSuffix(".mlmodelc") ? directory.appending(component: asset) : directory.appending(component: "\(asset).mlmodelc")
        case .coreMLPackage:
            if directory.pathExtension.lowercased() == "mlpackage" { return directory }
            return asset.hasSuffix(".mlpackage") ? directory.appending(component: asset) : directory.appending(component: "\(asset).mlpackage")
        case .synthetic:
            return directory
        }
    }

    private static func directCoreMLAssetKind(_ directory: URL) -> AssetKind? {
        guard FileManager.default.fileExists(atPath: directory.path) else { return nil }
        switch directory.pathExtension.lowercased() {
        case "mlmodelc": return .coreMLCompiled
        case "mlpackage": return .coreMLPackage
        default: return nil
        }
    }

    private static func resolveManifestURL(in directory: URL) -> (URL, Bool) {
        let primary = directory.appending(component: manifest)
        if FileManager.default.fileExists(atPath: primary.path) {
            return (primary, false)
        }
        let kokoro = directory.appending(component: kokoroANEManifest)
        if FileManager.default.fileExists(atPath: kokoro.path) {
            return (kokoro, true)
        }
        return (primary, false)
    }

    private static func resolveAssetKind(asset: String, manifest: [String: Any], in directory: URL) throws(BasaltError) -> AssetKind {
        if manifest["synthetic"] as? Bool == true {
            return .synthetic
        }
        let fm = FileManager.default
        let candidates: [(String, AssetKind)] = [
            (asset.hasSuffix(".aimodel") ? asset : "\(asset).aimodel", .coreAI),
            (asset.hasSuffix(".mlmodelc") ? asset : "\(asset).mlmodelc", .coreMLCompiled),
            (asset.hasSuffix(".mlpackage") ? asset : "\(asset).mlpackage", .coreMLPackage),
        ]
        for (file, kind) in candidates where fm.fileExists(atPath: directory.appending(component: file).path) {
            return kind
        }
        throw .modelConfigurationInvalid(
            path: directory.path,
            reason: "the ANE speech bundle has no \(asset).aimodel, \(asset).mlmodelc or \(asset).mlpackage"
        )
    }
}
