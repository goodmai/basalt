import Foundation

extension ModelCache {
    public static func importedRepoId(for repoId: String) -> String {
        repoId + "-mlx"
    }

    public static func isModelOpt(_ directory: URL) -> Bool {
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("hf_quant_config.json").path) { return true }
        return ((config(at: directory)?["quantization_config"] as? [String: Any])?["quant_method"] as? String) == "modelopt"
    }

    public static func importSource(of pack: URL) -> String? {
        guard let record = config(at: pack)?["basalt_import"] as? [String: Any], record["format"] as? String == "modelopt" else { return nil }
        return record["source"] as? String
    }

    public static func importedPack(for repoId: String, source: URL, root: URL = ModelCache.root) -> URL? {
        guard isModelOpt(source) else { return nil }
        let pack = cacheDir(for: importedRepoId(for: repoId), root: root)
        guard hasUsableConfig(at: pack), importSource(of: pack) == source.lastPathComponent else { return nil }
        return pack
    }

    public static func holdsDownloadedSnapshot(_ repoId: String, root: URL = ModelCache.root) -> Bool {
        let refs = root.appending(component: "models--\(repoId.replacingOccurrences(of: "/", with: "--"))/refs/main")
        guard let ref = try? String(contentsOf: refs, encoding: .utf8) else { return false }
        return !ref.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("local")
    }
}
