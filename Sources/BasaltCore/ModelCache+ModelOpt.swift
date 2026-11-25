import Foundation

extension ModelCache {
    public static func importedRepoId(for repoId: String) -> String {
        repoId + "-mlx"
    }

    public static func isModelOpt(_ directory: URL) -> Bool {
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("hf_quant_config.json").path) { return true }
        return ((config(at: directory)?["quantization_config"] as? [String: Any])?["quant_method"] as? String) == "modelopt"
    }

    public static func importedPack(for repoId: String, source: URL, root: URL = ModelCache.root) -> URL? {
        guard isModelOpt(source) else { return nil }
        let pack = cacheDir(for: importedRepoId(for: repoId), root: root)
        return hasUsableConfig(at: pack) ? pack : nil
    }
}
