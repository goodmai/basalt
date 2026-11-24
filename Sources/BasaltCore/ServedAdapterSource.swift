import Foundation

public enum ServedAdapterSource: Sendable, Equatable {
    case directory(URL)
    case repository(String)

    public struct Invalid: Error, Equatable, CustomStringConvertible {
        public let description: String
    }

    public static let repositoryPrefix = "hf:"

    public static func parse(_ specs: [String]) throws(Invalid) -> [String: ServedAdapterSource] {
        var sources: [String: ServedAdapterSource] = [:]
        for spec in specs {
            let parts = spec.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
                throw Invalid(description: "--adapter takes name=directory or name=hf:org/repo, got \(spec)")
            }
            guard sources[parts[0]] == nil else { throw Invalid(description: "--adapter \(parts[0]) is given twice") }
            sources[parts[0]] = try source(parts[1])
        }
        return sources
    }

    public static func isAdapterDirectory(_ directory: URL) -> Bool {
        let names = ["adapter_config.json", "adapter_model.safetensors", "adapters.safetensors", "edit.json", "edit.safetensors"]
        return hasAdapterFiles(names.filter { FileManager.default.fileExists(atPath: directory.appending(component: $0).path) })
    }

    public static func hasAdapterFiles(_ names: [String]) -> Bool {
        (names.contains("adapter_config.json") && (names.contains("adapter_model.safetensors") || names.contains("adapters.safetensors")))
            || (names.contains("edit.json") && names.contains("edit.safetensors"))
    }

    static func source(_ value: String) throws(Invalid) -> ServedAdapterSource {
        guard value.hasPrefix(repositoryPrefix) else { return .directory(URL(fileURLWithPath: (value as NSString).expandingTildeInPath)) }
        let repo = String(value.dropFirst(repositoryPrefix.count))
        guard repo.split(separator: "/", omittingEmptySubsequences: false).count == 2, HuggingFaceHub.isRepoId(repo) else {
            throw Invalid(description: "--adapter \(value): hf: takes a Hugging Face repo id of the form org/repo")
        }
        return .repository(repo)
    }
}
