import Foundation

public enum AdapterSource: Sendable, Equatable {
    case directory(URL)
    case repository(String)

    public struct Invalid: Error, Equatable, CustomStringConvertible {
        public let description: String
    }

    public static func parse(_ specs: [String], fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }) throws(Invalid) -> [String: AdapterSource] {
        var sources: [String: AdapterSource] = [:]
        for spec in specs {
            let parts = spec.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
                throw Invalid(description: "--adapter takes name=directory or name=org/repo, got \(spec)")
            }
            guard sources[parts[0]] == nil else { throw Invalid(description: "--adapter \(parts[0]) is given twice") }
            sources[parts[0]] = source(parts[1], fileExists: fileExists)
        }
        return sources
    }

    public static func isAdapterDirectory(_ directory: URL) -> Bool {
        let present = { (name: String) in FileManager.default.fileExists(atPath: directory.appending(component: name).path) }
        return present("adapter_config.json") && (present("adapter_model.safetensors") || present("adapters.safetensors"))
    }

    static func source(_ value: String, fileExists: (String) -> Bool) -> AdapterSource {
        let expanded = (value as NSString).expandingTildeInPath
        let pathLike = value.hasPrefix("/") || value.hasPrefix(".") || value.hasPrefix("~")
        let components = value.split(separator: "/", omittingEmptySubsequences: false)
        let segmentsValid = components.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
        let charactersValid = value.unicodeScalars.allSatisfy { !CharacterSet.whitespacesAndNewlines.contains($0) && !CharacterSet.controlCharacters.contains($0) }
        guard !pathLike, !fileExists(expanded), components.count == 2, segmentsValid, charactersValid else {
            return .directory(URL(fileURLWithPath: expanded))
        }
        return .repository(value)
    }
}
