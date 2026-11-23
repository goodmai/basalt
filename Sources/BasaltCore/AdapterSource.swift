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
        FileManager.default.fileExists(atPath: directory.appending(component: "adapter_config.json").path)
    }

    static func source(_ value: String, fileExists: (String) -> Bool) -> AdapterSource {
        let expanded = (value as NSString).expandingTildeInPath
        let pathLike = value.hasPrefix("/") || value.hasPrefix(".") || value.hasPrefix("~")
        let components = value.split(separator: "/", omittingEmptySubsequences: false)
        guard !pathLike, !fileExists(expanded), components.count == 2, components.allSatisfy({ !$0.isEmpty }) else {
            return .directory(URL(fileURLWithPath: expanded))
        }
        return .repository(value)
    }
}
