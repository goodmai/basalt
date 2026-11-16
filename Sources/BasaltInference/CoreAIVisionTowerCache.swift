import BasaltCore
import CryptoKit
import Foundation

public enum CoreAIVisionTowerCache {
    public struct Entry: Codable, Sendable, Equatable {
        public let cacheKey: String
        public let bundleSHA256: String
        public let osBuild: String
        public let asset: String
        public let functions: [String]
        public let bookmarkBase64: String?

        public init(
            cacheKey: String,
            bundleSHA256: String,
            osBuild: String,
            asset: String,
            functions: [String],
            bookmarkBase64: String? = nil
        ) {
            self.cacheKey = cacheKey
            self.bundleSHA256 = bundleSHA256
            self.osBuild = osBuild
            self.asset = asset
            self.functions = functions
            self.bookmarkBase64 = bookmarkBase64
        }
    }

    public static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/cache/coreai-vision")
    }

    public static func currentOSBuild() -> String {
        var size = 0
        sysctlbyname("kern.osversion", nil, &size, nil, 0)
        var raw = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("kern.osversion", &raw, &size, nil, 0)
        let build = String(decoding: raw.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        return build.isEmpty ? ProcessInfo.processInfo.operatingSystemVersionString : build
    }

    public static func bundleDigest(bundleDirectory: URL, asset: String? = nil) throws(BasaltError) -> String {
        let assetName: String
        if let asset {
            assetName = asset
        } else {
            assetName = try CoreAIVisionTowerBundle(directory: bundleDirectory).asset
        }
        let metadataURL = bundleDirectory.appending(component: "metadata.json")
        let assetURL = bundleDirectory.appending(component: "\(assetName).aimodel")
        guard FileManager.default.fileExists(atPath: metadataURL.path),
              FileManager.default.fileExists(atPath: assetURL.path) else {
            throw .modelConfigurationInvalid(
                path: bundleDirectory.path,
                reason: "vision tower bundle requires metadata.json and \(assetName).aimodel"
            )
        }
        let files = try bundleFiles(metadataURL: metadataURL, assetName: assetName, assetURL: assetURL, path: bundleDirectory.path)
        var hasher = SHA256()
        hasher.update(data: Data("coreai-vision-bundle-v1\0".utf8))
        for (relative, url) in files {
            hasher.update(data: Data("\(relative)\0".utf8))
            try hashFile(at: url, into: &hasher, bundlePath: bundleDirectory.path)
        }
        return hex(hasher.finalize())
    }

    public static func cacheKey(bundleSHA256: String, osBuild: String = currentOSBuild()) -> String {
        hex(SHA256.hash(data: Data("coreai-vision-key-v1\0\(bundleSHA256)\0\(osBuild)".utf8)))
    }

    public static func cacheKey(bundleDirectory: URL, osBuild: String = currentOSBuild()) throws(BasaltError) -> String {
        let digest = try bundleDigest(bundleDirectory: bundleDirectory)
        return cacheKey(bundleSHA256: digest, osBuild: osBuild)
    }

    public static func entryURL(forKey key: String, in cacheDirectory: URL = defaultDirectory) -> URL {
        cacheDirectory.appending(component: "\(key).json")
    }

    public static func loadEntry(forKey key: String, in cacheDirectory: URL = defaultDirectory) -> Entry? {
        let url = entryURL(forKey: key, in: cacheDirectory)
        guard let data = try? Data(contentsOf: url),
              let entry = try? JSONDecoder().decode(Entry.self, from: data),
              entry.cacheKey == key else {
            return nil
        }
        return entry
    }

    public static func saveEntry(_ entry: Entry, in cacheDirectory: URL = defaultDirectory) throws {
        try FileManager.default.createDirectory(
            at: cacheDirectory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(entry).write(to: entryURL(forKey: entry.cacheKey, in: cacheDirectory), options: .atomic)
    }

    private static func bundleFiles(
        metadataURL: URL, assetName: String, assetURL: URL, path: String
    ) throws(BasaltError) -> [(String, URL)] {
        var result: [(String, URL)] = [("metadata.json", metadataURL.resolvingSymlinksInPath())]
        let resolvedAsset = assetURL.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolvedAsset.path, isDirectory: &isDirectory) else {
            throw .modelConfigurationInvalid(path: path, reason: "unreadable \(assetName).aimodel")
        }
        guard isDirectory.boolValue else {
            result.append(("\(assetName).aimodel", resolvedAsset))
            return result
        }
        let prefix = resolvedAsset.standardizedFileURL.path + "/"
        let enumerator = FileManager.default.enumerator(at: resolvedAsset, includingPropertiesForKeys: [.isRegularFileKey])
        var nested: [(String, URL)] = []
        while let item = enumerator?.nextObject() as? URL {
            let values = try? item.resourceValues(forKeys: [.isRegularFileKey])
            guard values?.isRegularFile == true else { continue }
            let standardized = item.standardizedFileURL.path
            let suffix = standardized.hasPrefix(prefix) ? String(standardized.dropFirst(prefix.count)) : item.lastPathComponent
            nested.append(("\(assetName).aimodel/\(suffix)", item.resolvingSymlinksInPath()))
        }
        nested.sort { $0.0 < $1.0 }
        return result + nested
    }

    private static func hashFile(at url: URL, into hasher: inout SHA256, bundlePath: String) throws(BasaltError) {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw .modelConfigurationInvalid(path: bundlePath, reason: "cannot read \(url.lastPathComponent)")
        }
        defer { try? handle.close() }
        var total: UInt64 = 0
        while let chunk = try? handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            total += UInt64(chunk.count)
            hasher.update(data: chunk)
        }
        hasher.update(data: Data("\(total)\0".utf8))
    }

    private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
