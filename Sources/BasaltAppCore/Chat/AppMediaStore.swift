import CryptoKit
import Darwin
import Foundation

public struct AppMediaStore: Sendable, Equatable {
    public var directory: URL

    public static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/app/media", directoryHint: .isDirectory)
    }

    public init(directory: URL = Self.defaultDirectory) {
        self.directory = directory
    }

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    @discardableResult
    public func store(_ data: Data, format: String = "png") throws -> String {
        let ext = Self.sanitizedExtension(format)
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        _ = chmod(directory.path, 0o700)
        let hash = Self.sha256Hex(data)
        let destination = directory.appending(path: "\(hash).\(ext)")
        if !fm.fileExists(atPath: destination.path) {
            let staging = directory.appending(path: ".\(hash)-\(UUID().uuidString).tmp")
            try data.write(to: staging, options: .atomic)
            _ = chmod(staging.path, 0o600)
            do {
                try fm.moveItem(at: staging, to: destination)
            } catch {
                try? fm.removeItem(at: staging)
                if !fm.fileExists(atPath: destination.path) { throw error }
            }
        }
        return destination.path
    }

    public func read(assetID: String) -> Data? {
        let safe = (assetID as NSString).lastPathComponent
        guard !safe.isEmpty, safe == assetID, !safe.hasPrefix(".") else { return nil }
        return try? Data(contentsOf: directory.appending(path: safe))
    }

    private static func sanitizedExtension(_ format: String) -> String {
        let cleaned = format.lowercased().filter { $0.isASCII && ($0.isLetter || $0.isNumber) }
        return cleaned.isEmpty ? "png" : cleaned
    }
}
