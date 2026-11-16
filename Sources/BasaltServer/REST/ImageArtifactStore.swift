import CryptoKit
import Foundation
import BasaltCore

struct ImageArtifactStore: Sendable {
    struct PersistedAsset: Sendable, Equatable {
        let assetID: String
        let path: String
        let url: String
    }

    static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/images", directoryHint: .isDirectory)
    }

    static func persist(_ artifact: ImageCompletedArtifact, in directory: URL?) throws(BasaltError) -> PersistedAsset {
        let root = (directory ?? defaultDirectory).standardizedFileURL
        do {
            try FileManager.default.createDirectory(
                at: root,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
        } catch {
            throw .invalidRequestStructure(details: "image: failed to prepare store directory: \(error.localizedDescription)")
        }
        let digest = SHA256.hash(data: artifact.imageData).map { String(format: "%02x", $0) }.joined()
        let assetID = "\(digest).\(artifact.format.fileExtension)"
        let fileURL = root.appending(component: assetID)
        do {
            try artifact.imageData.write(to: fileURL, options: [.atomic])
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: fileURL.path)
        } catch {
            throw .modelInferenceError(details: "image: failed to persist generated artifact: \(error.localizedDescription)")
        }
        return PersistedAsset(assetID: assetID, path: fileURL.path, url: "/v1/images/content/\(assetID)")
    }

    static func parseAssetID(_ id: String) -> ImageOutputFormat? {
        let parts = id.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 2, parts[0].utf8.count == 64 else { return nil }
        let isHex = parts[0].unicodeScalars.allSatisfy {
            ("0" ... "9").contains($0) || ("a" ... "f").contains($0)
        }
        guard isHex else { return nil }
        return ImageOutputFormat(rawValue: String(parts[1]))
    }

    static func readValidatedAsset(id: String, in directory: URL?) -> (data: Data, format: ImageOutputFormat)? {
        guard let format = parseAssetID(id) else { return nil }
        let root = (directory ?? defaultDirectory).standardizedFileURL
        let fileURL = root.appending(component: id).standardizedFileURL
        guard fileURL.deletingLastPathComponent() == root else { return nil }
        guard let data = try? Data(contentsOf: fileURL, options: .mappedIfSafe) else { return nil }
        return (data, format)
    }

    static func decodeBase64Payload(_ raw: String, field: String) throws(BasaltError) -> Data {
        let stripped: Substring
        if raw.hasPrefix("data:"), let comma = raw.firstIndex(of: ",") {
            stripped = raw[raw.index(after: comma)...]
        } else {
            stripped = Substring(raw)
        }
        let compact = String(stripped.filter { !$0.isWhitespace })
        guard !compact.isEmpty, let decoded = Data(base64Encoded: compact) else {
            throw .invalidRequestStructure(details: "image: \(field) must be valid base64")
        }
        return decoded
    }
}
