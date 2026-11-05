import Foundation

public enum HubVerdict: String, Codable, Sendable {
    case verified = "true"
    case mismatch = "false"
    case unknown
}

public struct HubFileVerdict: Codable, Equatable, Sendable {
    public let name: String
    public let verified: HubVerdict
}

package struct HubTreeEntry: Decodable, Sendable {
    package struct LFS: Decodable, Sendable {
        package let oid: String
    }

    package let type: String
    package let path: String
    package let oid: String
    package let lfs: LFS?

    package var digest: HubDigest? {
        guard type == "file" else { return nil }
        return HubDigest(header: lfs?.oid ?? oid)
    }
}

package struct HubSnapshot: Sendable {
    package let directory: URL
    package let repoId: String?
    package let revision: String
    package let prefix: String

    package init(directory: URL) {
        self.directory = directory
        let parts = directory.standardizedFileURL.pathComponents
        guard let index = parts.lastIndex(of: "snapshots"), index > 0, index + 1 < parts.count, parts[index - 1].hasPrefix("models--") else {
            (repoId, revision, prefix) = (nil, "main", "")
            return
        }
        repoId = String(parts[index - 1].dropFirst("models--".count)).replacingOccurrences(of: "--", with: "/")
        revision = parts[index + 1]
        prefix = parts[(index + 2)...].map { "\($0)/" }.joined()
    }

    package func files() -> [String] {
        let all = FileManager.default.enumerator(atPath: directory.path)?.allObjects as? [String] ?? []
        return all.filter { path in
            let name = (path as NSString).lastPathComponent
            guard !(name.hasPrefix(".") && name.hasSuffix(".sha256")) else { return false }
            let resolved = directory.appendingPathComponent(path).resolvingSymlinksInPath()
            return (try? resolved.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) != true
        }.sorted()
    }

    package func verify(hub: HuggingFaceHub) async -> [HubFileVerdict] {
        let names = files()
        let recorded = names.map { HubSidecar.digest(for: directory.appendingPathComponent($0)) }
        let tree = recorded.contains(nil) ? await treeDigests(hub) : [:]
        return zip(names, recorded).map { name, sidecar in verdict(name, expected: sidecar ?? tree[prefix + name]) }
    }

    package func requireVerified(hub: HuggingFaceHub) async throws(HFError) {
        let refused = await verify(hub: hub).filter { $0.verified != .verified }.map(\.name)
        guard refused.isEmpty else { throw .unverifiedSnapshot(path: directory.path, files: refused) }
    }

    private func treeDigests(_ hub: HuggingFaceHub) async -> [String: HubDigest] {
        guard let repoId else { return [:] }
        do {
            return try await hub.treeDigests(repoId: repoId, revision: revision)
        } catch {
            HubSidecar.logger.warn("tree API for \(repoId)@\(revision) failed, files without a sidecar are unknown: \(error.localizedDescription)")
            return [:]
        }
    }

    private func verdict(_ name: String, expected: HubDigest?) -> HubFileVerdict {
        let start = ContinuousClock.now
        let verdict: HubVerdict = expected.map { digest in
            (try? digest.computed(fileAt: directory.appendingPathComponent(name))) == digest.hex ? .verified : .mismatch
        } ?? .unknown
        HubSidecar.logger.info("\(name): verified \(verdict.rawValue) against \(expected?.hex ?? "no digest"), hashed in \(start.duration(to: .now))")
        return HubFileVerdict(name: name, verified: verdict)
    }
}
