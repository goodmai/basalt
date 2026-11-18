import ArgumentParser
import BasaltCore
import Darwin
import Foundation

public struct ClusterNodeConfig: Sendable, Equatable, Codable {
    public var peerId: String
    public var bindHost: String
    public var port: Int
    public var secret: String
    public var identity: ClusterFrameAuth.NodeIdentity
    public var peers: [ClusterPeerEndpoint]

    public init(
        peerId: String,
        bindHost: String,
        port: Int,
        secret: String,
        identity: ClusterFrameAuth.NodeIdentity,
        peers: [ClusterPeerEndpoint] = []
    ) {
        self.peerId = peerId
        self.bindHost = bindHost
        self.port = port
        self.secret = secret
        self.identity = identity
        self.peers = peers
    }
}

public struct ClusterCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "cluster",
        abstract: "Manage authenticated multi-Mac cluster pairing, topology planning and status (#565)",
        subcommands: [Pair.self, Plan.self, Status.self]
    )

    public init() {}

    public static func defaultConfigURL() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appending(component: ".basalt")
            .appending(component: "cluster.json")
    }

    public static func resolveConfigURL(_ customPath: String?) -> URL {
        guard let customPath, !customPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return defaultConfigURL()
        }
        return URL(fileURLWithPath: (customPath as NSString).expandingTildeInPath)
    }

    public static func saveConfig(_ config: ClusterNodeConfig, to url: URL) throws {
        _ = try ClusterFrameAuth.validateBindHost(config.bindHost)
        _ = try ClusterFrameAuth.validateSecret(config.secret)
        guard (1 ... 65_535).contains(config.port) else {
            throw ClusterError.invalidEndpoint("port \(config.port) must be in 1...65535")
        }
        let dir = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(config)
        try writeOwnerOnly(data, to: url)
    }

    public static func loadConfig(from url: URL) throws -> ClusterNodeConfig {
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode(ClusterNodeConfig.self, from: data)
        _ = try ClusterFrameAuth.validateBindHost(decoded.bindHost)
        _ = try ClusterFrameAuth.validateSecret(decoded.secret)
        return decoded
    }

    private static func writeOwnerOnly(_ data: Data, to url: URL) throws {
        let tmp = url.deletingLastPathComponent()
            .appending(component: ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        let fd = open(tmp.path, O_CREAT | O_EXCL | O_WRONLY, 0o600)
        guard fd >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        let written = data.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
        guard written == data.count, fsync(fd) == 0 else {
            let errCode = errno
            close(fd)
            unlink(tmp.path)
            throw POSIXError(POSIXErrorCode(rawValue: errCode) ?? .EIO)
        }
        guard close(fd) == 0, rename(tmp.path, url.path) == 0 else {
            let errCode = errno
            unlink(tmp.path)
            throw POSIXError(POSIXErrorCode(rawValue: errCode) ?? .EIO)
        }
    }
}

extension ClusterCommand {
    public struct Pair: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "pair",
            abstract: "Generate Ed25519 identity and pre-shared HMAC secret bound to an explicit interface"
        )

        @Option(name: .long, help: "Explicit bind IP (loopback, Thunderbolt bridge 169.254.x.x or LAN IP; never 0.0.0.0)")
        var bind: String = "127.0.0.1"

        @Option(name: .long, help: "Cluster listener TCP port (1...65535)")
        var port: Int = 9440

        @Option(name: .customLong("peer-id"), help: "Unique peer identifier (default: hostname)")
        var peerId: String?

        @Option(name: .long, help: "Pre-shared HMAC secret (generated automatically if omitted)")
        var secret: String?

        @Option(name: .long, help: "Comma-separated remote peer endpoints (host:port)")
        var peers: [String] = []

        @Option(name: .long, help: "Cluster config path (default: ~/.basalt/cluster.json)")
        var config: String?

        public init() {}

        public func validate() throws {
            do {
                _ = try ClusterFrameAuth.validateBindHost(bind)
                guard (1 ... 65_535).contains(port) else {
                    throw ValidationError("--port must be in 1...65535, got \(port)")
                }
                if let secret {
                    _ = try ClusterFrameAuth.validateSecret(secret)
                }
                _ = try ClusterPeerEndpoint.parseList(peers)
            } catch let err as ValidationError {
                throw err
            } catch {
                throw ValidationError(error.localizedDescription)
            }
        }

        public mutating func run() async throws {
            let resolvedHost = try ClusterFrameAuth.validateBindHost(bind)
            let rawId = peerId?.trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedId = (rawId?.isEmpty == false ? rawId! : ProcessInfo.processInfo.hostName)
            let resolvedSecret: String
            if let secret {
                _ = try ClusterFrameAuth.validateSecret(secret)
                resolvedSecret = secret.trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                let raw = try ClusterFrameAuth.randomBytes(count: 32)
                resolvedSecret = "bclt_" + raw.map { String(format: "%02x", $0) }.joined()
            }
            let identity = try ClusterFrameAuth.NodeIdentity.generate(peerId: resolvedId)
            let parsedPeers = try ClusterPeerEndpoint.parseList(peers)
            let cfg = ClusterNodeConfig(
                peerId: resolvedId,
                bindHost: resolvedHost,
                port: port,
                secret: resolvedSecret,
                identity: identity,
                peers: parsedPeers)
            let url = ClusterCommand.resolveConfigURL(config)
            try ClusterCommand.saveConfig(cfg, to: url)
            let pubHex = identity.publicKeyRaw.prefix(8).map { String(format: "%02x", $0) }.joined()
            printSuccess("Paired cluster node '\(resolvedId)' on \(resolvedHost):\(port)")
            print("  Config:      \(url.path)")
            print("  Ed25519 Key: \(pubHex)…")
            print("  Peers:       \(parsedPeers.count)")
        }
    }
}

extension ClusterCommand {
    public struct Plan: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "plan",
            abstract: "Preview DynamicMicrobatchBalancer shard allocation across heterogeneous Macs"
        )

        @Option(name: .customLong("global-batch"), help: "Global calibration or training batch count")
        var globalBatch: Int = 16

        @Option(name: .long, help: "Peer capability spec <peerId>:<tflops>[:<gbps>[:<thermalFactor>]]")
        var peer: [String] = []

        public init() {}

        public static func parseCapability(_ spec: String) throws -> PeerCapability {
            let parts = spec.split(separator: ":").map(String.init)
            guard parts.count >= 2, parts.count <= 4,
                  !parts[0].isEmpty,
                  let tflops = Double(parts[1]), tflops.isFinite, tflops > 0 else {
                throw ValidationError("invalid --peer '\(spec)'; expected <id>:<tflops>[:<gbps>[:<thermal>]]")
            }
            let gbps = parts.count >= 3 ? (Double(parts[2]) ?? -1) : 80.0
            let thermal = parts.count == 4 ? (Double(parts[3]) ?? -1) : 1.0
            guard gbps.isFinite, gbps > 0, thermal.isFinite, (0.05 ... 1.0).contains(thermal) else {
                throw ValidationError("invalid bandwidth or thermal factor in --peer '\(spec)'")
            }
            return PeerCapability(
                peerId: parts[0], chipName: "Apple Silicon",
                unifiedMemoryBytes: 64 * 1_073_741_824,
                measuredTFLOPS: tflops, thermalFactor: thermal, linkGbps: gbps)
        }

        public func validate() throws {
            guard globalBatch > 0 else {
                throw ValidationError("--global-batch must be positive, got \(globalBatch)")
            }
            guard !peer.isEmpty else {
                throw ValidationError("provide at least one --peer <id>:<tflops>[:<gbps>[:<thermal>]]")
            }
            for item in peer { _ = try Self.parseCapability(item) }
        }

        public mutating func run() async throws {
            let caps = try peer.map(Self.parseCapability)
            let topology = try ClusterTopology(
                runId: "plan-preview", coordinatorId: caps[0].peerId, peers: caps)
            let shards = try DynamicMicrobatchBalancer.assignShards(
                globalBatchSize: globalBatch, peers: topology.peers)
            print("Cluster Topology (\(topology.peers.count) peer(s), globalBatch=\(globalBatch)):")
            for (idx, assignment) in shards.enumerated() {
                let score = String(format: "%.2f", topology.peers[idx].effectiveThroughputScore)
                print("  [\(assignment.rank)] \(assignment.peerId): R_p=\(score) -> shards=\(assignment.count) (\(assignment.range))")
            }
        }
    }

    public struct Status: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "status",
            abstract: "Inspect local cluster node identity and configured peers"
        )

        @Option(name: .long, help: "Cluster config path (default: ~/.basalt/cluster.json)")
        var config: String?

        public init() {}

        public mutating func run() async throws {
            let url = ClusterCommand.resolveConfigURL(config)
            guard FileManager.default.fileExists(atPath: url.path) else {
                print("Cluster not paired yet. Run 'basalt cluster pair --bind <ip>' first.")
                return
            }
            let cfg = try ClusterCommand.loadConfig(from: url)
            let pubHex = cfg.identity.publicKeyRaw.prefix(8).map { String(format: "%02x", $0) }.joined()
            print("Cluster Node: \(cfg.peerId)")
            print("  Bind:        \(cfg.bindHost):\(cfg.port)")
            print("  Ed25519 Key: \(pubHex)…")
            print("  Peers (\(cfg.peers.count)):   \(cfg.peers.map(\.urlString).joined(separator: ", "))")
        }
    }
}
