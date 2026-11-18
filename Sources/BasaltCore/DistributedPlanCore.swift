import Foundation

public enum ClusterError: Error, Sendable, Equatable, LocalizedError {
    case emptySecret
    case wildcardBindRejected(host: String)
    case invalidEndpoint(String)
    case invalidTopology(String)
    case unauthenticatedPeer(peerId: String)
    case signatureInvalid(peerId: String)
    case macMismatch
    case replaySequenceRejected(sequence: UInt64)
    case truncatedFrame(expected: Int, actual: Int)
    case malformedFrame(String)
    case runMismatch(expected: String, actual: String)
    case staleEpoch(expected: UInt64, actual: UInt64)
    case duplicateSubmission(peerId: String, epoch: UInt64)
    case peerDisconnected(peerId: String, reason: String)
    case quorumLost(remainingPeers: Int)

    public var errorDescription: String? {
        switch self {
        case .emptySecret:
            "Cluster pre-shared secret must not be empty."
        case .wildcardBindRejected(let host):
            "Refusing wildcard cluster bind address '\(host)'; specify an explicit loopback, Thunderbolt bridge, or LAN subnet IP."
        case .invalidEndpoint(let detail):
            "Invalid cluster peer endpoint: \(detail)"
        case .invalidTopology(let detail):
            "Invalid cluster topology: \(detail)"
        case .unauthenticatedPeer(let peerId):
            "Cluster peer '\(peerId)' is not authenticated for this run."
        case .signatureInvalid(let peerId):
            "Ed25519 challenge-response signature failed for peer '\(peerId)'."
        case .macMismatch:
            "Cluster frame HMAC-SHA256 authentication tag mismatch."
        case .replaySequenceRejected(let sequence):
            "Cluster frame sequence \(sequence) rejected by anti-replay window."
        case .truncatedFrame(let expected, let actual):
            "Truncated cluster frame: expected \(expected) bytes, received \(actual)."
        case .malformedFrame(let detail):
            "Malformed cluster frame: \(detail)"
        case .runMismatch(let expected, let actual):
            "Cluster run mismatch: expected '\(expected)', received '\(actual)'."
        case .staleEpoch(let expected, let actual):
            "Stale or unexpected step epoch \(actual); coordinator is at epoch \(expected)."
        case .duplicateSubmission(let peerId, let epoch):
            "Duplicate submission from peer '\(peerId)' in epoch \(epoch)."
        case .peerDisconnected(let peerId, let reason):
            "Cluster peer '\(peerId)' disconnected: \(reason)"
        case .quorumLost(let remaining):
            "Cluster quorum lost (\(remaining) surviving peer(s))."
        }
    }
}

public struct PeerCapability: Sendable, Equatable, Codable {
    public var peerId: String
    public var chipName: String
    public var unifiedMemoryBytes: Int64
    public var measuredTFLOPS: Double
    public var thermalFactor: Double
    public var linkGbps: Double

    public init(
        peerId: String,
        chipName: String,
        unifiedMemoryBytes: Int64,
        measuredTFLOPS: Double,
        thermalFactor: Double = 1.0,
        linkGbps: Double = 80.0
    ) {
        self.peerId = peerId
        self.chipName = chipName
        self.unifiedMemoryBytes = unifiedMemoryBytes
        self.measuredTFLOPS = measuredTFLOPS
        self.thermalFactor = thermalFactor
        self.linkGbps = linkGbps
    }

    public static func thermalFactor(for state: ProcessInfo.ThermalState) -> Double {
        switch state {
        case .nominal: 1.0
        case .fair: 0.9
        case .serious: 0.65
        case .critical: 0.25
        @unknown default: 1.0
        }
    }

    public var effectiveThroughputScore: Double {
        let compute = max(0.05, measuredTFLOPS.isFinite ? measuredTFLOPS : 0.05)
        let thermal = min(1.0, max(0.05, thermalFactor.isFinite ? thermalFactor : 1.0))
        let bw = max(0.1, linkGbps.isFinite ? linkGbps : 10.0)
        let linkScale = bw / (bw + 10.0)
        return compute * thermal * linkScale
    }
}

public struct ClusterPeerEndpoint: Sendable, Equatable, Codable {
    public var peerId: String
    public var scheme: String
    public var host: String
    public var port: Int

    public init(peerId: String, scheme: String = "http", host: String, port: Int) throws(ClusterError) {
        let trimmedId = peerId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedId.isEmpty else { throw .invalidEndpoint("peerId must not be empty") }
        let normalizedScheme = scheme.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard normalizedScheme == "http" || normalizedScheme == "https" || normalizedScheme == "loopback" else {
            throw .invalidEndpoint("unsupported scheme '\(scheme)'")
        }
        if normalizedScheme == "loopback" {
            self.peerId = trimmedId
            self.scheme = "loopback"
            self.host = "127.0.0.1"
            self.port = 0
            return
        }
        let validHost = try ClusterFrameAuth.validateBindHost(host)
        guard (1 ... 65_535).contains(port) else {
            throw .invalidEndpoint("port \(port) must be in 1...65535")
        }
        self.peerId = trimmedId
        self.scheme = normalizedScheme
        self.host = validHost
        self.port = port
    }

    public var urlString: String {
        if scheme == "loopback" { return "loopback://\(peerId)" }
        let formattedHost = host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host
        return "\(scheme)://\(formattedHost):\(port)"
    }
}

extension ClusterPeerEndpoint {
    public static func parse(_ raw: String) throws(ClusterError) -> ClusterPeerEndpoint {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw .invalidEndpoint("empty peer specification") }
        if trimmed.lowercased().hasPrefix("loopback://") {
            let id = String(trimmed.dropFirst("loopback://".count))
            return try ClusterPeerEndpoint(peerId: id, scheme: "loopback", host: "127.0.0.1", port: 0)
        }
        let (explicitId, target): (String?, String) = {
            guard let at = trimmed.firstIndex(of: "@"), !trimmed[..<at].contains("/") else { return (nil, trimmed) }
            return (String(trimmed[..<at]), String(trimmed[trimmed.index(after: at)...]))
        }()
        let withScheme = target.contains("://") ? target : "http://\(target)"
        guard let comps = URLComponents(string: withScheme),
              let scheme = comps.scheme?.lowercased(),
              let rawHost = comps.host, !rawHost.isEmpty,
              let port = comps.port else {
            throw .invalidEndpoint("expected [peerId@][http://]host:port, got '\(raw)'")
        }
        let peerId = explicitId ?? "\(rawHost):\(port)"
        return try ClusterPeerEndpoint(peerId: peerId, scheme: scheme, host: rawHost, port: port)
    }

    public static func parseList(_ specs: [String]) throws(ClusterError) -> [ClusterPeerEndpoint] {
        var out: [ClusterPeerEndpoint] = []
        var seen: Set<String> = []
        for entry in specs {
            for part in entry.split(separator: ",") {
                let ep = try parse(String(part))
                guard seen.insert(ep.peerId).inserted else {
                    throw .invalidEndpoint("duplicate peer '\(ep.peerId)'")
                }
                out.append(ep)
            }
        }
        return out
    }
}

public struct ClusterTopology: Sendable, Equatable, Codable {
    public var runId: String
    public var coordinatorId: String
    public var peers: [PeerCapability]
    public var stepEpoch: UInt64

    public init(
        runId: String,
        coordinatorId: String,
        peers: [PeerCapability],
        stepEpoch: UInt64 = 0
    ) throws(ClusterError) {
        let cleanRun = runId.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanCoord = coordinatorId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanRun.isEmpty else { throw .invalidTopology("runId must not be empty") }
        guard !cleanCoord.isEmpty else { throw .invalidTopology("coordinatorId must not be empty") }
        guard !peers.isEmpty else { throw .invalidTopology("peers must not be empty") }
        var ids: Set<String> = []
        for peer in peers {
            let pid = peer.peerId.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !pid.isEmpty else { throw .invalidTopology("peerId must not be empty") }
            guard peer.unifiedMemoryBytes > 0 else {
                throw .invalidTopology("unifiedMemoryBytes must be positive for '\(pid)'")
            }
            guard ids.insert(pid).inserted else {
                throw .invalidTopology("duplicate peerId '\(pid)'")
            }
        }
        self.runId = cleanRun
        self.coordinatorId = cleanCoord
        self.peers = peers
        self.stepEpoch = stepEpoch
    }

    public func removingPeer(_ failedPeerId: String) throws(ClusterError) -> ClusterTopology {
        let surviving = peers.filter { $0.peerId != failedPeerId }
        guard surviving.count < peers.count else {
            throw .invalidTopology("peer '\(failedPeerId)' is not part of topology")
        }
        guard !surviving.isEmpty else {
            throw .quorumLost(remainingPeers: 0)
        }
        let nextCoordinator = coordinatorId == failedPeerId ? surviving[0].peerId : coordinatorId
        return try ClusterTopology(
            runId: runId, coordinatorId: nextCoordinator, peers: surviving, stepEpoch: stepEpoch + 1)
    }

    public func advancingEpoch() throws(ClusterError) -> ClusterTopology {
        try ClusterTopology(runId: runId, coordinatorId: coordinatorId, peers: peers, stepEpoch: stepEpoch + 1)
    }
}

public enum DynamicMicrobatchBalancer: Sendable {
    public struct ShardAssignment: Sendable, Equatable, Codable {
        public let peerId: String
        public let rank: Int
        public let count: Int
        public let startIndex: Int

        public init(peerId: String, rank: Int, count: Int, startIndex: Int) {
            self.peerId = peerId
            self.rank = rank
            self.count = count
            self.startIndex = startIndex
        }

        public var range: Range<Int> { startIndex ..< (startIndex + count) }
    }

    public static func shardCounts(globalBatchSize: Int, peers: [PeerCapability]) throws(ClusterError) -> [Int] {
        guard globalBatchSize > 0 else {
            throw .invalidTopology("globalBatchSize must be positive, got \(globalBatchSize)")
        }
        guard !peers.isEmpty else {
            throw .invalidTopology("cannot balance microbatches across zero peers")
        }
        let scores = peers.map(\.effectiveThroughputScore)
        if globalBatchSize < peers.count {
            return subPeerCounts(globalBatchSize: globalBatchSize, scores: scores)
        }
        let totalScore = max(1e-9, scores.reduce(0, +))
        let ideals = scores.map { Double(globalBatchSize) * $0 / totalScore }
        var counts = ideals.map { max(1, Int($0.rounded())) }
        reconcileCounts(&counts, ideals: ideals, scores: scores, targetSum: globalBatchSize)
        return counts
    }

    public static func assignShards(
        globalBatchSize: Int,
        peers: [PeerCapability]
    ) throws(ClusterError) -> [ShardAssignment] {
        let counts = try shardCounts(globalBatchSize: globalBatchSize, peers: peers)
        var cursor = 0
        var assignments: [ShardAssignment] = []
        assignments.reserveCapacity(peers.count)
        for (rank, peer) in peers.enumerated() {
            let count = counts[rank]
            assignments.append(ShardAssignment(peerId: peer.peerId, rank: rank, count: count, startIndex: cursor))
            cursor += count
        }
        return assignments
    }

    private static func subPeerCounts(globalBatchSize: Int, scores: [Double]) -> [Int] {
        let ranked = scores.indices.sorted { lhs, rhs in
            scores[lhs] != scores[rhs] ? scores[lhs] > scores[rhs] : lhs < rhs
        }
        var counts = [Int](repeating: 0, count: scores.count)
        for idx in ranked.prefix(globalBatchSize) { counts[idx] = 1 }
        return counts
    }

    private static func reconcileCounts(
        _ counts: inout [Int],
        ideals: [Double],
        scores: [Double],
        targetSum: Int
    ) {
        var diff = targetSum - counts.reduce(0, +)
        while diff > 0 {
            let best = counts.indices.max { lhs, rhs in
                let rL = ideals[lhs] - Double(counts[lhs])
                let rR = ideals[rhs] - Double(counts[rhs])
                if rL != rR { return rL < rR }
                return scores[lhs] != scores[rhs] ? scores[lhs] < scores[rhs] : lhs > rhs
            }
            guard let best else { break }
            counts[best] += 1
            diff -= 1
        }
        while diff < 0 {
            let candidates = counts.indices.filter { counts[$0] > 1 }
            let worst = candidates.min { lhs, rhs in
                let rL = ideals[lhs] - Double(counts[lhs])
                let rR = ideals[rhs] - Double(counts[rhs])
                if rL != rR { return rL < rR }
                return scores[lhs] != scores[rhs] ? scores[lhs] < scores[rhs] : lhs > rhs
            }
            guard let worst else { break }
            counts[worst] -= 1
            diff += 1
        }
    }
}

public struct EpochQuorumTracker<Payload: Sendable>: Sendable {
    public enum Status: Sendable, Equatable {
        case pending(received: Int, required: Int)
        case complete
    }

    public private(set) var runId: String
    public private(set) var stepEpoch: UInt64
    public private(set) var expectedPeers: Set<String>
    public private(set) var authenticatedPeers: Set<String>
    public private(set) var submissions: [String: Payload] = [:]

    public init(
        runId: String,
        stepEpoch: UInt64,
        expectedPeers: Set<String>,
        authenticatedPeers: Set<String> = []
    ) {
        self.runId = runId
        self.stepEpoch = stepEpoch
        self.expectedPeers = expectedPeers
        self.authenticatedPeers = authenticatedPeers
    }

    public mutating func authenticate(peerId: String) {
        authenticatedPeers.insert(peerId)
    }

    public mutating func revoke(peerId: String) {
        authenticatedPeers.remove(peerId)
        submissions.removeValue(forKey: peerId)
    }

    public mutating func submit(
        peerId: String,
        runId incomingRunId: String,
        epoch incomingEpoch: UInt64,
        payload: Payload
    ) throws(ClusterError) -> Status {
        guard incomingRunId == runId else {
            throw .runMismatch(expected: runId, actual: incomingRunId)
        }
        guard authenticatedPeers.contains(peerId), expectedPeers.contains(peerId) else {
            throw .unauthenticatedPeer(peerId: peerId)
        }
        guard incomingEpoch == stepEpoch else {
            throw .staleEpoch(expected: stepEpoch, actual: incomingEpoch)
        }
        guard submissions[peerId] == nil else {
            throw .duplicateSubmission(peerId: peerId, epoch: stepEpoch)
        }
        submissions[peerId] = payload
        return submissions.count == expectedPeers.count
            ? .complete
            : .pending(received: submissions.count, required: expectedPeers.count)
    }

    public mutating func advanceEpoch(to nextEpoch: UInt64, expectedPeers nextPeers: Set<String>) throws(ClusterError) {
        guard nextEpoch > stepEpoch else {
            throw .staleEpoch(expected: stepEpoch + 1, actual: nextEpoch)
        }
        guard !nextPeers.isEmpty else {
            throw .quorumLost(remainingPeers: 0)
        }
        stepEpoch = nextEpoch
        expectedPeers = nextPeers
        authenticatedPeers.formIntersection(nextPeers)
        submissions.removeAll(keepingCapacity: true)
    }

    public var isComplete: Bool {
        !expectedPeers.isEmpty && submissions.count == expectedPeers.count
    }

    public var missingPeers: Set<String> {
        expectedPeers.subtracting(submissions.keys)
    }
}
