import BasaltCore
import Foundation
import MLX

package struct SerializedTensor: Sendable, Equatable, Codable {
    package let shape: [Int]
    package let floatBytes: Data

    package init(shape: [Int], floatBytes: Data) {
        self.shape = shape
        self.floatBytes = floatBytes
    }

    package static func fromFloats(_ values: [Float], shape: [Int]) -> SerializedTensor {
        var bytes = Data(capacity: values.count * 4)
        for val in values {
            var bits = val.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        return SerializedTensor(shape: shape, floatBytes: bytes)
    }

    package static func encode(_ array: MLXArray) -> SerializedTensor {
        let f32 = contiguous(array.asType(.float32))
        eval(f32)
        if f32.size == 0 {
            return SerializedTensor(shape: f32.shape, floatBytes: Data())
        }
        let floats = f32.asArray(Float.self)
        return fromFloats(floats, shape: f32.shape)
    }

    package func expectedElementCount() throws(ClusterError) -> Int {
        guard shape.allSatisfy({ $0 >= 0 }) else {
            throw .malformedFrame("negative tensor dimension in shape \(shape)")
        }
        var product = 1
        for dim in shape {
            let (next, overflow) = product.multipliedReportingOverflow(by: dim)
            guard !overflow else { throw .malformedFrame("tensor shape overflow \(shape)") }
            product = next
        }
        return product
    }

    package func decodeFloats() throws(ClusterError) -> [Float] {
        let count = try expectedElementCount()
        let expectedBytes = count * 4
        guard floatBytes.count == expectedBytes else {
            throw .malformedFrame("tensor byte count \(floatBytes.count) != expected \(expectedBytes)")
        }
        if count == 0 { return [] }
        let raw = floatBytes.bytes
        var out = [Float](repeating: 0, count: count)
        for idx in 0 ..< count {
            let bits = UInt32(littleEndian: raw.unsafeLoadUnaligned(fromByteOffset: idx * 4, as: UInt32.self))
            out[idx] = Float(bitPattern: bits)
        }
        return out
    }

    package func decodeMLXArray() throws(ClusterError) -> MLXArray {
        let floats = try decodeFloats()
        if floats.isEmpty {
            return MLXArray.zeros(shape, dtype: .float32)
        }
        return MLXArray(floats, shape)
    }
}

package struct ClusterPVBatchPayload: Sendable, Equatable, Codable {
    package var tokens: [Int32]
    package var topIDs: [Int32]
    package var topLogprobs: [Float]
    package var seqLen: Int
    package var topK: Int

    package init(from batch: PVTeacherBatch) {
        self.tokens = batch.tokens
        self.topIDs = batch.topIDs
        self.topLogprobs = batch.topLogprobs
        self.seqLen = batch.seqLen
        self.topK = batch.topK
    }

    package func toTeacherBatch() -> PVTeacherBatch {
        PVTeacherBatch(tokens: tokens, topIDs: topIDs, topLogprobs: topLogprobs, seqLen: seqLen, topK: topK)
    }
}

package enum ClusterWorkRequest: Sendable, Equatable, Codable {
    case hessian(blockIndices: [Int], tokenBatches: [[Int32]], maxSwitchRows: Int)
    case pvGradients(blockIndices: [Int], maxSwitchExperts: Int, batches: [ClusterPVBatchPayload])
}

package enum ClusterWorkResponse: Sendable, Equatable, Codable {
    case hessian(hessianSums: [String: SerializedTensor], tokenCounts: [String: Int])
    case pvGradients(meanGradients: [String: SerializedTensor], sampleCount: Int)
}

package protocol ClusterTransport: Sendable {
    func handshake(
        peerId: String,
        challenge: ClusterFrameAuth.HandshakeChallenge
    ) async throws(ClusterError) -> ClusterFrameAuth.HandshakeAccept

    func sendFrame(peerId: String, sealedFrame: Data) async throws(ClusterError) -> Data
    func disconnect(peerId: String) async
    func shutdown() async
}

package actor HTTPClusterTransport: ClusterTransport {
    package let bindHost: String
    private var peerBaseURLs: [String: URL]
    private let session: URLSession
    private var isClosed = false

    package init(
        bindHost: String = "127.0.0.1",
        peers: [ClusterPeerEndpoint],
        sessionConfiguration: URLSessionConfiguration = .ephemeral
    ) throws(ClusterError) {
        self.bindHost = try ClusterFrameAuth.validateBindHost(bindHost)
        var urls: [String: URL] = [:]
        for peer in peers {
            guard peer.scheme == "http" || peer.scheme == "https" else {
                throw .invalidEndpoint("HTTPClusterTransport requires http or https endpoint, got '\(peer.urlString)'")
            }
            _ = try ClusterFrameAuth.validateBindHost(peer.host)
            guard let url = URL(string: peer.urlString) else {
                throw .invalidEndpoint("invalid URL '\(peer.urlString)'")
            }
            urls[peer.peerId] = url
        }
        self.peerBaseURLs = urls
        self.session = URLSession(configuration: sessionConfiguration)
    }

    package func handshake(
        peerId: String,
        challenge: ClusterFrameAuth.HandshakeChallenge
    ) async throws(ClusterError) -> ClusterFrameAuth.HandshakeAccept {
        guard !isClosed, let base = peerBaseURLs[peerId] else {
            throw .peerDisconnected(peerId: peerId, reason: "transport closed or unknown peer")
        }
        guard let body = try? JSONEncoder().encode(challenge) else {
            throw .malformedFrame("failed to encode handshake challenge")
        }
        let data = try await post(
            url: base.appending(path: "v1/cluster/handshake"),
            peerId: peerId,
            contentType: "application/json",
            body: body)
        guard let accept = try? JSONDecoder().decode(ClusterFrameAuth.HandshakeAccept.self, from: data) else {
            throw .malformedFrame("invalid handshake response from '\(peerId)'")
        }
        return accept
    }

    package func sendFrame(peerId: String, sealedFrame: Data) async throws(ClusterError) -> Data {
        guard !isClosed, let base = peerBaseURLs[peerId] else {
            throw .peerDisconnected(peerId: peerId, reason: "transport closed or peer evicted")
        }
        return try await post(
            url: base.appending(path: "v1/cluster/frame"),
            peerId: peerId,
            contentType: "application/octet-stream",
            body: sealedFrame)
    }

    package func disconnect(peerId: String) async {
        let removed = peerBaseURLs.removeValue(forKey: peerId)
        guard let base = removed, !isClosed else { return }
        let sess = session
        let bind = bindHost
        await withTaskCancellationShield {
            var req = URLRequest(url: base.appending(path: "v1/cluster/disconnect"))
            req.httpMethod = "POST"
            req.timeoutInterval = 1.0
            req.setValue(bind, forHTTPHeaderField: "X-Basalt-Bind-Host")
            req.httpBody = Data(peerId.utf8)
            _ = try? await sess.data(for: req)
        }
    }

    package func shutdown() async {
        guard !isClosed else { return }
        isClosed = true
        peerBaseURLs.removeAll()
        let sess = session
        withTaskCancellationShield {
            sess.invalidateAndCancel()
        }
    }

    private func post(url: URL, peerId: String, contentType: String, body: Data) async throws(ClusterError) -> Data {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue(contentType, forHTTPHeaderField: "Content-Type")
        req.setValue("chunked", forHTTPHeaderField: "X-Basalt-Framing")
        req.setValue(bindHost, forHTTPHeaderField: "X-Basalt-Bind-Host")
        req.httpBody = body
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await session.data(for: req)
        } catch {
            throw .peerDisconnected(peerId: peerId, reason: error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else {
            throw .peerDisconnected(peerId: peerId, reason: "non-HTTP response")
        }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw .unauthenticatedPeer(peerId: peerId)
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw .peerDisconnected(peerId: peerId, reason: "HTTP status \(http.statusCode)")
        }
        return data
    }
}

struct CoordinatorSessionState: Sendable {
    var sessionKey: Data
    var nextOutboundSeq: UInt64 = 1
    var inboundReplay = ClusterFrameAuth.ReplayWindow()
}

struct CoordinatorValueCore: Sendable {
    var topology: ClusterTopology
    var sessions: [String: CoordinatorSessionState] = [:]
    var authenticatedPeers: Set<String> = []
    var isClosed = false

    mutating func registerAuthenticated(peerId: String, sessionKey: Data) throws(ClusterError) {
        guard !isClosed else { throw .quorumLost(remainingPeers: 0) }
        guard topology.peers.contains(where: { $0.peerId == peerId }) else {
            throw .invalidTopology("unexpected handshake peer '\(peerId)'")
        }
        sessions[peerId] = CoordinatorSessionState(sessionKey: sessionKey)
        authenticatedPeers.insert(peerId)
    }

    mutating func sealWorkFrames(
        workByPeer: [String: ClusterWorkRequest]
    ) throws(ClusterError) -> (epoch: UInt64, frames: [(peerId: String, frame: Data)]) {
        guard !isClosed else { throw .quorumLost(remainingPeers: 0) }
        let epoch = topology.stepEpoch
        var out: [(peerId: String, frame: Data)] = []
        out.reserveCapacity(workByPeer.count)
        for peerId in workByPeer.keys.sorted() {
            guard authenticatedPeers.contains(peerId), var sess = sessions[peerId] else {
                throw .unauthenticatedPeer(peerId: peerId)
            }
            guard let work = workByPeer[peerId], let payload = try? JSONEncoder().encode(work) else {
                throw .malformedFrame("failed to encode work request for '\(peerId)'")
            }
            let seq = sess.nextOutboundSeq
            sess.nextOutboundSeq += 1
            sessions[peerId] = sess
            let frame = try ClusterFrameAuth.sealFrame(
                kind: .tensor, epoch: epoch, sequence: seq, payload: payload, sessionKey: sess.sessionKey)
            out.append((peerId, frame))
        }
        return (epoch, out)
    }

    mutating func ingestResponseFrame(
        peerId: String,
        rawFrame: Data,
        expectedEpoch: UInt64,
        quorum: inout EpochQuorumTracker<ClusterWorkResponse>
    ) throws(ClusterError) {
        guard !isClosed else { throw .quorumLost(remainingPeers: 0) }
        guard topology.stepEpoch == expectedEpoch else {
            throw .staleEpoch(expected: topology.stepEpoch, actual: expectedEpoch)
        }
        guard var sess = sessions[peerId] else {
            throw .unauthenticatedPeer(peerId: peerId)
        }
        let decoded = try ClusterFrameAuth.openFrame(
            rawFrame.bytes, sessionKey: sess.sessionKey, replayWindow: &sess.inboundReplay)
        sessions[peerId] = sess
        guard decoded.kind == .tensor else {
            throw .malformedFrame("expected tensor response frame from '\(peerId)'")
        }
        guard let response = try? JSONDecoder().decode(ClusterWorkResponse.self, from: decoded.payload) else {
            throw .malformedFrame("invalid work response payload from '\(peerId)'")
        }
        _ = try quorum.submit(
            peerId: peerId, runId: topology.runId, epoch: decoded.epoch, payload: response)
    }

    mutating func evictPeer(_ failedPeerId: String) throws(ClusterError) -> ClusterTopology {
        sessions.removeValue(forKey: failedPeerId)
        authenticatedPeers.remove(failedPeerId)
        topology = try topology.removingPeer(failedPeerId)
        return topology
    }

    mutating func markClosed() {
        isClosed = true
        sessions.removeAll()
        authenticatedPeers.removeAll()
    }
}

package actor ClusterCoordinatorActor {
    private var core: CoordinatorValueCore
    private let identity: ClusterFrameAuth.NodeIdentity
    private let secret: String
    private let transport: any ClusterTransport

    package init(
        topology: ClusterTopology,
        identity: ClusterFrameAuth.NodeIdentity,
        secret: String,
        transport: any ClusterTransport
    ) throws(ClusterError) {
        _ = try ClusterFrameAuth.validateSecret(secret)
        self.core = CoordinatorValueCore(topology: topology)
        self.identity = identity
        self.secret = secret
        self.transport = transport
    }

    package func currentTopology() -> ClusterTopology {
        core.topology
    }

    package func authenticatedPeerIds() -> Set<String> {
        core.authenticatedPeers
    }

    package func connectAllPeers() async throws(ClusterError) {
        let snap = core.topology
        for peer in snap.peers {
            let challenge = try ClusterFrameAuth.makeChallenge(
                runId: snap.runId, challenger: identity, secret: secret)
            let accept = try await transport.handshake(peerId: peer.peerId, challenge: challenge)
            guard accept.responderId == peer.peerId else {
                throw .unauthenticatedPeer(peerId: peer.peerId)
            }
            let sessionKey = try ClusterFrameAuth.finalizeHandshake(
                challenge: challenge, accept: accept, secret: secret)
            try core.registerAuthenticated(peerId: peer.peerId, sessionKey: sessionKey)
        }
    }

    private var epochInFlight = false

    package func executeEpoch(
        workByPeer: [String: ClusterWorkRequest]
    ) async throws(ClusterError) -> (epoch: UInt64, responses: [String: ClusterWorkResponse]) {
        guard !epochInFlight else {
            throw .staleEpoch(expected: core.topology.stepEpoch, actual: core.topology.stepEpoch)
        }
        epochInFlight = true
        defer { epochInFlight = false }
        let prepared = try core.sealWorkFrames(workByPeer: workByPeer)
        let dispatchedEpoch = prepared.epoch
        let runId = core.topology.runId
        let expected = Set(workByPeer.keys)
        var quorum = EpochQuorumTracker<ClusterWorkResponse>(
            runId: runId,
            stepEpoch: dispatchedEpoch,
            expectedPeers: expected,
            authenticatedPeers: core.authenticatedPeers)

        let rawResults = try await collectRawResponses(frames: prepared.frames)
        guard !core.isClosed, core.topology.stepEpoch == dispatchedEpoch else {
            throw .staleEpoch(expected: core.topology.stepEpoch, actual: dispatchedEpoch)
        }
        for (peerId, rawFrame) in rawResults {
            try core.ingestResponseFrame(
                peerId: peerId, rawFrame: rawFrame, expectedEpoch: dispatchedEpoch, quorum: &quorum)
        }
        guard quorum.isComplete else {
            throw .quorumLost(remainingPeers: quorum.submissions.count)
        }
        core.topology = try core.topology.advancingEpoch()
        return (dispatchedEpoch, quorum.submissions)
    }

    package func evictDisconnectedPeer(_ failedPeerId: String) async throws(ClusterError) -> ClusterTopology {
        let updated = try core.evictPeer(failedPeerId)
        let transportRef = transport
        await withTaskCancellationShield {
            await transportRef.disconnect(peerId: failedPeerId)
        }
        return updated
    }

    package func shutdown() async {
        core.markClosed()
        let transportRef = transport
        await withTaskCancellationShield {
            await transportRef.shutdown()
        }
    }

    private func collectRawResponses(
        frames: [(peerId: String, frame: Data)]
    ) async throws(ClusterError) -> [(peerId: String, rawFrame: Data)] {
        let transportRef = transport
        do {
            return try await withThrowingTaskGroup(of: (String, Data).self) { group in
                for item in frames {
                    group.addTask {
                        let reply = try await transportRef.sendFrame(peerId: item.peerId, sealedFrame: item.frame)
                        return (item.peerId, reply)
                    }
                }
                var collected: [(String, Data)] = []
                collected.reserveCapacity(frames.count)
                for try await pair in group {
                    collected.append(pair)
                }
                return collected.sorted { $0.0 < $1.0 }
            }
        } catch let err as ClusterError {
            throw err
        } catch {
            throw .malformedFrame(error.localizedDescription)
        }
    }
}
