import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Synchronization
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("Cluster Topology, DynamicMicrobatchBalancer, EpochQuorumTracker & ClusterFrameAuth (#565)")
struct ClusterTopologyAndAuthTests {
    @Test("DynamicMicrobatchBalancer assigns proportional shard counts summing to globalBatchSize")
    func dynamicMicrobatchBalancerProportionalExactSum() throws {
        let peers2 = [
            PeerCapability(peerId: "m4-max", chipName: "M4 Max", unifiedMemoryBytes: 128 << 30, measuredTFLOPS: 36, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "m2-pro", chipName: "M2 Pro", unifiedMemoryBytes: 32 << 30, measuredTFLOPS: 12, thermalFactor: 1.0, linkGbps: 80),
        ]
        let shards2 = try DynamicMicrobatchBalancer.assignShards(globalBatchSize: 8, peers: peers2)
        #expect(shards2.map(\.count) == [6, 2])
        #expect(shards2[0].count != shards2[1].count)
        #expect(shards2.reduce(0) { $0 + $1.count } == 8)
        #expect(shards2[0].range == 0 ..< 6 && shards2[1].range == 6 ..< 8)

        let peers3 = [
            PeerCapability(peerId: "r0", chipName: "M4 Ultra", unifiedMemoryBytes: 192 << 30, measuredTFLOPS: 45, thermalFactor: 1.0, linkGbps: 100),
            PeerCapability(peerId: "r1", chipName: "M4 Pro", unifiedMemoryBytes: 48 << 30, measuredTFLOPS: 20, thermalFactor: 0.9, linkGbps: 80),
            PeerCapability(peerId: "r2", chipName: "M3", unifiedMemoryBytes: 24 << 30, measuredTFLOPS: 10, thermalFactor: 0.65, linkGbps: 10),
        ]
        for globalBatch in [1, 2, 3, 7, 8, 13, 64] {
            let assigned = try DynamicMicrobatchBalancer.assignShards(globalBatchSize: globalBatch, peers: peers3)
            #expect(assigned.reduce(0) { $0 + $1.count } == globalBatch)
            if globalBatch >= peers3.count {
                #expect(assigned.allSatisfy { $0.count >= 1 })
                #expect(assigned[0].count >= assigned[1].count && assigned[1].count >= assigned[2].count)
            }
        }
    }

    @Test("EpochQuorumTracker rejects stale epochs, duplicate ranks, run mismatches and unauthenticated peers")
    func epochQuorumTrackerFencesInvalidFrames() throws {
        var tracker = EpochQuorumTracker<Int>(
            runId: "run-565",
            stepEpoch: 3,
            expectedPeers: ["rank-0", "rank-1"],
            authenticatedPeers: ["rank-0"])

        #expect(throws: ClusterError.unauthenticatedPeer(peerId: "rank-1")) {
            _ = try tracker.submit(peerId: "rank-1", runId: "run-565", epoch: 3, payload: 10)
        }
        tracker.authenticate(peerId: "rank-1")

        #expect(throws: ClusterError.runMismatch(expected: "run-565", actual: "wrong-run")) {
            _ = try tracker.submit(peerId: "rank-0", runId: "wrong-run", epoch: 3, payload: 10)
        }
        #expect(throws: ClusterError.staleEpoch(expected: 3, actual: 2)) {
            _ = try tracker.submit(peerId: "rank-0", runId: "run-565", epoch: 2, payload: 10)
        }
        #expect(throws: ClusterError.staleEpoch(expected: 3, actual: 4)) {
            _ = try tracker.submit(peerId: "rank-0", runId: "run-565", epoch: 4, payload: 10)
        }

        let s1 = try tracker.submit(peerId: "rank-0", runId: "run-565", epoch: 3, payload: 42)
        #expect(s1 == .pending(received: 1, required: 2))
        #expect(!tracker.isComplete && tracker.missingPeers == ["rank-1"])

        #expect(throws: ClusterError.duplicateSubmission(peerId: "rank-0", epoch: 3)) {
            _ = try tracker.submit(peerId: "rank-0", runId: "run-565", epoch: 3, payload: 99)
        }

        let s2 = try tracker.submit(peerId: "rank-1", runId: "run-565", epoch: 3, payload: 58)
        #expect(s2 == .complete && tracker.isComplete)

        try tracker.advanceEpoch(to: 4, expectedPeers: ["rank-0"])
        #expect(throws: ClusterError.staleEpoch(expected: 4, actual: 3)) {
            _ = try tracker.submit(peerId: "rank-0", runId: "run-565", epoch: 3, payload: 1)
        }
        #expect(throws: ClusterError.unauthenticatedPeer(peerId: "rank-1")) {
            _ = try tracker.submit(peerId: "rank-1", runId: "run-565", epoch: 4, payload: 1)
        }
    }

    @Test("ClusterFrameAuth rejects wildcard 0.0.0.0 binds, empty secrets, tampered MACs and replayed frames")
    func clusterFrameAuthHandshakeAndFramingSecurity() throws {
        for badHost in ["0.0.0.0", "::", "[::]", "*", "0.0.0.0/0", "255.255.255.255", "0.1.2.3"] {
            #expect(throws: ClusterError.wildcardBindRejected(host: badHost)) {
                _ = try ClusterFrameAuth.validateBindHost(badHost)
            }
        }
        #expect(try ClusterFrameAuth.validateBindHost("127.0.0.1") == "127.0.0.1")
        #expect(try ClusterFrameAuth.validateBindHost("169.254.12.4") == "169.254.12.4")
        #expect(try ClusterFrameAuth.validateBindHost("192.168.10.22") == "192.168.10.22")
        #expect(throws: ClusterError.emptySecret) {
            _ = try ClusterFrameAuth.validateSecret("   ")
        }

        let coordId = try ClusterFrameAuth.NodeIdentity.generate(peerId: "coord")
        let workerId = try ClusterFrameAuth.NodeIdentity.generate(peerId: "worker-1")
        let challenge = try ClusterFrameAuth.makeChallenge(runId: "run-auth", challenger: coordId, secret: "tb5-psk-secret")
        #expect(throws: ClusterError.macMismatch) {
            _ = try ClusterFrameAuth.acceptChallenge(challenge, responder: workerId, secret: "wrong-secret")
        }
        let accepted = try ClusterFrameAuth.acceptChallenge(challenge, responder: workerId, secret: "tb5-psk-secret")
        let coordKey = try ClusterFrameAuth.finalizeHandshake(challenge: challenge, accept: accepted.accept, secret: "tb5-psk-secret")
        #expect(coordKey == accepted.sessionKey && coordKey.count == 32)

        let badSigAccept = ClusterFrameAuth.HandshakeAccept(
            runId: accepted.accept.runId, responderId: accepted.accept.responderId,
            nonces: accepted.accept.nonces, publicKey: coordId.publicKeyRaw,
            signature: accepted.accept.signature, mac: accepted.accept.mac)
        #expect(throws: ClusterError.self) {
            _ = try ClusterFrameAuth.finalizeHandshake(challenge: challenge, accept: badSigAccept, secret: "tb5-psk-secret")
        }

        try verifyFrameMACAndReplayWindow(sessionKey: coordKey)
    }

    private func verifyFrameMACAndReplayWindow(sessionKey: Data) throws {
        let payload = Data([1, 2, 3, 4, 5, 6, 7, 8])
        let frame1 = try ClusterFrameAuth.sealFrame(kind: .tensor, epoch: 7, sequence: 10, payload: payload, sessionKey: sessionKey)
        var tampered = frame1
        tampered[frame1.count - 1] ^= 0x01
        var window = ClusterFrameAuth.ReplayWindow()
        #expect(throws: ClusterError.macMismatch) {
            _ = try ClusterFrameAuth.openFrame(tampered.bytes, sessionKey: sessionKey, replayWindow: &window)
        }
        let opened1 = try ClusterFrameAuth.openFrame(frame1.bytes, sessionKey: sessionKey, replayWindow: &window)
        #expect(opened1.epoch == 7 && opened1.sequence == 10 && opened1.payload == payload)
        #expect(throws: ClusterError.replaySequenceRejected(sequence: 10)) {
            _ = try ClusterFrameAuth.openFrame(frame1.bytes, sessionKey: sessionKey, replayWindow: &window)
        }
        let frame8 = try ClusterFrameAuth.sealFrame(kind: .tensor, epoch: 7, sequence: 8, payload: payload, sessionKey: sessionKey)
        let opened8 = try ClusterFrameAuth.openFrame(frame8.bytes, sessionKey: sessionKey, replayWindow: &window)
        #expect(opened8.sequence == 8)
        #expect(throws: ClusterError.truncatedFrame(expected: ClusterFrameAuth.envelopeOverheadBytes, actual: 12)) {
            _ = try ClusterFrameAuth.openFrame(Data(repeating: 0, count: 12).bytes, sessionKey: sessionKey, replayWindow: &window)
        }
    }
}

@Suite("Distributed 2-Rank & 3-Rank Heterogeneous Hessian, PV-Tuning & Elastic Recovery (#565)", .serialized)
struct DistributedClusterCalibrationTests {
    @Test("Heterogeneous 2-rank and 3-rank clusters match single-node FP32 Hessians within 1e-5")
    func twoAndThreeRankHeterogeneousHessianParity() async throws {
        let batches = Self.makeCalibrationBatches(count: 8, seqLen: 16)
        let weightsURL = FileManager.default.temporaryDirectory.appending(path: "cluster-hessian-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: weightsURL) }
        let singleModel = try TinyModel.make(.qwen3)
        eval(singleModel as Module)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: (singleModel as Module).parameters().flattened()), url: weightsURL)
        let single = try DistributedHessianAggregator.computeHessianShard(
            model: singleModel, blockIndices: [0, 1], tokenBatches: batches)
        #expect(!single.hessianSums.isEmpty)

        let peers2 = [
            PeerCapability(peerId: "rank-0", chipName: "M4 Max", unifiedMemoryBytes: 128 << 30, measuredTFLOPS: 36, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "rank-1", chipName: "M2 Pro", unifiedMemoryBytes: 32 << 30, measuredTFLOPS: 12, thermalFactor: 1.0, linkGbps: 80),
        ]
        let harness2 = try await Self.makeLoopbackCluster(runId: "hessian-2r", peers: peers2, quantized: false, weightsURL: weightsURL)
        defer { Task { await harness2.coordinator.shutdown() } }
        let res2 = try await DistributedHessianAggregator.aggregateHessians(
            coordinator: harness2.coordinator, blockIndices: [0, 1], calibrationTokenBatches: batches)
        #expect(res2.shardCountsByPeer == ["rank-0": 6, "rank-1": 2])
        try Self.assertHessianParity(cluster: res2, single: single, tolerance: 1e-5)

        let peers3 = [
            PeerCapability(peerId: "rank-0", chipName: "M4 Ultra", unifiedMemoryBytes: 192 << 30, measuredTFLOPS: 40, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "rank-1", chipName: "M4 Pro", unifiedMemoryBytes: 48 << 30, measuredTFLOPS: 20, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "rank-2", chipName: "M3", unifiedMemoryBytes: 24 << 30, measuredTFLOPS: 10, thermalFactor: 1.0, linkGbps: 80),
        ]
        let harness3 = try await Self.makeLoopbackCluster(runId: "hessian-3r", peers: peers3, quantized: false, weightsURL: weightsURL)
        defer { Task { await harness3.coordinator.shutdown() } }
        let res3 = try await DistributedHessianAggregator.aggregateHessians(
            coordinator: harness3.coordinator, blockIndices: [0, 1], calibrationTokenBatches: batches)
        #expect(res3.shardCountsByPeer.values.reduce(0, +) == 8)
        #expect(res3.shardCountsByPeer["rank-0"] != res3.shardCountsByPeer["rank-2"])
        try Self.assertHessianParity(cluster: res3, single: single, tolerance: 1e-5)
    }

    @Test("Heterogeneous 2-rank and 3-rank clusters match single-node sample-weighted PV-Tuning gradients within 1e-5")
    func twoAndThreeRankHeterogeneousPVGradientParity() async throws {
        let weightsURL = FileManager.default.temporaryDirectory.appending(path: "cluster-pv-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: weightsURL) }
        let teacherContainer = try TinyModel.container(.qwen3)
        let teacherBatches = await PVTuner.collectTeacherBatches(
            teacherContainer, texts: PVTuner.defaultCalibrationCorpus, count: 8, seqLen: 16, topK: 16)
        #expect(teacherBatches.count == 8)

        let singleStudent = try TinyModel.make(.qwen3)
        quantize(model: singleStudent as Module, groupSize: 32, bits: 4, mode: .affine)
        eval(singleStudent as Module)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: (singleStudent as Module).parameters().flattened()), url: weightsURL)
        let singlePV = try DistributedHessianAggregator.computePVGradientShard(
            model: singleStudent, blockIndices: [0, 1], maxSwitchExperts: 8, batches: teacherBatches)
        #expect(!singlePV.meanGradients.isEmpty)

        let peers2 = [
            PeerCapability(peerId: "rank-0", chipName: "M4 Max", unifiedMemoryBytes: 128 << 30, measuredTFLOPS: 30, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "rank-1", chipName: "M2", unifiedMemoryBytes: 24 << 30, measuredTFLOPS: 10, thermalFactor: 1.0, linkGbps: 80),
        ]
        let harness2 = try await Self.makeLoopbackCluster(runId: "pv-2r", peers: peers2, quantized: true, weightsURL: weightsURL)
        defer { Task { await harness2.coordinator.shutdown() } }
        let pv2 = try await DistributedHessianAggregator.aggregatePVGradients(
            coordinator: harness2.coordinator, blockIndices: [0, 1], teacherBatches: teacherBatches, maxSwitchExperts: 8)
        #expect(pv2.totalSamples == 8 && pv2.shardCountsByPeer == ["rank-0": 6, "rank-1": 2])
        try Self.assertTensorMapParity(try pv2.mlxGradients(), reference: singlePV.meanGradients, tolerance: 1e-5)

        let peers3 = [
            PeerCapability(peerId: "rank-0", chipName: "M4 Ultra", unifiedMemoryBytes: 192 << 30, measuredTFLOPS: 40, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "rank-1", chipName: "M4 Pro", unifiedMemoryBytes: 48 << 30, measuredTFLOPS: 20, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "rank-2", chipName: "M3", unifiedMemoryBytes: 24 << 30, measuredTFLOPS: 10, thermalFactor: 1.0, linkGbps: 80),
        ]
        let harness3 = try await Self.makeLoopbackCluster(runId: "pv-3r", peers: peers3, quantized: true, weightsURL: weightsURL)
        defer { Task { await harness3.coordinator.shutdown() } }
        let pv3 = try await DistributedHessianAggregator.aggregatePVGradients(
            coordinator: harness3.coordinator, blockIndices: [0, 1], teacherBatches: teacherBatches, maxSwitchExperts: 8)
        #expect(pv3.totalSamples == 8)
        try Self.assertTensorMapParity(try pv3.mlxGradients(), reference: singlePV.meanGradients, tolerance: 1e-5)
    }

    @Test("Mid-step worker disconnect recovers cleanly via cancellation-shielded fallback and tampered reply fails MAC")
    func midStepDisconnectFallbackAndTamperRejection() async throws {
        let weightsURL = FileManager.default.temporaryDirectory.appending(path: "cluster-rec-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: weightsURL) }
        let batches = Self.makeCalibrationBatches(count: 8, seqLen: 16)
        let singleModel = try TinyModel.make(.qwen3)
        eval(singleModel as Module)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: (singleModel as Module).parameters().flattened()), url: weightsURL)
        let single = try DistributedHessianAggregator.computeHessianShard(
            model: singleModel, blockIndices: [0, 1], tokenBatches: batches)

        let peers3 = [
            PeerCapability(peerId: "rank-0", chipName: "M4 Max", unifiedMemoryBytes: 128 << 30, measuredTFLOPS: 36, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "rank-1", chipName: "M3 Pro", unifiedMemoryBytes: 36 << 30, measuredTFLOPS: 18, thermalFactor: 1.0, linkGbps: 80),
            PeerCapability(peerId: "rank-2", chipName: "M2", unifiedMemoryBytes: 24 << 30, measuredTFLOPS: 12, thermalFactor: 1.0, linkGbps: 80),
        ]
        let harness = try await Self.makeLoopbackCluster(
            runId: "recover-3r", peers: peers3, quantized: false, weightsURL: weightsURL, fault: .disconnectEpoch0("rank-1"))
        let recovered = try await DistributedHessianAggregator.aggregateHessians(
            coordinator: harness.coordinator, blockIndices: [0, 1], calibrationTokenBatches: batches)
        #expect(recovered.completedEpoch == 1)
        #expect(await harness.transport.disconnectedPeerIds() == ["rank-1"])
        #expect(recovered.shardCountsByPeer == ["rank-0": 6, "rank-2": 2])
        try Self.assertHessianParity(cluster: recovered, single: single, tolerance: 1e-5)

        let cancelledShutdown = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            await harness.coordinator.shutdown()
        }
        await cancelledShutdown.value
        #expect(await harness.transport.isShutdownComplete())

        let tamperHarness = try await Self.makeLoopbackCluster(
            runId: "tamper-2r", peers: Array(peers3.prefix(2)), quantized: false, weightsURL: weightsURL, fault: .corruptEpoch0("rank-1"))
        defer { Task { await tamperHarness.coordinator.shutdown() } }
        await #expect(throws: ClusterError.macMismatch) {
            _ = try await DistributedHessianAggregator.aggregateHessians(
                coordinator: tamperHarness.coordinator, blockIndices: [0, 1], calibrationTokenBatches: batches)
        }
    }
}

extension DistributedClusterCalibrationTests {
    struct LoopbackHarness {
        let coordinator: ClusterCoordinatorActor
        let transport: LoopbackClusterTransport
    }

    enum LoopbackFault {
        case none
        case disconnectEpoch0(String)
        case corruptEpoch0(String)
    }

    static func makeCalibrationBatches(count: Int, seqLen: Int) -> [[Int32]] {
        (0 ..< count).map { b in
            (0 ..< seqLen).map { t in Int32((b * 17 + t * 7 + 3) % 240 + 1) }
        }
    }

    static func makeLoopbackCluster(
        runId: String,
        peers: [PeerCapability],
        quantized: Bool,
        weightsURL: URL,
        fault: LoopbackFault = .none
    ) async throws -> LoopbackHarness {
        let secret = "cluster-test-psk-\(runId)"
        let gate = LoopbackComputeGate()
        var nodes: [LoopbackPeerNodeActor] = []
        for peer in peers {
            try await gate.registerRank(peerId: peer.peerId) {
                let model = try TinyModel.make(.qwen3)
                if quantized {
                    quantize(model: model as Module, groupSize: 32, bits: 4, mode: .affine)
                }
                let loaded = try MLX.loadArrays(url: weightsURL)
                (model as Module).update(parameters: ModuleParameters.unflattened(loaded))
                eval(model as Module)
                return model
            }
            let nodeId = try ClusterFrameAuth.NodeIdentity.generate(peerId: peer.peerId)
            var dropEpochs: Set<UInt64> = []
            var corruptEpochs: Set<UInt64> = []
            if case .disconnectEpoch0(let p) = fault, p == peer.peerId { dropEpochs = [0] }
            if case .corruptEpoch0(let p) = fault, p == peer.peerId { corruptEpochs = [0] }
            nodes.append(LoopbackPeerNodeActor(
                identity: nodeId, secret: secret, computeGate: gate,
                disconnectOnEpochs: dropEpochs, corruptMACOnEpochs: corruptEpochs))
        }
        let transport = LoopbackClusterTransport(nodes: nodes)
        let topology = try ClusterTopology(runId: runId, coordinatorId: peers[0].peerId, peers: peers)
        let coordIdentity = try ClusterFrameAuth.NodeIdentity.generate(peerId: "coordinator")
        let coordinator = try ClusterCoordinatorActor(
            topology: topology, identity: coordIdentity, secret: secret, transport: transport)
        try await coordinator.connectAllPeers()
        return LoopbackHarness(coordinator: coordinator, transport: transport)
    }

    static func assertHessianParity(
        cluster: AggregatedHessianResult,
        single: (hessianSums: [String: SerializedTensor], tokenCounts: [String: Int]),
        tolerance: Float
    ) throws {
        #expect(cluster.tokenCounts == single.tokenCounts)
        let clusterSums = try cluster.mlxHessianSums()
        try assertTensorMapParity(clusterSums, reference: single.hessianSums, tolerance: tolerance)
        let clusterMeans = try cluster.mlxMeanHessians()
        var expectedMeans: [String: SerializedTensor] = [:]
        for (key, sumTensor) in single.hessianSums {
            let n = max(1, single.tokenCounts[key] ?? 1)
            let scale = Float(0.5) / Float(n)
            expectedMeans[key] = SerializedTensor.fromFloats(
                try sumTensor.decodeFloats().map { $0 * scale }, shape: sumTensor.shape)
        }
        try assertTensorMapParity(clusterMeans, reference: expectedMeans, tolerance: tolerance)
    }

    static func assertTensorMapParity(
        _ actual: [String: MLXArray],
        reference: [String: SerializedTensor],
        tolerance: Float
    ) throws {
        #expect(Set(actual.keys) == Set(reference.keys))
        for (key, refTensor) in reference {
            let lhs = try #require(actual[key]).asType(.float32)
            let rhs = try refTensor.decodeMLXArray()
            #expect(lhs.shape == rhs.shape)
            let diff = abs(lhs - rhs).max()
            let scale = abs(rhs).max()
            eval(diff, scale)
            let maxDiff = diff.item(Float.self)
            let denom = max(scale.item(Float.self), Float(1.0))
            #expect(maxDiff.isFinite && (maxDiff / denom) < tolerance)
        }
    }
}

@Suite("HTTPClusterTransport & CLI / Converter Cluster Wiring (#565)")
struct ClusterTransportAndCLITests {
    @Test("HTTPClusterTransport validates bind hosts, authenticates over StubURLProtocol and shields teardown")
    func httpClusterTransportEndToEndOverStubURLProtocol() async throws {
        let validEp = try ClusterPeerEndpoint(peerId: "peer-a", scheme: "http", host: "127.0.0.1", port: 9440)
        #expect(throws: ClusterError.wildcardBindRejected(host: "0.0.0.0")) {
            _ = try HTTPClusterTransport(bindHost: "0.0.0.0", peers: [validEp])
        }

        let secret = "http-cluster-psk"
        let responderId = try ClusterFrameAuth.NodeIdentity.generate(peerId: "http-peer-0")
        let sessionKeyBox = StubSessionBox()
        let stubHost = StubURLProtocol.host { req, body in
            Self.handleStubPeerHTTP(req: req, body: body, responder: responderId, secret: secret, box: sessionKeyBox)
        }
        defer { StubURLProtocol.setHandler(for: stubHost, nil) }

        let ep = try ClusterPeerEndpoint(peerId: "http-peer-0", scheme: "http", host: stubHost, port: 9440)
        let transport = try HTTPClusterTransport(
            bindHost: "127.0.0.1", peers: [ep], sessionConfiguration: StubURLProtocol.configuration)
        let cap = PeerCapability(peerId: "http-peer-0", chipName: "M4", unifiedMemoryBytes: 32 << 30, measuredTFLOPS: 20)
        let topology = try ClusterTopology(runId: "http-run", coordinatorId: "http-peer-0", peers: [cap])
        let coord = try ClusterCoordinatorActor(
            topology: topology,
            identity: try ClusterFrameAuth.NodeIdentity.generate(peerId: "coord"),
            secret: secret,
            transport: transport)

        try await coord.connectAllPeers()
        let work: [String: ClusterWorkRequest] = [
            "http-peer-0": .hessian(blockIndices: [0], tokenBatches: [[1, 2, 3, 4]], maxSwitchRows: 64)
        ]
        let out = try await coord.executeEpoch(workByPeer: work)
        #expect(out.epoch == 0 && out.responses.count == 1)

        await transport.disconnect(peerId: "http-peer-0")
        await coord.shutdown()
        let paths = StubURLProtocol.requests(for: stubHost).compactMap { $0.0.url?.path }
        #expect(paths.contains("/v1/cluster/handshake"))
        #expect(paths.contains("/v1/cluster/frame"))
        #expect(paths.contains("/v1/cluster/disconnect"))
    }

    @Test("ClusterCommand config persistence and ModelsConvertCommand --cluster-peers validation")
    func clusterCLIConfigAndConvertPeersOption() throws {
        let tempDir = FileManager.default.temporaryDirectory.appending(path: "bclt-cfg-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let cfgURL = tempDir.appending(component: "cluster.json")
        let id = try ClusterFrameAuth.NodeIdentity.generate(peerId: "node-tb5")
        let peerEp = try ClusterPeerEndpoint.parse("worker-1@169.254.20.2:9440")
        let cfg = ClusterNodeConfig(
            peerId: "node-tb5", bindHost: "169.254.20.1", port: 9440,
            secret: "bclt_secret_123456", identity: id, peers: [peerEp])
        try ClusterCommand.saveConfig(cfg, to: cfgURL)
        let loaded = try ClusterCommand.loadConfig(from: cfgURL)
        #expect(loaded == cfg)
        let attrs = try FileManager.default.attributesOfItem(atPath: cfgURL.path)
        let perms = (attrs[.posixPermissions] as? NSNumber)?.intValue ?? 0
        #expect((perms & 0o777) == 0o600)

        let parsedCap = try ClusterCommand.Plan.parseCapability("m4-max:38.5:80:0.9")
        #expect(parsedCap.peerId == "m4-max" && parsedCap.measuredTFLOPS == 38.5)

        let cmd = try ConvertSubcommand.parse([
            "mlx-community/Qwen3.5-4B-4bit",
            "--out", tempDir.appending(component: "out").path,
            "--to", "macaq:affine:4:64",
            "--cluster-peers", "169.254.20.2:9440,169.254.20.3:9440",
        ])
        try cmd.validate()
        #expect(try cmd.resolvedClusterPeers() == ["http://169.254.20.2:9440", "http://169.254.20.3:9440"])

        #expect(throws: (any Error).self) {
            let bad = try ConvertSubcommand.parse([
                "mlx-community/Qwen3.5-4B-4bit",
                "--out", tempDir.appending(component: "out2").path,
                "--to", "affine:4:64",
                "--cluster-peers", "0.0.0.0:9440",
            ])
            try bad.validate()
        }

        var req = ModelConversionRequest(source: "org/src", outputId: "local/out", target: .macaqAffine(bits: 3, groupSize: 64))
        req.clusterPeers = ["http://169.254.20.2:9440"]
        let encoded = try JSONEncoder().encode(req)
        let roundTrip = try JSONDecoder().decode(ModelConversionRequest.self, from: encoded)
        #expect(roundTrip == req && roundTrip.clusterPeers == ["http://169.254.20.2:9440"])
    }
}

extension ClusterTransportAndCLITests {
    final class StubSessionBox: Sendable {
        let key = Synchronization.Mutex<Data>(Data())
    }

    static func handleStubPeerHTTP(
        req: URLRequest,
        body: Data?,
        responder: ClusterFrameAuth.NodeIdentity,
        secret: String,
        box: StubSessionBox
    ) -> (status: Int, headers: [String: String], body: Data) {
        let path = req.url?.path ?? ""
        if path == "/v1/cluster/handshake" {
            guard let body,
                  let chal = try? JSONDecoder().decode(ClusterFrameAuth.HandshakeChallenge.self, from: body),
                  let accepted = try? ClusterFrameAuth.acceptChallenge(chal, responder: responder, secret: secret),
                  let out = try? JSONEncoder().encode(accepted.accept) else {
                return (401, [:], Data())
            }
            box.key.withLock { $0 = accepted.sessionKey }
            return (200, ["Content-Type": "application/json"], out)
        }
        if path == "/v1/cluster/frame" {
            let sessionKey = box.key.withLock { $0 }
            var win = ClusterFrameAuth.ReplayWindow()
            guard let body,
                  let opened = try? ClusterFrameAuth.openFrame(body.bytes, sessionKey: sessionKey, replayWindow: &win) else {
                return (401, [:], Data())
            }
            let tensor = SerializedTensor.fromFloats([2, 0, 0, 2], shape: [2, 2])
            let resp = ClusterWorkResponse.hessian(hessianSums: ["block0": tensor], tokenCounts: ["block0": 4])
            guard let respBytes = try? JSONEncoder().encode(resp),
                  let reply = try? ClusterFrameAuth.sealFrame(
                      kind: .tensor, epoch: opened.epoch, sequence: 1, payload: respBytes, sessionKey: sessionKey) else {
                return (500, [:], Data())
            }
            return (200, ["Content-Type": "application/octet-stream"], reply)
        }
        return (200, [:], Data())
    }
}
