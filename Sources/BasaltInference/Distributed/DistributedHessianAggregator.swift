import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package actor LoopbackComputeGate {
    private var models: [String: any LanguageModel] = [:]

    package init() {}

    package func registerRank(
        peerId: String,
        buildModel: @Sendable () throws -> any LanguageModel
    ) throws {
        models[peerId] = try buildModel()
    }

    package func removeRank(peerId: String) {
        models.removeValue(forKey: peerId)
    }

    package func removeAllRanks() {
        models.removeAll()
    }

    package func execute(peerId: String, request: ClusterWorkRequest) throws(ClusterError) -> ClusterWorkResponse {
        guard let model = models[peerId] else {
            throw .peerDisconnected(peerId: peerId, reason: "rank model is not resident")
        }
        do {
            switch request {
            case .hessian(let blocks, let batches, _):
                let computed = try DistributedHessianAggregator.computeHessianShard(
                    model: model, blockIndices: Set(blocks), tokenBatches: batches)
                return .hessian(hessianSums: computed.hessianSums, tokenCounts: computed.tokenCounts)
            case .pvGradients(let blocks, let maxExperts, let payloads):
                let teacherBatches = payloads.map { $0.toTeacherBatch() }
                let computed = try DistributedHessianAggregator.computePVGradientShard(
                    model: model, blockIndices: Set(blocks), maxSwitchExperts: maxExperts, batches: teacherBatches)
                return .pvGradients(meanGradients: computed.meanGradients, sampleCount: computed.sampleCount)
            }
        } catch let err as ClusterError {
            throw err
        } catch {
            throw .malformedFrame("worker compute failure on '\(peerId)': \(error.localizedDescription)")
        }
    }
}

package actor LoopbackPeerNodeActor {
    package let identity: ClusterFrameAuth.NodeIdentity
    private let secret: String
    private let computeGate: LoopbackComputeGate
    private var sessionKey: Data?
    private var nextOutboundSeq: UInt64 = 1
    private var inboundReplay = ClusterFrameAuth.ReplayWindow()
    private var disconnectOnEpochs: Set<UInt64>
    private var corruptMACOnEpochs: Set<UInt64>
    private var disconnected = false

    package init(
        identity: ClusterFrameAuth.NodeIdentity,
        secret: String,
        computeGate: LoopbackComputeGate,
        disconnectOnEpochs: Set<UInt64> = [],
        corruptMACOnEpochs: Set<UInt64> = []
    ) {
        self.identity = identity
        self.secret = secret
        self.computeGate = computeGate
        self.disconnectOnEpochs = disconnectOnEpochs
        self.corruptMACOnEpochs = corruptMACOnEpochs
    }

    package func setDisconnectOnEpochs(_ epochs: Set<UInt64>) {
        disconnectOnEpochs = epochs
    }

    package func handleHandshake(
        _ challenge: ClusterFrameAuth.HandshakeChallenge
    ) throws(ClusterError) -> ClusterFrameAuth.HandshakeAccept {
        guard !disconnected else {
            throw .peerDisconnected(peerId: identity.peerId, reason: "peer offline")
        }
        let accepted = try ClusterFrameAuth.acceptChallenge(challenge, responder: identity, secret: secret)
        sessionKey = accepted.sessionKey
        nextOutboundSeq = 1
        inboundReplay = ClusterFrameAuth.ReplayWindow()
        return accepted.accept
    }

    package func handleFrame(_ sealedFrame: Data) async throws(ClusterError) -> Data {
        guard !disconnected else {
            throw .peerDisconnected(peerId: identity.peerId, reason: "peer disconnected")
        }
        guard let key = sessionKey else {
            throw .unauthenticatedPeer(peerId: identity.peerId)
        }
        let decoded = try ClusterFrameAuth.openFrame(
            sealedFrame.bytes, sessionKey: key, replayWindow: &inboundReplay)
        if disconnectOnEpochs.contains(decoded.epoch) {
            disconnected = true
            throw .peerDisconnected(peerId: identity.peerId, reason: "worker dropped during epoch \(decoded.epoch)")
        }
        guard let req = try? JSONDecoder().decode(ClusterWorkRequest.self, from: decoded.payload) else {
            throw .malformedFrame("unable to decode work request on '\(identity.peerId)'")
        }
        let response = try await computeGate.execute(peerId: identity.peerId, request: req)
        guard let outBytes = try? JSONEncoder().encode(response) else {
            throw .malformedFrame("unable to encode work response on '\(identity.peerId)'")
        }
        let seq = nextOutboundSeq
        nextOutboundSeq += 1
        var reply = try ClusterFrameAuth.sealFrame(
            kind: .tensor, epoch: decoded.epoch, sequence: seq, payload: outBytes, sessionKey: key)
        if corruptMACOnEpochs.contains(decoded.epoch), reply.count > 30 {
            reply[30] ^= 0xFF
        }
        return reply
    }

    package func markDisconnected() async {
        disconnected = true
        sessionKey = nil
        await computeGate.removeRank(peerId: identity.peerId)
    }
}

package actor LoopbackClusterTransport: ClusterTransport {
    private var nodes: [String: LoopbackPeerNodeActor]
    private var disconnectedIds: Set<String> = []
    private var didShutdown = false

    package init(nodes: [LoopbackPeerNodeActor]) {
        self.nodes = Dictionary(uniqueKeysWithValues: nodes.map { ($0.identity.peerId, $0) })
    }

    package func disconnectedPeerIds() -> Set<String> {
        disconnectedIds
    }

    package func isShutdownComplete() -> Bool {
        didShutdown
    }

    package func handshake(
        peerId: String,
        challenge: ClusterFrameAuth.HandshakeChallenge
    ) async throws(ClusterError) -> ClusterFrameAuth.HandshakeAccept {
        guard !didShutdown, let node = nodes[peerId] else {
            throw .peerDisconnected(peerId: peerId, reason: "unknown loopback peer")
        }
        return try await node.handleHandshake(challenge)
    }

    package func sendFrame(peerId: String, sealedFrame: Data) async throws(ClusterError) -> Data {
        guard !didShutdown, let node = nodes[peerId] else {
            throw .peerDisconnected(peerId: peerId, reason: "loopback peer unavailable")
        }
        return try await node.handleFrame(sealedFrame)
    }

    package func disconnect(peerId: String) async {
        disconnectedIds.insert(peerId)
        guard let node = nodes.removeValue(forKey: peerId) else { return }
        await withTaskCancellationShield {
            await node.markDisconnected()
        }
    }

    package func shutdown() async {
        guard !didShutdown else { return }
        didShutdown = true
        let remaining = Array(nodes.values)
        nodes.removeAll()
        await withTaskCancellationShield {
            for node in remaining {
                await node.markDisconnected()
            }
        }
    }
}

package struct AggregatedHessianResult: Sendable, Equatable {
    package let hessianSums: [String: SerializedTensor]
    package let meanHessians: [String: SerializedTensor]
    package let tokenCounts: [String: Int]
    package let completedEpoch: UInt64
    package let shardCountsByPeer: [String: Int]

    package init(
        hessianSums: [String: SerializedTensor],
        meanHessians: [String: SerializedTensor],
        tokenCounts: [String: Int],
        completedEpoch: UInt64,
        shardCountsByPeer: [String: Int]
    ) {
        self.hessianSums = hessianSums
        self.meanHessians = meanHessians
        self.tokenCounts = tokenCounts
        self.completedEpoch = completedEpoch
        self.shardCountsByPeer = shardCountsByPeer
    }

    package func mlxHessianSums() throws(ClusterError) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, tensor) in hessianSums {
            out[key] = try tensor.decodeMLXArray()
        }
        return out
    }

    package func mlxMeanHessians() throws(ClusterError) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, tensor) in meanHessians {
            out[key] = try tensor.decodeMLXArray()
        }
        return out
    }
}

package struct AggregatedPVGradientResult: Sendable, Equatable {
    package let gradients: [String: SerializedTensor]
    package let totalSamples: Int
    package let completedEpoch: UInt64
    package let shardCountsByPeer: [String: Int]

    package init(
        gradients: [String: SerializedTensor],
        totalSamples: Int,
        completedEpoch: UInt64,
        shardCountsByPeer: [String: Int]
    ) {
        self.gradients = gradients
        self.totalSamples = totalSamples
        self.completedEpoch = completedEpoch
        self.shardCountsByPeer = shardCountsByPeer
    }

    package func mlxGradients() throws(ClusterError) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, tensor) in gradients {
            out[key] = try tensor.decodeMLXArray()
        }
        return out
    }
}

package enum DistributedHessianAggregator {
    package static func computeHessianShard(
        model: any LanguageModel,
        blockIndices: Set<Int>,
        tokenBatches: [[Int32]]
    ) throws -> (hessianSums: [String: SerializedTensor], tokenCounts: [String: Int]) {
        let root = model as Module
        root.train(false)
        var boxes: [String: MACaQHessianBox] = [:]
        var swapped: [(String, Module)] = []
        defer {
            for (path, orig) in swapped { try? PVTuner.swapModule(path, with: orig, in: root) }
        }
        for (path, mod) in root.namedModules() {
            guard let bIdx = PVTuner.textBlockIndex(path), blockIndices.contains(bIdx),
                  !path.contains("router"), !path.hasSuffix("in_proj_b"), !path.hasSuffix("in_proj_a"),
                  let inF = linearInputFeatures(mod), inF > 0 else { continue }
            let groupKey = QTIPQuantizer.sharedInputGroupKey(path: path, inFeatures: inF, sharedMoERHT: false)
            guard boxes[groupKey] == nil else { continue }
            let box = MACaQHessianBox(inFeatures: inF)
            boxes[groupKey] = box
            try PVTuner.swapModule(path, with: MACaQLinearTap(wrapping: mod, box: box), in: root)
            swapped.append((path, mod))
        }
        guard !swapped.isEmpty else { return ([:], [:]) }
        for tokens in tokenBatches where !tokens.isEmpty {
            let out = model(LMInput.Text(tokens: MLXArray(tokens)[.newAxis]), cache: nil, state: nil)
            eval(out.logits)
        }
        var sums: [String: SerializedTensor] = [:]
        var counts: [String: Int] = [:]
        for (key, box) in boxes where box.tokenCount > 0 {
            let hSum = box.sumXXT * Float(2.0)
            eval(hSum)
            sums[key] = SerializedTensor.encode(hSum)
            counts[key] = box.tokenCount
        }
        return (sums, counts)
    }

    package static func computePVGradientShard(
        model: any LanguageModel,
        blockIndices: Set<Int>,
        maxSwitchExperts: Int = 16,
        batches: [PVTeacherBatch]
    ) throws -> (meanGradients: [String: SerializedTensor], sampleCount: Int) {
        guard !batches.isEmpty else { return ([:], 0) }
        let root = model as Module
        let targets = PVTuner.collectTargets(root, blocks: blockIndices, maxSwitchExperts: maxSwitchExperts)
        let chunks = PVTuner.chunkTargets(targets, budget: 150_000_000)
        let minBlock = blockIndices.min() ?? 0
        var serialized: [String: SerializedTensor] = [:]
        for chunk in chunks {
            let grads = try PVTuner.denseGradients(model, chunk: chunk, minBlock: minBlock, batches: batches)
            for (path, grad) in grads {
                serialized[path] = SerializedTensor.encode(grad)
            }
        }
        return (serialized, batches.count)
    }

    private static func linearInputFeatures(_ module: Module) -> Int? {
        if let nf4 = module as? NF4Linear { return nf4.shape.1 }
        if let ql = module as? QuantizedLinear, !(module is MACaQQuantizedLinear) {
            return (ql.weight.dim(1) * 32) / max(1, ql.bits)
        }
        if type(of: module) == Linear.self, let lin = module as? Linear { return lin.shape.1 }
        return nil
    }
}

extension DistributedHessianAggregator {
    package static func aggregateHessians(
        coordinator: ClusterCoordinatorActor,
        blockIndices: Set<Int>,
        calibrationTokenBatches: [[Int32]],
        maxSwitchRows: Int = 512
    ) async throws(ClusterError) -> AggregatedHessianResult {
        guard !calibrationTokenBatches.isEmpty else {
            throw .invalidTopology("calibrationTokenBatches must not be empty")
        }
        let sortedBlocks = blockIndices.sorted()
        while true {
            let topology = await coordinator.currentTopology()
            let assignments = try DynamicMicrobatchBalancer.assignShards(
                globalBatchSize: calibrationTokenBatches.count, peers: topology.peers)
            var workByPeer: [String: ClusterWorkRequest] = [:]
            var shardCounts: [String: Int] = [:]
            for item in assignments where item.count > 0 {
                let slice = Array(calibrationTokenBatches[item.range])
                workByPeer[item.peerId] = .hessian(
                    blockIndices: sortedBlocks, tokenBatches: slice, maxSwitchRows: maxSwitchRows)
                shardCounts[item.peerId] = item.count
            }
            do {
                let outcome = try await coordinator.executeEpoch(workByPeer: workByPeer)
                return try reduceHessians(
                    responses: outcome.responses, epoch: outcome.epoch, shardCounts: shardCounts)
            } catch .peerDisconnected(let failedPeerId, _) {
                _ = try await withTaskCancellationShield { () async throws(ClusterError) -> ClusterTopology in
                    try await coordinator.evictDisconnectedPeer(failedPeerId)
                }
            }
        }
    }

    package static func aggregatePVGradients(
        coordinator: ClusterCoordinatorActor,
        blockIndices: Set<Int>,
        teacherBatches: [PVTeacherBatch],
        maxSwitchExperts: Int = 16
    ) async throws(ClusterError) -> AggregatedPVGradientResult {
        guard !teacherBatches.isEmpty else {
            throw .invalidTopology("teacherBatches must not be empty")
        }
        let sortedBlocks = blockIndices.sorted()
        let dtoBatches = teacherBatches.map(ClusterPVBatchPayload.init)
        while true {
            let topology = await coordinator.currentTopology()
            let assignments = try DynamicMicrobatchBalancer.assignShards(
                globalBatchSize: dtoBatches.count, peers: topology.peers)
            var workByPeer: [String: ClusterWorkRequest] = [:]
            var shardCounts: [String: Int] = [:]
            for item in assignments where item.count > 0 {
                let slice = Array(dtoBatches[item.range])
                workByPeer[item.peerId] = .pvGradients(
                    blockIndices: sortedBlocks, maxSwitchExperts: maxSwitchExperts, batches: slice)
                shardCounts[item.peerId] = item.count
            }
            do {
                let outcome = try await coordinator.executeEpoch(workByPeer: workByPeer)
                return try reducePVGradients(
                    responses: outcome.responses, epoch: outcome.epoch, shardCounts: shardCounts)
            } catch .peerDisconnected(let failedPeerId, _) {
                _ = try await withTaskCancellationShield { () async throws(ClusterError) -> ClusterTopology in
                    try await coordinator.evictDisconnectedPeer(failedPeerId)
                }
            }
        }
    }
}

extension DistributedHessianAggregator {
    private static func reduceHessians(
        responses: [String: ClusterWorkResponse],
        epoch: UInt64,
        shardCounts: [String: Int]
    ) throws(ClusterError) -> AggregatedHessianResult {
        var accumFloats: [String: (shape: [Int], values: [Float])] = [:]
        var totalTokens: [String: Int] = [:]
        for peerId in responses.keys.sorted() {
            guard case .hessian(let sums, let counts) = responses[peerId] else {
                throw .malformedFrame("expected hessian response from '\(peerId)'")
            }
            for (layerKey, tensor) in sums {
                let decoded = try tensor.decodeFloats()
                if var existing = accumFloats[layerKey] {
                    guard existing.shape == tensor.shape, existing.values.count == decoded.count else {
                        throw .malformedFrame("hessian shape mismatch on '\(layerKey)'")
                    }
                    for i in 0 ..< decoded.count { existing.values[i] += decoded[i] }
                    accumFloats[layerKey] = existing
                } else {
                    accumFloats[layerKey] = (tensor.shape, decoded)
                }
                totalTokens[layerKey, default: 0] += counts[layerKey] ?? 0
            }
        }
        var sumTensors: [String: SerializedTensor] = [:]
        var meanTensors: [String: SerializedTensor] = [:]
        for (layerKey, entry) in accumFloats {
            let tok = max(1, totalTokens[layerKey] ?? 1)
            let invTwoN = Float(0.5) / Float(tok)
            sumTensors[layerKey] = SerializedTensor.fromFloats(entry.values, shape: entry.shape)
            meanTensors[layerKey] = SerializedTensor.fromFloats(
                entry.values.map { $0 * invTwoN }, shape: entry.shape)
        }
        return AggregatedHessianResult(
            hessianSums: sumTensors, meanHessians: meanTensors,
            tokenCounts: totalTokens, completedEpoch: epoch, shardCountsByPeer: shardCounts)
    }

    private static func reducePVGradients(
        responses: [String: ClusterWorkResponse],
        epoch: UInt64,
        shardCounts: [String: Int]
    ) throws(ClusterError) -> AggregatedPVGradientResult {
        var weightedFloats: [String: (shape: [Int], values: [Float])] = [:]
        var totalSamples = 0
        for peerId in responses.keys.sorted() {
            guard case .pvGradients(let grads, let count) = responses[peerId], count > 0 else {
                throw .malformedFrame("expected non-empty PV gradient response from '\(peerId)'")
            }
            let weight = Float(count)
            totalSamples += count
            for (layerKey, tensor) in grads {
                let decoded = try tensor.decodeFloats()
                if var existing = weightedFloats[layerKey] {
                    guard existing.shape == tensor.shape, existing.values.count == decoded.count else {
                        throw .malformedFrame("gradient shape mismatch on '\(layerKey)'")
                    }
                    for i in 0 ..< decoded.count { existing.values[i] += weight * decoded[i] }
                    weightedFloats[layerKey] = existing
                } else {
                    weightedFloats[layerKey] = (tensor.shape, decoded.map { weight * $0 })
                }
            }
        }
        guard totalSamples > 0 else { throw .malformedFrame("zero total PV samples") }
        let invN = Float(1.0) / Float(totalSamples)
        var out: [String: SerializedTensor] = [:]
        for (layerKey, entry) in weightedFloats {
            out[layerKey] = SerializedTensor.fromFloats(entry.values.map { $0 * invN }, shape: entry.shape)
        }
        return AggregatedPVGradientResult(
            gradients: out, totalSamples: totalSamples, completedEpoch: epoch, shardCountsByPeer: shardCounts)
    }
}
