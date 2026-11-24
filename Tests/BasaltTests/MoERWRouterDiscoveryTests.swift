import MLX
import MLXNN
import Testing
@testable import BasaltInference

@Suite("moe-rw router discovery and gate weights (#748, #757)")
struct MoERWRouterDiscoveryTests {
    @Test("Gemma 4 router (#748): moe-rw finds .router.proj, and each routed token's gate is the top-k softmax times per_expert_scale, so the expert Hessian differs from the unweighted one")
    func gemma4RouterWeightsTheExpertHessian() throws {
        let model = try TinyModel.make(.gemma4Moe)
        let root = model as Module
        let routerPath = try #require(root.namedModules().map { $0.0 }.first { $0.hasSuffix("layers.0.router") })
        let router = try #require(Dictionary(root.namedModules(), uniquingKeysWith: { a, _ in a })[routerPath])
        let scales: [Float] = [0.5, 1.5, 0.75, 2.0, 1.0, 0.25, 1.25, 3.0]
        try router.update(parameters: ModuleParameters.unflattened(["per_expert_scale": MLXArray(scales)]), verify: .noUnusedKeys)
        let site = try #require(MoERWPipeline.discoverRouters(in: root)[0])
        #expect(site.path == routerPath + ".proj")
        #expect(try #require(site.expertScale).asArray(Float.self) == scales)

        let switches = MoERWPipeline.collectCandidateTargets(root, blocks: [0]).switches
        let gate = try #require(switches.first { $0.path.hasSuffix("gate_proj") })
        let tokens = (0 ..< 48).map { Int32(($0 * 37 + 11) % 200) }
        let bundle = try MoERWPipeline.captureMoEGroup(
            model: model, root: root, targets: .init(linears: [], switches: [gate]),
            routers: [0: site], batches: [tokens], maxSwitchRows: 4096)
        let reservoir = try #require(bundle.switches[gate.path])
        let logits = try #require(reservoir.gateBox?.lastLogits)
        let top = MoERouterWeightedHessian.topKGating(logits: logits, topK: 2)
        let (ids, weights) = (top.indices.asArray(Int32.self), top.weights.asArray(Float.self))
        let estimator = MoERouterWeightedHessian()
        var checked = 0
        for expert in 0 ..< 8 where reservoir.countsByExpert[expert] >= estimator.minRoutedTokens {
            let expected = zip(ids, weights).filter { Int($0.0) == expert }.map { $0.1 * scales[expert] }.sorted()
            let captured = reservoir.routedC[expert].sorted()
            #expect(captured.count == expected.count)
            #expect(zip(captured, expected).allSatisfy { abs($0 - $1) < 1e-5 }, "expert \(expert)")
            let weighted = try #require(reservoir.expertHessian(expert: expert, estimator: estimator, routerWeight: nil)).hessian
            let rows = concatenated(reservoir.routedX[expert], axis: 0)
            let unweighted = estimator.hessian(activations: rows, gatingWeights: MLXArray.ones([rows.dim(0)], dtype: .float32))
            #expect(abs(weighted - unweighted).max().item(Float.self) > 1e-3, "expert \(expert)")
            checked += 1
        }
        #expect(checked >= 2)
    }

    @Test("Sigmoid routers (#757): with no Linear router to tap, moe-rw takes each routed token's gate from the model's routerTopK, so Nemotron-H's group top-k with scale and Inkling 2's normalised route_scale reach the expert Hessian",
          arguments: [(TinyModel.Family.sigmoidMoe, 1), (.inkling2Loop, 0)])
    func sigmoidRouterGatesComeFromTheModel(family: TinyModel.Family, block: Int) throws {
        let model = try TinyModel.make(family)
        let root = model as Module
        #expect(MoERWPipeline.discoverRouters(in: root)[block] == nil)
        let switches = MoERWPipeline.collectCandidateTargets(root, blocks: [block]).switches
        let input = try #require(switches.first { !$0.path.hasSuffix("down_proj") && !$0.path.hasSuffix("fc2") })
        let tokens = (0 ..< 48).map { Int32(($0 * 37 + 11) % 200) }

        let reference = MoERouterGateBox(numExperts: input.numExperts, inFeatures: 1)
        eval(MoERWRouterProbe.forward(model, tokens: tokens, probe: MoERWRouterProbe(boxes: [block: reference])))
        let routed = try #require(reference.lastRouted)
        let (ids, gates) = (routed.indices.asArray(Int32.self), routed.gates.asArray(Float.self))
        #expect(gates.contains { abs($0 - 1) > 1e-3 })

        let bundle = try MoERWPipeline.captureMoEGroup(
            model: model, root: root, targets: .init(linears: [], switches: [input]),
            routers: [:], batches: [tokens], maxSwitchRows: 4096)
        let reservoir = try #require(bundle.switches[input.path])
        #expect(reservoir.unweightedCalls == 0)
        for expert in 0 ..< input.numExperts {
            let expected = zip(ids, gates).filter { Int($0.0) == expert }.map(\.1).sorted()
            let captured = reservoir.routedC[expert].sorted()
            #expect(captured.count == expected.count, "expert \(expert)")
            #expect(zip(captured, expected).allSatisfy { abs($0 - $1) < 1e-5 }, "expert \(expert)")
        }
    }
}
