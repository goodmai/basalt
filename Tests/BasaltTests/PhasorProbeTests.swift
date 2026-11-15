import ComplexModule
import RealModule
import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Aizenberg MVN activation, complex XOR, and multi-sector phase parity (#365)")
struct MVNPhasorProbeAlgebraTests {
    @Test("Continuous activation normalizes onto S^1 with fallback 1+0i near zero, and discrete roots are idempotent",
          arguments: [2, 3, 4, 8, 16])
    func activationProperties(sectors: Int) {
        #expect(MVNActivation.continuousUnitCircle.apply(.zero) == Complex(1, 0))
        #expect(MVNActivation.continuousUnitCircle.apply(Complex(1e-15, -1e-15)) == Complex(1, 0))
        let scaled = MVNActivation.continuousUnitCircle.apply(Complex(3, 4))
        #expect((scaled - Complex(0.6, 0.8)).length < 1e-5)

        let discrete = MVNActivation.discreteRoots(sectors: sectors)
        for sector in 0 ..< sectors {
            let root = MVNActivation.rootOfUnity(sector: sector, sectors: sectors)
            let bisector = MVNActivation.wedgeBisector(wedge: sector, wedges: sectors)
            #expect(abs(root.length - 1) < 1e-5)
            #expect(discrete.sector(of: root) == sector)
            #expect(discrete.sector(of: bisector) == sector)
            #expect((discrete.apply(root) - root).length < 1e-5)
            #expect((discrete.apply(bisector) - root).length < 1e-5)
        }
    }

    @Test("Single-neuron MVN solves complex quadrant XOR at 100% while the real linear baseline stalls at <= 75%")
    func complexQuadrantXOR() {
        let trainStates: [[Float]] = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
        let trainLabels = [0, 0, 1, 1]
        let heldOutStates: [[Float]] = [
            [0.95, 1.05], [-1.05, -0.95], [1.05, -0.95], [-0.95, 1.05],
            [0.85, 0.90], [-0.90, -0.85], [0.90, -0.85], [-0.85, 0.90],
        ]
        let heldOutLabels = [0, 0, 1, 1, 0, 0, 1, 1]
        var request = PhasorProbeRequest(sectors: 2)
        request.folds = 2
        request.maxEpochs = 200
        request.seed = 42
        let dataset = PhasorLayerDataset(
            trainStates: trainStates,
            trainLabels: trainLabels,
            heldOutStates: heldOutStates,
            heldOutLabels: heldOutLabels
        )
        let report = PhasorProbeEvaluator.evaluateLayer(layer: 0, dataset: dataset, request: request)
        #expect(report.trainAccuracy == 1.0)
        #expect(report.heldOutAccuracy == 1.0)
        #expect(report.linearHeldOutAccuracy <= 0.75)
        #expect(report.vsLinearDelta >= 0.25)
        #expect(report.angularMAE < 0.2)
        #expect(report.epochsRun >= 1 && report.epochsRun <= 200)
    }

    @Test("Antipodal k-sector parity is solved at 100% by one periodic MVN while real linear probes cap at <= 75%",
          arguments: [2, 3, 4, 8])
    func antipodalSectorParity(sectors: Int) {
        let total = 2 * sectors
        var states: [[Float]] = []
        var labels: [Int] = []
        for wedge in 0 ..< total {
            let z = MVNActivation.wedgeBisector(wedge: wedge, wedges: total)
            states.append([z.real, z.imaginary])
            labels.append(wedge % sectors)
        }
        var request = PhasorProbeRequest(sectors: sectors)
        request.folds = 2
        request.maxEpochs = 300
        request.seed = 19
        let dataset = PhasorLayerDataset(
            trainStates: states,
            trainLabels: labels,
            heldOutStates: states,
            heldOutLabels: labels
        )
        let report = PhasorProbeEvaluator.evaluateLayer(layer: 2, dataset: dataset, request: request)
        #expect(report.trainAccuracy == 1.0)
        #expect(report.heldOutAccuracy == 1.0)
        #expect(report.linearHeldOutAccuracy <= 0.75)
        #expect(report.vsLinearDelta >= 0.25)
    }

    @Test("Two-variable discrete phase sum mod k reaches 100% with FHRR binding + MVN where raw linear readout fails",
          arguments: [2, 3, 4, 8])
    func discretePhaseSumModK(sectors: Int) {
        var boundSamples: [[Complex<Float>]] = []
        var rawFeatures: [[Float]] = []
        var labels: [Int] = []
        for a in 0 ..< sectors {
            let za = MVNActivation.rootOfUnity(sector: a, sectors: sectors)
            let lhs: InlineArray<1, Complex<Float>> = [za]
            for b in 0 ..< sectors {
                let zb = MVNActivation.rootOfUnity(sector: b, sectors: sectors)
                let rhs: InlineArray<1, Complex<Float>> = [zb]
                boundSamples.append(FHRRBinding.bind(lhs.span, rhs.span))
                rawFeatures.append([za.real, za.imaginary, zb.real, zb.imaginary])
                labels.append((a + b) % sectors)
            }
        }
        var mvn = MVNProbe(complexDimension: 1, activation: .discreteRoots(sectors: sectors), seed: 42)
        mvn.fit(samples: boundSamples, labels: labels, sectors: sectors, maxEpochs: 200)
        let mvnHits = zip(boundSamples, labels).filter { sample, label in
            mvn.predictSector(sample.span, sectors: sectors) == label
        }.count
        #expect(mvnHits == labels.count)

        var linear = LinearProbeBaseline(classes: sectors, featureDimension: 4, seed: 42)
        linear.fit(samples: rawFeatures, labels: labels, epochs: 300)
        let upperBound = Double(sectors * sectors - sectors + 1) / Double(sectors * sectors) + 1e-9
        #expect(linear.accuracy(samples: rawFeatures, labels: labels) <= upperBound)
    }
}

@Suite("FHRR holographic complex binding, unbinding, orthogonality, and superposition (#365)")
struct FHRRBindingAlgebraTests {
    @Test("Conjugate unbinding is an exact self-inverse on (S^1)^D and mismatched random keys are near-orthogonal at D=1024")
    func unbindingIdentityAndOrthogonality() {
        let role = PhasorVector.random(dimension: 1024, seed: 11)
        let filler = PhasorVector.random(dimension: 1024, seed: 29)
        let wrongKey = PhasorVector.random(dimension: 1024, seed: 97)

        let roleElems = role.elements
        let fillerElems = filler.elements
        let wrongElems = wrongKey.elements
        let bound = FHRRBinding.bind(roleElems.span, fillerElems.span)
        let recovered = FHRRBinding.unbind(bound: bound.span, key: roleElems.span)
        let exactSim = FHRRBinding.similarity(recovered.span, fillerElems.span)
        #expect(abs(exactSim - 1.0) < 1e-5)
        #expect(abs(role.bound(with: filler).unbound(key: role).similarity(to: filler) - 1.0) < 1e-5)

        let wrongRecovered = FHRRBinding.unbind(bound: bound.span, key: wrongElems.span)
        let wrongSim = FHRRBinding.similarity(wrongRecovered.span, fillerElems.span)
        #expect(abs(wrongSim) < 0.1)
    }

    @Test("Bundled superposition of M role-filler pairs at D=2048 recovers every filler > 5 sigma above random phase baseline",
          arguments: [2, 4, 8])
    func bundledSuperpositionRecovery(pairCount: Int) {
        let dimension = 2048
        let roles = (0 ..< pairCount).map { PhasorVector.random(dimension: dimension, seed: UInt64(100 + $0)) }
        let fillers = (0 ..< pairCount).map { PhasorVector.random(dimension: dimension, seed: UInt64(500 + $0)) }
        let bindings = zip(roles, fillers).map { role, filler in
            let r = role.elements, f = filler.elements
            return FHRRBinding.bind(r.span, f.span)
        }
        let superposition = FHRRBinding.bundle(bindings)
        let singleNullSigma = Float(1.0 / Double(2 * dimension).squareRoot())

        for index in 0 ..< pairCount {
            let roleElems = roles[index].elements
            let fillerElems = fillers[index].elements
            let unbound = FHRRBinding.unbind(bound: superposition.span, key: roleElems.span)
            let targetSim = FHRRBinding.similarity(unbound.span, fillerElems.span)
            #expect(targetSim > 5.0 * singleNullSigma)
        }

        let boundReal = Self.interleaveReal(superposition)
        let triples = (0 ..< pairCount).map { index in
            (
                bound: boundReal,
                role: Self.interleaveReal(roles[index].elements),
                filler: Self.interleaveReal(fillers[index].elements)
            )
        }
        let report = FHRRBinding.evaluate(layer: 3, triples: triples, seed: 77)
        #expect(report.layer == 3)
        #expect(report.zScore > 5.0)
        #expect(report.unbindingSimilarity > report.randomBaselineSimilarity + 0.15)
        #expect(report.unbindingSimilarity > report.additiveBaselineSimilarity)
    }

    private static func interleaveReal(_ complex: [Complex<Float>]) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(complex.count * 2)
        for z in complex {
            out.append(z.real)
            out.append(z.imaginary)
        }
        return out
    }
}

@Suite("TinyModel phasorProbe extraction and MCP phasor_probe research tool gating (#365)", .serialized)
struct PhasorProbeEngineAndMCPTests {
    @Test("TinyModel.engine(.qwen3) extracts per-layer residuals, evaluates MVN vs linear and FHRR triples, and leaves no overlays")
    func tinyModelEndToEnd() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let anatomy = try await engine.anatomy()
        let request = PhasorProbeRequest(
            train: [
                LabeledPrompt(prompt: "aaaa aaaa", sector: 0),
                LabeledPrompt(prompt: "aaaa aaab", sector: 0),
                LabeledPrompt(prompt: "zzzz zzzz", sector: 1),
                LabeledPrompt(prompt: "zzzz zzza", sector: 1),
            ],
            heldOut: [
                LabeledPrompt(prompt: "aaaa aaac", sector: 0),
                LabeledPrompt(prompt: "zzzz zzzb", sector: 1),
            ],
            triples: [
                RoleFillerTriple(bound: "capital France Paris", role: "capital France", filler: "Paris"),
                RoleFillerTriple(bound: "capital Japan Tokyo", role: "capital Japan", filler: "Tokyo"),
            ],
            sectors: 2
        )
        let summary = try await engine.phasorProbe(request)
        #expect(summary.layers.map(\.layer) == Array(0 ..< anatomy.layers.count))
        #expect(summary.bestLayer != nil)
        #expect(summary.layers.allSatisfy { (0.0 ... 1.0).contains($0.trainAccuracy) && $0.angularMAE.isFinite })
        #expect(summary.binding.map(\.layer) == Array(0 ..< anatomy.layers.count))
        #expect(summary.bestBindingLayer != nil)
        #expect(summary.binding.allSatisfy { $0.zScore.isFinite && (-1.0 ... 1.0).contains($0.unbindingSimilarity) })
        #expect(await engine.overlays().isEmpty)

        var subset = request
        subset.layers = [3, 1]
        subset.pooling = .meanPool
        subset.encoding = .rawPairs
        subset.continuous = true
        let narrow = try await engine.phasorProbe(subset)
        #expect(narrow.layers.map(\.layer) == [1, 3])
        #expect(narrow.binding.map(\.layer) == [1, 3])
    }

    @Test("Invalid PhasorProbeRequests are rejected before forward execution")
    func invalidEngineRequestsRejected() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let invalidCases: [PhasorProbeRequest] = [
            PhasorProbeRequest(),
            PhasorProbeRequest(train: [LabeledPrompt(prompt: "only one", sector: 0)], sectors: 2),
            PhasorProbeRequest(
                train: [LabeledPrompt(prompt: "a", sector: 0), LabeledPrompt(prompt: "b", sector: 2)],
                sectors: 2
            ),
            PhasorProbeRequest(
                train: [LabeledPrompt(prompt: "a", sector: 0), LabeledPrompt(prompt: "b", sector: 1)],
                sectors: 2,
                layers: [99]
            ),
        ]
        for bad in invalidCases {
            await #expect(throws: BasaltError.self) { try await engine.phasorProbe(bad) }
        }
    }

    @Test("phasor_probe MCP schema, argument parser, and --research-tools server gating behave per contract")
    func mcpParsingAndGating() async throws {
        let offNames = Set(MCPToolCatalog.catalog(researchTools: false).tools.map(\.name))
        let onTools = MCPToolCatalog.catalog(researchTools: true).tools
        #expect(!offNames.contains("phasor_probe"))
        let descriptor = try #require(onTools.first { $0.name == "phasor_probe" })
        let schema = try #require(descriptor.inputSchema.rawValue as? [String: Any])
        let properties = try #require(schema["properties"] as? [String: Any])
        #expect(properties.keys.contains("train") && properties.keys.contains("triples") && properties.keys.contains("folds"))

        let parsed = try PhasorProbeRequest(mcp: [
            "train": [["prompt": "p0", "sector": 0], ["prompt": "p1", "sector": 2]],
            "triples": [["bound": "ab", "role": "a", "filler": "b"]],
            "sectors": 3, "folds": 2, "encoding": "rawPairs", "pooling": "meanPool",
            "layers": [0, 1], "maxEpochs": 50, "learningRate": 0.5, "seed": 7,
        ])
        #expect(parsed.sectors == 3 && parsed.folds == 2 && parsed.encoding == .rawPairs && parsed.pooling == .meanPool)
        #expect(parsed.seed == 7 && parsed.maxEpochs == 50 && parsed.layers == [0, 1])

        let badFailure = #expect(throws: MCPFailure.self) { try PhasorProbeRequest(mcp: [:]) }
        #expect(badFailure?.code == -32602)

        let engine = try await TinyModel.engine(.qwen3)
        let capturer = OutputCapturer()
        let locked = MCPServer(orchestrator: ModelOrchestratorActor(engine: engine)) { capturer.capture($0) }
        let unlocked = MCPServer(orchestrator: ModelOrchestratorActor(engine: engine)) { capturer.capture($0) }
            .withResearchTools(true)
        let call = #"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"phasor_probe","arguments":{"triples":[{"bound":"ab","role":"a","filler":"b"}],"layers":[0]}}}"#
        await locked.dispatch(call)
        await unlocked.dispatch(call)
        #expect(capturer.lines[0].contains("-32601") && capturer.lines[0].contains("--research-tools"))
        #expect(capturer.lines[1].contains("unbindingSimilarity") && !capturer.lines[1].contains("\"error\""))
    }
}
