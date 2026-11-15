import Foundation
import Testing
import MLX
@testable import BasaltCore
@testable import BasaltHarness
@testable import BasaltInference
@testable import BasaltServer

@Suite("T-norm continuous satisfaction algebra, clause margins and crisp gate invariance (#362)")
struct TNormAlgebraTests {
    static let sampleGrid: [Double] = [0.0, 0.15, 0.4, 0.75, 0.89, 1.0]

    @Test("Boundary identity, zero annihilator, empty span and clamping hold for all T-norms", arguments: TNorm.allCases)
    func boundaryAndClamping(_ norm: TNorm) {
        let empty: [Double] = []
        #expect(norm.evaluate(empty.span) == 1.0)
        #expect(norm.tConorm(empty.span) == 0.0)
        for x in Self.sampleGrid {
            #expect(abs(norm.evaluate(1.0, x) - x) < 1e-15)
            #expect(abs(norm.evaluate(x, 1.0) - x) < 1e-15)
            #expect(norm.evaluate(0.0, x) == 0.0)
            #expect(norm.tConorm(0.0, x) == x)
            #expect(norm.tConorm(1.0, x) == 1.0)
        }
        let outOfBounds = [-2.0, 1.5]
        #expect(norm.evaluate(outOfBounds.span) == 0.0)
        let aboveUnit = [1.5, 2.5]
        #expect(norm.evaluate(aboveUnit.span) == 1.0)
        let withNaN = [0.9, Double.nan]
        #expect(norm.evaluate(withNaN.span) == 0.0)
    }

    @Test("Commutativity, associativity, monotonicity and residuation adjunction hold", arguments: TNorm.allCases)
    func algebraicAxioms(_ norm: TNorm) {
        for a in Self.sampleGrid {
            for b in Self.sampleGrid {
                #expect(abs(norm.evaluate(a, b) - norm.evaluate(b, a)) < 1e-12)
                for c in Self.sampleGrid {
                    let left = norm.evaluate(norm.evaluate(a, b), c)
                    let right = norm.evaluate(a, norm.evaluate(b, c))
                    #expect(abs(left - right) < 1e-12)
                    if a <= b {
                        #expect(norm.evaluate(a, c) <= norm.evaluate(b, c) + 1e-12)
                    }
                    let implication = norm.residuum(from: a, to: b)
                    let lhsHolds = norm.evaluate(a, c) <= b + 1e-12
                    let rhsHolds = c <= implication + 1e-12
                    #expect(lhsHolds == rhsHolds)
                }
            }
        }
    }

    @Test("De Morgan duality links evaluate and tConorm, and T_L <= T_P <= T_G holds everywhere")
    func dualityAndOrdering() {
        let vectors: [[Double]] = [
            [0.95, 0.90, 0.85],
            [0.4, 0.5, 0.6],
            [1.0, 0.89, 1.0, 0.75],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.99, 1.0],
            [1.0, 1.0, 1.0],
        ]
        for values in vectors {
            let complements = values.map { 1.0 - $0 }
            for norm in TNorm.allCases {
                let conorm = norm.tConorm(values.span)
                let dual = 1.0 - norm.evaluate(complements.span)
                #expect(abs(conorm - dual) < 1e-12)
            }
            let summary = TNormSummary(values.span)
            #expect(summary.checkCount == values.count)
            #expect(summary.lukasiewiczSatisfaction <= summary.productSatisfaction + 1e-12)
            #expect(summary.productSatisfaction <= summary.godelSatisfaction + 1e-12)
        }
    }

    @Test("Ten checks with 11 percent shortfall collapse Lukasiewicz to zero while Godel stays 0.89 and crisp GateReport.passed stays true")
    func deathByTenElevenPercentCuts() {
        let degrees = Array(repeating: 0.89, count: 10)
        let summary = TNormSummary(degrees.span)
        #expect(abs(summary.godelSatisfaction - 0.89) < 1e-12)
        #expect(abs(summary.productSatisfaction - Foundation.pow(0.89, 10.0)) < 1e-12)
        #expect(summary.lukasiewiczSatisfaction == 0.0)

        let passingChecks = (0 ..< 10).map { index in
            GateCheck(name: "check-\(index)", passed: true, value: 0.11, limit: 1.0, satisfaction: 0.89)
        }
        let passingReport = GateReport(checks: passingChecks)
        #expect(passingReport.passed)
        #expect(passingReport.tNorm?.lukasiewiczSatisfaction == 0.0)
        #expect(abs((passingReport.tNorm?.godelSatisfaction ?? 0) - 0.89) < 1e-12)

        var failingChecks = passingChecks
        failingChecks[9] = GateCheck(name: "check-9", passed: false, value: 1.2, limit: 1.0, satisfaction: 0.0)
        let failingReport = GateReport(checks: failingChecks)
        #expect(!failingReport.passed)
        #expect(failingReport.tNorm?.godelSatisfaction == 0.0)
    }

    @Test("ClauseMargin normalizes upper ceilings and lower floors and Harness reports keep crisp selection unchanged")
    func clauseMarginsAndHarnessCrispInvariance() {
        #expect(ClauseMargin.upperCeiling(actual: 0.0, limit: 0.5, ideal: 0.0) == 1.0)
        #expect(abs(ClauseMargin.upperCeiling(actual: 0.25, limit: 0.5, ideal: 0.0) - 0.5) < 1e-12)
        #expect(ClauseMargin.upperCeiling(actual: 0.6, limit: 0.5, ideal: 0.0) == 0.0)
        #expect(ClauseMargin.upperCeiling(actual: .nan, limit: 0.5, ideal: 0.0) == 0.0)
        #expect(abs(ClauseMargin.lowerFloor(actual: 0.6, floor: 0.2, ideal: 1.0) - 0.5) < 1e-12)
        #expect(ClauseMargin.lowerFloor(actual: 0.2, floor: 0.2, ideal: 1.0) == 0.0)

        let task = HarnessTask(slug: "s", prompt: "p", requiredTerms: ["alpha", "beta"])
        let cleanResponse = GenerationResponse(
            generatedText: "alpha\nbeta", promptTokens: 4, completionTokens: 4, tokensPerSecond: 1,
            generationTime: 1, timeToFirstToken: 1, memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop
        )
        let stutterResponse = GenerationResponse(
            generatedText: "alpha\nalpha\nbeta", promptTokens: 4, completionTokens: 6, tokensPerSecond: 1,
            generationTime: 1, timeToFirstToken: 1, memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop
        )
        let cleanView = HarnessRunView(candidateId: "a", response: cleanResponse, task: task)
        let stutterView = HarnessRunView(candidateId: "b", response: stutterResponse, task: task)
        #expect(cleanView.compareMeasured(stutterView) == 0)
        #expect((stutterView.tNorm?.godelSatisfaction ?? 1.0) < (cleanView.tNorm?.godelSatisfaction ?? 0.0))

        let tree = HarnessTree(files: [
            "CLAUDE.md": "Project rules.",
            ".claude/skills/orphan/SKILL.md": "---\nname: orphan\ndescription: Unreferenced skill\n---\nBody",
        ])
        let validation = HarnessValidator.validate(tree, spec: .claude)
        #expect(validation.isValid)
        #expect((validation.tNorm?.godelSatisfaction ?? 1.0) == 0.0)
    }
}

@Suite("Fisher-Rao Kahan half-angle logit drift metric (#361)")
struct FisherRaoDriftTests {
    @Test("Identity, symmetry and triangle inequality hold on categorical logit distributions")
    func metricAxioms() {
        let p = MLXArray([Float(2.0), -1.0, 0.5, 4.0, -3.0])
        let q = MLXArray([Float(-0.5), 3.0, 1.0, 0.0, 2.5])
        let r = MLXArray([Float(0.2), 0.1, -2.0, 1.5, 0.8])
        let extreme = MLXArray([Float(500.0), -500.0, 250.0, -250.0, 0.0])

        let selfDistance = LogitDrift.fisherRaoFromLogits(p, p).item(Float.self)
        let extremeSelf = LogitDrift.fisherRaoFromLogits(extreme, extreme).item(Float.self)
        #expect(selfDistance == 0)
        #expect(extremeSelf == 0)

        let pq = LogitDrift.fisherRaoFromLogits(p, q).item(Float.self)
        let qp = LogitDrift.fisherRaoFromLogits(q, p).item(Float.self)
        let qr = LogitDrift.fisherRaoFromLogits(q, r).item(Float.self)
        let pr = LogitDrift.fisherRaoFromLogits(p, r).item(Float.self)
        #expect(abs(pq - qp) < 1e-6)
        #expect(pr <= pq + qr + 1e-5)
    }

    @Test("Disjoint support converges to pi without NaN while KL diverges, and degenerate inputs stay in [0, pi]")
    func disjointSupportAndDegenerateInputs() {
        let logOneHotA = MLXArray([Float(0.0), -Float.infinity, -Float.infinity])
        let logOneHotB = MLXArray([-Float.infinity, Float(0.0), -Float.infinity])
        let exactOrthogonal = LogitDrift.fisherRao(logP: logOneHotA, logQ: logOneHotB).item(Float.self)
        #expect(abs(exactOrthogonal - Float.pi) < 1e-5)

        let sharpA = MLXArray([Float(80.0), -80.0, -80.0])
        let sharpB = MLXArray([Float(-80.0), 80.0, -80.0])
        let metrics = LogitDrift.metrics(
            logP: LogitDrift.normalizedLogProbabilities(sharpA),
            logQ: LogitDrift.normalizedLogProbabilities(sharpB)
        )
        #expect(abs(metrics.meanFisherRao - Float.pi) < 1e-5)
        #expect(metrics.klDivergence > 100)

        let degeneratePairs: [(MLXArray, MLXArray)] = [
            (MLXArray([Float.nan, Float.nan]), MLXArray([Float(0.0), 1.0])),
            (MLXArray([-Float.infinity, -Float.infinity]), MLXArray([Float.infinity, 0.0])),
            (MLXArray([Float(1000.0), 1000.0]), MLXArray([Float(-1000.0), -1000.0])),
        ]
        for (lhs, rhs) in degeneratePairs {
            let value = LogitDrift.fisherRaoFromLogits(lhs, rhs).item(Float.self)
            #expect(!value.isNaN)
            #expect(value >= 0 && value <= Float.pi)
        }
    }

    @Test("Small perturbations down to 1e-5 match sqrt(2 * KL) without acos Float32 underflow")
    func smallPerturbationMatchesFisherInformation() {
        for delta in [1e-4, 1e-5] {
            let pDouble = [0.5, 0.5]
            let qDouble = [0.5 + delta, 0.5 - delta]
            let logPF32 = pDouble.map { Float(Foundation.log($0)) }
            let logQF32 = qDouble.map { Float(Foundation.log($0)) }

            let dFR = Double(LogitDrift.fisherRao(logP: MLXArray(logPF32), logQ: MLXArray(logQF32)).item(Float.self))
            #expect(dFR > 0)

            let realizedP = logPF32.map { Foundation.exp(Double($0)) }
            let realizedQ = logQF32.map { Foundation.exp(Double($0)) }
            let pSum = realizedP.reduce(0, +)
            let qSum = realizedQ.reduce(0, +)
            let normP = realizedP.map { $0 / pSum }
            let normQ = realizedQ.map { $0 / qSum }
            let kl = zip(normP, normQ).reduce(0.0) { acc, pair in
                acc + pair.0 * (Foundation.log(pair.0) - Foundation.log(pair.1))
            }
            let localExpected = Foundation.sqrt(2.0 * kl)
            let ratio = dFR / localExpected
            #expect(abs(ratio - 1.0) < 0.03)
        }
    }
}

@Suite("Geodesic residual steering norm preservation, MCP wiring and TinyModel X-ray integration (#361)", .serialized)
struct GeodesicSteeringTests {
    @Test("Geodesic steering preserves L2 norm across scales [1e-2, 1e4], angles and collinear +/- directions")
    func normInvarianceAndCollinearFallback() {
        let baseDirection = [Float(1.0), -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75]
        let steerVector = MLXArray([Float(-0.5), 1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5])
        let baseUnitNorm = Foundation.sqrt(baseDirection.reduce(Float(0)) { $0 + $1 * $1 })
        let targetNorms: [Float] = [1e-2, 1.0, 42.0, 1e4]
        let angles: [Float] = [-2.4, -0.7, 0.0, 0.35, 1.25, Float.pi / 2, Float.pi]

        for targetNorm in targetNorms {
            let scaled = baseDirection.map { $0 * (targetNorm / baseUnitNorm) }
            let hidden = MLXArray(scaled).reshaped(1, 1, scaled.count)
            for angle in angles {
                let steered = GeodesicSteer.apply(hidden: hidden, direction: steerVector, scale: angle, geometry: .geodesic)
                let norm = MLX.sqrt(MLX.square(steered).sum()).item(Float.self)
                #expect(norm.isFinite)
                #expect(abs(norm - targetNorm) <= max(1e-5, 1e-5 * targetNorm))
            }
        }

        let collinearVectors: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            baseDirection,
        ]
        for vec in collinearVectors {
            let h = MLXArray(vec)
            let baseNorm = MLX.sqrt(MLX.square(h).sum()).item(Float.self)
            for sign in [Float(1.0), Float(-1.0)] {
                let u = MLXArray(vec.map { $0 * sign })
                let quarter = GeodesicSteer.apply(hidden: h, direction: u, scale: Float.pi / 2, geometry: .geodesic)
                let half = GeodesicSteer.apply(hidden: h, direction: u, scale: Float.pi, geometry: .geodesic)
                let quarterNorm = MLX.sqrt(MLX.square(quarter).sum()).item(Float.self)
                let halfValues = half.asArray(Float.self)
                #expect(quarterNorm.isFinite)
                #expect(abs(quarterNorm - baseNorm) <= max(1e-5, 1e-5 * baseNorm))
                #expect((quarter * h).sum().item(Float.self).magnitude < 1e-4 * baseNorm * baseNorm)
                for (got, orig) in zip(halfValues, vec) {
                    #expect(abs(got + orig) <= max(1e-5, 1e-5 * baseNorm))
                }
            }
        }
    }

    @Test("MCP intervene and steer_search parse geometry, expose schemas and reject unknown geometries")
    func mcpGeometryParsingAndSchemas() throws {
        let defaultSteer = try Intervention(mcp: ["kind": "steer", "layer": 1, "vector": [1.0, 0.0]])
        #expect(defaultSteer == .steer(layer: 1, vector: [1, 0], scale: 1, geometry: .additive))

        let geodesicSteer = try Intervention(mcp: ["kind": "steer", "layer": 2, "vector": [0.0, 1.0], "scale": 0.5, "geometry": "geodesic"])
        #expect(geodesicSteer == .steer(layer: 2, vector: [0, 1], scale: 0.5, geometry: .geodesic))

        let searchArgs: [String: Any] = ["positive": ["a", "b"], "negative": ["c", "d"], "heldOut": ["e"], "geometry": "geodesic"]
        let searchRequest = try ContrastRequest(mcp: searchArgs)
        #expect(searchRequest.geometry == .geodesic)

        for tool in [ResearchTool.intervene, ResearchTool.steerSearch] {
            let schema = try #require(tool.schema.rawValue as? [String: Any])
            let properties = try #require(schema["properties"] as? [String: Any])
            #expect(properties["geometry"] != nil)
        }

        let badIntervene = #expect(throws: MCPFailure.self) {
            try ResearchCall(.intervene, ["kind": "steer", "layer": 0, "vector": [1.0], "geometry": "hyperbolic"])
        }
        #expect(badIntervene?.code == -32602)
        #expect(badIntervene?.message.contains("geometry") == true)

        let badSearch = #expect(throws: MCPFailure.self) {
            try ResearchCall(.steerSearch, ["positive": ["a", "b"], "negative": ["c", "d"], "heldOut": ["e"], "geometry": 42])
        }
        #expect(badSearch?.code == -32602)
        #expect(badSearch?.message.contains("geometry") == true)
    }

    @Test("TinyModel X-ray geodesic intervene, experiment_compare, steer_search and gate_check report Fisher-Rao and T-norm")
    func tinyModelEndToEnd() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        await engine.skipJournalForDeltaTests()
        let hiddenSize = try await engine.anatomy().hiddenSize
        let prompts = ["The quick brown fox", "17 + 25 ="]

        let baselineCompare = try await engine.compare(CompareRequest(prompts: prompts, overlays: [], maxTokens: 4))
        #expect(baselineCompare.rows.allSatisfy { $0.fisherRaoDistance == 0 && $0.maxFisherRaoDistance == 0 })

        let vector = (0 ..< hiddenSize).map { Float(($0 % 5) - 2) }
        let overlay = try await engine.intervene(.steer(layer: 1, vector: vector, scale: 0.9, geometry: .geodesic))
        #expect(overlay.description.contains("geodesic"))

        let steeredCompare = try await engine.compare(CompareRequest(prompts: prompts, overlays: [overlay.id], maxTokens: 4))
        for row in steeredCompare.rows {
            #expect(row.fisherRaoDistance > 0 && row.fisherRaoDistance <= Float.pi)
            #expect(row.maxFisherRaoDistance >= row.fisherRaoDistance && row.maxFisherRaoDistance <= Float.pi)
        }
        try await engine.undo(overlay.id)

        var contrast = ContrastRequest(positive: ["aaaa aaaa", "aa aaaa a"], negative: ["zzzz zzzz", "zz zzzz z"], heldOut: ["the cat"], geometry: .geodesic)
        contrast.topHeads = 0
        contrast.scales = [0.4, 1.2]
        let contrastReport = try await engine.contrast(contrast)
        #expect(contrastReport.trials.allSatisfy { $0.fisherRaoDistance > 0 && $0.fisherRaoDistance <= Float.pi })

        let gateReport = try await engine.gate(GateRequest(heldOut: prompts))
        #expect(gateReport.fisherRaoDistance == 0)
        #expect(gateReport.tNorm?.checkCount == gateReport.checks.count)
    }
}
