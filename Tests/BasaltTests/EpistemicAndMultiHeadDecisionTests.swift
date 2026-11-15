import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI

actor EpistemicStubDecisionEngine: DecisionEngine {
    nonisolated let modelID = "org/epistemic-stub"
    private let noulScores: [String: Double]
    private(set) var requests: [SystemOneRequest] = []

    init(_ noulScores: [String: Double]) {
        self.noulScores = noulScores
    }

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        requests.append(request)
        var answers: [String: SystemOneAnswer] = [:]
        for (id, question) in request.questions {
            switch question.type {
            case .noul:
                let probability = EvidencePair.clamp(noulScores[id] ?? 0.5)
                var answer = SystemOneAnswer()
                answer.noul = probability
                answer.confidence = abs(2 * probability - 1)
                answers[id] = SystemOneCore.gated(answer, abstainBelow: request.abstainBelow, minMargin: request.minMargin)
            case .choice, .score:
                let count = SystemOneCore.options(question).keys.count
                let scores = (0 ..< count).map { Double(count - $0) * 3.0 }
                answers[id] = SystemOneCore.gated(
                    SystemOneCore.answer(question, scores: scores, temperature: 1.0),
                    abstainBelow: request.abstainBelow,
                    minMargin: request.minMargin
                )
            }
        }
        let usage = SystemOneUsage(prefixTokens: 12, candidateBranches: request.questions.count, prefixEvaluations: 1, branchForwardCalls: 0, suffixTokens: 0)
        return SystemOneResponse(model: modelID, answers: answers, usage: usage, calibrated: true)
    }
}

struct StubAnswerTarget: AnswerTarget {
    let modelName: String
    let replyText: String

    func answer(_ request: GenerationRequest) async throws(BasaltError) -> AnswerReply {
        AnswerReply(text: replyText, model: modelName, tokens: 6)
    }
}

@Suite("Epistemic algebra: FOUR bilattice laws, EvidencePair clamping, and verdict projection (#452)")
struct EpistemicAlgebraTests {
    @Test("Belnap FOUR satisfies commutativity, associativity, idempotency, absorption, De Morgan, and order laws")
    func bilatticeLaws() {
        let all = BelnapState.allCases
        for a in all {
            #expect(a.negated.negated == a)
            #expect(a.truthJoin(a) == a && a.truthMeet(a) == a)
            #expect(a.knowledgeJoin(a) == a && a.knowledgeMeet(a) == a)
            #expect(BelnapState.false.truthPrecedesOrEquals(a) && a.truthPrecedesOrEquals(.true))
            #expect(BelnapState.neither.knowledgePrecedesOrEquals(a) && a.knowledgePrecedesOrEquals(.both))
            for b in all {
                checkBinaryLaws(a, b)
                for c in all {
                    #expect(a.truthJoin(b).truthJoin(c) == a.truthJoin(b.truthJoin(c)))
                    #expect(a.truthMeet(b).truthMeet(c) == a.truthMeet(b.truthMeet(c)))
                    #expect(a.knowledgeJoin(b).knowledgeJoin(c) == a.knowledgeJoin(b.knowledgeJoin(c)))
                    #expect(a.knowledgeMeet(b).knowledgeMeet(c) == a.knowledgeMeet(b.knowledgeMeet(c)))
                }
            }
        }
    }

    func checkBinaryLaws(_ a: BelnapState, _ b: BelnapState) {
        #expect(a.truthJoin(b) == b.truthJoin(a))
        #expect(a.truthMeet(b) == b.truthMeet(a))
        #expect(a.knowledgeJoin(b) == b.knowledgeJoin(a))
        #expect(a.knowledgeMeet(b) == b.knowledgeMeet(a))
        #expect(a.truthJoin(a.truthMeet(b)) == a && a.truthMeet(a.truthJoin(b)) == a)
        #expect(a.knowledgeJoin(a.knowledgeMeet(b)) == a && a.knowledgeMeet(a.knowledgeJoin(b)) == a)
        #expect(a.truthJoin(b).negated == a.negated.truthMeet(b.negated))
        #expect(a.truthMeet(b).negated == a.negated.truthJoin(b.negated))
        #expect(a.knowledgeJoin(b).negated == a.negated.knowledgeJoin(b.negated))
        #expect(a.knowledgeMeet(b).negated == a.negated.knowledgeMeet(b.negated))
        #expect(a.truthPrecedesOrEquals(b) == (a.truthJoin(b) == b))
        #expect(a.knowledgePrecedesOrEquals(b) == (a.knowledgeJoin(b) == b))
    }

    @Test("EvidencePair clamps out-of-range and non-finite scores on init and JSON decode")
    func evidenceClampingAndCodable() throws {
        let clamped = EvidencePair(support: 1.7, refutation: -0.4)
        #expect(clamped.support == 1.0 && clamped.refutation == 0.0)
        #expect(EvidencePair(support: .nan, refutation: .infinity) == .none)
        let raw = Data(#"{"support":2.5,"refutation":-3.0}"#.utf8)
        let decoded = try JSONDecoder().decode(EvidencePair.self, from: raw)
        #expect(decoded == EvidencePair(support: 1.0, refutation: 0.0))
        let lhs = EvidencePair(support: 0.8, refutation: 0.2)
        let rhs = EvidencePair(support: 0.3, refutation: 0.9)
        #expect(lhs.knowledgeJoin(rhs) == EvidencePair(support: 0.8, refutation: 0.9))
        #expect(lhs.truthJoin(rhs) == EvidencePair(support: 0.8, refutation: 0.2))
        #expect(lhs.negated == EvidencePair(support: 0.2, refutation: 0.8))
    }

    @Test("EvidencePair projects through BelnapState into EpistemicVerdict with and without tools")
    func verdictProjection() {
        #expect(EvidencePair(support: 0.1, refutation: 0.2).belnap() == .neither)
        #expect(EvidencePair(support: 0.7, refutation: 0.2).belnap() == .true)
        #expect(EvidencePair(support: 0.2, refutation: 0.8).belnap() == .false)
        #expect(EvidencePair(support: 0.9, refutation: 0.9).belnap() == .both)
        #expect(EpistemicVerdict.project(state: .neither, needsToolScore: 0.9, hasTools: false) == .noEvidence)
        #expect(EpistemicVerdict.project(state: .neither, needsToolScore: 0.9, hasTools: true) == .needsTool)
        #expect(EpistemicVerdict.project(state: .true, needsToolScore: 0.1, hasTools: true) == .known)
        #expect(EpistemicVerdict.project(state: .true, needsToolScore: 0.8, hasTools: true) == .needsTool)
        #expect(EpistemicVerdict.project(state: .both, needsToolScore: 0.9, hasTools: true) == .conflict)
        #expect(EpistemicVerdict.project(state: .false, needsToolScore: 0.1, hasTools: false) == .conflict)
    }
}

@Suite("Pre-generation epistemic gate: zero-token abstention, conflict policy, escalation, and compose schema (#452)")
struct EpistemicPipelineGateTests {
    @Test("No-evidence with abstain short-circuits System 2 with zero decode calls and records epistemic receipt")
    func zeroTokenAbstain() async throws {
        let (engine, chat) = try await PipelineAPITests.chat()
        let decider = EpistemicStubDecisionEngine([
            EpistemicGateCore.supportID: 0.12,
            EpistemicGateCore.refutationID: 0.08,
            EpistemicGateCore.needsToolID: 0.15,
        ])
        var stages = PipelineStages()
        stages.epistemic = EpistemicGateSpec(onNoEvidence: .abstain).withAbstainMessage("Cannot verify from context.")
        let response = try await PipelineAPITests.post(
            chat,
            decisions: decider,
            stages: stages,
            ["text": "The memo is blank.", "task": "What was the Q3 revenue?", "questions": PipelineAPITests.questions]
        )
        #expect(response.status == .ok)
        #expect(await engine.getRequests().isEmpty)
        let receipt = try #require(response.json["receipt"] as? [String: Any])
        #expect(receipt["system2_calls"] as? Int == 0)
        #expect(PipelineAPITests.stages(response.json).map { $0["stage"] as? String } == ["decide"])
        let epistemic = try #require(receipt["epistemic"] as? [String: Any])
        #expect(epistemic["verdict"] as? String == "noEvidence")
        #expect(epistemic["state"] as? String == "neither")
        #expect(epistemic["action"] as? String == "abstain")
        #expect((response.json["system2"] as? [String: Any])?["text"] as? String == "Cannot verify from context.")
    }

    @Test("Conflict reports structured contradiction with zero System 2 calls or escalates when configured")
    func conflictReportAndEscalate() async throws {
        let (engine, chat) = try await PipelineAPITests.chat { _ in "Escalated synthesis." }
        let decider = EpistemicStubDecisionEngine([
            EpistemicGateCore.supportID: 0.85,
            EpistemicGateCore.refutationID: 0.91,
            EpistemicGateCore.needsToolID: 0.10,
        ])
        var reportStages = PipelineStages()
        reportStages.epistemic = EpistemicGateSpec(onConflict: .report)
        let reported = try await PipelineAPITests.post(
            chat,
            decisions: decider,
            stages: reportStages,
            ["text": "Page 1 says open; page 2 says closed.", "task": "Is the valve open?"]
        )
        let reportedRequests = await engine.getRequests()
        #expect(reported.status == .ok && reportedRequests.isEmpty)
        let reportText = (reported.json["system2"] as? [String: Any])?["text"] as? String ?? ""
        #expect(reportText.contains("support=0.85, refutation=0.91"))
        var escalateStages = PipelineStages()
        escalateStages.epistemic = EpistemicGateSpec(onConflict: .escalate)
        let escalated = try await PipelineAPITests.post(
            chat,
            decisions: decider,
            stages: escalateStages,
            ["text": "Page 1 says open; page 2 says closed.", "task": "Reconcile the discrepancy."]
        )
        let escalatedRequests = await engine.getRequests()
        #expect(escalated.status == .ok && escalatedRequests.count == 1)
        let receipt = try #require(escalated.json["receipt"] as? [String: Any])
        #expect(receipt["escalated"] as? Bool == true)
        #expect((receipt["epistemic"] as? [String: Any])?["action"] as? String == "escalate")
        #expect((escalated.json["system2"] as? [String: Any])?["text"] as? String == "Escalated synthesis.")
    }

    @Test("Known and needsTool verdicts proceed to System 2 and pass epistemic_verdict into situation")
    func knownAndNeedsToolProceed() async throws {
        let (_, chat) = try await PipelineAPITests.chat { _ in "Direct answer." }
        let decider = EpistemicStubDecisionEngine([
            EpistemicGateCore.supportID: 0.20,
            EpistemicGateCore.refutationID: 0.05,
            EpistemicGateCore.needsToolID: 0.92,
        ])
        let toolTarget = StubAnswerTarget(modelName: "org/tool-specialist", replyText: "Routed via tool specialist.")
        let fallbackTarget = StubAnswerTarget(modelName: "org/default-chat", replyText: "Fallback reply.")
        var stages = PipelineStages()
        stages.epistemic = EpistemicGateSpec(toolThreshold: 0.6)
        stages.hasTools = true
        stages.answers = AnswerRouter(
            routes: [AnswerRoute(when: .init(decide: ["epistemic_verdict": "needsTool"]), model: "tool")],
            targets: ["tool": toolTarget, "default": fallbackTarget],
            fallback: fallbackTarget
        )
        let response = try await PipelineAPITests.post(
            chat,
            decisions: decider,
            stages: stages,
            ["text": "User asks for live weather.", "task": "Fetch current temperature."]
        )
        #expect(response.status == .ok)
        let receipt = try #require(response.json["receipt"] as? [String: Any])
        let epistemic = try #require(receipt["epistemic"] as? [String: Any])
        #expect(epistemic["verdict"] as? String == "needsTool" && epistemic["action"] as? String == "routeTool")
        #expect((response.json["system2"] as? [String: Any])?["model"] as? String == "org/tool-specialist")
        #expect((response.json["system2"] as? [String: Any])?["text"] as? String == "Routed via tool specialist.")
    }

    @Test("Compose v2 parses decide.epistemic and v1 refuses it")
    func composeEpistemicSchema() throws {
        let yamlV2 = """
            version: 2
            name: gated-system
            models:
              chat:
                source: mlx-community/Qwen3-4B-4bit
              bank:
                source: /tmp/bank-bundle
            pipeline:
              decide:
                cascade: [bank]
                epistemic:
                  enabled: true
                  support_threshold: 0.6
                  refutation_threshold: 0.4
                  tool_threshold: 0.7
                  on_no_evidence: abstain
                  on_conflict: escalate
                  abstain_message: Not enough context.
              answer: chat
            """
        let parsed = try ComposeFile.parse(yamlV2, file: "v2.yaml")
        let gate = try #require(parsed.epistemic)
        #expect(gate.enabled && gate.supportThreshold == 0.6 && gate.refutationThreshold == 0.4 && gate.toolThreshold == 0.7)
        #expect(gate.onNoEvidence == .abstain && gate.onConflict == .escalate && gate.abstainMessage == "Not enough context.")
        let yamlV1 = yamlV2.replacingOccurrences(of: "version: 2", with: "version: 1")
        #expect(throws: ServeConfigError.self) {
            _ = try ComposeFile.parse(yamlV1, file: "v1.yaml")
        }
    }
}

@Suite("System One bank core: Span<Float> slicing for noul, choice, score64, and hybrid partitioning (#455)")
struct SystemOneBankCoreTests {
    @Test("Manifest validates version, slice bounds, distinct choice keys, and 64-bin score slots")
    func manifestValidation() throws {
        let valid = SystemOneBankManifest(
            baseModelID: "self",
            hiddenSize: 64,
            totalOutputs: 68,
            slots: [
                "flag": BankSlot(kind: .noul, slice: 0 ..< 1),
                "pick": BankSlot(kind: .choice, slice: 1 ..< 4, options: ["a", "b", "c"]),
                "rate": BankSlot(kind: .score64, slice: 4 ..< 68),
            ]
        )
        try valid.validate(path: "valid")
        var badVersion = valid
        badVersion.schemaVersion = 99
        #expect(throws: BasaltError.self) { try badVersion.validate(path: "v") }
        let outOfBounds = SystemOneBankManifest(
            baseModelID: "self", hiddenSize: 64, totalOutputs: 4, slots: ["q": BankSlot(kind: .score64, slice: 0 ..< 64)]
        )
        #expect(throws: BasaltError.self) { try outOfBounds.validate(path: "oob") }
        let dupChoice = SystemOneBankManifest(
            baseModelID: "self", hiddenSize: 64, totalOutputs: 2, slots: ["q": BankSlot(kind: .choice, slice: 0 ..< 2, options: ["x", "x"])]
        )
        #expect(throws: BasaltError.self) { try dupChoice.validate(path: "dup") }
    }

    @Test("Span<Float> decodes 1-logit and 2-logit noul with temperature and bias, plus choice tie-breaking")
    func noulAndChoiceDecoding() {
        let manifest = SystemOneBankManifest(
            baseModelID: "self",
            hiddenSize: 8,
            totalOutputs: 5,
            slots: [
                "single": BankSlot(kind: .noul, slice: 0 ..< 1, temperature: 2.0, bias: 0.5),
                "pair": BankSlot(kind: .noul, slice: 1 ..< 3, temperature: 1.0, bias: 0.0),
                "pick": BankSlot(kind: .choice, slice: 3 ..< 5, options: ["z_key", "a_key"], temperature: 1.0),
            ]
        )
        let request = SystemOneRequest(
            state: "evidence",
            questions: [
                "single": SystemOneQuestion(type: .noul, instructions: "Single?"),
                "pair": SystemOneQuestion(type: .noul, instructions: "Pair?"),
                "pick": SystemOneQuestion(type: .choice, instructions: "Pick", criteria: .named(["z_key": "Alpha", "a_key": "Zeta"])),
            ]
        )
        let covered = SystemOneBankCore.partition(request.questions, manifest: manifest).covered
        let logits: [Float] = [2.0, 1.0, 1.0 + Float(Foundation::log(3.0)), 4.0, 4.0]
        let decoded = SystemOneBankCore.decode(covered, logits: logits.span, request: request)
        let expectedSingle = 1.0 / (1.0 + Foundation::exp(-(2.0 / 2.0 + 0.5)))
        #expect(abs((decoded["single"]?.noul ?? 0) - expectedSingle) < 1e-6)
        #expect(abs((decoded["pair"]?.noul ?? 0) - 0.75) < 1e-5)
        #expect(decoded["pick"]?.choice == "z_key")
        #expect(abs(decoded["pick"]?.confidence ?? 1) < 1e-6)
    }

    @Test("Score64 expectation recovers ordinal targets across [0, 10], computes entropy confidence, and buckets levels")
    func score64Decoding() {
        let slot = BankSlot(kind: .score64, slice: 0 ..< BankSlot.scoreBins)
        let question = SystemOneQuestion(type: .score, instructions: "Quality", criteria: .levels(["poor", "fair", "great"]))
        for target in [0.0, 2.5, 5.0, 7.75, 10.0] {
            let probs = MultiHeadBankTrainer.score64Target(score: target, sigma: 0.15)
            let logits = probs.map { Float(Foundation::log(max(Double($0), 1e-12))) }
            let answer = SystemOneBankCore.decodeSlot(slot, question: question, slice: logits.span, calibrated: true)
            #expect(abs((answer.score ?? -1) - target) < 0.08)
            #expect((answer.confidence ?? 0) > 0.5)
            let bucketSum = (answer.probabilities ?? [:]).values.reduce(0, +)
            #expect(abs(bucketSum - 1.0) < 1e-6 && answer.legend == ["poor", "fair", "great"])
        }
        let uniform = [Float](repeating: 0, count: BankSlot.scoreBins)
        let flatAnswer = SystemOneBankCore.decodeSlot(slot, question: question, slice: uniform.span, calibrated: true)
        #expect(abs((flatAnswer.score ?? 0) - 5.0) < 1e-6 && (flatAnswer.confidence ?? 1) < 1e-6)
    }

    @Test("Partition separates covered bank slots from unknown IDs and mismatched choice option sets")
    func hybridPartition() {
        let manifest = SystemOneBankManifest(
            baseModelID: "self",
            hiddenSize: 16,
            totalOutputs: 3,
            slots: [
                "known_noul": BankSlot(kind: .noul, slice: 0 ..< 1),
                "known_choice": BankSlot(kind: .choice, slice: 1 ..< 3, options: ["left", "right"]),
            ]
        )
        let questions: [String: SystemOneQuestion] = [
            "known_noul": SystemOneQuestion(type: .noul, instructions: "ok?"),
            "known_choice": SystemOneQuestion(type: .choice, instructions: "dir?", criteria: .named(["left": "L", "right": "R", "up": "U"])),
            "ad_hoc": SystemOneQuestion(type: .noul, instructions: "dynamic?"),
        ]
        let split = SystemOneBankCore.partition(questions, manifest: manifest)
        #expect(split.covered.map(\.id) == ["known_noul"])
        #expect(Set(split.uncovered.keys) == ["known_choice", "ad_hoc"])
    }
}

@Suite("MultiHeadDecisionEngine & MultiHeadBankTrainer: TinyModel parity, hybrid fallback, bundle IO, and training (#455)", .serialized)
struct MultiHeadDecisionEngineTests {
    static func fixtureBank(hiddenSize: Int, baseModelID: String) -> (manifest: SystemOneBankManifest, weight: [Float], bias: [Float]) {
        let manifest = SystemOneBankManifest(
            baseModelID: baseModelID,
            hiddenSize: hiddenSize,
            totalOutputs: 70,
            slots: [
                EpistemicGateCore.supportID: BankSlot(kind: .noul, slice: 0 ..< 1, temperature: 1.1, bias: 0.1),
                EpistemicGateCore.refutationID: BankSlot(kind: .noul, slice: 1 ..< 2, temperature: 0.9, bias: -0.2),
                "route": BankSlot(kind: .choice, slice: 2 ..< 5, options: ["fast", "deep", "tool"], temperature: 0.8),
                "quality": BankSlot(kind: .score64, slice: 5 ..< 69, temperature: 1.0),
                EpistemicGateCore.needsToolID: BankSlot(kind: .noul, slice: 69 ..< 70),
            ],
            modelID: "bank-fixture"
        )
        var rng = SplitMix64(seed: 455)
        let weight = (0 ..< 70 * hiddenSize).map { _ in Float(Int64(bitPattern: rng.next() % 200) - 100) / 250.0 }
        let bias = (0 ..< 70).map { index in Float(index % 7 - 3) * 0.1 }
        return (manifest, weight, bias)
    }

    static var bankQuestions: [String: SystemOneQuestion] {
        var questions = EpistemicGateCore.probes(includeTools: true)
        questions["route"] = SystemOneQuestion(
            type: .choice,
            instructions: "Select execution tier.",
            criteria: .named(["fast": "Fast path", "deep": "Deep reasoning", "tool": "External tool"])
        )
        questions["quality"] = SystemOneQuestion(
            type: .score,
            instructions: "Context clarity.",
            criteria: .levels(["poor", "fair", "great"])
        )
        return questions
    }

    @Test(
        "Single-matmul bank matches explicit MLX matmul + bias with zero branch forwards on Qwen3 and Gemma4",
        arguments: [TinyModel.Family.qwen3, .gemma4]
    )
    func tinyModelParity(family: TinyModel.Family) async throws {
        let container = try TinyModel.container(family)
        let bank = Self.fixtureBank(hiddenSize: 64, baseModelID: "tiny-\(family.rawValue)")
        let engine = try MultiHeadDecisionEngine(manifest: bank.manifest, weight: bank.weight, bias: bank.bias, container: container)
        let request = SystemOneRequest(state: "Quarterly revenue grew 18% to $42M with zero debt.", questions: Self.bankQuestions)
        let actual = try await engine.decide(request)
        #expect(actual.usage.branchForwardCalls == 0 && actual.usage.suffixTokens == 0 && actual.usage.prefixEvaluations == 1)
        let (manifest, weight, bias) = (bank.manifest, bank.weight, bank.bias)
        let rawReference: [Float] = try await container.perform { context in
            let ids = MultiHeadDecisionEngine.encodeState(request.state, tokenizer: context.tokenizer)
            let hidden = try MultiHeadDecisionEngine.lastHidden(context.model, tokens: ids)
            let wMat = MLXArray(weight, [manifest.totalOutputs, manifest.hiddenSize])
            let logits = (matmul(hidden, wMat.T) + MLXArray(bias, [manifest.totalOutputs])).asType(.float32)
            eval(logits)
            return logits.asArray(Float.self)
        }
        let covered = SystemOneBankCore.partition(request.questions, manifest: bank.manifest).covered
        let expected = SystemOneBankCore.decode(covered, logits: rawReference.span, request: request)
        for key in Self.bankQuestions.keys {
            let lhs = try #require(actual.answers[key])
            let rhs = try #require(expected[key])
            #expect(lhs.choice == rhs.choice)
            #expect(abs((lhs.noul ?? 0) - (rhs.noul ?? 0)) < 1e-5)
            #expect(abs((lhs.score ?? 0) - (rhs.score ?? 0)) < 1e-4)
        }
    }

    @Test("BASALT_BENCH=1 micro-benchmark comparing MultiHeadDecisionEngine vs 5-branch ReadoutDecisionEngine")
    func envGatedBankVsReadoutBenchmark() async throws {
        guard ProcessInfo.processInfo.environment["BASALT_BENCH"] == "1" else { return }
        for family in [TinyModel.Family.qwen3, .gemma4] {
            let mlxEngine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
            let container = ModelContainer(context: ModelContext(
                configuration: ModelConfiguration(id: "tiny-\(family.rawValue)"),
                model: try TinyModel.make(family),
                processor: ByteInputProcessor(),
                tokenizer: ASCIITokenizer()
            ))
            await mlxEngine.attach(container)
            let bank = Self.fixtureBank(hiddenSize: 64, baseModelID: "tiny-\(family.rawValue)")
            let bankEngine = try MultiHeadDecisionEngine(manifest: bank.manifest, weight: bank.weight, bias: bank.bias, container: container)
            let readoutEngine = ReadoutDecisionEngine(engine: mlxEngine, modelID: "tiny-\(family.rawValue)", readout: .uncalibrated)
            let request = SystemOneRequest(state: "Quarterly revenue grew 18% to $42M with zero debt and audited margins.", questions: Self.bankQuestions)
            for _ in 0 ..< 3 {
                _ = try await bankEngine.decide(request)
                _ = try await readoutEngine.decide(request)
            }
            let clock = ContinuousClock()
            var bankSamples: [Double] = []
            var readoutSamples: [Double] = []
            for _ in 0 ..< 9 {
                let dBank = try await clock.measure { _ = try await bankEngine.decide(request) }
                let dRead = try await clock.measure { _ = try await readoutEngine.decide(request) }
                bankSamples.append(Double(dBank.components.seconds) * 1e3 + Double(dBank.components.attoseconds) * 1e-15)
                readoutSamples.append(Double(dRead.components.seconds) * 1e3 + Double(dRead.components.attoseconds) * 1e-15)
            }
            bankSamples.sort()
            readoutSamples.sort()
            let (bMs, rMs) = (bankSamples[4], readoutSamples[4])
            print(String(format: "[BankBench] %@ bank=%.2fms readout=%.2fms speedup=%.2fx",
                         family.rawValue, bMs, rMs, rMs / max(bMs, 1e-6)))
            #expect(bMs > 0 && rMs > 0)
        }
    }

    @Test("Hybrid request answers covered slots in one matmul and delegates uncovered slots to fallback")
    func hybridFallbackAndBundleRoundtrip() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "bank-bundle-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let container = try TinyModel.container(.qwen3)
        let bank = Self.fixtureBank(hiddenSize: 64, baseModelID: "tiny-qwen3")
        try SystemOneBankBundle.save(manifest: bank.manifest, weight: bank.weight, bias: bank.bias, to: dir)
        #expect(MultiHeadDecisionEngine.isBundle(dir))
        #expect(try MultiHeadDecisionEngine.baseModel(of: dir) == "tiny-qwen3")
        let fallback = EpistemicStubDecisionEngine(["dynamic_flag": 0.93])
        let engine = try MultiHeadDecisionEngine(directory: dir, container: container, fallback: fallback)
        var mixed = Self.bankQuestions
        mixed["dynamic_flag"] = SystemOneQuestion(type: .noul, instructions: "Ad-hoc runtime check.")
        let response = try await engine.decide(SystemOneRequest(state: "System nominal.", questions: mixed))
        #expect(response.answers.count == mixed.count)
        #expect(abs((response.answers["dynamic_flag"]?.noul ?? 0) - 0.93) < 1e-6)
        #expect(response.usage.stages?.count == 2)
        let fallbackSeen = await fallback.requests
        #expect(fallbackSeen.count == 1 && Array(fallbackSeen[0].questions.keys) == ["dynamic_flag"])
    }

    @Test("MultiHeadBankTrainer converges on synthetic [N, D] states and saved bundle reproduces teacher targets")
    func trainerConvergence() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "bank-trained-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let (manifest, dataset, prototypes) = Self.syntheticTrainingData()
        let config = MultiHeadBankTrainConfig(iterations: 200, batchSize: 24, learningRate: 0.08, weightDecay: 1e-4, brierWeight: 0.1, seed: 11)
        let report = try MultiHeadBankTrainer.train(manifest: manifest, dataset: dataset, config: config, saveTo: dir)
        #expect(report.finalLoss < report.initialLoss * 0.25)
        #expect(SystemOneBankBundle.isBundle(dir))
        let loaded = try SystemOneBankBundle.loadHead(from: dir, manifest: manifest)
        let bias = try #require(loaded.bias)
        let questions: [String: SystemOneQuestion] = [
            "flag": SystemOneQuestion(type: .noul, instructions: "Flag"),
            "tier": SystemOneQuestion(type: .choice, instructions: "Tier", criteria: .named(["alpha": "A", "beta": "B", "gamma": "G"])),
            "score": SystemOneQuestion(type: .score, instructions: "Score"),
        ]
        let covered = SystemOneBankCore.partition(questions, manifest: manifest).covered
        let expectedTiers = ["alpha", "beta", "gamma"]
        let expectedScores = [1.5, 5.0, 8.8]
        let expectedFlags = [true, false, true]
        for index in 0 ..< 3 {
            let logitsTensor = matmul(MLXArray(prototypes[index], [16]), MLXArray(loaded.weight, [68, 16]).T) + MLXArray(bias, [68])
            eval(logitsTensor)
            let raw = logitsTensor.asArray(Float.self)
            let decoded = SystemOneBankCore.decode(covered, logits: raw.span, request: SystemOneRequest(state: "s", questions: questions))
            #expect(decoded["tier"]?.choice == expectedTiers[index])
            #expect(abs((decoded["score"]?.score ?? -10) - expectedScores[index]) < 0.6)
            #expect(((decoded["flag"]?.noul ?? 0.5) > 0.5) == expectedFlags[index])
        }
    }

    static func syntheticTrainingData() -> (SystemOneBankManifest, MultiHeadBankDataset, [[Float]]) {
        let manifest = SystemOneBankManifest(
            baseModelID: "self",
            hiddenSize: 16,
            totalOutputs: 68,
            slots: [
                "flag": BankSlot(kind: .noul, slice: 0 ..< 1),
                "tier": BankSlot(kind: .choice, slice: 1 ..< 4, options: ["alpha", "beta", "gamma"]),
                "score": BankSlot(kind: .score64, slice: 4 ..< 68),
            ]
        )
        let prototypes: [[Float]] = (0 ..< 3).map { group in
            (0 ..< 16).map { dim in dim / 5 == group ? Float(1.5) : Float(-0.2) }
        }
        let teacherRows: [[Float]] = [
            [0.96, 0.94, 0.03, 0.03] + MultiHeadBankTrainer.score64Target(score: 1.5, sigma: 0.25),
            [0.04, 0.03, 0.94, 0.03] + MultiHeadBankTrainer.score64Target(score: 5.0, sigma: 0.25),
            [0.92, 0.03, 0.03, 0.94] + MultiHeadBankTrainer.score64Target(score: 8.8, sigma: 0.25),
        ]
        var rng = SplitMix64(seed: 99)
        var hiddenStates: [Float] = []
        var targets: [Float] = []
        for sample in 0 ..< 48 {
            let group = sample % 3
            let noisy = prototypes[group].map { $0 + Float(Int64(bitPattern: rng.next() % 21) - 10) * 0.01 }
            hiddenStates.append(contentsOf: noisy)
            targets.append(contentsOf: teacherRows[group])
        }
        let dataset = MultiHeadBankDataset(sampleCount: 48, hiddenSize: 16, totalOutputs: 68, hiddenStates: hiddenStates, targets: targets)
        return (manifest, dataset, prototypes)
    }
}
