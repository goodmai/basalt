import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltBench
@testable import BasaltCore
@testable import BasaltInference

@Suite("Self-rewriting P1: math & translation graders, paired stats, reserved split, budget, tensor registry, and Gemma 4 session tools (#676, #770)", .serialized)
struct SelfRewritingP1Tests {
    static let textOProj = "model.layers.0.self_attn.o_proj"

    static func encode(_ text: String) -> [Int] {
        ByteTokenizer().encode(text: text, addSpecialTokens: false) + [ByteTokenizer.endOfUserTurn]
    }

    static func decode(_ tokens: [Int]) -> String {
        ByteTokenizer().decode(tokenIds: tokens)
    }

    static func makeProblems(_ count: Int, prefix: String = "p") -> [PairedProblem] {
        (0 ..< count).map { idx in
            PairedProblem(
                id: "\(prefix)-\(idx)",
                prompt: "Compute \(idx) + \(idx) carefully and explain step by step.",
                expectedAnswer: "\(idx * 2)"
            )
        }
    }
}

extension SelfRewritingP1Tests {
    @Test("MathAnswerGrader passes golden table across boxed, nested LaTeX, fractions, decimals, signs, percents, and rejects mismatches")
    func mathAnswerGraderGoldenTable() {
        let grader = MathAnswerGrader()
        let matches: [(response: String, expected: String)] = [
            ("Step 1. Therefore \\boxed{42}.", "42"),
            ("Wrong \\boxed{10}, wait actually \\boxed{\\frac{1}{2}}.", "0.5"),
            ("We get \\boxed{\\dfrac{-3}{4}}", "-0.75"),
            ("Hence \\boxed{-\\tfrac{6}{8}}.", "-3/4"),
            ("Total is \\boxed{1,250}.", "1250"),
            ("Final answer: $x = 17$.", "17"),
            ("Computation:\n#### 50%", "1/2"),
            ("Result: \\boxed{50\\%}", "0.5"),
            ("Thus \\boxed{3\\sqrt{2}}", "3 * \\sqrt{2}"),
            ("Angle is \\boxed{\\frac{\\pi}{4}}", "\\pi/4"),
            ("Negative angle \\boxed{-\\frac{\\pi}{4}}", "-\\pi/4"),
            ("Rationalized \\boxed{\\frac{\\sqrt{2}}{2}}", "1/\\sqrt{2}"),
            ("Value \\boxed{-0.0}", "0"),
            ("Answer: \\left(\\text{42}\\right)", "(42)"),
            ("We find \\fbox{\\mathrm{2.5}}", "5/2"),
            ("The answer is 100.", "100"),
            ("Simply $7/2$", "3.5"),
        ]
        for (idx, pair) in matches.enumerated() {
            let problem = PairedProblem(id: "g-\(idx)", prompt: "q", expectedAnswer: pair.expected)
            let result = grader.grade(response: pair.response, problem: problem)
            #expect(result.passed, "expected match for case \(idx): \(pair.response) vs \(pair.expected) (\(result.detail ?? ""))")
        }

        let mismatches: [(response: String, expected: String)] = [
            ("\\boxed{41}", "42"),
            ("\\boxed{50%}", "50"),
            ("\\boxed{\\frac{1}{3}}", "0.33"),
            ("\\boxed{(1, 2)}", "(1, 3)"),
            ("", "5"),
            ("\\boxed{3\\sqrt{2}}", "3\\sqrt{3}"),
        ]
        for (idx, pair) in mismatches.enumerated() {
            let problem = PairedProblem(id: "m-\(idx)", prompt: "q", expectedAnswer: pair.expected)
            let result = grader.grade(response: pair.response, problem: problem)
            #expect(!result.passed, "expected mismatch for case \(idx): \(pair.response) vs \(pair.expected)")
        }
    }

    @Test("PairedTestStatistics matches hand-computed mean, paired SE, 95% CI, and McNemar counts for k=1 and k=2")
    func pairedStatisticsMatchesHandComputedExample() {
        let k1Samples: [PairedSamplePair] = [
            PairedSamplePair(problemId: "p1", seed: 1, baseGrade: PairedGradeResult(passed: false), editedGrade: PairedGradeResult(passed: true)),
            PairedSamplePair(problemId: "p2", seed: 1, baseGrade: PairedGradeResult(passed: false), editedGrade: PairedGradeResult(passed: true)),
            PairedSamplePair(problemId: "p3", seed: 1, baseGrade: PairedGradeResult(passed: true), editedGrade: PairedGradeResult(passed: true)),
            PairedSamplePair(problemId: "p4", seed: 1, baseGrade: PairedGradeResult(passed: true), editedGrade: PairedGradeResult(passed: false)),
        ]
        let s1 = PairedTestStatistics.summarize(k1Samples)
        #expect(s1.problemCount == 4)
        #expect(abs(s1.baseAccuracy - 0.5) < 1e-9)
        #expect(abs(s1.editedAccuracy - 0.75) < 1e-9)
        #expect(abs(s1.delta - 0.25) < 1e-9)
        let expectedSE = sqrt((0.5625 + 0.5625 + 0.0625 + 1.5625) / 3.0 / 4.0)
        #expect(abs(s1.standardError - expectedSE) < 1e-9)
        #expect(s1.wrongToRight == 2 && s1.rightToWrong == 1)
        #expect(!PairedTestStatistics.seExcludesZero(s1))

        let k2Samples: [PairedSamplePair] = [
            PairedSamplePair(problemId: "a", seed: 1, baseGrade: PairedGradeResult(passed: false), editedGrade: PairedGradeResult(passed: true)),
            PairedSamplePair(problemId: "a", seed: 2, baseGrade: PairedGradeResult(passed: false), editedGrade: PairedGradeResult(passed: true)),
            PairedSamplePair(problemId: "b", seed: 1, baseGrade: PairedGradeResult(passed: false), editedGrade: PairedGradeResult(passed: true)),
            PairedSamplePair(problemId: "b", seed: 2, baseGrade: PairedGradeResult(passed: true), editedGrade: PairedGradeResult(passed: true)),
            PairedSamplePair(problemId: "c", seed: 1, baseGrade: PairedGradeResult(passed: false), editedGrade: PairedGradeResult(passed: true)),
            PairedSamplePair(problemId: "c", seed: 2, baseGrade: PairedGradeResult(passed: false), editedGrade: PairedGradeResult(passed: true)),
        ]
        let s2 = PairedTestStatistics.summarize(k2Samples)
        #expect(s2.problemCount == 3)
        #expect(abs(s2.delta - (5.0 / 6.0)) < 1e-9)
        #expect(PairedTestStatistics.seExcludesZero(s2))
        #expect(PairedTestStatistics.ci95ExcludesZero(s2))
        let ci = PairedTestStatistics.confidenceInterval95(s2)
        #expect(ci.low > 0 && ci.high > ci.low)
        #expect(PairedTestStatistics.summarize([]).problemCount == 0)
    }
}

extension SelfRewritingP1Tests {
    @Test("PairedDatasetSplit enforces reserved split isolation, single reserved evaluation, decontamination, and budget caps")
    func datasetSplitIsolationDecontaminationAndBudget() throws {
        let candidates = [
            PairedProblem(id: "clean-1", prompt: "Find the sum of the first twenty odd positive integers.", expectedAnswer: "400"),
            PairedProblem(id: "exact-dup", prompt: "What is the value of 2023 factorial modulo 2024?", expectedAnswer: "0"),
            PairedProblem(id: "ngram-dup", prompt: "Consider a triangle ABC where the inscribed circle touches sides AB and BC at points D and E respectively.", expectedAnswer: "12"),
            PairedProblem(id: "clean-2", prompt: "Solve for x when three x minus seven equals twenty.", expectedAnswer: "9"),
            PairedProblem(id: "clean-3", prompt: "How many distinct prime factors divide two hundred ten?", expectedAnswer: "4"),
            PairedProblem(id: "clean-4", prompt: "Evaluate the integral of cosine squared from zero to pi.", expectedAnswer: "\\pi/2"),
        ]
        let benchmarks = [
            "What is the value of 2023 factorial modulo 2024?",
            "In AMC 2023: Consider a triangle ABC where the inscribed circle touches sides AB and BC at points D and E.",
        ]
        let (split, removed) = PairedDatasetSplit.decontaminated(
            candidates: candidates,
            benchmarks: benchmarks,
            practiceCount: 2,
            ngramSize: 6,
            seed: 676
        )
        #expect(removed == 2)
        #expect(split.practice.count == 2 && split.reservedCount == 2)

        let reservedId = try #require(candidates.map(\.id).first { id in
            id.hasPrefix("clean-") && !split.practice.contains { $0.id == id }
        })
        let accessErr = #expect(throws: SelfRewriteBudgetError.self) {
            _ = try split.practiceProblem(id: reservedId)
        }
        #expect(accessErr == .reservedSplitAccessDenied(problemId: reservedId))
        #expect(try split.practiceBatch(ids: [split.practice[0].id]).count == 1)

        var budget = try SelfRewriteBudget(maxForwardPasses: 4, maxToolCalls: 2, maxReservedTests: 1, maxEditApplies: 1, maxCloudUSD: 0.50)
        let summary = try split.evaluateReserved(budget: &budget) { problems in
            problems.map { PairedSamplePair(problemId: $0.id, seed: 1, baseGrade: PairedGradeResult(passed: false), editedGrade: PairedGradeResult(passed: true)) }
        }
        #expect(summary.problemCount == 2 && summary.delta == 1.0)
        #expect(throws: SelfRewriteBudgetError.self) {
            _ = try split.evaluateReserved(budget: &budget) { _ in [] }
        }
        try budget.consumeForwardPasses(4)
        #expect(budget.isExhausted)
        #expect(throws: SelfRewriteBudgetError.self) { try budget.consumeForwardPasses(1) }
        try budget.consumeToolCall()
        try budget.consumeToolCall()
        #expect(throws: SelfRewriteBudgetError.self) { try budget.consumeToolCall() }
        try budget.consumeEditApply()
        #expect(throws: SelfRewriteBudgetError.self) { try budget.consumeEditApply() }
        try budget.consumeCloudCost(usd: 0.40)
        #expect(throws: SelfRewriteBudgetError.self) { try budget.consumeCloudCost(usd: 0.20) }
    }

    @Test("TranslationPairedGrader grades chrF++, tech_fidelity, repetition loops, and Belnap SIXTEEN_3 states")
    func translationPairedGraderAndBelnapStates() {
        let grader = TranslationPairedGrader()
        let problem = PairedProblem(
            id: "tr-1",
            prompt: "Translate to Russian.",
            expectedAnswer: "Запустите `basalt serve` и откройте https://example.com для проверки.",
            sourceText: "Run `basalt serve` and open https://example.com to verify."
        )
        let good = grader.evaluate(
            response: "Запустите `basalt serve` и откройте https://example.com для проверки.",
            problem: problem
        )
        #expect(good.passed && good.belnapState == .true && good.score.techFidelity == 100.0 && good.repetitionRatio == 0.0)

        let droppedCode = grader.evaluate(
            response: "Запустите сервер и откройте сайт для проверки.",
            problem: problem
        )
        #expect(!droppedCode.passed && droppedCode.score.techFidelity < 90.0 && (droppedCode.belnapState == .false || droppedCode.belnapState == .both))

        let highChrfBrokenCode = grader.evaluate(
            response: "Запустите basalt serve и откройте https://example.com для проверки.",
            problem: problem
        )
        #expect(!highChrfBrokenCode.passed && highChrfBrokenCode.belnapState == .both)

        let strictGrader = TranslationPairedGrader(chrfThreshold: 30.0, strongChrfThreshold: 85.0)
        let borderline = strictGrader.evaluate(
            response: "Включите `basalt serve`, затем перейдите по ссылке https://example.com.",
            problem: problem
        )
        #expect(borderline.passed && borderline.belnapState == .neither)

        let looping = grader.evaluate(
            response: "Запустите `basalt serve` https://example.com проверка снова проверка снова проверка снова проверка снова проверка снова проверка снова",
            problem: problem
        )
        #expect(!looping.passed && looping.repetitionRatio > grader.maxRepetitionRatio && (looping.belnapState == .false || looping.belnapState == .both))

        let gradeResult = grader.grade(response: "Запустите `basalt serve` и откройте https://example.com для проверки.", problem: problem)
        #expect(gradeResult.passed && gradeResult.score > 0.8 && (gradeResult.detail?.contains("belnap=true") == true))
    }
}

extension SelfRewritingP1Tests {
    @Test("SessionTensorRegistry registers contrast and top-left SVD handles, enforces caps, and feeds WeightCopy.apply directly")
    func sessionTensorRegistryAndWeightCopyIntegration() throws {
        var registry = SessionTensorRegistry(maxEntries: 3, maxTotalBytes: 8192)
        let pos = [MLXArray([Float(2), 0, 0, 0]), MLXArray([Float(4), 0, 0, 0])]
        let neg = [MLXArray([Float(0), 0, 0, 0]), MLXArray([Float(0), 0, 0, 0])]
        let contrast = try registry.registerActivationContrast(key: "probe_u_L0", positive: pos, negative: neg)
        #expect(contrast.shape == [4] && abs(contrast.stats.l2Norm - 1.0) < 1e-5 && contrast.topAbsIndices.first == 0)

        let model = try TinyModel.make(.gemma4)
        let oProj = try #require(model.namedModules().first(where: { $0.0 == Self.textOProj })?.1 as? Linear)
        let svdSummary = try registry.registerTopLeftSingularVector(key: "svd_u_o0", matrix: oProj.weight, iterations: 24)
        #expect(svdSummary.shape == [64] && abs(svdSummary.stats.l2Norm - 1.0) < 1e-4)

        let program = EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "svd_u_o0", alpha: 0.2)])
        let operands = try registry.resolveOperands(for: program)
        var copy = try WeightCopy.makeSync(from: model)
        _ = try copy.apply(program, tensors: operands, verifying: model)
        try copy.release(verifying: model)

        #expect(throws: TensorRegistryError.self) {
            _ = try registry.register(MLXArray([Float.nan, 1.0]), forKey: "bad_nan")
        }
        _ = try registry.register(MLXArray([Float(1), 2]), forKey: "slot_3")
        #expect(throws: TensorRegistryError.self) {
            _ = try registry.register(MLXArray([Float(1), 2]), forKey: "slot_4")
        }
        registry.remove(key: "slot_3")
        #expect(registry.summary(forKey: "slot_3") == nil && registry.keys.count == 2)
        #expect(throws: TensorRegistryError.self) {
            _ = try registry.resolveOperands(for: EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "missing", alpha: 0.1)]))
        }
        registry.clear()
        #expect(registry.keys.isEmpty && registry.totalBytes == 0)
    }

    @Test("Gemma4SessionInspector reports topology blast-radius warnings, tensor stats, and MoE router histograms on Gemma 4 fixtures")
    func gemma4SessionInspectorTopologyAndMoERouterSummary() throws {
        let gemma4 = try TinyModel.make(.gemma4)
        let configObj = try #require(JSONSerialization.jsonObject(with: Data(TinyModel.Family.gemma4.config.utf8)) as? [String: Any])
        let summary = Gemma4SessionInspector.describeForEditing(model: gemma4, config: configObj)
        let topology = try #require(summary.topology)
        #expect(topology.kvDonorForLayers == [0: [2], 1: [3]])
        #expect(summary.editableLayers.contains { $0.path == Self.textOProj })
        #expect(!summary.editableLayers.contains { $0.path.contains("per_layer_model_projection") })

        let donorInspection = try Gemma4SessionInspector.inspectTensor(
            "model.layers.0.self_attn.k_proj.weight",
            in: gemma4,
            topology: topology
        )
        #expect(donorInspection.blastRadiusWarning?.contains("shared layers [2]") == true)
        #expect(donorInspection.forbiddenKind == nil && !donorInspection.isScale && donorInspection.stats.l2Norm > 0)

        let forbiddenInspection = try Gemma4SessionInspector.inspectTensor(
            "model.embed_tokens_per_layer.weight",
            in: gemma4,
            topology: topology
        )
        #expect(forbiddenInspection.forbiddenKind == .embedding)

        let moe = try TinyModel.make(.gemma4Moe)
        let tokens = Self.encode("Inspect Gemma 4 MoE routing distribution.").map(Int32.init)
        let routerSummaries = Gemma4SessionInspector.summarizeMoERouter(model: moe, tokens: tokens, layers: [0, 1])
        #expect(routerSummaries.count == 2)
        for item in routerSummaries {
            #expect(item.numExperts == 8)
            #expect(item.tokensObserved == tokens.count)
            #expect(item.distribution.activeExpertCount > 0 && item.distribution.activeExpertCount <= 8)
            #expect(item.distribution.normalizedEntropy >= 0.0 && item.distribution.normalizedEntropy <= 1.0 + 1e-6)
            #expect(item.perExpertScale?.minValue == 1.0 && item.perExpertScale?.maxValue == 1.0)
        }
    }

    @Test("PairedTestRunner produces exact zero delta on unedited Gemma 4 WeightCopy, detects edit shift, and enforces reserved split budget")
    func pairedTestRunnerEndToEndOnGemma4() throws {
        let original = try TinyModel.make(.gemma4)
        var copy = try WeightCopy.makeSync(from: original)
        let problems = Self.makeProblems(4, prefix: "g4")
        let split = PairedDatasetSplit(practice: Array(problems[0 ..< 2]), reserved: Array(problems[2 ..< 4]))
        let spec = PairedEvaluationSpec(
            encode: Self.encode,
            decode: Self.decode,
            grader: MathAnswerGrader(),
            config: PairedTestRunConfig(seeds: [7, 11], maxTokens: 4, temperature: 0.0, stopTokenIds: [ByteTokenizer.endOfUserTurn], budgetCost: 1)
        )
        var budget = try SelfRewriteBudget(maxForwardPasses: 3, maxToolCalls: 20, maxReservedTests: 1)
        let unedited = try PairedTestRunner.evaluatePractice(
            problems: split.practice,
            original: original,
            copy: copy,
            spec: spec,
            budget: &budget
        )
        #expect(unedited.samples.count == 4)
        #expect(unedited.summary.problemCount == 2 && unedited.summary.delta == 0.0 && unedited.summary.standardError == 0.0)
        #expect(unedited.summary.wrongToRight == 0 && unedited.summary.rightToWrong == 0)
        #expect(budget.forwardPassesUsed == 1)

        var registry = SessionTensorRegistry()
        let oProj = try #require(original.namedModules().first(where: { $0.0 == Self.textOProj })?.1 as? Linear)
        _ = try registry.registerTopLeftSingularVector(key: "u", matrix: oProj.weight, iterations: 12)
        let editProg = EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u", alpha: 0.5)])
        _ = try copy.apply(editProg, tensors: try registry.resolveOperands(for: editProg), verifying: original)

        let reservedSummary = try PairedTestRunner.evaluateReserved(
            split: split,
            original: original,
            copy: copy,
            spec: spec,
            budget: &budget
        )
        #expect(reservedSummary.problemCount == 2 && budget.forwardPassesUsed == 2 && budget.reservedTestsUsed == 1)
        #expect(throws: SelfRewriteBudgetError.self) {
            _ = try PairedTestRunner.evaluateReserved(split: split, original: original, copy: copy, spec: spec, budget: &budget)
        }
        #expect(budget.forwardPassesUsed == 2)
        try copy.release(verifying: original)
    }
}
