import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import Synchronization
import Testing
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltCore
@testable import BasaltInference

@Suite("Teacher-guided System One calibration: planted recovery, canonical bundle JSON, JSONL teacher cache (#448)")
struct TeacherCalibrationCoreTests {
    @Test("SystemOneReadout canonical encode/decode round-trips deterministically and integrates with calibration(for:in:)")
    func readoutRoundtrip() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "readout-cal-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(SystemOneReadout.calibration(for: "org/custom-chat", in: dir) == .uncalibrated)
        let original = SystemOneReadout(temperature: 2.4, noulTemperature: 1.8, noulBias: -0.65)
        let bytes = try original.encode()
        let secondBytes = try original.encode()
        #expect(bytes == secondBytes)
        let text = String(decoding: bytes, as: UTF8.self)
        #expect(text.contains("\"noul_bias\"") && text.contains("\"noul_temperature\"") && text.contains("\"temperature\""))
        let decoded = try SystemOneReadout.decode(bytes)
        #expect(decoded == original)

        try bytes.write(to: dir.appending(path: SystemOneReadout.bundleFile), options: .atomic)
        #expect(SystemOneReadout.bundled(in: dir) == original)
        #expect(SystemOneReadout.calibration(for: "org/custom-chat", in: dir) == original)
        #expect(throws: BasaltError.self) { _ = try SystemOneReadout(temperature: 0).encode() }
        #expect(throws: BasaltError.self) { _ = try SystemOneReadout.decode(Data(#"{"temperature": -1}"#.utf8)) }
    }

    @Test("SystemOneCalibrationCore recovers planted temperature, noulTemperature and noulBias within < 0.05 and halves ECE")
    func plantedParameterRecovery() {
        let planted = SystemOneReadout(temperature: 2.4, noulTemperature: 1.8, noulBias: -0.65)
        let samples = Self.syntheticMiscalibratedSamples(planted: planted, countPerKind: 600)
        let fit = SystemOneCalibrationCore.fit(samples)

        #expect(fit.sampleCount == 1200)
        #expect(abs(fit.readout.temperature - 2.4) < 0.05)
        #expect(abs(fit.readout.noulTemperature - 1.8) < 0.05)
        #expect(abs(fit.readout.noulBias - (-0.65)) < 0.05)
        #expect(fit.klAfter < fit.klBefore)
        #expect(fit.eceAfter <= 0.5 * fit.eceBefore)
        #expect(fit.brierAfter < fit.brierBefore)
        #expect(fit.binsBefore.count == 10 && fit.binsAfter.count == 10)

        let softSamples = samples.map {
            CalibrationSample(stateHash: $0.stateHash, questionID: $0.questionID, kind: $0.kind, rawScores: $0.rawScores, teacherProbabilities: $0.teacherProbabilities, goldIndex: nil)
        }
        let softFit = SystemOneCalibrationCore.fit(softSamples, holdoutFraction: 0.2)
        #expect(softFit.sampleCount == 240)
        #expect(abs(softFit.readout.temperature - 2.4) < 0.05)
        #expect(abs(softFit.readout.noulTemperature - 1.8) < 0.05)
        #expect(abs(softFit.readout.noulBias - (-0.65)) < 0.05)
        #expect(softFit.klAfter < softFit.klBefore && softFit.eceAfter <= 0.5 * softFit.eceBefore && softFit.brierAfter < softFit.brierBefore)
    }

    @Test("Single-logit and two-logit noul rawScores agree bit-for-bit, and JSONL + SHA-256 cache keys round-trip deterministically")
    func jsonlAndCacheKeyDeterminism() throws {
        let readout = SystemOneReadout(temperature: 1.7, noulTemperature: 1.4, noulBias: -0.35)
        let two = SystemOneCalibrationCore.probabilities(kind: .noul, rawScores: [2.5, -0.7], readout: readout)
        let one = SystemOneCalibrationCore.probabilities(kind: .noul, rawScores: [3.2], readout: readout)
        #expect(abs(two[0] - one[0]) < 1e-14 && abs(two[1] - one[1]) < 1e-14)

        let qA = SystemOneQuestion(type: .choice, instructions: "Pick team", criteria: .named(["shipping": "Ship", "billing": "Bill"]))
        let qB = SystemOneQuestion(type: .choice, instructions: "Pick team", criteria: .named(["billing": "Bill", "shipping": "Ship"]))
        let h1 = SystemOneCalibrationCore.stateHash(for: "Customer wants refund")
        let k1 = SystemOneCalibrationCore.cacheKey(teacherModel: "gemini/gemini-decision-1", stateHash: h1, question: qA)
        let k2 = SystemOneCalibrationCore.cacheKey(teacherModel: "gemini/gemini-decision-1", stateHash: h1, question: qB)
        let kOther = SystemOneCalibrationCore.cacheKey(teacherModel: "openrouter/jev", stateHash: h1, question: qA)
        #expect(h1.count == 64 && k1.count == 64 && k1 == k2 && k1 != kOther)

        let samples = [
            CalibrationSample(stateHash: h1, questionID: "team", kind: .choice, rawScores: [2.1, -0.4], teacherProbabilities: [0.8, 0.2], goldIndex: 0),
            CalibrationSample(stateHash: h1, questionID: "safe", kind: .noul, rawScores: [1.5, -0.5], teacherProbabilities: [0.7, 0.3], goldIndex: nil)
        ]
        let encoded1 = try SystemOneCalibrationCore.encodeJSONL(samples)
        let encoded2 = try SystemOneCalibrationCore.encodeJSONL(samples)
        #expect(encoded1 == encoded2)
        #expect(try SystemOneCalibrationCore.decodeJSONL(encoded1) == samples)
        #expect(try SystemOneCalibrationCore.decodeJSONL(Data()).isEmpty)
    }

    private static func syntheticMiscalibratedSamples(planted: SystemOneReadout, countPerKind: Int) -> [CalibrationSample] {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func nextUnit() -> Double {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            return Double(z >> 11) / Double(1 << 53)
        }
        func drawIndex(_ probs: [Double]) -> Int {
            let u = nextUnit()
            var acc = 0.0
            for (i, p) in probs.enumerated() {
                acc += p
                if u <= acc || i == probs.count - 1 { return i }
            }
            return 0
        }
        var out: [CalibrationSample] = []
        for i in 0 ..< countPerKind {
            let kind: SystemOneKind = i.isMultiple(of: 2) ? .choice : .score
            let k = kind == .choice ? 4 : 3
            let raw = (0 ..< k).map { _ in (nextUnit() - 0.5) * 8.0 }
            let q = SystemOneCalibrationCore.probabilities(kind: kind, rawScores: raw, readout: planted)
            out.append(CalibrationSample(stateHash: "s\(i)", questionID: "m\(i)", kind: kind, rawScores: raw, teacherProbabilities: q, goldIndex: drawIndex(q)))

            let zYes = (nextUnit() - 0.5) * 9.0
            let zNo = (nextUnit() - 0.5) * 9.0
            let noulRaw = i.isMultiple(of: 3) ? [zYes - zNo] : [zYes, zNo]
            let qNoul = SystemOneCalibrationCore.probabilities(kind: .noul, rawScores: [zYes, zNo], readout: planted)
            out.append(CalibrationSample(stateHash: "n\(i)", questionID: "b\(i)", kind: .noul, rawScores: noulRaw, teacherProbabilities: qNoul, goldIndex: drawIndex(qNoul)))
        }
        return out
    }
}

@Suite("Teacher-guided System One calibration: TinyModel readout parity, serve --decision-model self, and calibrate-decide CLI (#448)", .serialized)
struct TeacherCalibrationEngineAndCLITests {
    private static func makeTinyQwen3Engine() async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen3"),
            model: try TinyModel.make(.qwen3),
            processor: ByteInputProcessor(),
            tokenizer: ASCIITokenizer())))
        return engine
    }

    @Test("TinyModel.make(.qwen3) ReadoutDecisionEngine matches SystemOneCalibrationCore probabilities and serve --decision-model self loads basalt_readout.json")
    func tinyModelReadoutParityAndSelfWiring() async throws {
        let engine = try await Self.makeTinyQwen3Engine()
        let dir = FileManager.default.temporaryDirectory.appending(path: "self-cal-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let questions: [String: SystemOneQuestion] = [
            "team": SystemOneQuestion(type: .choice, instructions: "Which team?", criteria: .named(["billing": "Charges.", "shipping": "Parcels.", "support": "Help."])),
            "risk": SystemOneQuestion(type: .score, instructions: "Severity?", criteria: .levels(["low", "medium", "high"])),
            "safe": SystemOneQuestion(type: .noul, instructions: "The request is safe.")
        ]
        let req = SystemOneRequest(state: "Order 42 arrived damaged.", questions: questions)
        let samples = try await engine.collectCalibrationSamples(state: req.state, questions: questions)
        #expect(samples.count == 3)

        let uncalServe = try ServeCommand.parse(["--model-path", dir.path, "--decision-model", "self"])
        let uncalDecider = try #require(try await uncalServe.loadDecisions(chat: engine, chatPath: dir.path) as? ReadoutDecisionEngine)
        #expect(uncalDecider.readout == .uncalibrated)
        let uncalRes = try await uncalDecider.decide(req)
        #expect(!uncalRes.calibrated)

        let custom = SystemOneReadout(temperature: 2.25, noulTemperature: 1.65, noulBias: -0.45)
        try custom.encode().write(to: dir.appending(path: SystemOneReadout.bundleFile), options: .atomic)
        let calServe = try ServeCommand.parse(["--model-path", dir.path, "--decision-model", "self"])
        let calDecider = try #require(try await calServe.loadDecisions(chat: engine, chatPath: dir.path) as? ReadoutDecisionEngine)
        #expect(calDecider.readout == custom)
        let calRes = try await calDecider.decide(req)
        #expect(calRes.calibrated)

        let byID = Dictionary(uniqueKeysWithValues: samples.map { ($0.questionID, $0) })
        let teamProbs = SystemOneCalibrationCore.probabilities(for: try #require(byID["team"]), readout: custom)
        let teamKeys = SystemOneReadout.options(try #require(questions["team"])).map(\.key)
        for (idx, key) in teamKeys.enumerated() {
            #expect(abs((calRes.answers["team"]?.probabilities?[key] ?? -1) - teamProbs[idx]) < 1e-12)
        }
        #expect(abs((calRes.answers["team"]?.confidence ?? -1) - SystemOneCore.choiceConfidence(teamProbs)) < 1e-12)

        let riskProbs = SystemOneCalibrationCore.probabilities(for: try #require(byID["risk"]), readout: custom)
        #expect(abs((calRes.answers["risk"]?.confidence ?? -1) - SystemOneCore.scoreConfidence(riskProbs)) < 1e-12)

        let safeProbs = SystemOneCalibrationCore.probabilities(for: try #require(byID["safe"]), readout: custom)
        #expect(abs((calRes.answers["safe"]?.noul ?? -1) - safeProbs[0]) < 1e-12)
    }

    @Test("basalt models calibrate-decide parses flags, caches teacher distributions by SHA-256, enforces --max-usd, and writes basalt_readout.json")
    func calibrateDecideRunnerAndCache() async throws {
        let parsed = try CalibrateDecideSubcommand.parse(["org/model", "--dataset", "/tmp/eval.jsonl", "--teacher", "cloud:gemini/gemini-decision-1"])
        #expect(parsed.keyEnvName == "GEMINI_API_KEY" && parsed.maxUsd == 1.0 && parsed.holdoutFraction == 0.2 && parsed.out == nil)
        #expect(throws: (any Error).self) { _ = try CalibrateDecideSubcommand.parse(["org/model", "--dataset", "/tmp/eval.jsonl", "--holdout-fraction", "1.0"]) }
        #expect(throws: (any Error).self) { _ = try CalibrateDecideSubcommand.parse(["org/model", "--dataset", "/tmp/eval.jsonl", "--max-usd", "-0.5"]) }

        let root = FileManager.default.temporaryDirectory.appending(path: "cal-cli-\(UUID().uuidString)")
        let cacheDir = root.appending(path: "cache")
        let datasetURL = root.appending(path: "dataset.jsonl")
        let outURL = root.appending(path: SystemOneReadout.bundleFile)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let lines = (0 ..< 6).map { i in
            #"{"state":"Case \#(i)","questions":{"team":{"type":"choice","instructions":"Route?","criteria":{"a":"Alpha","b":"Beta"}},"ok":{"type":"noul","instructions":"Safe?"}}}"#
        }.joined(separator: "\n") + "\n"
        try Data(lines.utf8).write(to: datasetURL)

        let engine = try await Self.makeTinyQwen3Engine()
        let teacher = StubTeacherDecisionEngine(costPerCall: 0.05)
        let config = CalibrateDecideRunner.Config(
            modelPath: root.path, datasetPath: datasetURL.path, teacherSpec: "cloud:gemini/gemini-decision-1",
            keyEnvName: "GEMINI_API_KEY", maxUSD: 1.0, cacheDir: cacheDir, holdoutFraction: 0.2, outputURL: outURL)

        let first = try await CalibrateDecideRunner.run(config, student: engine, teacher: teacher)
        #expect(first.totalSamples == 12 && first.teacherCalls == 6 && first.cachedHits == 0 && abs(first.spentUSD - 0.30) < 1e-9)
        #expect(SystemOneReadout.bundled(in: root) == first.fit.readout)
        let summary = CalibrateDecideRunner.summary(first)
        #expect(summary.contains("KL:") && summary.contains("ECE:") && summary.contains("Brier:") && summary.contains("temperature="))

        var cachedOnly = config
        cachedOnly.maxUSD = 0.0
        let second = try await CalibrateDecideRunner.run(cachedOnly, student: engine, teacher: teacher)
        #expect(second.teacherCalls == 0 && second.cachedHits == 12 && second.spentUSD == 0 && teacher.callCount == 6)
        #expect(second.fit == first.fit)

        let tightCache = root.appending(path: "empty-cache")
        var tight = config
        (tight.cacheDir, tight.maxUSD) = (tightCache, 0.02)
        await #expect(throws: (any Error).self) { _ = try await CalibrateDecideRunner.run(tight, student: engine, teacher: teacher) }
    }
}

private final class StubTeacherDecisionEngine: DecisionEngine {
    let modelID = "stub-teacher"
    let costPerCall: Double
    private let counter = Mutex(0)
    var callCount: Int { counter.withLock { $0 } }

    init(costPerCall: Double) { self.costPerCall = costPerCall }

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        counter.withLock { $0 += 1 }
        var answers: [String: SystemOneAnswer] = [:]
        for (id, q) in request.questions {
            var ans = SystemOneAnswer()
            if q.type == .noul {
                ans.noul = 0.75
            } else {
                let keys = SystemOneReadout.options(q).map(\.key)
                ans.choice = keys.first
                ans.probabilities = Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($0.element, $0.offset == 0 ? 0.8 : 0.2 / Double(max(keys.count - 1, 1))) })
            }
            answers[id] = ans
        }
        var usage = SystemOneUsage(prefixTokens: 4, candidateBranches: request.questions.count, prefixEvaluations: 1, branchForwardCalls: request.questions.count, suffixTokens: 4)
        usage.cost = costPerCall
        return SystemOneResponse(model: modelID, answers: answers, usage: usage, calibrated: true)
    }
}
