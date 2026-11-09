import Foundation
import Testing
import MLX
import MLXLLM
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("System One decisions: OpenJev candidate branches over a shared prefix, calibrated noul/choice/score (#234)")
struct JevDecisionTests {
    @Test("Options follow the reference: choice by name, score by level index, noul false/true with or without criteria")
    func options() {
        let choice = SystemOneQuestion(type: .choice, instructions: "Which?", criteria: .named(["b": "Beta", "a": "Alpha"]))
        #expect(SystemOneCore.options(choice).keys == ["a", "b"] && SystemOneCore.options(choice).descriptions == ["Alpha", "Beta"])
        let score = SystemOneQuestion(type: .score, instructions: "How good?", criteria: .levels(["bad", "ok", "good"]))
        #expect(SystemOneCore.options(score).keys == ["0", "1", "2"])
        let noul = SystemOneQuestion(type: .noul, instructions: "It rains.")
        #expect(SystemOneCore.options(noul).descriptions == ["The following statement is false: It rains.", "The following statement is true: It rains."])
        #expect(SystemOneCore.branch(noul, candidate: "x", tail: "<T>") == "\n\nQuestion:\nIt rains.\n\nCandidate answer:\nx\n\nIs this candidate correct? Answer Yes or No.<T>")
    }

    @Test("Answers: noul is p(true), choice breaks ties by description, score is the expected level; temperature divides the scores")
    func answers() {
        let noul = SystemOneCore.answer(SystemOneQuestion(type: .noul, instructions: "x"), scores: [0, log(3)], temperature: 1)
        #expect(abs((noul.noul ?? 0) - 0.75) < 1e-12)
        let choice = SystemOneCore.answer(
            SystemOneQuestion(type: .choice, instructions: "x", criteria: .named(["k1": "zeta", "k2": "Alpha"])), scores: [2, 2], temperature: 1)
        #expect(choice.choice == "k2" && abs((choice.confidence ?? 0) - 0.5) < 1e-12)
        let score = SystemOneCore.answer(
            SystemOneQuestion(type: .score, instructions: "x", criteria: .levels(["a", "b", "c"])), scores: [0, 0, 0], temperature: 2)
        #expect(abs((score.score ?? 0) - 1) < 1e-12 && score.legend == ["a", "b", "c"])
        let hot = SystemOneCore.answer(SystemOneQuestion(type: .noul, instructions: "x"), scores: [0, 2], temperature: 2)
        #expect(abs((hot.noul ?? 0) - 1 / (1 + exp(-1))) < 1e-12)
    }

    @Test("Requests the reference refuses are refused: blank state, one choice, a noul with other keys, duplicate descriptions")
    func validation() {
        func refused(_ question: SystemOneQuestion, state: String = "s") -> Bool {
            (try? SystemOneCore.validate(SystemOneRequest(state: state, questions: ["q": question]))) == nil
        }
        #expect(refused(SystemOneQuestion(type: .noul, instructions: "x"), state: " "))
        #expect(refused(SystemOneQuestion(type: .choice, instructions: "x", criteria: .named(["a": "A"]))))
        #expect(refused(SystemOneQuestion(type: .noul, instructions: "x", criteria: .named(["yes": "Y", "no": "N"]))))
        #expect(refused(SystemOneQuestion(type: .score, instructions: "x", criteria: .levels(["same", "same"]))))
        #expect(!refused(SystemOneQuestion(type: .score, instructions: "x", criteria: .levels(["low", "high"]))))
    }

    @Test("A PEFT LoRA merges into the base weight as W + (alpha/r)·B·A, and a missing base weight is refused")
    func loraMerge() throws {
        let (w, a, b) = (MLXRandom.normal([6, 4]), MLXRandom.normal([2, 4]), MLXRandom.normal([6, 2]))
        let merged = try JevBundle.mergedLoRA(
            ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a, "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b],
            base: ["model.layers.0.self_attn.q_proj.weight": w], scale: 2)
        let expected = w + 2 * matmul(b, a)
        #expect(abs(merged["model.layers.0.self_attn.q_proj.weight"]! - expected).max().item(Float.self) < 1e-5)
        #expect(throws: BasaltError.self) {
            try JevBundle.mergedLoRA(["x.lora_A.weight": a, "x.lora_B.weight": b], base: [:], scale: 1)
        }
    }

    @Test("Batched branches over the shared prefix cache score like one uncached forward per candidate (tiny Qwen3)")
    func branchesMatchFullForward() throws {
        let model = try #require(try TinyModel.make(.qwen3) as? Qwen3Model)
        let head = MLXRandom.normal([model.model.callAsFunction(MLXArray([Int32(1)]).reshaped(1, 1)).dim(-1)])
        let prefix = [3, 7, 11, 2, 9]
        let suffixes = [[4, 5], [6, 8, 10, 12], [13]]
        let batched = try JevEvaluator.scores(model, head: head, prefix: prefix, suffixes: suffixes, batch: 2)
        let full = suffixes.map { suffix -> Double in
            let ids = (prefix + suffix).map(Int32.init)
            let hidden = model.model(MLXArray(ids).reshaped(1, -1), cache: nil)
            return Double(matmul(hidden[0, ids.count - 1].asType(.float32), head).item(Float.self))
        }
        #expect(zip(batched, full).allSatisfy { abs($0 - $1) < 1e-3 }, "\(batched) vs \(full)")
    }

    @Test("The REST body takes state as text or as an object, which is serialized compactly with sorted keys")
    func restBody() throws {
        let body = #"{"state": {"b": 1, "a": "x/y"}, "questions": {"q": {"type": "noul", "instructions": "ok?"}}, "calibrated": false}"#
        let request = try #require(SystemOneController.request(from: Data(body.utf8)))
        #expect(request.state == #"{"a":"x/y","b":1}"#)
        #expect(request.calibrated == false && request.questions["q"]?.type == .noul)
        #expect(SystemOneController.request(from: Data(#"{"state": 3, "questions": {}}"#.utf8)) == nil)
    }

    @Test("Abstain fires below the confidence floor or under the top-two margin; escalate is set when any question abstains")
    func abstain() {
        let close = SystemOneCore.answer(SystemOneQuestion(type: .choice, instructions: "x", criteria: .named(["a": "A", "b": "B", "c": "C"])), scores: [1, 0.9, -5], temperature: 1)
        let gated = SystemOneCore.gated(close, abstainBelow: nil, minMargin: 0.1)
        #expect(gated.abstain == true && (gated.margin ?? 1) < 0.1)
        #expect(SystemOneCore.gated(close, abstainBelow: nil, minMargin: nil).abstain == false)
        let sure = SystemOneCore.gated(SystemOneCore.answer(SystemOneQuestion(type: .noul, instructions: "x"), scores: [0, 4], temperature: 1), abstainBelow: 0.9, minMargin: nil)
        #expect(sure.abstain == false && abs((sure.margin ?? 0) - (2 / (1 + exp(-4)) - 1)) < 1e-12)
        let unsure = SystemOneCore.gated(SystemOneCore.answer(SystemOneQuestion(type: .noul, instructions: "x"), scores: [0, 0.2], temperature: 1), abstainBelow: 0.9, minMargin: nil)
        let usage = SystemOneUsage(prefixTokens: 1, candidateBranches: 2, prefixEvaluations: 1, branchForwardCalls: 1, suffixTokens: 2)
        #expect(SystemOneResponse(model: "m", answers: ["a": sure, "b": unsure], usage: usage, calibrated: true).escalate)
        #expect(!SystemOneResponse(model: "m", answers: ["a": sure], usage: usage, calibrated: true).escalate)
        #expect((try? SystemOneCore.validate(SystemOneRequest(state: "s", questions: ["q": SystemOneQuestion(type: .noul, instructions: "x")], abstainBelow: 1.5))) == nil)
    }

    @Test("Every decision is written to the session log: answers, abstain and escalate, the state as a digest in meta mode and as text only in full mode")
    func decisionLog() async throws {
        let question = SystemOneQuestion(type: .noul, instructions: "x")
        let request = SystemOneRequest(state: "secret state", questions: ["q": question], abstainBelow: 0.9)
        let answer = SystemOneCore.gated(SystemOneCore.answer(question, scores: [0, 0.2], temperature: 1), abstainBelow: 0.9, minMargin: nil)
        let response = SystemOneResponse(
            model: "org/jev", answers: ["q": answer], usage: SystemOneUsage(prefixTokens: 9, candidateBranches: 2, prefixEvaluations: 1, branchForwardCalls: 1, suffixTokens: 4), calibrated: true)
        for mode in [SessionLogMode.meta, .full] {
            let dir = FileManager.default.temporaryDirectory.appending(path: "jev-log-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: dir) }
            let writer = SessionLogWriter(mode: mode, directory: dir, defaultSession: "d")
            await writer.logDecision(origin: RequestOrigin(source: "systemone", endpoint: "/v1/systemone", stream: false), sessionId: nil, request: request, response: response, latencyMs: 12)
            await writer.flush()
            await writer.close()
            let file = try #require(SessionLogReader.file(for: "d", in: dir))
            let text = try String(contentsOf: file, encoding: .utf8)
            let line = try #require(text.split(separator: "\n").first { $0.contains("\"type\":\"decision\"") })
            let record = try #require(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
            #expect(record["escalate"] as? Bool == true && record["model"] as? String == "org/jev")
            #expect(((record["questions"] as? [String: [String: Any]])?["q"]?["abstain"]) as? Bool == true)
            #expect((record["state_text"] as? String == "secret state") == (mode == .full))
        }
    }
}
