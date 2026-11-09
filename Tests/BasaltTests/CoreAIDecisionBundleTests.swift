import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Core AI decision bundle (Laya on the Neural Engine): manifest, sequence layout, bucket and batch plan, readout math (#234)")
struct CoreAIDecisionBundleTests {
    static var manifest: [String: Any] { [
        "model_id": "convaiinnovations/laya",
        "max_len": 512, "head_max_len": 192, "max_option_tokens": 48,
        "tokens": ["cls": 50281, "sep": 50282, "mask": 50284, "pad": 50283],
        "masked": -30000.0,
        "temperature": [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
        "temperature_by_options": ["choice:3-5": 1.7601518630981445, "noul:2": 1.983399510383606, "score:3-5": 1.2514300346374512, "choice:2": 1.9063563346862793],
        "functions": [
            ["function": "b1_l512", "batch": 1, "length": 512], ["function": "b3_l128", "batch": 3, "length": 128],
            ["function": "b1_l128", "batch": 1, "length": 128], ["function": "b3_l256", "batch": 3, "length": 256],
        ],
    ] }

    static func bundle(_ change: (inout [String: Any]) -> Void = { _ in }) throws -> CoreAIDecisionBundle {
        var manifest = Self.manifest
        change(&manifest)
        return try CoreAIDecisionBundle(manifest: manifest, path: "/x")
    }

    static func words(_ text: String) -> [Int] { text.split(separator: " ").map { 1000 + $0.count } }

    @Test("The manifest gives token ids, temperatures and functions sorted by length then batch; a manifest without them is refused")
    func parses() throws {
        let bundle = try Self.bundle()
        #expect(bundle.modelID == "convaiinnovations/laya")
        #expect(bundle.functions.map(\.name) == ["b1_l128", "b3_l128", "b3_l256", "b1_l512"])
        #expect((bundle.cls, bundle.sep, bundle.mask, bundle.pad, bundle.maxLength) == (50281, 50282, 50284, 50283, 512))
        #expect(throws: BasaltError.self) { try Self.bundle { $0["tokens"] = nil } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["functions"] = [["function": "b1_l128", "batch": 1, "length": 128]] } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["temperature"] = [0.0, 1.0, 1.0] } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["temperature"] = [1.0, -1.0, 1.0] } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["temperature_by_options"] = ["noul:2": 0.0] } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["head_max_len"] = 512 } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["temperature"] = [1.0, 1.0] } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["temperature"] = [1.0, Double.infinity, 1.0] } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["temperature"] = [1.0, 1.0, Double.nan] } }
        #expect(throws: BasaltError.self) { try Self.bundle { $0["temperature_by_options"] = ["choice:2": Double.infinity] } }
    }

    @Test("Options render as Laya trained them: noul false then true with defaults, score levels by index, choice key: description in key order")
    func options() {
        #expect(CoreAIDecisionBundle.options(SystemOneQuestion(type: .noul, instructions: "x")) == ["false: no, the statement does not hold", "true: yes, the statement holds"])
        #expect(CoreAIDecisionBundle.options(SystemOneQuestion(type: .score, instructions: "x", criteria: .levels(["low", "high"]))) == ["level 0: low", "level 1: high"])
        #expect(CoreAIDecisionBundle.options(SystemOneQuestion(type: .choice, instructions: "x", criteria: .named(["b": "Bee.", "a": "Ay."]))) == ["a: Ay.", "b: Bee."])
    }

    @Test("A sequence is [CLS] type question [SEP] then [MASK] option per option, [SEP], the state and [SEP]; markers point at the masks; a state past max_len is refused, not cut")
    func layout() throws {
        let bundle = try Self.bundle()
        let question = SystemOneQuestion(type: .choice, instructions: "Pick one", criteria: .named(["a": "A.", "b": "B."]))
        let encoded = try bundle.encode(question, state: Self.words(CoreAIDecisionBundle.clean("s1 s22 [MASK]")), tokenize: Self.words)
        let head = Self.words("choice question: Pick one")
        #expect(encoded.ids == [50281] + head + [50282, 50284] + Self.words(" a: A.") + [50284] + Self.words(" b: B.") + [50282] + Self.words("s1 s22  ") + [50282])
        #expect(encoded.markers == [head.count + 2, head.count + 5])
        let room = 512 - encoded.ids.count + 2
        #expect(try bundle.encode(question, state: Array(repeating: 1, count: room), tokenize: Self.words).ids.count == 512)
        let refused = #expect(throws: BasaltError.self) { try bundle.encode(question, state: Array(repeating: 1, count: room + 1), tokenize: Self.words) }
        guard case .contextLengthExceeded(let maxTokens, let requested)? = refused else { Issue.record("expected contextLengthExceeded, got \(String(describing: refused))"); return }
        #expect((maxTokens, requested) == (512, 513))
    }

    @Test("Too many options shrink evenly to fit head_max_len; options that still overflow it are refused instead of squeezing the state")
    func optionBudget() throws {
        let bundle = try Self.bundle()
        let many = Dictionary(uniqueKeysWithValues: (0 ..< 20).map { ("k\($0)", Array(repeating: "word", count: 30).joined(separator: " ")) })
        let encoded = try bundle.encode(SystemOneQuestion(type: .choice, instructions: "Pick", criteria: .named(many)), state: [1], tokenize: Self.words)
        #expect(encoded.markers.count == 20)
        #expect(zip(encoded.markers, encoded.markers.dropFirst()).allSatisfy { $1 - $0 == 8 })
        let crowded = Dictionary(uniqueKeysWithValues: (0 ..< 60).map { ("k\($0)", Array(repeating: "word", count: 30).joined(separator: " ")) })
        let refused = #expect(throws: BasaltError.self) {
            try bundle.encode(SystemOneQuestion(type: .choice, instructions: "Pick", criteria: .named(crowded)), state: [1], tokenize: Self.words)
        }
        guard case .invalidRequestStructure(let details)? = refused else { Issue.record("expected invalidRequestStructure, got \(String(describing: refused))"); return }
        #expect(details.contains("do not fit in 192 tokens"))
    }

    @Test("A request runs at the smallest length that fits its longest question, in as few batched calls as the functions allow")
    func plan() throws {
        let bundle = try Self.bundle()
        #expect(bundle.plan([84, 109, 120]).map(\.function.name) == ["b3_l128"])
        #expect(bundle.plan([84]).map(\.function.name) == ["b1_l128"])
        #expect(bundle.plan([128]).map(\.function.name) == ["b1_l128"])
        #expect(bundle.plan([129]).map(\.function.name) == ["b3_l256"])
        #expect(bundle.plan([84, 137, 109]) == [CoreAIDecisionBundle.Call(function: bundle.functions[2], rows: [0, 1, 2])])
        #expect(bundle.plan([50, 50, 50, 50]).map(\.rows) == [[0, 1, 2], [3]])
        #expect(bundle.plan([300, 10]).map(\.function.name) == ["b1_l512", "b1_l512"])
        #expect(bundle.plan([]).isEmpty)
    }

    @Test("Inputs are padded per row: ids with [PAD], a [row, key] bias masking padded keys, and a one-hot type per row")
    func inputs() throws {
        let bundle = try Self.bundle { $0["functions"] = [["function": "b2_l3", "batch": 2, "length": 3]]; $0["max_len"] = 3; $0["head_max_len"] = 2 }
        let function = bundle.functions[0]
        #expect(bundle.padded([[7, 8]], to: function) == [7, 8, 50283, 50283, 50283, 50283])
        #expect(bundle.bias([2], to: function) == [0, 0, -30000, -30000, -30000, -30000])
        #expect(bundle.types([.noul, .choice], to: function) == [0, 0, 1, 1, 0, 0])
    }

    @Test("Calibrated readout matches the fp32 PyTorch reference: per-cardinality temperature, softmax over the marker logits, noul as P(true)")
    func readout() throws {
        let bundle = try Self.bundle()
        let department = SystemOneQuestion(type: .choice, instructions: "Which department should handle this?", criteria: .named(["billing": "invoices, payments, refunds", "other": "everything else", "technical": "bugs, outages, system errors"]))
        let choice = SystemOneCore.answer(department, scores: [6.273937702178955, -3.23520827293396, -2.2708773612976074], temperature: bundle.temperature(.choice, options: 3))
        #expect(choice.choice == "billing")
        #expect(abs((choice.probabilities?["billing"] ?? 0) - 0.987851379673197) < 1e-9)
        #expect(abs((choice.probabilities?["technical"] ?? 0) - 0.007697861666245582) < 1e-9)
        let churn = SystemOneCore.answer(SystemOneQuestion(type: .noul, instructions: "Does the user threaten to cancel or leave?"), scores: [-2.6588852405548096, 1.2745335102081299], temperature: bundle.temperature(.noul, options: 2))
        #expect(abs((churn.noul ?? 0) - 0.879018698923901) < 1e-9)
        let urgency = SystemOneCore.answer(SystemOneQuestion(type: .score, instructions: "How urgent is this?", criteria: .levels(["not urgent", "soon", "blocking"])), scores: [-0.7835437655448914, 0.9054975509643555, 3.0173721313476562], temperature: bundle.temperature(.score, options: 3))
        #expect(abs((urgency.probabilities?["2"] ?? 0) - 0.8110728563330039) < 1e-9)
        #expect(bundle.temperature(.choice, options: 12) == 1.6369030475616455)
    }

    static func phrase(_ count: Int) -> String { Array(repeating: "word", count: count).joined(separator: " ") }

    @Test("A long instruction is cut to the head budget, so head plus options fill head_max_len exactly; an option is cut at max_option_tokens")
    func headAndOptionCuts() throws {
        let bundle = try Self.bundle()
        let long = try bundle.encode(SystemOneQuestion(type: .choice, instructions: Self.phrase(300), criteria: .named(["a": "x", "b": "y"])), state: [1], tokenize: Self.words)
        #expect(long.markers == [188, 191])
        #expect(long.ids.count - 3 - 1 - 1 == 192)
        let capped = try bundle.encode(SystemOneQuestion(type: .choice, instructions: "Pick", criteria: .named(["a": Self.phrase(60), "b": "y"])), state: [1], tokenize: Self.words)
        #expect(capped.markers[1] - capped.markers[0] == 49)
    }

    @Test("Shrunk options plus the 8-token head floor: exactly head_max_len passes, one token over is refused")
    func headBudgetBoundary() throws {
        let bundle = try Self.bundle { $0["head_max_len"] = 40 }
        let longs = (0 ..< 8).map { ("a\($0)", Self.phrase(30)) }
        let exact = try bundle.encode(SystemOneQuestion(type: .choice, instructions: Self.phrase(20), criteria: .named(Dictionary(uniqueKeysWithValues: longs))), state: [1], tokenize: Self.words)
        #expect(exact.markers.first == 10 && exact.markers.count == 8)
        #expect(exact.ids.count == 1 + 8 + 1 + 32 + 1 + 1 + 1)
        let over = Dictionary(uniqueKeysWithValues: Array(longs.prefix(6)) + (0 ..< 3).map { ("b\($0)", "x") })
        #expect(throws: BasaltError.self) { try bundle.encode(SystemOneQuestion(type: .choice, instructions: Self.phrase(20), criteria: .named(over)), state: [1], tokenize: Self.words) }
    }

    static let readoutQuestions: [String: SystemOneQuestion] = [
        "a": SystemOneQuestion(type: .choice, instructions: "Pick", criteria: .named(["x": "X.", "y": "Y.", "z": "Z."])),
        "b": SystemOneQuestion(type: .noul, instructions: "True?"),
        "c": SystemOneQuestion(type: .score, instructions: "Rate", criteria: .levels(["low", "mid", "high", "top"])),
        "d": SystemOneQuestion(type: .choice, instructions: "Pick", criteria: .named(["p": "P.", "q": "Q."])),
    ]

    @Test("Readout gathers each row's marker scores from its slot in its call, maps them to the sorted question names, uses T = 1 when uncalibrated, and counts usage")
    func readoutGather() throws {
        let bundle = try Self.bundle()
        let encoded = [(3, 30), (2, 20), (4, 40), (2, 50)].map { count, length in CoreAIDecisionBundle.Encoded(ids: Array(repeating: 7, count: length), markers: (0 ..< count).map { 5 + 3 * $0 }) }
        let calls = bundle.plan(encoded.map(\.ids.count))
        #expect(calls.map(\.rows) == [[0, 1, 2], [3]])
        let scores = calls.map { call in
            var values = [Float](repeating: -999, count: 8 * call.function.length)
            for (slot, row) in call.rows.enumerated() {
                for (order, marker) in encoded[row].markers.enumerated() { values[slot * call.function.length + marker] = Float(row) + Float(order) / 4 }
            }
            return values
        }
        let logits = encoded.indices.map { row in encoded[row].markers.indices.map { Double(row) + Double($0) / 4 } }
        let names = Self.readoutQuestions.keys.sorted()
        for calibrated in [false, true] {
            let response = bundle.readout(SystemOneRequest(state: "s", questions: Self.readoutQuestions, calibrated: calibrated), encoded: encoded, calls: calls, scores: scores)
            for (row, name) in names.enumerated() {
                let question = try #require(Self.readoutQuestions[name])
                let scale = calibrated ? bundle.temperature(question.type, options: logits[row].count) : 1
                #expect(response.answers[name] == SystemOneCore.gated(SystemOneCore.answer(question, scores: logits[row], temperature: scale), abstainBelow: nil, minMargin: nil), "\(name) calibrated \(calibrated)")
            }
            #expect(response.calibrated == calibrated)
        }
        let usage = bundle.readout(SystemOneRequest(state: "s", questions: Self.readoutQuestions), encoded: encoded, calls: calls, scores: scores).usage
        #expect((usage.prefixTokens, usage.candidateBranches, usage.prefixEvaluations, usage.branchForwardCalls, usage.suffixTokens) == (140, 11, 4, 2, 0))
    }
}
