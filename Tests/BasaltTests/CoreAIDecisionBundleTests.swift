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
    }

    @Test("Options render as Laya trained them: noul false then true with defaults, score levels by index, choice key: description in key order")
    func options() {
        #expect(CoreAIDecisionBundle.options(SystemOneQuestion(type: .noul, instructions: "x")) == ["false: no, the statement does not hold", "true: yes, the statement holds"])
        #expect(CoreAIDecisionBundle.options(SystemOneQuestion(type: .score, instructions: "x", criteria: .levels(["low", "high"]))) == ["level 0: low", "level 1: high"])
        #expect(CoreAIDecisionBundle.options(SystemOneQuestion(type: .choice, instructions: "x", criteria: .named(["b": "Bee.", "a": "Ay."]))) == ["a: Ay.", "b: Bee."])
    }

    @Test("A sequence is [CLS] type question [SEP] then [MASK] option per option, [SEP], the state and [SEP]; markers point at the masks and the state is cut at max_len")
    func layout() throws {
        let bundle = try Self.bundle()
        let question = SystemOneQuestion(type: .choice, instructions: "Pick one", criteria: .named(["a": "A.", "b": "B."]))
        let encoded = try bundle.encode(question, state: "s1 s22 [MASK]", tokenize: Self.words)
        let head = Self.words("choice question: Pick one")
        #expect(encoded.ids == [50281] + head + [50282, 50284] + Self.words(" a: A.") + [50284] + Self.words(" b: B.") + [50282] + Self.words("s1 s22  ") + [50282])
        #expect(encoded.markers == [head.count + 2, head.count + 5])
        let long = try bundle.encode(question, state: Array(repeating: "w", count: 900).joined(separator: " "), tokenize: Self.words)
        #expect(long.ids.count == 512 && long.ids.last == 50282)
    }

    @Test("Too many options shrink evenly to fit head_max_len, and options that still do not fit are refused")
    func optionBudget() throws {
        let bundle = try Self.bundle()
        let many = Dictionary(uniqueKeysWithValues: (0 ..< 20).map { ("k\($0)", Array(repeating: "word", count: 30).joined(separator: " ")) })
        let encoded = try bundle.encode(SystemOneQuestion(type: .choice, instructions: "Pick", criteria: .named(many)), state: "s", tokenize: Self.words)
        #expect(encoded.markers.count == 20)
        #expect(zip(encoded.markers, encoded.markers.dropFirst()).allSatisfy { $1 - $0 == 8 })
        let tight = try Self.bundle { $0["max_len"] = 128; $0["head_max_len"] = 400 }
        let wide = Dictionary(uniqueKeysWithValues: (0 ..< 4).map { ("k\($0)", Array(repeating: "word", count: 60).joined(separator: " ")) })
        #expect(throws: BasaltError.self) { try tight.encode(SystemOneQuestion(type: .choice, instructions: "Pick", criteria: .named(wide)), state: "s", tokenize: Self.words) }
    }

    @Test("A request runs at the smallest length that fits its longest question, in as few batched calls as the functions allow")
    func plan() throws {
        let bundle = try Self.bundle()
        #expect(bundle.plan([84, 109, 120]).map(\.function.name) == ["b3_l128"])
        #expect(bundle.plan([84]).map(\.function.name) == ["b1_l128"])
        #expect(bundle.plan([84, 137, 109]) == [CoreAIDecisionBundle.Call(function: bundle.functions[2], rows: [0, 1, 2])])
        #expect(bundle.plan([50, 50, 50, 50]).map(\.rows) == [[0, 1, 2], [3]])
        #expect(bundle.plan([300, 10]).map(\.function.name) == ["b1_l512", "b1_l512"])
        #expect(bundle.plan([]).isEmpty)
    }

    @Test("Inputs are padded per row: ids with [PAD], a [row, key, 1, query] bias masking padded keys, and a one-hot type per row")
    func inputs() throws {
        let bundle = try Self.bundle { $0["functions"] = [["function": "b2_l3", "batch": 2, "length": 3]]; $0["max_len"] = 3 }
        let function = bundle.functions[0]
        #expect(bundle.padded([[7, 8]], to: function) == [7, 8, 50283, 50283, 50283, 50283])
        #expect(bundle.bias([2], to: function) == [0, 0, 0, 0, 0, 0, -30000, -30000, -30000] + Array(repeating: -30000, count: 9))
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
}
