import Foundation
import BasaltCore

struct CoreAIDecisionBundle: Sendable, Equatable {
    struct Function: Sendable, Equatable {
        var name: String
        var batch: Int
        var length: Int
    }

    struct Call: Sendable, Equatable {
        var function: Function
        var rows: [Int]
    }

    struct Encoded: Sendable, Equatable {
        var ids: [Int]
        var markers: [Int]
    }

    var modelID: String
    var maxLength: Int
    var headMaxLength: Int
    var maxOptionTokens: Int
    var cls: Int
    var sep: Int
    var mask: Int
    var pad: Int
    var masked: Float
    var temperatures: [Double]
    var temperaturesByOptions: [String: Double]
    var functions: [Function]

    static let manifest = "laya_decision.json"
    static let maskText = "[MASK]"

    static func isBundle(_ directory: URL) -> Bool {
        [manifest, "decision.aimodel", "token_table.aimodel"].allSatisfy { FileManager.default.fileExists(atPath: directory.appending(component: $0).path) }
    }

    init(directory: URL) throws(BasaltError) {
        guard let data = try? Data(contentsOf: directory.appending(component: Self.manifest)),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let body = root["laya_decision"] as? [String: Any]
        else { throw .modelConfigurationInvalid(path: directory.path, reason: "\(Self.manifest) has no \"laya_decision\" section") }
        try self.init(manifest: body, path: directory.path)
    }

    init(manifest: [String: Any], path: String) throws(BasaltError) {
        let tokens = manifest["tokens"] as? [String: Int] ?? [:]
        let functions = (manifest["functions"] as? [[String: Any]] ?? []).compactMap { entry -> Function? in
            guard let name = entry["function"] as? String, let batch = entry["batch"] as? Int, let length = entry["length"] as? Int, batch > 0, length > 0 else { return nil }
            return Function(name: name, batch: batch, length: length)
        }.sorted { ($0.length, $0.batch) < ($1.length, $1.batch) }
        let temperatures = manifest["temperature"] as? [Double] ?? []
        guard let cls = tokens["cls"], let sep = tokens["sep"], let mask = tokens["mask"], let pad = tokens["pad"], !functions.isEmpty, temperatures.count == 3,
              let maxLength = manifest["max_len"] as? Int, let headMaxLength = manifest["head_max_len"] as? Int, maxLength <= functions[functions.count - 1].length
        else { throw .modelConfigurationInvalid(path: path, reason: "the decision bundle needs cls/sep/mask/pad token ids, three temperatures, max_len and head_max_len, and functions up to max_len") }
        modelID = manifest["model_id"] as? String ?? URL(fileURLWithPath: path).lastPathComponent
        (self.maxLength, self.headMaxLength, maxOptionTokens) = (maxLength, headMaxLength, manifest["max_option_tokens"] as? Int ?? 48)
        (self.cls, self.sep, self.mask, self.pad) = (cls, sep, mask, pad)
        masked = Float(manifest["masked"] as? Double ?? -30000)
        self.temperatures = temperatures
        temperaturesByOptions = manifest["temperature_by_options"] as? [String: Double] ?? [:]
        self.functions = functions
    }

    static func typeIndex(_ kind: SystemOneKind) -> Int {
        switch kind {
        case .choice: 0
        case .score: 1
        case .noul: 2
        }
    }

    static func options(_ question: SystemOneQuestion) -> [String] {
        switch (question.type, question.criteria) {
        case (.noul, let criteria):
            let named: [String: String] = if case .named(let map)? = criteria { map } else { [:] }
            return ["false: " + (named["false"] ?? "no, the statement does not hold"), "true: " + (named["true"] ?? "yes, the statement holds")]
        case (.score, _):
            return SystemOneCore.options(question).descriptions.enumerated().map { "level \($0.offset): \($0.element)" }
        case (.choice, _):
            let (keys, descriptions) = SystemOneCore.options(question)
            return zip(keys, descriptions).map { $1.isEmpty ? $0 : "\($0): \($1)" }
        }
    }

    func temperature(_ kind: SystemOneKind, options count: Int) -> Double {
        let size = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+"
        return temperaturesByOptions["\(kind.rawValue):\(size)"] ?? temperatures[Self.typeIndex(kind)]
    }

    func encode(_ question: SystemOneQuestion, state: String, tokenize: (String) -> [Int]) throws(BasaltError) -> Encoded {
        let clean = { (text: String) in text.replacingOccurrences(of: Self.maskText, with: " ") }
        var options = Self.options(question).map { [mask] + tokenize(" " + clean($0)).prefix(maxOptionTokens) }
        var budget = headMaxLength - options.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let each = max(4, (headMaxLength - 16) / max(1, options.count))
            options = options.map { Array($0.prefix(each)) }
            budget = headMaxLength - options.reduce(0) { $0 + $1.count }
        }
        var ids = [cls] + tokenize("\(question.type.rawValue) question: \(clean(question.instructions))").prefix(max(8, budget)) + [sep]
        var markers: [Int] = []
        for option in options {
            markers.append(ids.count)
            ids += option
        }
        ids.append(sep)
        ids += tokenize(clean(state)).prefix(max(0, maxLength - ids.count - 1))
        ids = Array((ids + [sep]).prefix(maxLength))
        markers = markers.filter { $0 < maxLength }
        guard markers.count == options.count else {
            throw .invalidRequestStructure(details: "systemone: the options of a question do not fit in \(headMaxLength) tokens of \(modelID)")
        }
        return Encoded(ids: ids, markers: markers)
    }

    func plan(_ lengths: [Int]) -> [Call] {
        guard let longest = lengths.max() else { return [] }
        let length = (functions.first { $0.length >= longest } ?? functions[functions.count - 1]).length
        let shapes = functions.filter { $0.length == length }
        var (calls, next) = ([Call](), 0)
        while next < lengths.count {
            let left = lengths.count - next
            let function = shapes.first { $0.batch >= left } ?? shapes[shapes.count - 1]
            let rows = Array(next ..< min(next + function.batch, lengths.count))
            calls.append(Call(function: function, rows: rows))
            next += rows.count
        }
        return calls
    }

    func padded(_ rows: [[Int]], to function: Function) -> [Int32] {
        (0 ..< function.batch).flatMap { row in
            let ids = row < rows.count ? Array(rows[row].prefix(function.length)) : []
            return (ids + Array(repeating: pad, count: function.length - ids.count)).map(Int32.init)
        }
    }

    func bias(_ reals: [Int], to function: Function) -> [Float] {
        (0 ..< function.batch).flatMap { row in
            let real = row < reals.count ? reals[row] : 0
            return (0 ..< function.length).flatMap { key in Array(repeating: key < real ? 0 : masked, count: function.length) }
        }
    }

    func types(_ kinds: [SystemOneKind], to function: Function) -> [Float] {
        (0 ..< function.batch).flatMap { row in
            let index = row < kinds.count ? Self.typeIndex(kinds[row]) : 0
            return (0 ..< 3).map { $0 == index ? Float(1) : 0 }
        }
    }
}
