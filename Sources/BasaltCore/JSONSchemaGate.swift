import Foundation

public struct UnsupportedJSONSchema: Error, Equatable, Sendable {
    public let message: String
}

public struct PreparedJSONSchema: Sendable, Equatable {
    public let schema: String
    public let relaxations: [String]
}

public enum JSONNesting {
    public static func depth(of data: Data) -> Int {
        var depth = 0
        var deepest = 0
        var inString = false
        var escaped = false
        for byte in data {
            if inString {
                if escaped { escaped = false } else if byte == UInt8(ascii: "\\") { escaped = true } else if byte == UInt8(ascii: "\"") { inString = false }
            } else if byte == UInt8(ascii: "\"") {
                inString = true
            } else if byte == UInt8(ascii: "[") || byte == UInt8(ascii: "{") {
                depth += 1
                deepest = max(deepest, depth)
            } else if byte == UInt8(ascii: "]") || byte == UInt8(ascii: "}") {
                depth -= 1
            }
        }
        return deepest
    }
}

public indirect enum OrderedJSON: Equatable, Sendable {
    public struct Member: Equatable, Sendable {
        public let key: String
        let quotedKey: String
        public var value: OrderedJSON
    }

    case object([Member])
    case array([OrderedJSON])
    case scalar(String)

    public static func parse(_ data: Data) -> OrderedJSON? {
        guard JSONNesting.depth(of: data) <= JSONSchemaGate.depthLimit,
              (try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)) != nil else { return nil }
        var reader = Reader(bytes: Array(data))
        return reader.value()
    }

    public subscript(key: String) -> OrderedJSON? {
        guard case .object(let members) = self else { return nil }
        return members.first { $0.key == key }?.value
    }

    public var text: String {
        switch self {
        case .scalar(let raw): raw
        case .array(let items): "[" + items.map(\.text).joined(separator: ",") + "]"
        case .object(let members): "{" + members.map { $0.quotedKey + ":" + $0.value.text }.joined(separator: ",") + "}"
        }
    }

    struct Reader {
        let bytes: [UInt8]
        var index = 0

        mutating func value() -> OrderedJSON {
            skipWhitespace()
            switch bytes[index] {
            case UInt8(ascii: "{"):
                index += 1
                var members: [Member] = []
                while true {
                    skipWhitespace()
                    if bytes[index] == UInt8(ascii: "}") { index += 1; return .object(members) }
                    if bytes[index] == UInt8(ascii: ",") { index += 1; continue }
                    let quoted = string()
                    skipWhitespace()
                    index += 1
                    members.append(Member(key: Self.decoded(quoted), quotedKey: quoted, value: value()))
                }
            case UInt8(ascii: "["):
                index += 1
                var items: [OrderedJSON] = []
                while true {
                    skipWhitespace()
                    if bytes[index] == UInt8(ascii: "]") { index += 1; return .array(items) }
                    if bytes[index] == UInt8(ascii: ",") { index += 1; continue }
                    items.append(value())
                }
            case UInt8(ascii: "\""):
                return .scalar(string())
            default:
                let start = index
                while index < bytes.count, !Array(",]} \t\r\n".utf8).contains(bytes[index]) { index += 1 }
                return .scalar(String(decoding: bytes[start..<index], as: UTF8.self))
            }
        }

        mutating func string() -> String {
            let start = index
            index += 1
            while bytes[index] != UInt8(ascii: "\"") { index += bytes[index] == UInt8(ascii: "\\") ? 2 : 1 }
            index += 1
            return String(decoding: bytes[start..<index], as: UTF8.self)
        }

        mutating func skipWhitespace() {
            while index < bytes.count, Array(" \t\r\n".utf8).contains(bytes[index]) { index += 1 }
        }

        static func decoded(_ quoted: String) -> String {
            guard quoted.contains("\\") else { return String(quoted.dropFirst().dropLast()) }
            return (try? JSONSerialization.jsonObject(with: Data(quoted.utf8), options: .fragmentsAllowed)) as? String ?? quoted
        }
    }
}

public enum JSONSchemaGate {
    static let depthLimit = 128
    static let nestingLimit = 64
    static let nodeLimit = 4096
    static let enumValueLimit = 1024
    static let enumBytesLimit = 64 * 1024

    static let enforced: Set<String> = [
        "type", "properties", "required", "additionalProperties", "patternProperties", "propertyNames",
        "unevaluatedProperties", "minProperties", "maxProperties", "items", "prefixItems", "unevaluatedItems",
        "minItems", "maxItems", "enum", "const", "anyOf", "oneOf", "allOf", "$ref", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern", "format",
    ]
    static let annotations: Set<String> = [
        "title", "description", "default", "examples", "$schema", "$id", "$comment", "deprecated", "readOnly",
        "writeOnly", "$defs", "definitions",
    ]
    static let schemaMaps: Set<String> = ["properties", "patternProperties", "$defs", "definitions"]
    static let schemaValues: Set<String> = ["items", "additionalProperties", "propertyNames", "unevaluatedProperties", "unevaluatedItems"]
    static let schemaLists: Set<String> = ["prefixItems", "anyOf", "oneOf", "allOf"]

    public static func prepare(_ schemaJSON: String, strict: Bool) throws(UnsupportedJSONSchema) -> PreparedJSONSchema {
        let data = Data(schemaJSON.utf8)
        guard JSONNesting.depth(of: data) <= depthLimit else {
            throw UnsupportedJSONSchema(message: "the schema nests deeper than \(depthLimit) JSON levels")
        }
        guard let document = OrderedJSON.parse(data) else {
            throw UnsupportedJSONSchema(message: "the schema is not valid JSON")
        }
        var walk = Walk(document: document, strict: strict)
        let prepared = try walk.visit(document, at: "#")
        return PreparedJSONSchema(schema: prepared.text, relaxations: walk.relaxations)
    }

    struct Walk {
        let document: OrderedJSON
        let strict: Bool
        var relaxations: [String] = []
        var nodes = 0
        var nesting = 0
        var followed: Set<String> = []

        init(document: OrderedJSON, strict: Bool) {
            self.document = document
            self.strict = strict
        }

        mutating func relax(_ problem: String, _ relaxation: String) throws(UnsupportedJSONSchema) {
            guard !strict else { throw UnsupportedJSONSchema(message: problem) }
            relaxations.append(relaxation)
        }

        mutating func visit(_ schema: OrderedJSON, at path: String) throws(UnsupportedJSONSchema) -> OrderedJSON {
            nesting += 1
            defer { nesting -= 1 }
            guard nesting <= JSONSchemaGate.nestingLimit else {
                throw UnsupportedJSONSchema(message: "schema at \(path) nests deeper than \(JSONSchemaGate.nestingLimit) levels, counting $ref")
            }
            nodes += 1
            guard nodes <= JSONSchemaGate.nodeLimit else {
                throw UnsupportedJSONSchema(message: "the schema has more than \(JSONSchemaGate.nodeLimit) subschemas")
            }
            guard case .object(var members) = schema else {
                if schema == .scalar("true") { return schema }
                throw UnsupportedJSONSchema(message: "schema at \(path) must be an object")
            }
            for keyword in members.map(\.key) where !JSONSchemaGate.enforced.contains(keyword) && !JSONSchemaGate.annotations.contains(keyword) {
                try relax("unsupported JSON Schema keyword '\(keyword)' at \(path)", "'\(keyword)' at \(path) is not enforced")
            }
            if case .array(let all)? = schema["allOf"], all.count != 1 {
                try relax("allOf at \(path) must list exactly one schema", "allOf at \(path) accepts any value")
            }
            if let reference = schema["$ref"] {
                try follow(reference, at: path)
            }
            try checkLiterals(schema, at: path)
            for index in members.indices {
                let key = members[index].key
                let at = "\(path)/\(key)"
                switch members[index].value {
                case .object(var map) where JSONSchemaGate.schemaMaps.contains(key):
                    for entry in map.indices { map[entry].value = try visit(map[entry].value, at: "\(at)/\(map[entry].key)") }
                    members[index].value = .object(map)
                case .array(let list) where JSONSchemaGate.schemaLists.contains(key):
                    var visited: [OrderedJSON] = []
                    for (position, item) in list.enumerated() { visited.append(try visit(item, at: "\(at)/\(position)")) }
                    members[index].value = .array(visited)
                case .object where JSONSchemaGate.schemaValues.contains(key):
                    members[index].value = try visit(members[index].value, at: at)
                default:
                    break
                }
            }
            return .object(members + openings(schema))
        }

        mutating func follow(_ reference: OrderedJSON, at path: String) throws(UnsupportedJSONSchema) {
            guard case .scalar(let quoted) = reference, quoted.hasPrefix("\"#") else {
                throw UnsupportedJSONSchema(message: "$ref at \(path) is not local to the schema")
            }
            let pointer = OrderedJSON.Reader.decoded(quoted)
            guard followed.insert(pointer).inserted else { return }
            var target = document
            for token in pointer.dropFirst().split(separator: "/", omittingEmptySubsequences: true) {
                let key = token.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
                if let next = target[key] {
                    target = next
                } else if case .array(let items) = target, let index = Int(key), items.indices.contains(index) {
                    target = items[index]
                } else {
                    throw UnsupportedJSONSchema(message: "$ref '\(pointer)' does not resolve")
                }
            }
            _ = try visit(target, at: pointer)
        }

        mutating func checkLiterals(_ schema: OrderedJSON, at path: String) throws(UnsupportedJSONSchema) {
            let values: [OrderedJSON]
            if case .array(let listed)? = schema["enum"] { values = listed } else if let constant = schema["const"] { values = [constant] } else { return }
            guard values.count <= JSONSchemaGate.enumValueLimit, values.reduce(0, { $0 + $1.text.utf8.count }) <= JSONSchemaGate.enumBytesLimit else {
                throw UnsupportedJSONSchema(message:
                    "enum at \(path) lists more than \(JSONSchemaGate.enumValueLimit) values or \(JSONSchemaGate.enumBytesLimit) bytes")
            }
            for case .scalar(let number) in values {
                guard let spelling = Self.xgrammarSpelling(number), spelling != number else { continue }
                try relax("enum at \(path) lists \(number), which XGrammar can only emit as \(spelling)",
                          "enum value \(number) at \(path) is emitted as \(spelling)")
            }
        }

        static func xgrammarSpelling(_ number: String) -> String? {
            guard let first = number.first, first == "-" || first.isNumber, Int64(number) == nil, let value = Double(number) else { return nil }
            return abs(value) < 9_007_199_254_740_992 && value.rounded() == value ? String(format: "%.f", value) : String(format: "%.17g", value)
        }

        func openings(_ schema: OrderedJSON) -> [OrderedJSON.Member] {
            var types: [String] = []
            switch schema["type"] {
            case .scalar(let type)?: types = [type]
            case .array(let list)?: types = list.map(\.text)
            default: break
            }
            var added: [OrderedJSON.Member] = []
            if types.contains("\"object\""), ["properties", "additionalProperties", "patternProperties", "unevaluatedProperties"].allSatisfy({ schema[$0] == nil }) {
                added.append(OrderedJSON.Member(key: "additionalProperties", quotedKey: "\"additionalProperties\"", value: .scalar("true")))
            }
            if types.contains("\"array\""), ["items", "prefixItems", "unevaluatedItems"].allSatisfy({ schema[$0] == nil }) {
                added.append(OrderedJSON.Member(key: "items", quotedKey: "\"items\"", value: .scalar("true")))
            }
            return added
        }
    }
}
