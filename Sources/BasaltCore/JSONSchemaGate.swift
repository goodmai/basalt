import Foundation

public struct UnsupportedJSONSchema: Error, Equatable, Sendable {
    public let message: String
}

public struct PreparedJSONSchema: Sendable, Equatable {
    public let schema: String
    public let relaxations: [String]
}

public enum JSONNesting {
    public static func containsZero(_ data: Data) -> Bool {
        data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return false }
            return memchr(base, 0, bytes.count) != nil
        }
    }

    public static func depth(of data: Data) -> Int {
        data.withUnsafeBytes { bytes in
            var depth = 0
            var deepest = 0
            var index = 0
            while index < bytes.count {
                switch bytes[index] {
                case UInt8(ascii: "\""): index = closingQuote(in: bytes, after: index)
                case UInt8(ascii: "["), UInt8(ascii: "{"):
                    depth += 1
                    deepest = max(deepest, depth)
                case UInt8(ascii: "]"), UInt8(ascii: "}"): depth -= 1
                default: break
                }
                index += 1
            }
            return deepest
        }
    }

    static func closingQuote(in bytes: UnsafeRawBufferPointer, after open: Int) -> Int {
        guard let base = bytes.baseAddress else { return bytes.count }
        var from = open + 1
        while from < bytes.count, let hit = memchr(base + from, Int32(UInt8(ascii: "\"")), bytes.count - from) {
            let quote = base.distance(to: UnsafeRawPointer(hit))
            var backslashes = 0
            while quote - backslashes - 1 > open, bytes[quote - backslashes - 1] == UInt8(ascii: "\\") { backslashes += 1 }
            if backslashes.isMultiple(of: 2) { return quote }
            from = quote + 1
        }
        return bytes.count
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

    public static func parse(_ data: Data, keeping topLevelKeys: Set<String>? = nil) -> OrderedJSON? {
        guard !JSONNesting.containsZero(data), JSONNesting.depth(of: data) <= JSONSchemaGate.depthLimit,
              (try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)) != nil else { return nil }
        let bytes = Array(data)
        var reader = Reader(bytes: bytes.starts(with: [0xEF, 0xBB, 0xBF]) ? Array(bytes.dropFirst(3)) : bytes)
        return try? reader.value(keeping: topLevelKeys)
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
        struct Malformed: Error {}

        static let whitespace: Set<UInt8> = Set(" \t\r\n".utf8)
        static let scalarEnd: Set<UInt8> = whitespace.union(",]}:".utf8)

        let bytes: [UInt8]
        var index = 0
        var depth = 0

        func current() throws(Malformed) -> UInt8 {
            guard index < bytes.count else { throw Malformed() }
            return bytes[index]
        }

        mutating func value(keeping topLevelKeys: Set<String>? = nil) throws(Malformed) -> OrderedJSON {
            skipWhitespace()
            switch try current() {
            case UInt8(ascii: "{"):
                try enter()
                defer { depth -= 1 }
                var members: [Member] = []
                while true {
                    skipWhitespace()
                    switch try current() {
                    case UInt8(ascii: "}"): index += 1; return .object(members)
                    case UInt8(ascii: ","): index += 1; continue
                    default: break
                    }
                    let quoted = try string()
                    skipWhitespace()
                    guard try current() == UInt8(ascii: ":") else { throw Malformed() }
                    index += 1
                    let key = Self.decoded(quoted)
                    if let topLevelKeys, !topLevelKeys.contains(key) {
                        try skipValue()
                    } else {
                        members.append(Member(key: key, quotedKey: quoted, value: try value()))
                    }
                }
            case UInt8(ascii: "["):
                try enter()
                defer { depth -= 1 }
                var items: [OrderedJSON] = []
                while true {
                    skipWhitespace()
                    switch try current() {
                    case UInt8(ascii: "]"): index += 1; return .array(items)
                    case UInt8(ascii: ","): index += 1; continue
                    default: items.append(try value())
                    }
                }
            case UInt8(ascii: "\""):
                return .scalar(try string())
            default:
                let start = index
                while index < bytes.count, !Self.scalarEnd.contains(bytes[index]) { index += 1 }
                guard index > start else { throw Malformed() }
                return .scalar(String(decoding: bytes[start..<index], as: UTF8.self))
            }
        }

        mutating func enter() throws(Malformed) {
            depth += 1
            guard depth <= JSONSchemaGate.depthLimit else { throw Malformed() }
            index += 1
        }

        mutating func skipValue() throws(Malformed) {
            var open = 0
            while true {
                switch try current() {
                case UInt8(ascii: "\""): try skipString()
                case UInt8(ascii: "{"), UInt8(ascii: "["): open += 1; index += 1
                case UInt8(ascii: "}"), UInt8(ascii: "]"):
                    guard open > 0 else { return }
                    open -= 1; index += 1
                case UInt8(ascii: ",") where open == 0: return
                default: index += 1
                }
            }
        }

        mutating func skipString() throws(Malformed) {
            index += 1
            while try current() != UInt8(ascii: "\"") { index += bytes[index] == UInt8(ascii: "\\") ? 2 : 1 }
            index += 1
        }

        mutating func string() throws(Malformed) -> String {
            let start = index
            guard try current() == UInt8(ascii: "\"") else { throw Malformed() }
            try skipString()
            return String(decoding: bytes[start..<index], as: UTF8.self)
        }

        mutating func skipWhitespace() {
            while index < bytes.count, Self.whitespace.contains(bytes[index]) { index += 1 }
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
            try checkDroppedKeywords(schema, keys: members.map(\.key), at: path)
            for index in members.indices {
                members[index].value = try visitSubschemas(members[index].value, under: members[index].key, at: Self.pointer(path, members[index].key))
            }
            return .object(members + openings(schema))
        }

        mutating func visitSubschemas(_ value: OrderedJSON, under key: String, at path: String) throws(UnsupportedJSONSchema) -> OrderedJSON {
            switch value {
            case .object(var map) where JSONSchemaGate.schemaMaps.contains(key):
                for entry in map.indices { map[entry].value = try visit(map[entry].value, at: Self.pointer(path, map[entry].key)) }
                return .object(map)
            case .array(let list) where JSONSchemaGate.schemaLists.contains(key):
                var visited: [OrderedJSON] = []
                for (position, item) in list.enumerated() { visited.append(try visit(item, at: Self.pointer(path, String(position)))) }
                return .array(visited)
            case .object where JSONSchemaGate.schemaValues.contains(key):
                return try visit(value, at: path)
            default:
                return value
            }
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

        func openings(_ schema: OrderedJSON) -> [OrderedJSON.Member] {
            let types = Self.declaredTypes(schema)
            var added: [OrderedJSON.Member] = []
            if types.contains("object"), ["properties", "additionalProperties", "patternProperties", "unevaluatedProperties"].allSatisfy({ schema[$0] == nil }) {
                added.append(OrderedJSON.Member(key: "additionalProperties", quotedKey: "\"additionalProperties\"", value: .scalar("true")))
            }
            if types.contains("array"), ["items", "prefixItems", "unevaluatedItems"].allSatisfy({ schema[$0] == nil }) {
                added.append(OrderedJSON.Member(key: "items", quotedKey: "\"items\"", value: .scalar("true")))
            }
            return added
        }
    }
}

extension JSONSchemaGate {
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
    static let exclusiveKeywords = ["$ref", "const", "enum", "anyOf", "oneOf", "allOf"]
    static let lengthKeywords = ["pattern", "minLength", "maxLength"]
    static let formats: Set<String> = [
        "email", "date", "time", "date-time", "duration", "ipv4", "ipv6", "hostname", "uuid", "uri", "uri-reference",
        "uri-template", "json-pointer", "relative-json-pointer",
    ]
}

extension JSONSchemaGate.Walk {
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

    mutating func checkDroppedKeywords(_ schema: OrderedJSON, keys: [String], at path: String) throws(UnsupportedJSONSchema) {
        let types = Self.declaredTypes(schema)
        if let winner = JSONSchemaGate.exclusiveKeywords.first(where: { schema[$0] != nil }) {
            try checkExclusive(winner, in: schema, keys: keys, types: types, at: path)
            return
        }
        if types.contains("string") { try checkStringKeywords(schema, at: path) }
        try checkRequired(schema, at: path)
    }

    mutating func checkExclusive(
        _ winner: String, in schema: OrderedJSON, keys: [String], types: [String], at path: String
    ) throws(UnsupportedJSONSchema) {
        let literal = winner == "enum" || winner == "const"
        for key in keys where key != winner && !JSONSchemaGate.annotations.contains(key) && !(literal && key == "type") {
            try relax("'\(key)' next to '\(winner)' at \(path) is ignored by XGrammar",
                      "'\(key)' at \(path) is not enforced next to '\(winner)'")
        }
        guard literal, !types.isEmpty else { return }
        let values: [OrderedJSON] = if case .array(let listed)? = schema["enum"] { listed } else { schema["const"].map { [$0] } ?? [] }
        for value in values where !types.contains(where: { Self.value(value, isOf: $0) }) {
            try relax("\(winner) at \(path) lists \(value.text), which is not of the declared type",
                      "\(winner) value \(value.text) at \(path) is accepted despite the declared type")
        }
    }

    mutating func checkStringKeywords(_ schema: OrderedJSON, at path: String) throws(UnsupportedJSONSchema) {
        let format = schema["format"].map { OrderedJSON.Reader.decoded($0.text) }
        if let format, !JSONSchemaGate.formats.contains(format) {
            try relax("format '\(format)' at \(path) is not supported by XGrammar", "format '\(format)' at \(path) is not enforced")
        }
        let leader = format.map(JSONSchemaGate.formats.contains) == true ? "format" : schema["pattern"] != nil ? "pattern" : nil
        guard let leader else { return }
        for key in JSONSchemaGate.lengthKeywords where key != leader && schema[key] != nil {
            try relax("'\(key)' next to '\(leader)' at \(path) is ignored by XGrammar",
                      "'\(key)' at \(path) is not enforced next to '\(leader)'")
        }
    }

    mutating func checkRequired(_ schema: OrderedJSON, at path: String) throws(UnsupportedJSONSchema) {
        guard case .array(let required)? = schema["required"] else { return }
        let declared: Set<String> = if case .object(let members)? = schema["properties"] { Set(members.map(\.key)) } else { [] }
        for case .scalar(let quoted) in required where !declared.contains(OrderedJSON.Reader.decoded(quoted)) {
            try relax("required \(quoted) at \(path) is not among its properties, which XGrammar ignores",
                      "required \(quoted) at \(path) is not enforced")
        }
    }
}

extension JSONSchemaGate.Walk {
    static func declaredTypes(_ schema: OrderedJSON) -> [String] {
        switch schema["type"] {
        case .scalar(let type)?: [OrderedJSON.Reader.decoded(type)]
        case .array(let list)?: list.map { OrderedJSON.Reader.decoded($0.text) }
        case .object?, nil: []
        }
    }

    static func value(_ value: OrderedJSON, isOf type: String) -> Bool {
        switch (value, type) {
        case (.object, "object"), (.array, "array"): return true
        case (.scalar(let raw), _):
            switch type {
            case "string": return raw.hasPrefix("\"")
            case "boolean": return raw == "true" || raw == "false"
            case "null": return raw == "null"
            case "number": return Double(raw) != nil
            case "integer": return Double(raw).map { $0.rounded() == $0 } ?? false
            default: return false
            }
        default: return false
        }
    }

    static func pointer(_ path: String, _ token: String) -> String {
        path + "/" + token.replacingOccurrences(of: "~", with: "~0").replacingOccurrences(of: "/", with: "~1")
    }

    static func xgrammarSpelling(_ number: String) -> String? {
        guard let first = number.first, first == "-" || first.isNumber, Int64(number) == nil, let value = Double(number) else { return nil }
        return abs(value) < 9_007_199_254_740_992 && value.rounded() == value ? String(format: "%.f", value) : String(format: "%.17g", value)
    }
}
