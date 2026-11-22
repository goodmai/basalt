import Foundation

public struct UnsupportedJSONSchema: Error, Equatable, Sendable {
    public let message: String
}

public struct PreparedJSONSchema: Sendable, Equatable {
    public let schema: String
    public let relaxations: [String]
}

public enum JSONNesting {
    private static let ones: UInt64 = 0x0101_0101_0101_0101
    private static let highs: UInt64 = 0x8080_8080_8080_8080

    public static func containsZero(_ bytes: Span<UInt8>) -> Bool { nextByte(0, in: bytes, from: 0) != nil }
    public static func containsZero(_ bytes: RawSpan) -> Bool { containsZero(Span(viewing: bytes)) }
    public static func containsZero(_ data: Data) -> Bool { containsZero(data.span) }

    public static func depth(of bytes: Span<UInt8>) -> Int {
        var depth = 0, deepest = 0, index = 0
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

    public static func depth(of bytes: RawSpan) -> Int { depth(of: Span(viewing: bytes)) }
    public static func depth(of data: Data) -> Int { depth(of: data.span) }

    static func closingQuote(in bytes: Span<UInt8>, after open: Int) -> Int {
        var from = open + 1
        while let quote = nextByte(UInt8(ascii: "\""), in: bytes, from: from) {
            var backslashes = 0
            while quote - backslashes - 1 > open, bytes[quote - backslashes - 1] == UInt8(ascii: "\\") { backslashes += 1 }
            if backslashes.isMultiple(of: 2) { return quote }
            from = quote + 1
        }
        return bytes.count
    }

    static func nextByte(_ target: UInt8, in bytes: Span<UInt8>, from start: Int) -> Int? {
        let raw = bytes.bytes
        let pattern = UInt64(target) &* ones
        var index = start
        while index + 8 <= bytes.count {
            let xored = UInt64(littleEndian: raw.unsafeLoadUnaligned(fromByteOffset: index, as: UInt64.self)) ^ pattern
            let mask = (xored &- ones) & ~xored & highs
            if mask != 0 { return index + (mask.trailingZeroBitCount >> 3) }
            index += 8
        }
        while index < bytes.count {
            if bytes[index] == target { return index }
            index += 1
        }
        return nil
    }
}

public indirect enum OrderedJSON: Equatable, Sendable {
    public struct Member: Equatable, Sendable {
        public let key: String
        let quotedKey: String
        public var value: OrderedJSON
    }

    struct Cursor {
        var index = 0
        var depth = 0
    }

    struct Reader {
        struct Malformed: Error {}

        let bytes: [UInt8]
        var index = 0
        var depth = 0

        mutating func value(keeping topLevelKeys: Set<String>? = nil) throws(Malformed) -> OrderedJSON {
            var cursor = Cursor(index: index, depth: depth)
            let parsed = try OrderedJSON.readValue(in: bytes.span, cursor: &cursor, keeping: topLevelKeys)
            (index, depth) = (cursor.index, cursor.depth)
            return parsed
        }

        static func decoded(_ quoted: String) -> String {
            let utf8 = quoted.utf8Span
            return (try? OrderedJSON.unescapeQuoted(utf8.span)) ?? quoted
        }
    }

    case object([Member])
    case array([OrderedJSON])
    case scalar(String)

    public static func parse(_ bytes: Span<UInt8>, keeping topLevelKeys: Set<String>? = nil) -> OrderedJSON? {
        guard !JSONNesting.containsZero(bytes), JSONNesting.depth(of: bytes) <= JSONSchemaGate.depthLimit,
              (try? UTF8Span(validating: bytes)) != nil else { return nil }
        let hasBOM = bytes.count >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF
        let body = hasBOM ? bytes.extracting(droppingFirst: 3) : bytes
        var cursor = Cursor()
        guard let root = try? readValue(in: body, cursor: &cursor, keeping: topLevelKeys) else { return nil }
        skipWhitespace(in: body, at: &cursor.index)
        return cursor.index == body.count ? root : nil
    }

    public static func parse(_ data: Data, keeping topLevelKeys: Set<String>? = nil) -> OrderedJSON? {
        parse(data.span, keeping: topLevelKeys)
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
}

extension OrderedJSON {
    static func readValue(
        in bytes: Span<UInt8>, cursor: inout Cursor, keeping topLevelKeys: Set<String>? = nil, discarding: Bool = false
    ) throws(Reader.Malformed) -> OrderedJSON {
        skipWhitespace(in: bytes, at: &cursor.index)
        switch try current(in: bytes, at: cursor.index) {
        case UInt8(ascii: "{"): return try readObject(in: bytes, cursor: &cursor, keeping: topLevelKeys, discarding: discarding)
        case UInt8(ascii: "["): return try readArray(in: bytes, cursor: &cursor, discarding: discarding)
        case UInt8(ascii: "\""):
            let range = try scanString(in: bytes, cursor: &cursor)
            return .scalar(discarding ? "" : try sliceString(in: bytes, range: range))
        case UInt8(ascii: "t"): return .scalar(try scanLiteral("true", in: bytes, cursor: &cursor))
        case UInt8(ascii: "f"): return .scalar(try scanLiteral("false", in: bytes, cursor: &cursor))
        case UInt8(ascii: "n"): return .scalar(try scanLiteral("null", in: bytes, cursor: &cursor))
        default: return .scalar(try readNumber(in: bytes, cursor: &cursor))
        }
    }

    private static func readObject(
        in bytes: Span<UInt8>, cursor: inout Cursor, keeping topLevelKeys: Set<String>?, discarding: Bool
    ) throws(Reader.Malformed) -> OrderedJSON {
        try enterContainer(cursor: &cursor)
        defer { cursor.depth -= 1 }
        skipWhitespace(in: bytes, at: &cursor.index)
        if try current(in: bytes, at: cursor.index) == UInt8(ascii: "}") { cursor.index += 1; return .object([]) }
        var members: [Member] = []
        while true {
            let keyRange = try scanString(in: bytes, cursor: &cursor)
            try consumeColon(in: bytes, at: &cursor.index)
            if discarding {
                _ = try readValue(in: bytes, cursor: &cursor, discarding: true)
            } else {
                let key = try unescapeQuoted(bytes.extracting(keyRange))
                if let topLevelKeys, !topLevelKeys.contains(key) {
                    _ = try readValue(in: bytes, cursor: &cursor, discarding: true)
                } else {
                    members.append(Member(key: key, quotedKey: try sliceString(in: bytes, range: keyRange),
                                          value: try readValue(in: bytes, cursor: &cursor)))
                }
            }
            if try finishStep(in: bytes, at: &cursor.index, close: UInt8(ascii: "}"), quoteAfterComma: true) { return .object(members) }
        }
    }

    private static func readArray(in bytes: Span<UInt8>, cursor: inout Cursor, discarding: Bool) throws(Reader.Malformed) -> OrderedJSON {
        try enterContainer(cursor: &cursor)
        defer { cursor.depth -= 1 }
        skipWhitespace(in: bytes, at: &cursor.index)
        if try current(in: bytes, at: cursor.index) == UInt8(ascii: "]") { cursor.index += 1; return .array([]) }
        var items: [OrderedJSON] = []
        while true {
            let element = try readValue(in: bytes, cursor: &cursor, discarding: discarding)
            if !discarding { items.append(element) }
            if try finishStep(in: bytes, at: &cursor.index, close: UInt8(ascii: "]"), quoteAfterComma: false) { return .array(items) }
        }
    }

    private static func enterContainer(cursor: inout Cursor) throws(Reader.Malformed) {
        cursor.depth += 1
        guard cursor.depth <= JSONSchemaGate.depthLimit else { throw Reader.Malformed() }
        cursor.index += 1
    }

    private static func consumeColon(in bytes: Span<UInt8>, at index: inout Int) throws(Reader.Malformed) {
        skipWhitespace(in: bytes, at: &index)
        guard try current(in: bytes, at: index) == UInt8(ascii: ":") else { throw Reader.Malformed() }
        index += 1
    }

    private static func finishStep(in bytes: Span<UInt8>, at index: inout Int, close: UInt8, quoteAfterComma: Bool) throws(Reader.Malformed) -> Bool {
        skipWhitespace(in: bytes, at: &index)
        let token = try current(in: bytes, at: index)
        if token == close { index += 1; return true }
        guard token == UInt8(ascii: ",") else { throw Reader.Malformed() }
        index += 1
        skipWhitespace(in: bytes, at: &index)
        let next = try current(in: bytes, at: index)
        guard quoteAfterComma ? (next == UInt8(ascii: "\"")) : (next != close) else { throw Reader.Malformed() }
        return false
    }
}

extension OrderedJSON {
    static func scanString(in bytes: Span<UInt8>, cursor: inout Cursor) throws(Reader.Malformed) -> Range<Int> {
        let start = cursor.index
        guard try current(in: bytes, at: cursor.index) == UInt8(ascii: "\"") else { throw Reader.Malformed() }
        cursor.index += 1
        while true {
            let byte = try current(in: bytes, at: cursor.index)
            guard byte >= 0x20 else { throw Reader.Malformed() }
            if byte == UInt8(ascii: "\"") { cursor.index += 1; return start..<cursor.index }
            cursor.index += 1
            if byte == UInt8(ascii: "\\") { _ = try consumeEscapedUnit(in: bytes, at: &cursor.index) }
        }
    }

    static func unescapeQuoted(_ quoted: Span<UInt8>) throws(Reader.Malformed) -> String {
        guard quoted.count >= 2, quoted[0] == UInt8(ascii: "\""), quoted[quoted.count - 1] == UInt8(ascii: "\"") else { throw Reader.Malformed() }
        let inner = quoted.extracting(1 ..< quoted.count - 1)
        if JSONNesting.nextByte(UInt8(ascii: "\\"), in: inner, from: 0) == nil { return try sliceString(in: inner, range: 0..<inner.count) }
        var out: [UInt8] = [], index = 0
        out.reserveCapacity(inner.count)
        while index < inner.count {
            if inner[index] != UInt8(ascii: "\\") { out.append(inner[index]); index += 1; continue }
            index += 1
            out.append(contentsOf: (try consumeEscapedUnit(in: inner, at: &index)).utf8)
        }
        return try sliceString(in: out.span, range: 0..<out.count)
    }

    private static func consumeEscapedUnit(in bytes: Span<UInt8>, at index: inout Int) throws(Reader.Malformed) -> UnicodeScalar {
        let esc = try current(in: bytes, at: index)
        index += 1
        switch esc {
        case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): return UnicodeScalar(esc)
        case UInt8(ascii: "b"): return UnicodeScalar(0x08)
        case UInt8(ascii: "f"): return UnicodeScalar(0x0C)
        case UInt8(ascii: "n"): return UnicodeScalar(0x0A)
        case UInt8(ascii: "r"): return UnicodeScalar(0x0D)
        case UInt8(ascii: "t"): return UnicodeScalar(0x09)
        case UInt8(ascii: "u"): return try decodeScalarEscape(in: bytes, at: &index)
        default: throw Reader.Malformed()
        }
    }

    private static func decodeScalarEscape(in bytes: Span<UInt8>, at index: inout Int) throws(Reader.Malformed) -> UnicodeScalar {
        let unit = try parseHex4(in: bytes, at: &index)
        if let scalar = UnicodeScalar(unit) { return scalar }
        guard (0xD800...0xDBFF).contains(unit), index + 2 <= bytes.count,
              bytes[index] == UInt8(ascii: "\\"), bytes[index + 1] == UInt8(ascii: "u") else { throw Reader.Malformed() }
        index += 2
        let low = try parseHex4(in: bytes, at: &index)
        guard (0xDC00...0xDFFF).contains(low),
              let combined = UnicodeScalar(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)) else { throw Reader.Malformed() }
        return combined
    }

    private static func parseHex4(in bytes: Span<UInt8>, at index: inout Int) throws(Reader.Malformed) -> UInt32 {
        guard index + 4 <= bytes.count else { throw Reader.Malformed() }
        var value: UInt32 = 0
        for offset in 0..<4 {
            let b = bytes[index + offset]
            let digit: UInt32 = if b >= UInt8(ascii: "0"), b <= UInt8(ascii: "9") { UInt32(b - UInt8(ascii: "0")) }
            else if b >= UInt8(ascii: "a"), b <= UInt8(ascii: "f") { UInt32(b - UInt8(ascii: "a") + 10) }
            else if b >= UInt8(ascii: "A"), b <= UInt8(ascii: "F") { UInt32(b - UInt8(ascii: "A") + 10) }
            else { throw Reader.Malformed() }
            value = (value << 4) | digit
        }
        index += 4
        return value
    }

    static func scanLiteral(_ literal: StaticString, in bytes: Span<UInt8>, cursor: inout Cursor) throws(Reader.Malformed) -> String {
        let count = literal.utf8CodeUnitCount
        guard cursor.index + count <= bytes.count, literal.withUTF8Buffer({ target in
            for offset in 0..<count where bytes[cursor.index + offset] != target[offset] { return false }
            return true
        }) else { throw Reader.Malformed() }
        cursor.index += count
        guard cursor.index == bytes.count || isScalarBoundary(bytes[cursor.index]) else { throw Reader.Malformed() }
        return String(describing: literal)
    }

    static func readNumber(in bytes: Span<UInt8>, cursor: inout Cursor) throws(Reader.Malformed) -> String {
        let start = cursor.index
        if cursor.index < bytes.count, bytes[cursor.index] == UInt8(ascii: "-") { cursor.index += 1 }
        let first = try current(in: bytes, at: cursor.index)
        if first == UInt8(ascii: "0") {
            cursor.index += 1
            if cursor.index < bytes.count, isDigit(bytes[cursor.index]) { throw Reader.Malformed() }
        } else if first >= UInt8(ascii: "1"), first <= UInt8(ascii: "9") {
            cursor.index += 1
            consumeDigits(in: bytes, at: &cursor.index)
        } else { throw Reader.Malformed() }
        try consumeFractionAndExponent(in: bytes, at: &cursor.index)
        guard cursor.index == bytes.count || isScalarBoundary(bytes[cursor.index]) else { throw Reader.Malformed() }
        let raw = try sliceString(in: bytes, range: start..<cursor.index)
        guard Int64(raw) != nil || Double(raw)?.isFinite == true else { throw Reader.Malformed() }
        return raw
    }

    private static func consumeFractionAndExponent(in bytes: Span<UInt8>, at index: inout Int) throws(Reader.Malformed) {
        if index < bytes.count, bytes[index] == UInt8(ascii: ".") {
            index += 1
            guard index < bytes.count, isDigit(bytes[index]) else { throw Reader.Malformed() }
            consumeDigits(in: bytes, at: &index)
        }
        if index < bytes.count, bytes[index] == UInt8(ascii: "e") || bytes[index] == UInt8(ascii: "E") {
            index += 1
            if index < bytes.count, bytes[index] == UInt8(ascii: "+") || bytes[index] == UInt8(ascii: "-") { index += 1 }
            guard index < bytes.count, isDigit(bytes[index]) else { throw Reader.Malformed() }
            consumeDigits(in: bytes, at: &index)
        }
    }

    static func sliceString(in bytes: Span<UInt8>, range: Range<Int>) throws(Reader.Malformed) -> String {
        guard let utf8 = try? UTF8Span(validating: bytes.extracting(range)) else { throw Reader.Malformed() }
        return String(copying: utf8)
    }

    static func current(in bytes: Span<UInt8>, at index: Int) throws(Reader.Malformed) -> UInt8 {
        guard index < bytes.count else { throw Reader.Malformed() }
        return bytes[index]
    }

    static func skipWhitespace(in bytes: Span<UInt8>, at index: inout Int) { while index < bytes.count, isWhitespace(bytes[index]) { index += 1 } }
    static func consumeDigits(in bytes: Span<UInt8>, at index: inout Int) { while index < bytes.count, isDigit(bytes[index]) { index += 1 } }
    static func isDigit(_ byte: UInt8) -> Bool { byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9") }
    static func isWhitespace(_ byte: UInt8) -> Bool { byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D }
    static func isScalarBoundary(_ byte: UInt8) -> Bool { isWhitespace(byte) || byte == 0x2C || byte == 0x5D || byte == 0x7D }
}

public enum JSONSchemaGate {
    static let depthLimit = 128
    static let nestingLimit = 64
    static let nodeLimit = 4096
    static let enumValueLimit = 1024
    static let enumBytesLimit = 64 * 1024

    public static func prepare(_ schemaBytes: Span<UInt8>, strict: Bool) throws(UnsupportedJSONSchema) -> PreparedJSONSchema {
        guard JSONNesting.depth(of: schemaBytes) <= depthLimit else {
            throw UnsupportedJSONSchema(message: "the schema nests deeper than \(depthLimit) JSON levels")
        }
        guard let document = OrderedJSON.parse(schemaBytes) else {
            throw UnsupportedJSONSchema(message: "the schema is not valid JSON")
        }
        var walk = Walk(document: document, strict: strict)
        let prepared = try walk.visit(document, at: "#")
        return PreparedJSONSchema(schema: prepared.text, relaxations: walk.relaxations)
    }

    public static func prepare(_ schemaData: Data, strict: Bool) throws(UnsupportedJSONSchema) -> PreparedJSONSchema {
        try prepare(schemaData.span, strict: strict)
    }

    public static func prepare(_ schemaJSON: String, strict: Bool) throws(UnsupportedJSONSchema) -> PreparedJSONSchema {
        let utf8 = schemaJSON.utf8Span
        return try prepare(utf8.span, strict: strict)
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
            guard pointer == "#" || pointer.hasPrefix("#/") else {
                throw UnsupportedJSONSchema(message: "$ref '\(pointer)' is not a JSON Pointer")
            }
            guard followed.insert(pointer).inserted else { return }
            var target = document
            let tokens = pointer == "#" ? [] : pointer.dropFirst(2).split(separator: "/", omittingEmptySubsequences: false)
            for token in tokens {
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
