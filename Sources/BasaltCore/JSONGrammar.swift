import Foundation

public struct UnsupportedJSONSchema: Error, Equatable, Sendable {
    public let message: String
}

public final class JSONGrammar: Sendable {
    enum Node: Sendable {
        case string
        case number(integerOnly: Bool)
        case literals(LiteralTrie)
        case object(ObjectShape)
        case array(items: Int)
        case choice([Int])
    }

    struct LiteralTrie: Sendable {
        var children: [[UInt8: Int32]] = [[:]]
        var terminal: [Int32] = [-1]
        var reachable: [UInt64] = [0]

        init(_ literals: [String]) {
            for (index, literal) in literals.enumerated() {
                var cursor = 0
                reachable[0] |= Self.bit(index)
                for byte in literal.utf8 {
                    if let next = children[cursor][byte] {
                        cursor = Int(next)
                    } else {
                        children.append([:])
                        terminal.append(-1)
                        reachable.append(0)
                        children[cursor][byte] = Int32(children.count - 1)
                        cursor = children.count - 1
                    }
                    reachable[cursor] |= Self.bit(index)
                }
                if terminal[cursor] < 0 { terminal[cursor] = Int32(index) }
            }
        }

        static func bit(_ index: Int) -> UInt64 { index < 64 ? 1 << UInt64(index) : .max }
    }

    struct ObjectShape: Sendable {
        let keys: Int
        let declared: Bool
        let values: [Int]
        let required: UInt64
        let all: UInt64
    }

    static let schemaDepthLimit = 128
    static let nestingLimit = 64
    static let nodeLimit = 4096
    static let enumValueLimit = 1024
    static let enumBytesLimit = 64 * 1024

    let nodes: [Node]
    let root: Int
    let starts: [[Int32]]
    public let relaxations: [String]

    init(nodes: [Node], root: Int, starts: [[Int32]], relaxations: [String]) {
        self.nodes = nodes
        self.root = root
        self.starts = starts
        self.relaxations = relaxations
    }

    public static func compile(schemaJSON: String, strict: Bool = true) throws(UnsupportedJSONSchema) -> JSONGrammar {
        let data = Data(schemaJSON.utf8)
        guard JSONNesting.depth(of: data) <= schemaDepthLimit else {
            throw UnsupportedJSONSchema(message: "the schema nests deeper than \(schemaDepthLimit) JSON levels")
        }
        guard let parsed = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed),
              let schema = JSONValue(parsed) else {
            throw UnsupportedJSONSchema(message: "the schema is not valid JSON")
        }
        var compiler = SchemaCompiler(document: schema, strict: strict)
        let root = try compiler.resolve("#")
        return try compiler.grammar(root: root)
    }

    public static func anyObject() -> JSONGrammar {
        var compiler = SchemaCompiler(document: .bool(true), strict: true)
        let any = compiler.anyValue()
        let root = compiler.freeObject(values: any)
        return try! compiler.grammar(root: root)
    }
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

indirect enum JSONValue: Equatable, Sendable {
    case null
    case bool(Bool)
    case number(String)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init?(_ value: Any) {
        switch value {
        case is NSNull:
            self = .null
        case let number as NSNumber:
            self = CFGetTypeID(number) == CFBooleanGetTypeID() ? .bool(number.boolValue) : .number(Self.text(of: number))
        case let string as String:
            self = .string(string)
        case let items as [Any]:
            var values: [JSONValue] = []
            for item in items {
                guard let value = JSONValue(item) else { return nil }
                values.append(value)
            }
            self = .array(values)
        case let members as [String: Any]:
            var values: [String: JSONValue] = [:]
            for (key, member) in members {
                guard let value = JSONValue(member) else { return nil }
                values[key] = value
            }
            self = .object(values)
        default:
            return nil
        }
    }

    static func text(of number: NSNumber) -> String {
        let type = number.objCType.pointee
        guard !(number is NSDecimalNumber), type == CChar(UInt8(ascii: "d")) || type == CChar(UInt8(ascii: "f")) else {
            return number.stringValue
        }
        let value = number.doubleValue
        return value.rounded() == value && abs(value) < 9_007_199_254_740_992 ? String(Int64(value)) : "\(value)"
    }

    var literalText: String? {
        switch self {
        case .null: "null"
        case .bool(let value): value ? "true" : "false"
        case .number(let text): text
        case .string(let value): Self.quoted(value)
        case .array, .object: nil
        }
    }

    static func quoted(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case _ where scalar.value < 0x20: out += String(format: "\\u%04x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }
}

struct SchemaCompiler {
    static let annotations: Set<String> = [
        "title", "description", "default", "examples", "$schema", "$id", "$comment", "format",
        "deprecated", "readOnly", "writeOnly", "$defs", "definitions",
    ]
    static let keywords: Set<String> = [
        "type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf", "oneOf", "$ref",
    ]
    static let valueConstraints: Set<String> = [
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern",
        "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties", "propertyNames",
        "contentEncoding", "contentMediaType",
    ]

    let document: JSONValue
    let strict: Bool
    var nodes: [JSONGrammar.Node] = []
    var references: [String: Int] = [:]
    var shared: [String: Int] = [:]
    var relaxations: [String] = []
    private var nesting = 0

    init(document: JSONValue, strict: Bool) {
        self.document = document
        self.strict = strict
    }

    mutating func add(_ node: JSONGrammar.Node) -> Int {
        nodes.append(node)
        return nodes.count - 1
    }

    mutating func sharedNode(_ name: String, _ make: () -> JSONGrammar.Node) -> Int {
        if let known = shared[name] { return known }
        let index = add(make())
        shared[name] = index
        return index
    }

    mutating func anyValue() -> Int {
        if let known = shared["any"] { return known }
        let any = add(.choice([]))
        shared["any"] = any
        let object = freeObject(values: any)
        let array = add(.array(items: any))
        let string = sharedNode("string") { .string }
        let number = sharedNode("number") { .number(integerOnly: false) }
        let literals = sharedNode("true|false|null") { .literals(JSONGrammar.LiteralTrie(["true", "false", "null"])) }
        nodes[any] = .choice([object, array, string, number, literals])
        return any
    }

    mutating func freeObject(values: Int) -> Int {
        let keys = sharedNode("string") { .string }
        return add(.object(JSONGrammar.ObjectShape(keys: keys, declared: false, values: [values], required: 0, all: 0)))
    }

    mutating func anyValue(startingWith byte: UInt8) -> Int {
        guard case .choice(let kinds) = nodes[anyValue()], kinds.count == 5 else { return anyValue() }
        switch byte {
        case UInt8(ascii: "{"): return kinds[0]
        case UInt8(ascii: "["): return kinds[1]
        case UInt8(ascii: "\""): return kinds[2]
        case UInt8(ascii: "t"), UInt8(ascii: "f"), UInt8(ascii: "n"): return kinds[4]
        default: return kinds[3]
        }
    }

    mutating func resolve(_ reference: String) throws(UnsupportedJSONSchema) -> Int {
        if let known = references[reference] { return known }
        guard reference.hasPrefix("#") else {
            throw UnsupportedJSONSchema(message: "$ref '\(reference)' is not local to the schema")
        }
        var target = document
        for token in reference.dropFirst().split(separator: "/", omittingEmptySubsequences: true) {
            let key = token.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
            switch target {
            case .object(let members) where members[key] != nil:
                target = members[key]!
            case .array(let items) where Int(key).map(items.indices.contains) == true:
                target = items[Int(key)!]
            default:
                throw UnsupportedJSONSchema(message: "$ref '\(reference)' does not resolve")
            }
        }
        let alias = add(.choice([]))
        references[reference] = alias
        let resolved = try compile(target, at: reference)
        nodes[alias] = .choice([resolved])
        return alias
    }

    mutating func compile(_ schema: JSONValue, at path: String) throws(UnsupportedJSONSchema) -> Int {
        nesting += 1
        defer { nesting -= 1 }
        guard nesting <= JSONGrammar.nestingLimit else {
            throw UnsupportedJSONSchema(message: "schema at \(path) nests deeper than \(JSONGrammar.nestingLimit) levels, counting $ref")
        }
        guard nodes.count < JSONGrammar.nodeLimit else {
            throw UnsupportedJSONSchema(message: "the schema needs more than \(JSONGrammar.nodeLimit) grammar nodes")
        }
        guard case .object(let members) = schema else {
            if schema == .bool(true) { return anyValue() }
            throw UnsupportedJSONSchema(message: "schema at \(path) must be an object")
        }
        for keyword in members.keys.sorted() where !Self.keywords.contains(keyword) && !Self.annotations.contains(keyword) {
            guard !strict, Self.valueConstraints.contains(keyword) else {
                throw UnsupportedJSONSchema(message: "unsupported JSON Schema keyword '\(keyword)' at \(path)")
            }
            relaxations.append("'\(keyword)' at \(path) is not enforced")
        }
        if case .string(let reference)? = members["$ref"] {
            return try resolve(reference)
        }
        if let values = members["enum"].map({ value -> [JSONValue] in
            if case .array(let items) = value { return items }
            return [value]
        }) ?? members["const"].map({ [$0] }) {
            let literals = values.compactMap(\.literalText)
            guard !literals.isEmpty, literals.count == values.count else {
                throw UnsupportedJSONSchema(message: "enum/const at \(path) must list strings, numbers, booleans or null")
            }
            guard literals.count <= JSONGrammar.enumValueLimit,
                  literals.reduce(0, { $0 + $1.utf8.count }) <= JSONGrammar.enumBytesLimit else {
                throw UnsupportedJSONSchema(message:
                    "enum at \(path) lists more than \(JSONGrammar.enumValueLimit) values or \(JSONGrammar.enumBytesLimit) bytes")
            }
            return add(.literals(JSONGrammar.LiteralTrie(literals)))
        }
        if case .array(let alternatives)? = members["anyOf"] ?? members["oneOf"] {
            var compiled: [Int] = []
            for (index, alternative) in alternatives.enumerated() {
                compiled.append(try compile(alternative, at: "\(path)/anyOf/\(index)"))
            }
            return add(.choice(compiled))
        }
        var types: [String] = []
        switch members["type"] {
        case nil:
            return anyValue()
        case .string(let type)?:
            types = [type]
        case .array(let list)? where !list.isEmpty:
            for entry in list {
                guard case .string(let type) = entry else {
                    throw UnsupportedJSONSchema(message: "type at \(path) must list only strings")
                }
                types.append(type)
            }
        default:
            throw UnsupportedJSONSchema(message: "type at \(path) must be a string or a non-empty array of strings")
        }
        var compiled: [Int] = []
        for type in types {
            compiled.append(try typed(type, members, at: path))
        }
        return compiled.count == 1 ? compiled[0] : add(.choice(compiled))
    }

    private mutating func typed(_ type: String, _ members: [String: JSONValue], at path: String) throws(UnsupportedJSONSchema) -> Int {
        switch type {
        case "string": return sharedNode("string") { .string }
        case "number": return sharedNode("number") { .number(integerOnly: false) }
        case "integer": return sharedNode("integer") { .number(integerOnly: true) }
        case "boolean": return sharedNode("true|false") { .literals(JSONGrammar.LiteralTrie(["true", "false"])) }
        case "null": return sharedNode("null") { .literals(JSONGrammar.LiteralTrie(["null"])) }
        case "array": return add(.array(items: try compile(members["items"] ?? .bool(true), at: "\(path)/items")))
        case "object": return try object(members, at: path)
        default: throw UnsupportedJSONSchema(message: "unsupported type '\(type)' at \(path)")
        }
    }

    private mutating func object(_ members: [String: JSONValue], at path: String) throws(UnsupportedJSONSchema) -> Int {
        var properties: [String: JSONValue] = [:]
        if case .object(let declared)? = members["properties"] { properties = declared }
        let additional = members["additionalProperties"]
        if properties.isEmpty, members["required"] == nil, additional != .bool(false) {
            return freeObject(values: try compile(additional ?? .bool(true), at: "\(path)/additionalProperties"))
        }
        if let additional, additional != .bool(false) {
            guard !strict else {
                throw UnsupportedJSONSchema(message: "additionalProperties at \(path) must be false when properties are declared")
            }
            relaxations.append("additionalProperties at \(path): only the declared properties are generated")
        }
        guard properties.count <= 64 else {
            throw UnsupportedJSONSchema(message: "more than 64 properties at \(path)")
        }
        let names = properties.keys.sorted()
        var required: UInt64 = 0
        if case .array(let list)? = members["required"] {
            for case .string(let name) in list {
                guard let index = names.firstIndex(of: name) else {
                    throw UnsupportedJSONSchema(message: "required property '\(name)' at \(path) is not in properties")
                }
                required |= 1 << UInt64(index)
            }
        }
        var values: [Int] = []
        for name in names {
            values.append(try compile(properties[name]!, at: "\(path)/properties/\(name)"))
        }
        let keys = add(.literals(JSONGrammar.LiteralTrie(names.map(JSONValue.quoted))))
        let all: UInt64 = names.isEmpty ? 0 : UInt64.max >> UInt64(64 - names.count)
        return add(.object(JSONGrammar.ObjectShape(keys: keys, declared: true, values: values, required: required, all: all)))
    }

    mutating func grammar(root: Int) throws(UnsupportedJSONSchema) -> JSONGrammar {
        var starts: [[Int32]?] = []
        var index = 0
        while index < nodes.count {
            var visiting: Set<Int> = []
            _ = try dispatch(index, into: &starts, visiting: &visiting)
            index += 1
        }
        return JSONGrammar(nodes: nodes, root: root, starts: starts.map { $0! }, relaxations: relaxations)
    }

    private mutating func dispatch(
        _ index: Int, into starts: inout [[Int32]?], visiting: inout Set<Int>
    ) throws(UnsupportedJSONSchema) -> [Int32] {
        if starts.count < nodes.count { starts += [[Int32]?](repeating: nil, count: nodes.count - starts.count) }
        if let known = starts[index] { return known }
        var table = [Int32](repeating: -1, count: 256)
        let concrete = Int32(index)
        switch nodes[index] {
        case .string:
            table[Int(UInt8(ascii: "\""))] = concrete
        case .number:
            for byte in Array("-0123456789".utf8) { table[Int(byte)] = concrete }
        case .literals(let trie):
            for byte in trie.children[0].keys { table[Int(byte)] = concrete }
        case .object:
            table[Int(UInt8(ascii: "{"))] = concrete
        case .array:
            table[Int(UInt8(ascii: "["))] = concrete
        case .choice(let alternatives):
            guard visiting.insert(index).inserted else {
                throw UnsupportedJSONSchema(message: "a $ref or anyOf that refers to itself without an object or array in between")
            }
            var widenedFirsts: [Character] = []
            for alternative in alternatives {
                let inner = try dispatch(alternative, into: &starts, visiting: &visiting)
                for byte in 0..<256 where inner[byte] >= 0 && table[byte] != inner[byte] {
                    guard table[byte] >= 0 else { table[byte] = inner[byte]; continue }
                    let first = Character(UnicodeScalar(UInt8(byte)))
                    guard !strict else {
                        throw UnsupportedJSONSchema(message: "anyOf/oneOf/type alternatives that can start with the same character '\(first)'")
                    }
                    let widened = Int32(anyValue(startingWith: UInt8(byte)))
                    if table[byte] != widened { widenedFirsts.append(first) }
                    table[byte] = widened
                }
            }
            if !widenedFirsts.isEmpty {
                relaxations.append("anyOf/oneOf/type alternatives that overlap on '\(String(widenedFirsts))' accept any JSON value starting with those")
            }
            visiting.remove(index)
        }
        if starts.count < nodes.count { starts += [[Int32]?](repeating: nil, count: nodes.count - starts.count) }
        starts[index] = table
        return table
    }
}
