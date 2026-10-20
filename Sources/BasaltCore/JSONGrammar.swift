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

    let nodes: [Node]
    let root: Int
    let starts: [[Int32]]

    init(nodes: [Node], root: Int) throws(UnsupportedJSONSchema) {
        self.nodes = nodes
        self.root = root
        var starts: [[Int32]] = []
        for index in nodes.indices {
            var visiting: Set<Int> = []
            starts.append(try Self.dispatch(index, nodes: nodes, visiting: &visiting))
        }
        self.starts = starts
    }

    private static func dispatch(_ index: Int, nodes: [Node], visiting: inout Set<Int>) throws(UnsupportedJSONSchema) -> [Int32] {
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
            for alternative in alternatives {
                let inner = try dispatch(alternative, nodes: nodes, visiting: &visiting)
                for byte in 0..<256 where inner[byte] >= 0 {
                    guard table[byte] < 0 else {
                        throw UnsupportedJSONSchema(message:
                            "anyOf/oneOf/type alternatives that can start with the same character '\(Character(UnicodeScalar(UInt8(byte))))'")
                    }
                    table[byte] = inner[byte]
                }
            }
            visiting.remove(index)
        }
        return table
    }

    public static func compile(schemaJSON: String) throws(UnsupportedJSONSchema) -> JSONGrammar {
        let schema: JSONValue
        do {
            schema = try JSONDecoder().decode(JSONValue.self, from: Data(schemaJSON.utf8))
        } catch {
            throw UnsupportedJSONSchema(message: "the schema is not valid JSON")
        }
        var compiler = SchemaCompiler(document: schema)
        let root = try compiler.resolve("#")
        return try JSONGrammar(nodes: compiler.nodes, root: root)
    }

    public static func anyObject() -> JSONGrammar {
        var compiler = SchemaCompiler(document: .bool(true))
        let any = compiler.anyValue()
        let root = compiler.freeObject(values: any)
        return try! JSONGrammar(nodes: compiler.nodes, root: root)
    }
}

indirect enum JSONValue: Decodable, Equatable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    var literalText: String? {
        switch self {
        case .null: "null"
        case .bool(let value): value ? "true" : "false"
        case .number(let value):
            value.rounded() == value && abs(value) < 1e15 ? String(Int64(value)) : "\(value)"
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

    let document: JSONValue
    var nodes: [JSONGrammar.Node] = []
    var references: [String: Int] = [:]
    var shared: [String: Int] = [:]

    init(document: JSONValue) {
        self.document = document
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
        guard case .object(let members) = schema else {
            if schema == .bool(true) { return anyValue() }
            throw UnsupportedJSONSchema(message: "schema at \(path) must be an object")
        }
        if let unsupported = members.keys.sorted().first(where: { !Self.keywords.contains($0) && !Self.annotations.contains($0) }) {
            throw UnsupportedJSONSchema(message: "unsupported JSON Schema keyword '\(unsupported)' at \(path)")
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
            return add(.literals(JSONGrammar.LiteralTrie(literals)))
        }
        if case .array(let alternatives)? = members["anyOf"] ?? members["oneOf"] {
            var compiled: [Int] = []
            for (index, alternative) in alternatives.enumerated() {
                compiled.append(try compile(alternative, at: "\(path)/anyOf/\(index)"))
            }
            return add(.choice(compiled))
        }
        let types: [String]
        switch members["type"] {
        case nil: return anyValue()
        case .string(let type)?: types = [type]
        case .array(let list)?:
            types = list.compactMap { if case .string(let type) = $0 { type } else { nil } }
        default: throw UnsupportedJSONSchema(message: "type at \(path) must be a string or an array of strings")
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
        let allowsAdditional = additional != nil && additional != .bool(false)
        if allowsAdditional {
            guard properties.isEmpty, additional == .bool(true) || additional == .object([:]) else {
                throw UnsupportedJSONSchema(message:
                    "additionalProperties at \(path) must be false, or true on an object without properties")
            }
            return freeObject(values: anyValue())
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
}
