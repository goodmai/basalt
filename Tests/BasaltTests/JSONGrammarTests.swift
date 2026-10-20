import Foundation
import Testing
import Synchronization
@testable import BasaltCore

func grammar(_ schema: String) throws -> JSONGrammar {
    try JSONGrammar.compile(schemaJSON: schema)
}

func matches(_ grammar: JSONGrammar, _ text: String) -> Bool {
    var matcher = JSONGrammarMatcher(grammar: grammar)
    return matcher.accept(Array(text.utf8)) && matcher.isAccepting
}

func isPrefix(_ grammar: JSONGrammar, _ text: String) -> Bool {
    var matcher = JSONGrammarMatcher(grammar: grammar)
    return matcher.accept(Array(text.utf8))
}

@Suite("JSON Schema pushdown automaton (#42)")
struct JSONGrammarTests {
    static let tree = #"""
        {"$defs": {"node": {"type": "object",
            "properties": {"value": {"type": "integer"}, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}},
            "required": ["value", "children"]}},
         "$ref": "#/$defs/node"}
        """#

    @Test("A recursive $ref tree accepts any depth and rejects a wrong type, a missing field or an extra field deep inside")
    func recursiveTree() throws {
        let tree = try grammar(Self.tree)
        #expect(matches(tree, #"{"value":1,"children":[]}"#))
        #expect(matches(tree, #"{"value":1,"children":[{"value":2,"children":[{"value":3,"children":[]}]},{"children":[],"value":4}]}"#))
        #expect(!matches(tree, #"{"value":1,"children":[{"value":"2","children":[]}]}"#))
        #expect(!matches(tree, #"{"value":1,"children":[{"value":2}]}"#))
        #expect(!matches(tree, #"{"value":1,"children":[{"value":2,"children":[],"extra":0}]}"#))
        #expect(!isPrefix(tree, #"{"value":1,"children":[{"value":2,"children":[]},]"#))
    }

    @Test("Nested arrays and objects, an array of objects with an enum, any key order, no duplicates")
    func nestedContainers() throws {
        let schema = try grammar(#"""
            {"type": "object", "properties": {
                "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "items": {"type": "array", "items": {"type": "object",
                    "properties": {"kind": {"enum": ["red", "green", 12]}, "ok": {"type": "boolean"}},
                    "required": ["kind"]}}},
             "required": ["matrix", "items"]}
            """#)
        #expect(matches(schema, #"{"matrix":[[1,2.5],[],[-3e2]],"items":[{"kind":"red","ok":true},{"kind":12}]}"#))
        #expect(matches(schema, #"{"items":[],"matrix":[]}"#))
        #expect(!matches(schema, #"{"matrix":[],"items":[{"kind":"blue"}]}"#))
        #expect(!matches(schema, #"{"matrix":[],"items":[{"kind":1}]}"#))
        #expect(!matches(schema, #"{"matrix":[],"items":[{"ok":true}]}"#))
        #expect(!isPrefix(schema, #"{"matrix":[],"matrix""#))
        #expect(!isPrefix(schema, #"{"matrix":[[1,2],]"#))
        #expect(!isPrefix(schema, #"{}"#))
    }

    @Test("String escapes, unicode escapes and UTF-8 are checked byte by byte")
    func strings() throws {
        let string = try grammar(#"{"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}"#)
        #expect(matches(string, #"{"s":"a\"b\\c\/\u00e9\n\t"}"#))
        #expect(matches(string, "{\"s\":\"привет 👋\"}"))
        #expect(!isPrefix(string, #"{"s":"\x"#))
        #expect(!isPrefix(string, #"{"s":"\u12g"#))
        #expect(!isPrefix(string, "{\"s\":\"a\nb"))
        for invalid: [UInt8] in [[0xC0, 0x80], [0x80], [0xED, 0xA0, 0x80], [0xF4, 0x90], [0xE2, 0x82, 0x22]] {
            var matcher = JSONGrammarMatcher(grammar: string)
            let accepted = matcher.accept(Array(#"{"s":""#.utf8) + invalid)
            #expect(!accepted, "\(invalid)")
        }
    }

    @Test("Numbers follow the JSON grammar; integer rejects a fraction or an exponent")
    func numbers() throws {
        let number = try grammar(#"{"type": "number"}"#)
        for valid in ["0", "-0", "12.5e-3", "1E+2", "-7", "0.25"] { #expect(matches(number, valid), "\(valid)") }
        for invalid in ["01", "1.", ".5", "-", "1e", "+1", "1.e3", "--1"] { #expect(!matches(number, invalid), "\(invalid)") }
        let integer = try grammar(#"{"type": "integer"}"#)
        #expect(matches(integer, "-42"))
        #expect(!matches(integer, "1.5"))
        #expect(!matches(integer, "1e3"))
    }

    @Test("An enum prefix of another enum value completes only where the grammar ends it")
    func enumPrefixes() throws {
        let numbers = try grammar(#"{"type": "array", "items": {"enum": [1, 12, "a"]}}"#)
        #expect(matches(numbers, "[1,12,\"a\"]"))
        #expect(!matches(numbers, "[123]"))
        #expect(!matches(numbers, "[\"ab\"]"))
    }

    @Test("Whitespace is allowed between tokens up to a bounded run, never after the top-level value")
    func whitespace() throws {
        let schema = try grammar(#"{"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}}}"#)
        #expect(matches(schema, "\n{\n  \"a\" : [ 1 ,\n\t2 ]\n}"))
        #expect(!isPrefix(schema, "{" + String(repeating: " ", count: 33)))
        #expect(isPrefix(schema, "{" + String(repeating: " ", count: 32)))
        #expect(!isPrefix(schema, "{} "))
        #expect(matches(schema, "{}"))
    }

    @Test("anyOf and type arrays dispatch on the first byte, so optional fields can be null")
    func unions() throws {
        let schema = try grammar(#"""
            {"type": "object", "properties": {
                "a": {"type": ["string", "null"]},
                "b": {"anyOf": [{"type": "integer"}, {"type": "object", "properties": {"x": {"type": "boolean"}}, "required": ["x"]}]}},
             "required": ["a", "b"]}
            """#)
        #expect(matches(schema, #"{"a":null,"b":3}"#))
        #expect(matches(schema, #"{"a":"s","b":{"x":false}}"#))
        #expect(!matches(schema, #"{"a":1,"b":3}"#))
        #expect(!matches(schema, #"{"a":null,"b":"3"}"#))
    }

    @Test("json_object accepts any object with any nesting and rejects a top-level array")
    func anyObject() {
        let any = JSONGrammar.anyObject()
        #expect(matches(any, #"{"k":[1,{"x":null,"y":[true,false,"z"]}],"k":{}}"#))
        #expect(!matches(any, "[1]"))
        #expect(!matches(any, #"{"k":}"#))
    }

    @Test("Unsupported schema features fail with a message naming the keyword and where it is")
    func unsupported() {
        func message(_ schema: String) -> String? {
            do { _ = try JSONGrammar.compile(schemaJSON: schema); return nil } catch { return error.message }
        }
        #expect(message(#"{"type": "object", "properties": {"n": {"type": "string", "minLength": 2}}}"#)
            == "unsupported JSON Schema keyword 'minLength' at #/properties/n")
        #expect(message(#"{"type": "object", "properties": {"a": {}}, "additionalProperties": {"type": "string"}}"#)?.contains("additionalProperties") == true)
        #expect(message(#"{"anyOf": [{"type": "string"}, {"enum": ["a"]}]}"#)?.contains("same character") == true)
        #expect(message(##"{"$ref": "#"}"##)?.contains("refers to itself") == true)
        #expect(message(#"{"$ref": "https://example.com/s.json"}"#)?.contains("not local") == true)
        #expect(message(#"{"type": "object", "properties": {}, "required": ["x"]}"#)?.contains("'x'") == true)
        #expect(message(#"{"type": "object", "properties": {"d": {"type": "string", "description": "kept", "format": "uri"}}}"#) == nil)
    }

    static func onServerSizedStack<T: Sendable>(_ work: @escaping @Sendable () -> T) -> T {
        let done = DispatchSemaphore(value: 0)
        let result = Mutex<T?>(nil)
        let thread = Thread {
            result.withLock { $0 = work() }
            done.signal()
        }
        thread.stackSize = 512 * 1024
        thread.start()
        done.wait()
        return result.withLock { $0! }
    }

    static func timedMessage(_ schema: String) -> (message: String?, seconds: Double) {
        onServerSizedStack {
            let start = ContinuousClock.now
            let message: String?
            do { _ = try JSONGrammar.compile(schemaJSON: schema); message = nil } catch { message = (error as? UnsupportedJSONSchema)?.message ?? "\(error)" }
            return (message, (ContinuousClock.now - start) / .seconds(1))
        }
    }

    @Test("Deep nesting, a long $ref chain and a huge enum are refused quickly on a 512 KB stack instead of crashing")
    func hostileSchemasAreBounded() {
        let deepArrays = String(repeating: #"{"type": "array", "items": "#, count: 300) + "{}" + String(repeating: "}", count: 300)
        let raw = String(repeating: "[", count: 300) + String(repeating: "]", count: 300)
        func chain(_ length: Int) -> String {
            let defs = (0..<length).map { "\"d\($0)\": {\"$ref\": \"#/$defs/d\($0 + 1)\"}" }.joined(separator: ", ")
            return "{\"$defs\": {\(defs), \"d\(length)\": {\"type\": \"string\"}}, \"$ref\": \"#/$defs/d0\"}"
        }
        let hugeEnum = #"{"enum": ["# + (0..<200_000).map { "\"v\($0)\"" }.joined(separator: ",") + "]}"
        for (schema, expected) in [(deepArrays, "nests deeper"), (raw, "nests deeper"), (chain(1000), "counting $ref"),
                                   (chain(5000), "counting $ref"), (hugeEnum, "more than 1024 values")] {
            let (message, seconds) = Self.timedMessage(schema)
            #expect(message?.contains(expected) == true, "\(message ?? "compiled")")
            #expect(seconds < 1, "\(seconds) s")
        }
        let (fits, _) = Self.timedMessage(chain(60))
        #expect(fits == nil)
    }

    @Test("type object without properties is any object; additionalProperties with a schema constrains its values")
    func freeObjects() throws {
        let object = try grammar(#"{"type": "object"}"#)
        #expect(matches(object, #"{"a":1,"b":{"c":[null]}}"#))
        #expect(matches(object, "{}"))
        #expect(!matches(object, "[]"))
        let strings = try grammar(#"{"type": "object", "additionalProperties": {"type": "string"}}"#)
        #expect(matches(strings, #"{"a":"x","b":"y"}"#))
        #expect(!matches(strings, #"{"a":1}"#))
        let closed = try grammar(#"{"type": "object", "additionalProperties": false}"#)
        #expect(matches(closed, "{}"))
        #expect(!matches(closed, #"{"a":1}"#))
    }

    @Test("type must be a string or a non-empty list of strings")
    func typeLists() {
        func message(_ schema: String) -> String? {
            do { _ = try JSONGrammar.compile(schemaJSON: schema); return nil } catch { return error.message }
        }
        #expect(message(#"{"type": []}"#) == "type at # must be a string or a non-empty array of strings")
        #expect(message(#"{"type": ["string", 1]}"#) == "type at # must list only strings")
        #expect(message(#"{"type": ["string", "null"]}"#) == nil)
    }

    @Test("Numeric enum values keep their exact digits")
    func exactEnumNumbers() throws {
        let big = try grammar(#"{"enum": [12345678901234567890, 0.1, 123456789012345678901234567890]}"#)
        #expect(matches(big, "12345678901234567890"))
        #expect(!matches(big, "12345678901234567168"))
        #expect(matches(big, "0.1"))
        #expect(matches(big, "123456789012345678901234567890"))
    }

    @Test("Escaped surrogates must pair: a high surrogate needs a low one, a lone low one is rejected")
    func surrogatePairs() throws {
        let string = try grammar(#"{"type": "string"}"#)
        #expect(matches(string, #""\uD83D\uDE00""#))
        #expect(matches(string, #""\ud83d\ude00 \u00e9""#))
        #expect(!isPrefix(string, #""\uD83D""#))
        #expect(!isPrefix(string, #""\uD83Dx"#))
        #expect(!isPrefix(string, #""\uD83D\u0041"#))
        #expect(!isPrefix(string, #""\uDE00"#))
    }

    @Test("Without strict, constraint keywords are skipped and an ambiguous anyOf widens to any value of those kinds")
    func nonStrict() throws {
        let schema = #"{"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}, "v": {"anyOf": [{"type": "integer"}, {"type": "number"}, {"type": "string", "maxLength": 3}, {"enum": ["long value"]}]}}, "required": ["n", "v"]}"#
        #expect(throws: UnsupportedJSONSchema.self) { try JSONGrammar.compile(schemaJSON: schema) }
        let lenient = try JSONGrammar.compile(schemaJSON: schema, strict: false)
        #expect(matches(lenient, #"{"n":0,"v":2.5}"#))
        #expect(matches(lenient, #"{"n":0,"v":"anything"}"#))
        #expect(!matches(lenient, #"{"n":0,"v":true}"#))
        #expect(lenient.relaxations.contains("'minimum' at #/properties/n is not enforced"))
        #expect(lenient.relaxations.contains { $0.contains("overlap") })
        #expect(throws: UnsupportedJSONSchema.self) {
            try JSONGrammar.compile(schemaJSON: #"{"allOf": [{"type": "string"}]}"#, strict: false)
        }
    }
}
