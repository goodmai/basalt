import Foundation
import Testing
import Synchronization
import MLXGuidedGeneration
@testable import BasaltCore

@Suite("JSON Schema gate in front of XGrammar (#42)")
struct JSONSchemaGateTests {
    static func message(_ schema: String, strict: Bool = true) -> String? {
        do { _ = try JSONSchemaGate.prepare(schema, strict: strict); return nil } catch { return error.message }
    }

    @Test("Strict refuses a keyword XGrammar would only warn about, names it and where it is; without strict it is a relaxation")
    func strictKeywords() throws {
        #expect(Self.message(#"{"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 2}}}"#)
            == "unsupported JSON Schema keyword 'multipleOf' at #/properties/n")
        #expect(Self.message(#"{"type": "array", "items": {"not": {"type": "null"}}}"#) == "unsupported JSON Schema keyword 'not' at #/items")
        #expect(Self.message(#"{"allOf": [{"type": "string"}, {"maxLength": 3}]}"#) == "allOf at # must list exactly one schema")
        #expect(Self.message(#"{"$ref": "https://example.com/s.json"}"#)?.contains("not local") == true)
        #expect(Self.message(#"{"type": "object", "properties": {"d": {"type": "string", "description": "kept", "format": "uri", "pattern": "^h", "minLength": 2}}}"#) == nil)
        let lenient = try JSONSchemaGate.prepare(#"{"type": "integer", "multipleOf": 2, "uniqueItems": true}"#, strict: false)
        #expect(lenient.relaxations == ["'multipleOf' at # is not enforced", "'uniqueItems' at # is not enforced"])
    }

    @Test("type object or array with nothing declared is any object or any array, as JSON Schema says, not XGrammar's strict-mode empty one")
    func freeContainers() throws {
        #expect(try SchemaCheck.accepts(#"{"type": "object"}"#, #"{"a":1,"b":{"c":[null]}}"#))
        #expect(try SchemaCheck.accepts(#"{"type": "object"}"#, "{}"))
        #expect(try !SchemaCheck.accepts(#"{"type": "object"}"#, "[]"))
        #expect(try SchemaCheck.accepts(#"{"type": "array"}"#, #"[1,"x",{}]"#))
        #expect(try SchemaCheck.accepts(#"{"type": "object", "additionalProperties": {"type": "string"}}"#, #"{"a":"x"}"#))
        #expect(try !SchemaCheck.accepts(#"{"type": "object", "additionalProperties": {"type": "string"}}"#, #"{"a":1}"#))
        #expect(try !SchemaCheck.accepts(#"{"type": "object", "additionalProperties": false}"#, #"{"a":1}"#))
        #expect(try !SchemaCheck.accepts(#"{"type": "object", "properties": {"a": {"type": "integer"}}}"#, #"{"a":1,"b":2}"#))
    }

    @Test("Properties keep the order they were declared in, since XGrammar emits them in exactly that order")
    func declaredOrder() throws {
        let schema = #"{"type": "object", "properties": {"reasoning": {"type": "string"}, "answer": {"type": "integer"}}, "required": ["reasoning", "answer"]}"#
        #expect(try JSONSchemaGate.prepare(schema, strict: true).schema
            == #"{"type":"object","properties":{"reasoning":{"type":"string"},"answer":{"type":"integer"}},"required":["reasoning","answer"]}"#)
        #expect(try SchemaCheck.accepts(schema, #"{"reasoning":"r","answer":1}"#))
        #expect(try !SchemaCheck.accepts(schema, #"{"answer":1,"reasoning":"r"}"#))
        let body = Data(#"{"response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": \#(schema)}}}"#.utf8)
        #expect(OrderedJSON.parse(body)?["response_format"]?["json_schema"]?["schema"]?.text == (try JSONSchemaGate.prepare(schema, strict: true).schema))
    }

    @Test("A recursive $ref tree, enum and const literals and integer bounds are enforced by XGrammar")
    func xgrammarEnforces() throws {
        let tree = #"""
            {"$defs": {"node": {"type": "object",
                "properties": {"value": {"type": "integer", "minimum": 0}, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}},
                "required": ["value", "children"]}},
             "$ref": "#/$defs/node"}
            """#
        #expect(try SchemaCheck.accepts(tree, #"{"value":1,"children":[{"value":2,"children":[]}]}"#))
        #expect(try !SchemaCheck.accepts(tree, #"{"value":1,"children":[{"value":-2,"children":[]}]}"#))
        #expect(try !SchemaCheck.accepts(tree, #"{"value":1,"children":[{"value":2}]}"#))
        #expect(try SchemaCheck.accepts(#"{"enum": ["fast", "slow"]}"#, #""slow""#))
        #expect(try !SchemaCheck.accepts(#"{"enum": ["fast", "slow"]}"#, #""fastest""#))
    }

    @Test("Integer enum values within Int64 keep their digits; other numbers XGrammar respells, so strict refuses them and lenient logs the spelling")
    func enumNumbers() throws {
        #expect(try SchemaCheck.accepts(#"{"enum": [9007199254740993, -3, 2.5]}"#, "9007199254740993", strict: true))
        #expect(try !SchemaCheck.accepts(#"{"enum": [9007199254740993, -3, 2.5]}"#, "9007199254740992", strict: true))
        #expect(try SchemaCheck.accepts(#"{"enum": [9007199254740993, -3, 2.5]}"#, "2.5", strict: true))
        #expect(Self.message(#"{"enum": [12345678901234567890]}"#)
            == "enum at # lists 12345678901234567890, which XGrammar can only emit as 1.2345678901234567e+19")
        #expect(Self.message(#"{"const": 0.1}"#) == "enum at # lists 0.1, which XGrammar can only emit as 0.10000000000000001")
        let lenient = try JSONSchemaGate.prepare(#"{"enum": [0.1, 123456789012345678901234567890]}"#, strict: false)
        #expect(lenient.relaxations == ["enum value 0.1 at # is emitted as 0.10000000000000001",
                                        "enum value 123456789012345678901234567890 at # is emitted as 1.2345678901234568e+29"])
        #expect(try SchemaCheck.accepts(lenient.schema, "0.10000000000000001"))
        #expect(try !SchemaCheck.accepts(lenient.schema, "0.1"))
    }

    @Test("XGrammar takes paired surrogate escapes and, unlike strict JSON parsers, a lone one too")
    func surrogates() throws {
        let string = #"{"type": "string"}"#
        #expect(try SchemaCheck.accepts(string, #""\uD83D\uDE00""#))
        #expect(try SchemaCheck.accepts(string, #""\ud83d\ude00 \u00e9 ok""#))
        #expect(try SchemaCheck.accepts(string, #""\uD83D""#))
        #expect(try SchemaCheck.accepts(string, #""\uDE00""#))
        #expect(try !SchemaCheck.accepts(string, #""\uD83""#))
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

    static func timed(_ schema: String) -> (message: String?, seconds: Double) {
        onServerSizedStack {
            let start = ContinuousClock.now
            let message: String?
            do {
                let prepared = try JSONSchemaGate.prepare(schema, strict: false)
                _ = try GrammarConstraint(tokenizer: SchemaCheck.bytes, jsonSchema: prepared.schema)
                message = nil
            } catch {
                message = (error as? UnsupportedJSONSchema)?.message ?? "\(error)"
            }
            return (message, (ContinuousClock.now - start) / .seconds(1))
        }
    }

    @Test("Deep nesting, a long $ref chain, a huge enum and too many subschemas are refused before XGrammar, quickly, on a 512 KB stack")
    func hostileSchemasAreBounded() {
        let deepArrays = String(repeating: #"{"type": "array", "items": "#, count: 300) + "{}" + String(repeating: "}", count: 300)
        let raw = String(repeating: "[", count: 300) + String(repeating: "]", count: 300)
        func chain(_ length: Int) -> String {
            let defs = (0..<length).map { "\"d\($0)\": {\"$ref\": \"#/$defs/d\($0 + 1)\"}" }.joined(separator: ", ")
            return "{\"$defs\": {\(defs), \"d\(length)\": {\"type\": \"string\"}}, \"$ref\": \"#/$defs/d0\"}"
        }
        let hugeEnum = #"{"enum": ["# + (0..<200_000).map { "\"v\($0)\"" }.joined(separator: ",") + "]}"
        let wide = #"{"type": "object", "properties": {"# + (0..<5000).map { "\"p\($0)\": {\"type\": \"string\"}" }.joined(separator: ",") + "}}"
        for (schema, expected) in [(deepArrays, "nests deeper"), (raw, "nests deeper"), (chain(1000), "counting $ref"),
                                   (chain(5000), "counting $ref"), (hugeEnum, "more than 1024 values"), (wide, "more than 4096")] {
            let (message, seconds) = Self.timed(schema)
            #expect(message?.contains(expected) == true, "\(message ?? "compiled")")
            #expect(seconds < 1, "\(seconds) s")
        }
        let deepest = String(repeating: #"{"type": "array", "items": "#, count: 63) + #"{"type": "string"}"# + String(repeating: "}", count: 63)
        for schema in [chain(60), deepest] {
            let (fits, seconds) = Self.timed(schema)
            #expect(fits == nil)
            #expect(seconds < 1, "\(seconds) s")
        }
    }

    @Test("The order-keeping reader never traps: UTF-16 and UTF-32 bodies, truncations and random bytes come back as nil or a value")
    func orderedReaderNeverTraps() throws {
        var generator = SystemRandomNumberGenerator()
        func value(_ depth: Int) -> Any {
            switch Int.random(in: 0..<6, using: &generator) {
            case 0 where depth < 4: return (0..<Int.random(in: 0..<4, using: &generator)).map { _ in value(depth + 1) }
            case 1 where depth < 4: return Dictionary(uniqueKeysWithValues: (0..<Int.random(in: 0..<4, using: &generator)).map { ("k\($0)", value(depth + 1)) })
            case 2: return "s\\\"\(Int.random(in: 0..<100, using: &generator))"
            case 3: return Int.random(in: -100...100, using: &generator)
            case 4: return true
            default: return NSNull()
            }
        }
        for round in 0..<500 {
            let document = try JSONSerialization.data(
                withJSONObject: ["response_format": value(0), "b": [value(0)]], options: round % 2 == 0 ? [.prettyPrinted] : [])
            let text = String(decoding: document, as: UTF8.self)
            #expect(OrderedJSON.parse(document) != nil)
            for encoding: String.Encoding in [.utf16LittleEndian, .utf16BigEndian, .utf16, .utf32LittleEndian] {
                #expect(OrderedJSON.parse(try #require(text.data(using: encoding))) == nil)
            }
            for cut in stride(from: 0, to: document.count, by: max(1, document.count / 7)) {
                _ = OrderedJSON.parse(document.prefix(cut))
                var reader = OrderedJSON.Reader(bytes: Array(document.prefix(cut)))
                _ = try? reader.value()
            }
            var noise = OrderedJSON.Reader(bytes: (0..<Int.random(in: 0..<64, using: &generator)).map { _ in
                Array("{}[]\",:\\ a1".utf8).randomElement(using: &generator)!
            })
            _ = try? noise.value()
        }
    }

    @Test("Keeping only response_format reads that subtree and skips the rest of the body, strings with brackets included")
    func keepsTheFormatSubtree() throws {
        let body = Data(#"{"messages": [{"content": "}]{[\"x"}], "response_format": {"type": "json_object", "n": [1, {"a": "]"}]}, "tail": 7}"#.utf8)
        let kept = try #require(OrderedJSON.parse(body, keeping: ["response_format"]))
        #expect(kept.text == #"{"response_format":{"type":"json_object","n":[1,{"a":"]"}]}}"#)
        #expect(OrderedJSON.parse(body)?["response_format"] == kept["response_format"])
    }
}

