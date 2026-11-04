import Foundation
import BasaltCore
import BasaltHarness

struct RawMessage: Codable {
    let jsonrpc: String
    let id: JSONRPCId?
    let method: String?
    let params: AnyCodable?
    let result: AnyCodable?
    let error: RPCError?
}

enum JSONRPCId: Codable, Sendable {
    case string(String)
    case int(Int)
    case null

    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let i = try? c.decode(Int.self)    { self = .int(i);    return }
        self = .null
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let s): try c.encode(s)
        case .int(let i):    try c.encode(i)
        case .null:          try c.encodeNil()
        }
    }
}

struct RPCError: Codable, Sendable {
    let code: Int
    let message: String
}

struct MCPFailure: Error, Sendable, Equatable {
    let code: Int
    let message: String

    static func invalidParams(_ message: String) -> MCPFailure { MCPFailure(code: -32602, message: message) }
    static func internalError(_ message: String) -> MCPFailure { MCPFailure(code: -32603, message: message) }
    static func methodNotFound(_ message: String) -> MCPFailure { MCPFailure(code: -32601, message: message) }
    static let parseError = MCPFailure(code: -32700, message: "Parse error")
    static let libraryMissing = internalError("Experience Library is not configured on this server")

    init(code: Int, message: String) {
        self.code = code
        self.message = message
    }

    init(_ error: BasaltError) {
        message = error.localizedDescription
        code = switch error {
        case .modelNotFound:            -32001
        case .outOfMemory:              -32002
        case .inferenceHardwareFailure: -32003
        case .invalidRequestStructure:  -32602
        case .contextLengthExceeded:    -32004
        case .weightsCorrupted:         -32005
        case .rateLimitExceeded:        -32006
        case .authenticationFailed:     -32007
        case .modelInferenceError, .nonFiniteLogits: -32008
        case .inferenceError:           -32009
        case .modelNotCached:           -32010
        case .modelSwitching:           -32011
        case .serverBusy, .modelConfigurationInvalid: -32012
        case .memoryPressure:           -32013
        }
    }

    static func knowledge(_ error: any Error) -> MCPFailure {
        if let error = error as? HarnessLibraryError, case .invalidArgument = error {
            return invalidParams(error.description)
        }
        return internalError("Failed to persist knowledge: \(error)")
    }

    static func optimization(_ error: any Error) -> MCPFailure {
        switch error {
        case let error as BasaltError: MCPFailure(error)
        case let error as HarnessBudgetError: invalidParams("Harness optimization refused: \(error)")
        default: internalError("Harness optimization failed: \(error)")
        }
    }
}

enum MCPRoute: Sendable {
    case failed(id: JSONRPCId?, MCPFailure)
    case ignored
    case initialize(id: JSONRPCId?)
    case toolsList(id: JSONRPCId?)
    case toolsCall(id: JSONRPCId?, params: AnyCodable?)

    init(line: String) {
        guard let data = line.data(using: .utf8),
              let message = try? MCPWire.decoder.decode(RawMessage.self, from: data)
        else {
            self = .failed(id: nil, .parseError)
            return
        }
        switch message.method {
        case nil, "notifications/initialized", "initialized": self = .ignored
        case "initialize": self = .initialize(id: message.id)
        case "tools/list": self = .toolsList(id: message.id)
        case "tools/call": self = .toolsCall(id: message.id, params: message.params)
        case let method?: self = .failed(id: message.id, .methodNotFound("Method not found: \(method)"))
        }
    }
}

enum MCPWire {
    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        return e
    }()
    static let decoder = JSONDecoder()

    static func result(id: JSONRPCId?, _ result: some Encodable) -> String {
        do {
            let data = try encoder.encode(result)
            guard let json = String(data: data, encoding: .utf8) else {
                return error(id: id, .internalError("Internal: UTF-8 conversion failed"))
            }
            return envelope(id: id, field: "result", valueJSON: json)
        } catch {
            return self.error(id: id, .internalError("Internal: \(error.localizedDescription)"))
        }
    }

    static func error(id: JSONRPCId?, _ failure: MCPFailure) -> String {
        envelope(id: id, field: "error", valueJSON: "{\"code\":\(failure.code),\"message\":\(jsonString(failure.message))}")
    }

    static func envelope(id: JSONRPCId?, field: String, valueJSON: String) -> String {
        var s = "{\"jsonrpc\":\"2.0\""
        if let id { s += ",\"id\":\(idJSON(id))" }
        s += ",\"\(field)\":\(valueJSON)}"
        return s
    }

    static func idJSON(_ id: JSONRPCId) -> String {
        switch id {
        case .int(let i):    return "\(i)"
        case .null:          return "null"
        case .string(let s): return jsonString(s)
        }
    }

    static func jsonString(_ s: String) -> String {
        if let d = try? encoder.encode(s), let str = String(data: d, encoding: .utf8) { return str }
        return "\"\""
    }
}

enum AnyCodable: Sendable, Codable {
    case int(Int)
    case double(Double)
    case bool(Bool)
    case string(String)
    case array([AnyCodable])
    case object([String: AnyCodable])
    case null

    init(_ rawValue: Any) {
        switch rawValue {
        case let i as Int:           self = .int(i)
        case let d as Double:        self = .double(d)
        case let b as Bool:          self = .bool(b)
        case let s as String:        self = .string(s)
        case let a as [Any]:         self = .array(a.map(AnyCodable.init))
        case let m as [String: Any]: self = .object(m.mapValues(AnyCodable.init))
        default:                     self = .null
        }
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let i = try? c.decode(Int.self)                  { self = .int(i);    return }
        if let d = try? c.decode(Double.self)               { self = .double(d); return }
        if let b = try? c.decode(Bool.self)                 { self = .bool(b);   return }
        if let s = try? c.decode(String.self)               { self = .string(s); return }
        if let a = try? c.decode([AnyCodable].self)         { self = .array(a);  return }
        if let m = try? c.decode([String: AnyCodable].self) { self = .object(m); return }
        self = .null
    }

    var rawValue: Any {
        switch self {
        case .int(let i):    i
        case .double(let d): d
        case .bool(let b):   b
        case .string(let s): s
        case .array(let a):  a.map(\.rawValue)
        case .object(let m): m.mapValues(\.rawValue)
        case .null:          NSNull()
        }
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .int(let i):    try c.encode(i)
        case .double(let d): try c.encode(d)
        case .bool(let b):   try c.encode(b)
        case .string(let s): try c.encode(s)
        case .array(let a):  try c.encode(a)
        case .object(let m): try c.encode(m)
        case .null:          try c.encodeNil()
        }
    }
}
