import Foundation

public struct SSEEvent: Sendable, Equatable {
    public var name: String?
    public var data: String

    public init(name: String? = nil, data: String) {
        (self.name, self.data) = (name, data)
    }
}

public struct SSEParser: Sendable {
    private var line: [UInt8] = []
    private var name: String?
    private var data: [String] = []

    public init() {}

    public mutating func feed(_ bytes: some Sequence<UInt8>) -> [SSEEvent] {
        var events: [SSEEvent] = []
        for byte in bytes {
            guard byte == UInt8(ascii: "\n") else {
                line.append(byte)
                continue
            }
            if line.last == UInt8(ascii: "\r") { line.removeLast() }
            if let event = finish(String(decoding: line, as: UTF8.self)) { events.append(event) }
            line.removeAll(keepingCapacity: true)
        }
        return events
    }

    private mutating func finish(_ text: String) -> SSEEvent? {
        guard !text.isEmpty else {
            defer { (name, data) = (nil, []) }
            return data.isEmpty ? nil : SSEEvent(name: name, data: data.joined(separator: "\n"))
        }
        guard !text.hasPrefix(":") else { return nil }
        let field = text.prefix { $0 != ":" }
        var value = text.dropFirst(field.count + 1)
        if value.first == " " { value = value.dropFirst() }
        switch field {
        case "data": data.append(String(value))
        case "event": name = String(value)
        default: break
        }
        return nil
    }
}
