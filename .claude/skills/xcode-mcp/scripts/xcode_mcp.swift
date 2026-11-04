#!/usr/bin/env swift
import Foundation

typealias JSON = [String: Any]

let handshake: [JSON] = [
    ["jsonrpc": "2.0", "id": 0, "method": "initialize",
     "params": ["protocolVersion": "2025-06-18", "capabilities": [String: Any](), "clientInfo": ["name": "claude-code", "version": "1"]]],
    ["jsonrpc": "2.0", "method": "notifications/initialized"],
]

func isAnswer(_ reply: JSON) -> Bool {
    (reply["id"] as? Int) == 1 || ((reply["id"] == nil || reply["id"] is NSNull) && reply["error"] != nil)
}

func parse(_ line: Data) -> JSON {
    let trimmed = String(decoding: line, as: UTF8.self).trimmingCharacters(in: .whitespaces)
    guard trimmed.hasPrefix("{") else { return [:] }
    return (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8))) as? JSON ?? [:]
}

func call(_ request: JSON, timeout: Double) throws -> JSON {
    let bridge = Process()
    bridge.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    bridge.arguments = ["xcrun", "mcpbridge"]
    let input = Pipe(), output = Pipe()
    bridge.standardInput = input
    bridge.standardOutput = output
    try bridge.run()
    defer { bridge.terminate() }
    for message in handshake + [request] {
        input.fileHandleForWriting.write(try JSONSerialization.data(withJSONObject: message) + Data("\n".utf8))
    }
    let deadline = Date().addingTimeInterval(timeout)
    var buffer = Data()
    while Date() < deadline {
        let chunk = output.fileHandleForReading.availableData
        if chunk.isEmpty { break }
        buffer += chunk
        while let newline = buffer.firstIndex(of: UInt8(ascii: "\n")) {
            let reply = parse(buffer[buffer.startIndex ..< newline])
            buffer = Data(buffer[(newline + 1)...])
            if isAnswer(reply) { return reply }
        }
    }
    return ["error": ["message": "no reply within \(String(format: "%.0f", timeout)) s"]]
}

func text(_ reply: JSON) -> String {
    let result = reply["result"] as? JSON ?? reply["error"] as? JSON ?? [:]
    let parts = (result["content"] as? [JSON] ?? []).map { $0["text"] as? String ?? "" } + [result["message"] as? String ?? ""]
    return parts.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
}

func toolCall(_ name: String, _ arguments: Any) -> JSON {
    ["jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["name": name, "arguments": arguments]]
}

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    FileHandle.standardError.write(Data("usage: xcode_mcp.swift list | <Tool> '<json arguments>' [timeout seconds]\n".utf8))
    exit(2)
}
let tool = arguments[1]
let toolArguments = try arguments.count > 2 ? JSONSerialization.jsonObject(with: Data(arguments[2].utf8), options: .fragmentsAllowed) : JSON()
let request = tool == "list" ? ["jsonrpc": "2.0", "id": 1, "method": "tools/list"] : toolCall(tool, toolArguments)
let timeout = arguments.count > 3 ? Double(arguments[3]) ?? 120 : 120
var reply = try call(request, timeout: timeout)
if text(reply).contains("isn't approved") {
    _ = try call(toolCall("XcodeOpenWorkspace", ["path": FileManager.default.currentDirectoryPath]), timeout: 60)
    reply = try call(request, timeout: timeout)
}
if tool == "list" {
    for item in (reply["result"] as? JSON)?["tools"] as? [JSON] ?? [] {
        let description = (item["description"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        print("\(item["name"] as? String ?? ""): \(description.split(whereSeparator: \.isNewline).first ?? "")")
    }
    exit(0)
}
print(text(reply))
exit((reply["result"] as? JSON)?["isError"] as? Bool == true || reply["error"] != nil ? 1 : 0)
