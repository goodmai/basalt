import Foundation

/// Deterministic parser and repairer for partially corrupted JSON tool calls
/// commonly produced by quantized (4-bit, 3-bit, 2-bit) models.
public enum ToolCallRepairer: Sendable {


    /// A tool call rebuilt from one the model's tool-call parser rejected as malformed. The payload
    /// is the whole protocol span (e.g. `<tool_call>{…}</tool_call>`): its protocol markers are
    /// stripped, then it is repaired and read with the same schemas as `extractAndRepairToolCall`.
    /// The name must be one the request declared — a repaired call skips the parser's own name
    /// and schema checks. nil when nothing safe can be rebuilt: a truncated payload (repairing
    /// half a call would invent the rest), no recognizable call, or an undeclared name.
    public static func repairToolCall(
        name parserName: String?,
        callID: String?,
        rawText: String,
        truncated: Bool,
        declaredTools: Set<String>
    ) -> ToolCallOutput? {
        guard !truncated else { return nil }
        // Strip only the protocol's own markers. Cutting at the last brace instead would drop the
        // tail of an argument that contains one ("git log --format={h} -n 5" losing " -n 5").
        var payload = rawText
        for tag in ["</tool_call>", "<end_function_call>", "<|call|>", "[TOOL_CALLS]"] {
            payload = payload.replacingOccurrences(of: tag, with: "")
        }
        // Mistral: `name[ARGS]{…}` — the arguments object follows the marker.
        if let args = payload.range(of: "[ARGS]") { payload = String(payload[args.upperBound...]) }

        let name: String
        let arguments: [String: Any]
        if let extracted = extractAndRepairToolCall(from: payload) {
            name = parserName ?? extracted.name
            arguments = extracted.arguments
        } else if let parserName,
                  let repaired = repairJSON(payload),
                  let object = try? JSONSerialization.jsonObject(with: Data(repaired.utf8)) as? [String: Any] {
            // The parser knew the name; the payload is the bare arguments object.
            name = parserName
            arguments = object
        } else {
            return nil
        }
        guard declaredTools.contains(name),
              let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return ToolCallOutput(id: callID ?? "call_\(UUID().uuidString.prefix(24))", name: name, arguments: json)
    }

    /// Repairs a potentially corrupted JSON string.
    ///
    /// Handles:
    /// - Markdown code fences (```json ... ``` or ``` ...)
    /// - Leading/trailing non-JSON preamble/chatter
    /// - Single-quoted strings and keys (`{'key': 'value'}` -> `{"key": "value"}`)
    /// - Unquoted object keys (`{name: "test"}` -> `{"name": "test"}`)
    /// - Python/JS literals (`True`, `False`, `None` -> `true`, `false`, `null`)
    /// - Trailing commas before `}` or `]` (`{"a": 1,}` -> `{"a": 1}`)
    /// - Unclosed string literals (`{"name": "run_command`)
    /// - Unbalanced braces and brackets (appends missing `}` and `]`)
    public static func repairJSON(_ raw: String) -> String? {
        var str = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if str.isEmpty { return nil }

        str = stripMarkdownFences(str)

        guard let firstBrace = str.firstIndex(where: { $0 == "{" || $0 == "[" }) else {
            return nil
        }
        str = sliceBalancedRoot(String(str[firstBrace...]))

        if (try? JSONSerialization.jsonObject(with: Data(str.utf8))) != nil {
            return str
        }

        str = convertSingleQuotes(str)
        str = transformOutsideStrings(str) { segment in
            var s = normalizeLiterals(segment)
            s = quoteUnquotedKeys(s)
            s = removeTrailingCommas(s)
            return s
        }
        str = balanceEnclosures(str)

        // 8. Final verification with JSONSerialization
        if let data = str.data(using: .utf8),
           (try? JSONSerialization.jsonObject(with: data)) != nil {
            return str
        }

        return nil
    }

    public static func extractAndRepairToolCall(from raw: String) -> (name: String, arguments: [String: Any])? {
        if let xmlCall = extractXMLArgKeyToolCall(from: raw) {
            return xmlCall
        }
        guard let repaired = repairJSON(raw),
              let data = repaired.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        // Schema A: standard tool_call object `{"name": "...", "arguments": {...}}`
        if let name = json["name"] as? String {
            if let args = json["arguments"] as? [String: Any] {
                return (name, args)
            } else if let argsStr = json["arguments"] as? String {
                if let repairedArgs = repairJSON(argsStr),
                   let argsData = repairedArgs.data(using: .utf8),
                   let dict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
                    return (name, dict)
                }
            }
        }

        // Schema B: function calling format `{"function": {"name": "...", "arguments": ...}}`
        if let fn = json["function"] as? [String: Any],
           let name = fn["name"] as? String {
            if let args = fn["arguments"] as? [String: Any] {
                return (name, args)
            } else if let argsStr = fn["arguments"] as? String {
                if let repairedArgs = repairJSON(argsStr),
                   let argsData = repairedArgs.data(using: .utf8),
                   let dict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
                    return (name, dict)
                }
            }
        }

        if let name = (json["tool"] ?? json["tool_name"]) as? String,
           let args = (json["args"] ?? json["parameters"] ?? json["input"]) as? [String: Any] {
            return (name, args)
        }

        return nil
    }
}

extension ToolCallRepairer {
    public static func extractXMLArgKeyToolCall(from raw: String) -> (name: String, arguments: [String: Any])? {
        var body = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let open = body.range(of: "<tool_call>") {
            body = String(body[open.upperBound...])
        }
        if let close = body.range(of: "</tool_call>") {
            body = String(body[..<close.lowerBound])
        }
        guard body.contains("<arg_key>") else { return nil }
        guard let firstArg = body.range(of: "<arg_key>") else { return nil }
        let name = body[..<firstArg.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !name.contains("<") else { return nil }
        var arguments: [String: Any] = [:]
        var cursor = firstArg.lowerBound
        while let kOpen = body.range(of: "<arg_key>", range: cursor ..< body.endIndex),
              let kClose = body.range(of: "</arg_key>", range: kOpen.upperBound ..< body.endIndex),
              let vOpen = body.range(of: "<arg_value>", range: kClose.upperBound ..< body.endIndex),
              let vClose = body.range(of: "</arg_value>", range: vOpen.upperBound ..< body.endIndex)
        {
            let key = body[kOpen.upperBound ..< kClose.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            let rawVal = String(body[vOpen.upperBound ..< vClose.lowerBound])
            if !key.isEmpty {
                arguments[key] = decodeXMLArgValue(rawVal)
            }
            cursor = vClose.upperBound
        }
        return arguments.isEmpty ? nil : (name, arguments)
    }

    private static func decodeXMLArgValue(_ rawVal: String) -> Any {
        let trimmed = rawVal.trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = trimmed.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        {
            return parsed
        }
        if (trimmed.hasPrefix("{") || trimmed.hasPrefix("[")),
           let repaired = repairJSON(trimmed),
           let data = repaired.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data)
        {
            return parsed
        }
        return rawVal
    }
}

extension ToolCallRepairer {
    static func stripMarkdownFences(_ text: String) -> String {
        var str = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if str.hasPrefix("```") {
            if let firstNewline = str.firstIndex(of: "\n") {
                str = String(str[str.index(after: firstNewline)...])
            } else {
                str = str.replacingOccurrences(of: "^```[a-zA-Z]*", with: "", options: .regularExpression)
            }
        }
        if str.hasSuffix("```") {
            str = String(str.dropLast(3)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return str
    }

    static func normalizeLiterals(_ text: String) -> String {
        var s = text
        s = s.replacingOccurrences(of: ":\\s*True\\b", with: ": true", options: .regularExpression)
        s = s.replacingOccurrences(of: ":\\s*False\\b", with: ": false", options: .regularExpression)
        s = s.replacingOccurrences(of: ":\\s*None\\b", with: ": null", options: .regularExpression)
        return s
    }

    static func convertSingleQuotes(_ text: String) -> String {
        if !text.contains("'") { return text }

        var result = ""
        var inDoubleQuotes = false
        var inSingleQuotes = false
        var isEscaped = false

        for ch in text {
            if isEscaped {
                if inSingleQuotes && ch == "'" {
                    result.removeLast()
                }
                result.append(ch)
                isEscaped = false
                continue
            }
            if ch == "\\" {
                result.append(ch)
                isEscaped = true
                continue
            }

            if ch == "\"" && !inSingleQuotes {
                inDoubleQuotes.toggle()
                result.append(ch)
            } else if ch == "\"" && inSingleQuotes {
                result.append("\\\"")
            } else if ch == "'" && !inDoubleQuotes {
                inSingleQuotes.toggle()
                result.append("\"")
            } else {
                result.append(ch)
            }
        }
        return result
    }

    static func quoteUnquotedKeys(_ text: String) -> String {
        let pattern = "([\\{\\,]\\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\\s*:)"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "$1\"$2\"$3")
    }

    static func removeTrailingCommas(_ text: String) -> String {
        let pattern = ",\\s*([\\}\\]])"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "$1")
    }

    static func balanceEnclosures(_ text: String) -> String {
        var stack: [Character] = []
        var inString = false
        var isEscaped = false

        for ch in text {
            if isEscaped {
                isEscaped = false
                continue
            }
            if ch == "\\" {
                isEscaped = true
                continue
            }

            if ch == "\"" {
                inString.toggle()
                continue
            }

            if !inString {
                if ch == "{" {
                    stack.append("}")
                } else if ch == "[" {
                    stack.append("]")
                } else if ch == "}" || ch == "]" {
                    if let last = stack.last, last == ch {
                        stack.removeLast()
                    }
                }
            }
        }

        var repaired = text
        // If string was left open, close it
        if inString {
            repaired.append("\"")
        }

        // Close any remaining unclosed braces/brackets in reverse order
        while let needed = stack.popLast() {
            repaired.append(needed)
        }

        return repaired
    }
}

private extension ToolCallRepairer {
    static func sliceBalancedRoot(_ text: String) -> String {
        var depth = 0
        var quote: Character?
        var escaped = false
        for idx in text.indices {
            let ch = text[idx]
            if escaped { escaped = false; continue }
            if ch == "\\" && quote != nil { escaped = true; continue }
            if let q = quote {
                if ch == q { quote = nil }
                continue
            }
            if ch == "\"" || ch == "'" { quote = ch; continue }
            if ch == "{" || ch == "[" { depth += 1 }
            else if ch == "}" || ch == "]" {
                depth -= 1
                if depth == 0 { return String(text[...idx]) }
            }
        }
        return text
    }

    static func transformOutsideStrings(_ text: String, _ transform: (String) -> String) -> String {
        var out = ""
        var outside = ""
        var inString = false
        var escaped = false
        for ch in text {
            if inString {
                out.append(ch)
                if escaped { escaped = false }
                else if ch == "\\" { escaped = true }
                else if ch == "\"" { inString = false }
            } else if ch == "\"" {
                if !outside.isEmpty { out.append(transform(outside)); outside = "" }
                inString = true
                out.append(ch)
            } else {
                outside.append(ch)
            }
        }
        if !outside.isEmpty { out.append(transform(outside)) }
        return out
    }
}
