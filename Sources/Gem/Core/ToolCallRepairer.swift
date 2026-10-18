import Foundation

/// Deterministic parser and repairer for partially corrupted JSON tool calls
/// commonly produced by quantized (4-bit, 3-bit, 2-bit) models.
public enum ToolCallRepairer: Sendable {

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
    /// A tool call rebuilt from a call the model's tool-call parser rejected as malformed:
    /// repairs the raw payload, then takes its `arguments` (or `parameters`) object, or the whole
    /// object when it carries neither. nil when nothing safe can be rebuilt — no name, a truncated
    /// payload (repairing half a call would invent the rest), or no JSON object at all.
    public static func repairToolCall(name: String?, callID: String?, rawText: String, truncated: Bool) -> ToolCallOutput? {
        guard let name, !name.isEmpty, !truncated,
              let repaired = repairJSON(rawText),
              let object = try? JSONSerialization.jsonObject(with: Data(repaired.utf8)) as? [String: Any] else {
            return nil
        }
        let arguments: Any
        if let nested = object["arguments"] ?? object["parameters"] {
            // Some formats carry arguments as a JSON string inside the object.
            if let text = nested as? String, let decoded = repairJSON(text).flatMap({
                try? JSONSerialization.jsonObject(with: Data($0.utf8))
            }) {
                arguments = decoded
            } else {
                arguments = nested
            }
        } else {
            arguments = object.filter { $0.key != "name" }
        }
        guard JSONSerialization.isValidJSONObject(arguments),
              let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return ToolCallOutput(id: callID ?? "call_\(UUID().uuidString.prefix(24))", name: name, arguments: json)
    }

    public static func repairJSON(_ raw: String) -> String? {
        var str = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if str.isEmpty { return nil }

        // 1. Strip markdown code fences
        str = stripMarkdownFences(str)

        // 2. Locate outermost JSON boundaries '{' or '['
        guard let firstBrace = str.firstIndex(where: { $0 == "{" || $0 == "[" }) else {
            return nil
        }
        str = String(str[firstBrace...])

        // Valid JSON needs no repair — and must not get one: the quote and literal rewrites below
        // would corrupt string values that contain an apostrophe, "True" or a brace.
        if (try? JSONSerialization.jsonObject(with: Data(str.utf8))) != nil {
            return str
        }

        // 3. Normalize Python/JS literals
        str = normalizeLiterals(str)

        // 4. Convert single-quoted keys and strings to double quotes
        str = convertSingleQuotes(str)

        // 5. Quote unquoted keys (e.g. `{command: "ls"}` -> `{"command": "ls"}`)
        str = quoteUnquotedKeys(str)

        // 6. Remove trailing commas before closing braces/brackets
        str = removeTrailingCommas(str)

        // 7. Fix unclosed strings and balance braces/brackets
        str = balanceEnclosures(str)

        // 8. Final verification with JSONSerialization
        if let data = str.data(using: .utf8),
           (try? JSONSerialization.jsonObject(with: data)) != nil {
            return str
        }

        return nil
    }

    /// Extract tool call name and arguments dictionary from raw assistant response text
    public static func extractAndRepairToolCall(from raw: String) -> (name: String, arguments: [String: Any])? {
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

        // Schema C: tool invocation format `{"tool": "...", "args": {...}}`
        if let name = (json["tool"] ?? json["tool_name"]) as? String,
           let args = (json["args"] ?? json["parameters"] ?? json["input"]) as? [String: Any] {
            return (name, args)
        }

        return nil
    }

    // MARK: — Internal Helpers

    private static func stripMarkdownFences(_ text: String) -> String {
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

    private static func normalizeLiterals(_ text: String) -> String {
        var s = text
        s = s.replacingOccurrences(of: ":\\s*True\\b", with: ": true", options: .regularExpression)
        s = s.replacingOccurrences(of: ":\\s*False\\b", with: ": false", options: .regularExpression)
        s = s.replacingOccurrences(of: ":\\s*None\\b", with: ": null", options: .regularExpression)
        return s
    }

    private static func convertSingleQuotes(_ text: String) -> String {
        // If string already contains double quotes and no single quotes, leave untouched
        if !text.contains("'") { return text }

        var result = ""
        var inDoubleQuotes = false
        var inSingleQuotes = false
        var isEscaped = false

        for ch in text {
            if isEscaped {
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
            } else if ch == "'" && !inDoubleQuotes {
                inSingleQuotes.toggle()
                result.append("\"")
            } else {
                result.append(ch)
            }
        }
        return result
    }

    private static func quoteUnquotedKeys(_ text: String) -> String {
        // Matches { key: or , key: where key is alphanumeric/_
        let pattern = "([\\{\\,]\\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\\s*:)"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "$1\"$2\"$3")
    }

    private static func removeTrailingCommas(_ text: String) -> String {
        let pattern = ",\\s*([\\}\\]])"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "$1")
    }

    private static func balanceEnclosures(_ text: String) -> String {
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
