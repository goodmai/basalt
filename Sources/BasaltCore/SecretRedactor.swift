import Foundation

public enum SecretRedactor {
    static let patterns: [(label: String, regex: NSRegularExpression)] = [
        ("anthropic_key", #"sk-ant-[A-Za-z0-9_\-]{20,}"#),
        ("openai_key", #"sk-[A-Za-z0-9_\-]{20,}"#),
        ("github_token", #"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"#),
        ("aws_access_key", #"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"#),
        ("slack_token", #"xox[bpas]-[A-Za-z0-9\-]{10,}"#),
        ("bearer", #"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{16,}"#),
    ].map { ($0.0, try! NSRegularExpression(pattern: $0.1)) }

    static let pemBegin = "-----BEGIN "
    static let pemPrivate = "PRIVATE KEY-----"
    static let pemEnd = "-----END "

    public static func redact(_ text: String) -> String {
        var out = redactingPEM(text)
        for (label, regex) in patterns {
            let range = NSRange(out.startIndex..., in: out)
            guard regex.firstMatch(in: out, range: range) != nil else { continue }
            out = regex.stringByReplacingMatches(in: out, range: range, withTemplate: "[REDACTED:\(label)]")
        }
        return out
    }

    static func redactingPEM(_ text: String) -> String {
        guard text.contains(pemPrivate) else { return text }
        var out = ""
        var cursor = text.startIndex
        while let begin = text.range(of: pemBegin, range: cursor..<text.endIndex) {
            let lineEnd = text[begin.upperBound...].firstIndex(where: \.isNewline) ?? text.endIndex
            guard text[begin.upperBound..<lineEnd].hasSuffix(pemPrivate) else {
                out += text[cursor..<begin.upperBound]
                cursor = begin.upperBound
                continue
            }
            out += text[cursor..<begin.lowerBound] + "[REDACTED:private_key]"
            guard let end = text.range(of: pemEnd, range: lineEnd..<text.endIndex) else {
                return out
            }
            let afterEnd = text[end.upperBound...].range(of: "-----")?.upperBound ?? text.endIndex
            cursor = afterEnd
        }
        return out + text[cursor...]
    }

    public static func redact(_ messages: [ChatMessageInput]) -> [ChatMessageInput] {
        messages.map { message in
            ChatMessageInput(
                role: message.role, text: redact(message.text),
                toolCalls: message.toolCalls.map(redact), toolCallId: message.toolCallId, toolName: message.toolName)
        }
    }

    public static func redact(_ calls: [ToolCallOutput]) -> [ToolCallOutput] {
        calls.map { ToolCallOutput(id: $0.id, name: $0.name, arguments: redact($0.arguments)) }
    }
}
