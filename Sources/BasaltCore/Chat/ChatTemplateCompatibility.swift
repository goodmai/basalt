import Foundation

public enum ChatTemplateCompatibility {
    static let reservedLoopVariable = "call"
    static let renamedLoopVariable = "call_item"

    nonisolated(unsafe) private static let jinjaSegment = try! NSRegularExpression(
        pattern: #"\{[{%].*?[%}]\}"#, options: .dotMatchesLineSeparators)
    nonisolated(unsafe) private static let loopOpening = try! NSRegularExpression(pattern: #"^\{%[-+]?\s*for\s+(.+?)\s+in\b"#)
    nonisolated(unsafe) private static let loopClosing = try! NSRegularExpression(pattern: #"^\{%[-+]?\s*endfor\b"#)
    nonisolated(unsafe) private static let stringLiteral = try! NSRegularExpression(
        pattern: #"'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*""#, options: .dotMatchesLineSeparators)
    nonisolated(unsafe) private static let reservedIdentifier = try! NSRegularExpression(
        pattern: #"(?<![\w.])"# + reservedLoopVariable + #"(?!\w)"#)
    public static func renamingReservedLoopVariables(_ template: String) -> String {
        let source = template as NSString
        var output = ""
        var cursor = 0
        var loopsBindingReserved: [Bool] = []
        for match in jinjaSegment.matches(in: template, range: NSRange(location: 0, length: source.length)) {
            output += source.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            var segment = source.substring(with: match.range)
            let whole = NSRange(location: 0, length: (segment as NSString).length)
            if let opening = loopOpening.firstMatch(in: segment, range: whole) {
                loopsBindingReserved.append(bindsReserved(targets: (segment as NSString).substring(with: opening.range(at: 1))))
            }
            if loopsBindingReserved.contains(true) { segment = renamingOutsideStringLiterals(segment) }
            if loopClosing.firstMatch(in: segment, range: whole) != nil { _ = loopsBindingReserved.popLast() }
            output += segment
            cursor = match.range.location + match.range.length
        }
        return output + source.substring(from: cursor)
    }

    private static func bindsReserved(targets: String) -> Bool {
        targets.split(whereSeparator: { ",()".contains($0) })
            .contains { $0.trimmingCharacters(in: .whitespaces) == reservedLoopVariable }
    }

    private static func renamingOutsideStringLiterals(_ segment: String) -> String {
        let source = segment as NSString
        var output = ""
        var cursor = 0
        func renamingCode(upTo end: Int) {
            let code = source.substring(with: NSRange(location: cursor, length: end - cursor))
            output += reservedIdentifier.stringByReplacingMatches(
                in: code, range: NSRange(location: 0, length: (code as NSString).length), withTemplate: renamedLoopVariable)
        }
        for literal in stringLiteral.matches(in: segment, range: NSRange(location: 0, length: source.length)) {
            renamingCode(upTo: literal.range.location)
            output += source.substring(with: literal.range)
            cursor = literal.range.location + literal.range.length
        }
        renamingCode(upTo: source.length)
        return output
    }
}

extension ChatTemplateCompatibility {
    public static func compatible(_ template: String) -> String {
        renamingReservedLoopVariables(applyingWhitespaceControl(template))
    }

    public static func applyingWhitespaceControl(_ template: String) -> String {
        let source = Array(template.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n"))
        let (texts, tags) = TemplateScanner(source).segments()
        var output = ""
        for (index, text) in texts.enumerated() {
            let previous = index > 0 ? tags[index - 1] : nil
            let next = index < tags.count ? tags[index] : nil
            let settled = settledWhitespace(text, isFirst: index == 0, previous: previous, next: next)
            output += previous?.emitted ?? ""
            output += previous?.kind == .rawOpening ? literal(settled) : protected(settled, isFirst: index == 0, previous: previous, next: next)
        }
        return output
    }

    private static func settledWhitespace(_ original: String, isFirst: Bool, previous: TemplateTag?, next: TemplateTag?) -> String {
        var text = original
        switch previous {
        case let tag? where tag.trailing == "-":
            text = String(text.drop(while: \.isWhitespace))
        case let tag? where tag.trimsFollowingNewline && text.first == "\n":
            text.removeFirst()
        default:
            break
        }
        switch next {
        case let tag? where tag.leading == "-":
            text = String(text.reversed().drop(while: \.isWhitespace).reversed())
        case let tag? where tag.stripsLineIndentation && (isFirst || original.contains("\n")):
            text.removeLast(indentation(of: text).count)
        default:
            break
        }
        return text
    }

    private static func indentation(of text: String) -> Substring {
        let lastLine = text.split(separator: "\n", omittingEmptySubsequences: false).last ?? ""
        return lastLine.allSatisfy { $0 == " " || $0 == "\t" } ? lastLine : ""
    }

    private static func protected(_ text: String, isFirst: Bool, previous: TemplateTag?, next: TemplateTag?) -> String {
        var middle = Substring(text)
        var head = "", tail = ""
        if let previous, previous.kind != .expression, middle.first == "\n" {
            head = "\n"
            middle = middle.dropFirst()
        }
        if let next, next.kind != .expression, isFirst || !head.isEmpty || middle.contains("\n") {
            let indent = indentation(of: String(middle))
            tail = String(indent)
            middle = middle.dropLast(indent.count)
        }
        if next != nil, middle.last == "{" {
            tail = "{" + tail
            middle = middle.dropLast()
        }
        return literal(head) + middle + literal(tail)
    }

    static func literal(_ text: String) -> String {
        guard !text.isEmpty else { return "" }
        var escaped = ""
        for character in text {
            switch character {
            case "\\": escaped += "\\\\"
            case "'": escaped += "\\'"
            case "\n": escaped += "\\n"
            case "-", "{", "}", "%", "#": escaped += "\\\(character)"
            default: escaped.append(character)
            }
        }
        return "{{ '\(escaped)' }}"
    }
}

struct TemplateTag {
    enum Kind { case expression, statement, comment, rawOpening, rawClosing }

    let kind: Kind
    let leading: Character?
    let trailing: Character?
    let body: String

    var trimsFollowingNewline: Bool { kind != .expression && kind != .rawOpening && trailing != "+" }
    var stripsLineIndentation: Bool { kind != .expression && leading != "+" }

    private var paddedBody: String {
        let modifiers: Set<Character> = ["-", "+"]
        return (body.first.map(modifiers.contains) == true ? " " : "") + body + (body.last.map(modifiers.contains) == true ? " " : "")
    }

    var emitted: String {
        switch kind {
        case .expression: return "{{\(paddedBody)}}"
        case .statement: return "{%\(paddedBody)%}"
        case .comment: return "{# #}"
        case .rawOpening, .rawClosing: return ""
        }
    }
}

struct TemplateScanner {
    let source: [Character]
    private static let openingNeighbours: Set<Character> = ["{", "%", "#"]
    private static let closingNeighbours: Set<Character> = ["}", "%", "#"]

    init(_ source: [Character]) { self.source = source }

    func segments() -> (texts: [String], tags: [TemplateTag]) {
        var texts: [String] = [], tags: [TemplateTag] = []
        var text = ""
        var position = 0
        while position < source.count {
            guard let (tag, end) = tag(at: position) else {
                text.append(source[position])
                position += 1
                continue
            }
            texts.append(text)
            tags.append(tag)
            text = ""
            position = end
            if tag.kind == .statement, tag.body.trimmingCharacters(in: .whitespaces) == "raw",
               let (closing, contentEnd, closingEnd) = rawClosing(from: end) {
                tags[tags.count - 1] = TemplateTag(kind: .rawOpening, leading: tag.leading, trailing: tag.trailing, body: "")
                texts.append(String(source[end..<contentEnd]))
                tags.append(closing)
                position = closingEnd
            }
        }
        texts.append(text)
        return (texts, tags)
    }

    private func tag(at position: Int) -> (TemplateTag, Int)? {
        guard position + 1 < source.count, source[position] == "{" else { return nil }
        let opener = source[position + 1]
        let kind: TemplateTag.Kind
        let closer: Character
        switch opener {
        case "{": (kind, closer) = (.expression, "}")
        case "%": (kind, closer) = (.statement, "%")
        case "#": (kind, closer) = (.comment, "#")
        default: return nil
        }
        var bodyStart = position + 2
        var leading: Character?
        if bodyStart < source.count, source[bodyStart] == "-" || (kind != .expression && source[bodyStart] == "+") {
            leading = source[bodyStart]
            bodyStart += 1
        }
        guard let closing = closingDelimiter(from: bodyStart, closer: closer, skipsStrings: kind != .comment) else { return nil }
        var bodyEnd = closing
        var trailing: Character?
        if bodyEnd > bodyStart, source[bodyEnd - 1] == "-" || (kind != .expression && source[bodyEnd - 1] == "+") {
            trailing = source[bodyEnd - 1]
            bodyEnd -= 1
        }
        let body = kind == .comment ? "" : sanitizedStringLiterals(in: source[bodyStart..<bodyEnd])
        return (TemplateTag(kind: kind, leading: leading, trailing: trailing, body: body), closing + 2)
    }

    private func closingDelimiter(from start: Int, closer: Character, skipsStrings: Bool) -> Int? {
        var position = start
        var quote: Character?
        var depth = 0
        while position + 1 < source.count {
            let character = source[position]
            if let open = quote {
                if character == "\\" { position += 1 } else if character == open { quote = nil }
            } else if !skipsStrings {
                if character == closer, source[position + 1] == "}" { return position }
            } else if character == "'" || character == "\"" {
                quote = character
            } else if "([{".contains(character) {
                depth += 1
            } else if depth > 0, ")]}".contains(character) {
                depth -= 1
            } else if character == closer, source[position + 1] == "}" {
                return position
            }
            position += 1
        }
        return nil
    }

    private func sanitizedStringLiterals(in body: ArraySlice<Character>) -> String {
        var output = ""
        var quote: Character?
        var position = body.startIndex
        while position < body.endIndex {
            let character = body[position]
            if let open = quote {
                if character == "\\", position + 1 < body.endIndex {
                    output.append(character)
                    output.append(body[position + 1])
                    position += 2
                    continue
                }
                if character == open { quote = nil }
                let previous = position > body.startIndex ? body[position - 1] : nil
                if character == "\n" {
                    output += "\\n"
                } else if character == "-" && previous.map(Self.openingNeighbours.contains) == true
                            || previous == "-" && Self.closingNeighbours.contains(character) {
                    output += "\\\(character)"
                } else {
                    output.append(character)
                }
            } else {
                if character == "'" || character == "\"" { quote = character }
                output.append(character)
            }
            position += 1
        }
        return output
    }

    private func rawClosing(from start: Int) -> (TemplateTag, Int, Int)? {
        var position = start
        while position + 1 < source.count {
            if source[position] == "{", source[position + 1] == "%",
               let (tag, end) = tag(at: position), tag.body.trimmingCharacters(in: .whitespaces) == "endraw" {
                return (TemplateTag(kind: .rawClosing, leading: tag.leading, trailing: tag.trailing, body: ""), position, end)
            }
            position += 1
        }
        return nil
    }
}
