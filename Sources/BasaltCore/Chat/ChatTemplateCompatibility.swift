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
