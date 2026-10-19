import Foundation

public enum ChatTemplateCompatibility {
    static let reservedLoopVariable = "call"
    static let renamedLoopVariable = "call_item"

    nonisolated(unsafe) private static let jinjaSegment = try! NSRegularExpression(
        pattern: #"\{[{%].*?[%}]\}"#, options: .dotMatchesLineSeparators)
    nonisolated(unsafe) private static let loopOpening = try! NSRegularExpression(pattern: #"^\{%-?\s*for\s+(\w+)\s+in\b"#)
    nonisolated(unsafe) private static let loopClosing = try! NSRegularExpression(pattern: #"^\{%-?\s*endfor\b"#)
    nonisolated(unsafe) private static let reservedIdentifier = try! NSRegularExpression(
        pattern: #"(?<![\w.'"])"# + reservedLoopVariable + #"(?![\w'"])"#)

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
                loopsBindingReserved.append((segment as NSString).substring(with: opening.range(at: 1)) == reservedLoopVariable)
            }
            if loopsBindingReserved.contains(true) {
                segment = reservedIdentifier.stringByReplacingMatches(
                    in: segment, range: whole, withTemplate: renamedLoopVariable)
            }
            if loopClosing.firstMatch(in: segment, range: whole) != nil { _ = loopsBindingReserved.popLast() }
            output += segment
            cursor = match.range.location + match.range.length
        }
        return output + source.substring(from: cursor)
    }
}
