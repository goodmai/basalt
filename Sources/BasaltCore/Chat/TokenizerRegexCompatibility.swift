import Foundation

public enum TokenizerRegexCompatibility {
    public static func compatible(_ pattern: String) -> String {
        guard (try? NSRegularExpression(pattern: pattern)) == nil else { return pattern }
        return escapingBareBraces(pattern)
    }

    static func escapingBareBraces(_ pattern: String) -> String {
        let chars = Array(pattern)
        var output = ""
        var index = 0
        var inClass = false
        while index < chars.count {
            let length = tokenLength(chars, at: index, inClass: inClass)
            let token = String(chars[index ..< index + length])
            output += !inClass && (token == "{" || token == "}") ? "\\" + token : token
            if token == "[" { inClass = true } else if token == "]" { inClass = false }
            index += length
        }
        return output
    }

    private static func tokenLength(_ chars: [Character], at index: Int, inClass: Bool) -> Int {
        if chars[index] == "\\" {
            guard index + 1 < chars.count else { return 1 }
            if index + 2 < chars.count, chars[index + 1].isLetter, chars[index + 2] == "{",
               let close = chars[(index + 2)...].firstIndex(of: "}") {
                return close - index + 1
            }
            return 2
        }
        if !inClass, chars[index] == "{", let close = chars[index...].firstIndex(of: "}"),
           String(chars[(index + 1) ..< close]).wholeMatch(of: /\d+(,\d*)?/) != nil {
            return close - index + 1
        }
        return 1
    }
}
