import Foundation

public struct MathAnswerGrader: PairedAnswerGrader, Sendable {
    public let numericTolerance: Double

    public init(numericTolerance: Double = 1e-6) {
        self.numericTolerance = numericTolerance
    }

    public func grade(response: String, problem: PairedProblem) -> PairedGradeResult {
        guard let extracted = extractAnswer(from: response) else {
            return PairedGradeResult(passed: false, score: 0.0, extractedAnswer: nil, detail: "no answer extracted")
        }
        let ok = equivalent(extracted, problem.expectedAnswer)
        return PairedGradeResult(
            passed: ok,
            score: ok ? 1.0 : 0.0,
            extractedAnswer: extracted,
            detail: ok ? "matched" : "expected \(normalize(problem.expectedAnswer)), got \(normalize(extracted))"
        )
    }

    public func extractAnswer(from response: String) -> String? {
        if let boxed = Self.lastBalancedCommand(in: response, names: ["\\boxed", "\\fbox"]) {
            let trimmed = boxed.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        if let hashAnswer = Self.lastPrefixedLine(in: response, prefix: "####") {
            return hashAnswer
        }
        if let labeled = Self.lastLabeledAnswer(in: response) {
            return labeled
        }
        if let mathSpan = Self.lastDollarSpan(in: response) {
            return mathSpan
        }
        let lines = response.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return lines.last
    }

    public func normalize(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        while s.hasSuffix(".") || s.hasSuffix(";") {
            s.removeLast()
        }
        s = Self.stripMathDelimiters(s)
        s = Self.stripLeadingAssignment(s)
        s = Self.unwrapCommands(s, names: ["\\text", "\\mathrm", "\\mathbf", "\\textbf", "\\operatorname", "\\mbox"])
        let replacements: [(String, String)] = [
            ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
            ("\\left", ""), ("\\right", ""),
            ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""), ("\\ ", ""),
            ("\\$", ""), ("\\%", "%"),
            ("\\cdot", "*"), ("\\times", "*"),
            ("−", "-"), ("–", "-"),
        ]
        for (old, new) in replacements {
            s = s.replacingOccurrences(of: old, with: new)
        }
        s = Self.stripThousandsCommas(s)
        s = s.replacingOccurrences(of: " ", with: "")
        while s.hasSuffix(".") || s.hasSuffix(";") {
            s.removeLast()
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func equivalent(_ candidate: String, _ expected: String) -> Bool {
        let lhs = normalize(candidate)
        let rhs = normalize(expected)
        guard !lhs.isEmpty, !rhs.isEmpty else { return false }
        if lhs == rhs { return true }
        if let rl = Self.parseRational(lhs), let rr = Self.parseRational(rhs) {
            return rl == rr
        }
        if let dl = Self.evaluateNumeric(lhs), let dr = Self.evaluateNumeric(rhs) {
            let scale = max(1.0, max(abs(dl), abs(dr)))
            return abs(dl - dr) <= numericTolerance * scale
        }
        return Self.canonicalSymbolic(lhs) == Self.canonicalSymbolic(rhs)
    }
}

extension MathAnswerGrader {
    struct Rational: Equatable {
        let num: Int64
        let den: Int64

        init?(_ num: Int64, _ den: Int64) {
            guard den != 0 else { return nil }
            if num == 0 {
                self.num = 0
                self.den = 1
                return
            }
            let sign: Int64 = (num < 0) == (den < 0) ? 1 : -1
            let a = num.magnitude
            let b = den.magnitude
            let g = Self.gcd(a, b)
            let rn = a / g
            let rd = b / g
            guard rn <= UInt64(Int64.max), rd <= UInt64(Int64.max) else { return nil }
            self.num = sign * Int64(rn)
            self.den = Int64(rd)
        }

        private static func gcd(_ a: UInt64, _ b: UInt64) -> UInt64 {
            var x = a
            var y = b
            while y != 0 {
                let t = x % y
                x = y
                y = t
            }
            return max(1, x)
        }
    }

    static func parseRational(_ expr: String) -> Rational? {
        var s = expr
        var percentDiv: Int64 = 1
        if s.hasSuffix("%") {
            s.removeLast()
            percentDiv = 100
        }
        var sign: Int64 = 1
        while s.hasPrefix("+") || s.hasPrefix("-") {
            if s.removeFirst() == "-" { sign = -sign }
        }
        guard !s.isEmpty else { return nil }
        if let (numStr, denStr) = extractFracArgs(s) {
            guard let n = parseRational(numStr), let d = parseRational(denStr), d.num != 0 else { return nil }
            let (nn, o1) = n.num.multipliedReportingOverflow(by: d.den)
            let (dd1, o2) = n.den.multipliedReportingOverflow(by: d.num)
            let (dd, o3) = dd1.multipliedReportingOverflow(by: percentDiv)
            let (sn, o4) = nn.multipliedReportingOverflow(by: sign)
            guard !o1, !o2, !o3, !o4 else { return nil }
            return Rational(sn, dd)
        }
        if let slash = s.firstIndex(of: "/"), s[s.index(after: slash)...].firstIndex(of: "/") == nil {
            let lhs = String(s[..<slash])
            let rhs = String(s[s.index(after: slash)...])
            guard let n = parseDecimalRational(lhs), let d = parseDecimalRational(rhs), d.num != 0 else { return nil }
            let (nn, o1) = n.num.multipliedReportingOverflow(by: d.den)
            let (dd1, o2) = n.den.multipliedReportingOverflow(by: d.num)
            let (dd, o3) = dd1.multipliedReportingOverflow(by: percentDiv)
            let (sn, o4) = nn.multipliedReportingOverflow(by: sign)
            guard !o1, !o2, !o3, !o4 else { return nil }
            return Rational(sn, dd)
        }
        guard let dec = parseDecimalRational(s) else { return nil }
        let (sn, o1) = dec.num.multipliedReportingOverflow(by: sign)
        let (dd, o2) = dec.den.multipliedReportingOverflow(by: percentDiv)
        guard !o1, !o2 else { return nil }
        return Rational(sn, dd)
    }

    private static func parseDecimalRational(_ token: String) -> Rational? {
        guard !token.isEmpty else { return nil }
        var s = token
        var sign: Int64 = 1
        while s.hasPrefix("+") || s.hasPrefix("-") {
            if s.removeFirst() == "-" { sign = -sign }
        }
        let parts = s.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count <= 2 else { return nil }
        let intPart = String(parts[0])
        let fracPart = parts.count == 2 ? String(parts[1]) : ""
        let digits = intPart + fracPart
        guard !digits.isEmpty, digits.allSatisfy(\.isASCII), digits.allSatisfy(\.isNumber), fracPart.count <= 12 else {
            return nil
        }
        guard let magnitude = Int64(digits) else { return nil }
        var denom: Int64 = 1
        for _ in 0 ..< fracPart.count {
            let (next, overflow) = denom.multipliedReportingOverflow(by: 10)
            guard !overflow else { return nil }
            denom = next
        }
        return Rational(sign * magnitude, denom)
    }
}

extension MathAnswerGrader {
    static func evaluateNumeric(_ expr: String) -> Double? {
        if let r = parseRational(expr) { return Double(r.num) / Double(r.den) }
        var s = expr
        if s.hasSuffix("%") {
            s.removeLast()
            return evaluateNumeric(s).map { $0 / 100.0 }
        }
        if s.hasPrefix("-") {
            return evaluateNumeric(String(s.dropFirst())).map { -$0 }
        }
        if s.hasPrefix("+") {
            return evaluateNumeric(String(s.dropFirst()))
        }
        if let (numStr, denStr) = extractFracArgs(s) {
            guard let n = evaluateNumeric(numStr), let d = evaluateNumeric(denStr), d != 0 else { return nil }
            return n / d
        }
        if let slash = s.firstIndex(of: "/"), !s.contains("\\frac") {
            let lhs = String(s[..<slash])
            let rhs = String(s[s.index(after: slash)...])
            if let n = evaluateNumeric(lhs), let d = evaluateNumeric(rhs), d != 0 { return n / d }
        }
        if s == "\\pi" || s == "pi" { return Double.pi }
        if let val = Double(s), val.isFinite { return val }
        return evaluateSimpleProduct(s)
    }

    private static func evaluateSimpleProduct(_ expr: String) -> Double? {
        var s = expr
        var sign = 1.0
        if s.hasPrefix("-") {
            sign = -1.0
            s.removeFirst()
        }
        if s.hasPrefix("\\sqrt{"), s.hasSuffix("}") {
            let inner = String(s.dropFirst(6).dropLast())
            guard let v = evaluateNumeric(inner), v >= 0 else { return nil }
            return sign * sqrt(v)
        }
        if let range = s.range(of: "\\sqrt{"), s.hasSuffix("}") {
            let prefix = String(s[..<range.lowerBound]).replacingOccurrences(of: "*", with: "")
            let inner = String(s[range.upperBound...].dropLast())
            guard let coeff = prefix.isEmpty ? 1.0 : evaluateNumeric(prefix),
                  let v = evaluateNumeric(inner), v >= 0 else { return nil }
            return sign * coeff * sqrt(v)
        }
        if s.hasSuffix("\\pi") {
            let prefix = String(s.dropLast(3)).replacingOccurrences(of: "*", with: "")
            guard let coeff = prefix.isEmpty ? 1.0 : evaluateNumeric(prefix) else { return nil }
            return sign * coeff * Double.pi
        }
        return nil
    }
}

extension MathAnswerGrader {
    static func canonicalSymbolic(_ expr: String) -> String {
        var s = expr
        if let (num, den) = extractFracArgs(s) {
            s = "(\(canonicalSymbolic(num)))/(\(canonicalSymbolic(den)))"
        }
        return s.lowercased()
    }

    static func extractFracArgs(_ expr: String) -> (String, String)? {
        guard expr.hasPrefix("\\frac") else { return nil }
        let afterCmd = expr.index(expr.startIndex, offsetBy: 5)
        guard let (first, afterFirst) = readBracedGroup(in: expr, from: afterCmd),
              let (second, endIdx) = readBracedGroup(in: expr, from: afterFirst),
              endIdx == expr.endIndex else {
            return nil
        }
        return (first, second)
    }

    static func lastBalancedCommand(in text: String, names: [String]) -> String? {
        var best: String?
        var idx = text.startIndex
        while idx < text.endIndex {
            var matched: String?
            for name in names where text[idx...].hasPrefix(name) {
                matched = name
                break
            }
            guard let cmd = matched else {
                idx = text.index(after: idx)
                continue
            }
            let afterCmd = text.index(idx, offsetBy: cmd.count)
            var cursor = afterCmd
            while cursor < text.endIndex, text[cursor].isWhitespace {
                cursor = text.index(after: cursor)
            }
            if let (content, next) = readBracedGroup(in: text, from: cursor) {
                best = content
                idx = next
            } else {
                idx = afterCmd
            }
        }
        return best
    }

    static func readBracedGroup(in text: String, from start: String.Index) -> (String, String.Index)? {
        guard start < text.endIndex, text[start] == "{" else { return nil }
        var depth = 0
        var idx = start
        let contentStart = text.index(after: start)
        while idx < text.endIndex {
            let ch = text[idx]
            if ch == "{" { depth += 1 }
            else if ch == "}" {
                depth -= 1
                if depth == 0 {
                    return (String(text[contentStart ..< idx]), text.index(after: idx))
                }
            }
            idx = text.index(after: idx)
        }
        return nil
    }

    static func unwrapCommands(_ text: String, names: [String]) -> String {
        var current = text
        var changed = true
        while changed {
            changed = false
            for name in names {
                guard let range = current.range(of: name),
                      let (inner, next) = readBracedGroup(in: current, from: range.upperBound) else { continue }
                current.replaceSubrange(range.lowerBound ..< next, with: inner)
                changed = true
                break
            }
        }
        return current
    }

    static func stripMathDelimiters(_ text: String) -> String {
        var s = text
        let pairs: [(String, String)] = [("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")]
        for (open, close) in pairs {
            if s.hasPrefix(open), s.hasSuffix(close), s.count >= open.count + close.count {
                s = String(s.dropFirst(open.count).dropLast(close.count)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return s
    }

    static func stripLeadingAssignment(_ text: String) -> String {
        guard let eq = text.firstIndex(of: "=") else { return text }
        let lhs = text[..<eq].trimmingCharacters(in: .whitespaces)
        guard !lhs.isEmpty, lhs.count <= 3, lhs.allSatisfy(\.isLetter) else { return text }
        return String(text[text.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
    }

    static func stripThousandsCommas(_ text: String) -> String {
        let chars = Array(text)
        guard chars.count >= 5 else { return text }
        var out: [Character] = []
        out.reserveCapacity(chars.count)
        for i in chars.indices {
            if chars[i] == ",", i > 0, i + 3 < chars.count,
               chars[i - 1].isNumber, chars[i + 1].isNumber, chars[i + 2].isNumber, chars[i + 3].isNumber,
               (i + 4 == chars.count || !chars[i + 4].isNumber) {
                continue
            }
            out.append(chars[i])
        }
        return String(out)
    }

    static func lastPrefixedLine(in text: String, prefix: String) -> String? {
        for line in text.split(whereSeparator: \.isNewline).reversed() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix(prefix) {
                let val = trimmed.dropFirst(prefix.count).trimmingCharacters(in: .whitespaces)
                if !val.isEmpty { return val }
            }
        }
        return nil
    }

    static func lastLabeledAnswer(in text: String) -> String? {
        let markers = ["final answer:", "answer:", "the answer is"]
        var bestRange: Range<String.Index>?
        for marker in markers {
            if let range = text.range(of: marker, options: [.backwards, .caseInsensitive]) {
                if bestRange == nil || range.lowerBound > bestRange!.lowerBound {
                    bestRange = range
                }
            }
        }
        guard let found = bestRange else { return nil }
        let tail = text[found.upperBound...].split(whereSeparator: \.isNewline).first ?? ""
        let trimmed = tail.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static func lastDollarSpan(in text: String) -> String? {
        let parts = text.split(separator: "$", omittingEmptySubsequences: false)
        guard parts.count >= 3 else { return nil }
        let idx = parts.count - 2
        let candidate = parts[idx].trimmingCharacters(in: .whitespacesAndNewlines)
        return candidate.isEmpty ? nil : candidate
    }
}
