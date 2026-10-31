import Foundation
import Testing

public struct TokenDiff: Equatable, CustomTestReflectable, CustomTestStringConvertible, Sendable {
    public let tokens: [Int]
    public let expected: [Int]?

    public init(_ tokens: [Int], expected: [Int]? = nil) {
        self.tokens = tokens
        self.expected = expected
    }

    public static func == (lhs: TokenDiff, rhs: TokenDiff) -> Bool {
        lhs.tokens == rhs.tokens
    }

    public var customTestMirror: Mirror {
        if let expected = expected, tokens != expected {
            let firstDiff = zip(tokens.indices, zip(tokens, expected))
                .first { $1.0 != $1.1 }?
                .0 ?? min(tokens.count, expected.count)

            let start = max(0, firstDiff - 4)
            let actualEnd = min(tokens.count, firstDiff + 5)
            let expectedEnd = min(expected.count, firstDiff + 5)

            let actualWindow = Array(tokens[start..<actualEnd])
            let expectedWindow = Array(expected[start..<expectedEnd])

            return Mirror(self, children: [
                "divergenceIndex": firstDiff,
                "count": "actual \(tokens.count) vs expected \(expected.count)",
                "actualWindow": actualWindow,
                "expectedWindow": expectedWindow,
                "tokens": tokens,
            ])
        }
        return Mirror(self, children: [
            "count": tokens.count,
            "tokens": tokens,
        ])
    }

    public var testDescription: String {
        if let expected = expected, tokens != expected {
            let firstDiff = zip(tokens.indices, zip(tokens, expected))
                .first { $1.0 != $1.1 }?
                .0 ?? min(tokens.count, expected.count)
            let start = max(0, firstDiff - 4)
            let actualEnd = min(tokens.count, firstDiff + 5)
            let expectedEnd = min(expected.count, firstDiff + 5)
            let actualWindow = Array(tokens[start..<actualEnd])
            let expectedWindow = Array(expected[start..<expectedEnd])
            return "TokenDiff(divergenceIndex: \(firstDiff), actual: \(actualWindow), expected: \(expectedWindow))"
        }
        return "TokenDiff(\(tokens.count) tokens)"
    }
}

public func expectTokenSequence(
    _ actual: [Int],
    _ expected: [Int],
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let diffActual = TokenDiff(actual, expected: expected)
    let diffExpected = TokenDiff(expected, expected: actual)
    #expect(diffActual == diffExpected, sourceLocation: sourceLocation)
}
