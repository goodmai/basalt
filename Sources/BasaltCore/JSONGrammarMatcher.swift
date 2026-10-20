import Foundation

public struct GrammarFrame: Hashable, Sendable {
    var node: Int32
    var phase: UInt8 = 0
    var whitespace: UInt8 = 0
    var cursor: Int32 = 0
    var seen: UInt64 = 0

    static let document: Int32 = -1
}

public struct JSONGrammarMatcher: Sendable {
    public enum Step: Sendable, Equatable {
        case consumed
        case rejected
        case needsContext
    }

    static let whitespaceRunLimit: UInt8 = 32

    let grammar: JSONGrammar
    public private(set) var stack: [GrammarFrame]

    public init(grammar: JSONGrammar) {
        self.grammar = grammar
        self.stack = [GrammarFrame(node: GrammarFrame.document)]
    }

    init(grammar: JSONGrammar, stack: [GrammarFrame]) {
        self.grammar = grammar
        self.stack = stack
    }

    public mutating func accept(_ bytes: some Sequence<UInt8>) -> Bool {
        var trial = self
        for byte in bytes where trial.advance(byte) != .consumed { return false }
        self = trial
        return true
    }

    public var isAccepting: Bool {
        guard stack[0].node == GrammarFrame.document, stack[0].phase == 1 else { return false }
        return stack.count == 1 || (stack.count == 2 && isComplete(stack[1]))
    }

    private enum Transition {
        case stay(GrammarFrame)
        case push(GrammarFrame, child: GrammarFrame)
        case complete
        case reject
    }

    public mutating func advance(_ byte: UInt8) -> Step {
        while true {
            let top = stack.count - 1
            switch transition(stack[top], byte) {
            case .stay(let frame):
                stack[top] = frame
                popFinishedFrames()
                return .consumed
            case .push(let frame, let child):
                stack[top] = frame
                stack.append(child)
            case .complete:
                guard top > 0 else { return .needsContext }
                let child = stack.removeLast()
                stack[top - 1] = completed(stack[top - 1], by: child)
            case .reject:
                return .rejected
            }
        }
    }

    private mutating func popFinishedFrames() {
        while stack.count > 1, isFinal(stack[stack.count - 1]) {
            let child = stack.removeLast()
            stack[stack.count - 1] = completed(stack[stack.count - 1], by: child)
        }
    }

    private static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09
    }

    private func skippingWhitespace(_ frame: GrammarFrame) -> Transition {
        guard frame.whitespace < Self.whitespaceRunLimit else { return .reject }
        var next = frame
        next.whitespace += 1
        return .stay(next)
    }

    private func entering(_ frame: GrammarFrame, phase: UInt8) -> GrammarFrame {
        var next = frame
        next.phase = phase
        next.whitespace = 0
        return next
    }

    private func pushValue(_ node: Int, byte: UInt8, from frame: GrammarFrame, phase: UInt8) -> Transition {
        let start = grammar.starts[node][Int(byte)]
        guard start >= 0 else { return .reject }
        return .push(entering(frame, phase: phase), child: GrammarFrame(node: start))
    }

    private func transition(_ frame: GrammarFrame, _ byte: UInt8) -> Transition {
        guard frame.node != GrammarFrame.document else {
            guard frame.phase == 0 else { return .reject }
            if Self.isWhitespace(byte) { return skippingWhitespace(frame) }
            return pushValue(grammar.root, byte: byte, from: frame, phase: 1)
        }
        switch grammar.nodes[Int(frame.node)] {
        case .string: return stringTransition(frame, byte)
        case .number(let integerOnly): return numberTransition(frame, byte, integerOnly: integerOnly)
        case .literals(let trie): return literalTransition(frame, byte, trie)
        case .object(let shape): return objectTransition(frame, byte, shape)
        case .array(let items): return arrayTransition(frame, byte, items: items)
        case .choice: return .reject
        }
    }

    private func moving(_ frame: GrammarFrame, to phase: UInt8) -> Transition {
        .stay(entering(frame, phase: phase))
    }

    private static let escapable = Set("\"\\/bfnrt".utf8)
    private static let hexDigits = Set("0123456789abcdefABCDEF".utf8)

    private func stringTransition(_ frame: GrammarFrame, _ byte: UInt8) -> Transition {
        switch (frame.phase, byte) {
        case (0, UInt8(ascii: "\"")): return moving(frame, to: 1)
        case (1, UInt8(ascii: "\"")): return moving(frame, to: 7)
        case (1, UInt8(ascii: "\\")): return moving(frame, to: 2)
        case (1, 0x20..<0x80): return .stay(frame)
        case (1, 0xC2...0xDF): return moving(frame, to: 10)
        case (1, 0xE0): return moving(frame, to: 13)
        case (1, 0xE1...0xEC), (1, 0xEE...0xEF): return moving(frame, to: 11)
        case (1, 0xED): return moving(frame, to: 14)
        case (1, 0xF0): return moving(frame, to: 15)
        case (1, 0xF1...0xF3): return moving(frame, to: 12)
        case (1, 0xF4): return moving(frame, to: 16)
        case (2, _) where Self.escapable.contains(byte): return moving(frame, to: 1)
        case (2, UInt8(ascii: "u")): return moving(frame, to: 3)
        case (3...6, _) where Self.hexDigits.contains(byte): return moving(frame, to: frame.phase == 6 ? 1 : frame.phase + 1)
        case (10, 0x80...0xBF): return moving(frame, to: 1)
        case (11, 0x80...0xBF): return moving(frame, to: 10)
        case (12, 0x80...0xBF): return moving(frame, to: 11)
        case (13, 0xA0...0xBF): return moving(frame, to: 10)
        case (14, 0x80...0x9F): return moving(frame, to: 10)
        case (15, 0x90...0xBF): return moving(frame, to: 11)
        case (16, 0x80...0x8F): return moving(frame, to: 11)
        case (7, _): return .complete
        default: return .reject
        }
    }

    private static let numberEnds: Set<UInt8> = [2, 3, 5, 8]

    private func numberTransition(_ frame: GrammarFrame, _ byte: UInt8, integerOnly: Bool) -> Transition {
        let digit = (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
        let exponent = !integerOnly && (byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E"))
        switch frame.phase {
        case 0 where byte == UInt8(ascii: "-"): return moving(frame, to: 1)
        case 0, 1:
            if byte == UInt8(ascii: "0") { return moving(frame, to: 2) }
            if digit { return moving(frame, to: 3) }
        case 2, 3:
            if frame.phase == 3 && digit { return .stay(frame) }
            if !integerOnly && byte == UInt8(ascii: ".") { return moving(frame, to: 4) }
            if exponent { return moving(frame, to: 6) }
        case 4, 5:
            if digit { return moving(frame, to: 5) }
            if frame.phase == 5 && exponent { return moving(frame, to: 6) }
        case 6 where byte == UInt8(ascii: "+") || byte == UInt8(ascii: "-"): return moving(frame, to: 7)
        case 6, 7, 8:
            if digit { return moving(frame, to: 8) }
        default: break
        }
        return Self.numberEnds.contains(frame.phase) ? .complete : .reject
    }

    private func literalTransition(_ frame: GrammarFrame, _ byte: UInt8, _ trie: JSONGrammar.LiteralTrie) -> Transition {
        if let child = trie.children[Int(frame.cursor)][byte], trie.reachable[Int(child)] & ~frame.seen != 0 {
            var next = frame
            next.cursor = child
            return .stay(next)
        }
        return isComplete(frame) ? .complete : .reject
    }

    private func objectTransition(_ frame: GrammarFrame, _ byte: UInt8, _ shape: JSONGrammar.ObjectShape) -> Transition {
        let unseen = !shape.declared || shape.all & ~frame.seen != 0
        let closable = shape.required & ~frame.seen == 0
        switch frame.phase {
        case 0:
            return byte == UInt8(ascii: "{") ? moving(frame, to: 1) : .reject
        case 1, 2, 4, 5, 7:
            if Self.isWhitespace(byte) { return skippingWhitespace(frame) }
        default:
            return .complete
        }
        switch (frame.phase, byte) {
        case (1, UInt8(ascii: "}")) where closable, (7, UInt8(ascii: "}")) where closable:
            return moving(frame, to: 8)
        case (1, UInt8(ascii: "\"")) where unseen, (2, UInt8(ascii: "\"")) where unseen:
            return .push(entering(frame, phase: 3), child: GrammarFrame(node: Int32(shape.keys), seen: frame.seen))
        case (4, UInt8(ascii: ":")):
            return moving(frame, to: 5)
        case (5, _):
            let value = shape.declared ? shape.values[Int(frame.cursor)] : shape.values[0]
            return pushValue(value, byte: byte, from: frame, phase: 6)
        case (7, UInt8(ascii: ",")) where unseen:
            return moving(frame, to: 2)
        default:
            return .reject
        }
    }

    private func arrayTransition(_ frame: GrammarFrame, _ byte: UInt8, items: Int) -> Transition {
        switch frame.phase {
        case 0:
            return byte == UInt8(ascii: "[") ? moving(frame, to: 1) : .reject
        case 1, 3, 4:
            if Self.isWhitespace(byte) { return skippingWhitespace(frame) }
        default:
            return .complete
        }
        switch (frame.phase, byte) {
        case (1, UInt8(ascii: "]")), (3, UInt8(ascii: "]")): return moving(frame, to: 5)
        case (3, UInt8(ascii: ",")): return moving(frame, to: 4)
        case (1, _), (4, _): return pushValue(items, byte: byte, from: frame, phase: 2)
        default: return .reject
        }
    }

    private func completed(_ parent: GrammarFrame, by child: GrammarFrame) -> GrammarFrame {
        guard parent.node != GrammarFrame.document else { return parent }
        switch grammar.nodes[Int(parent.node)] {
        case .object(let shape) where parent.phase == 3:
            var next = entering(parent, phase: 4)
            if shape.declared, case .literals(let trie) = grammar.nodes[Int(child.node)] {
                next.cursor = trie.terminal[Int(child.cursor)]
                next.seen |= 1 << UInt64(next.cursor)
            }
            return next
        case .object:
            return entering(parent, phase: 7)
        default:
            return entering(parent, phase: 3)
        }
    }

    private func isFinal(_ frame: GrammarFrame) -> Bool {
        switch grammar.nodes[Int(frame.node)] {
        case .string: frame.phase == 7
        case .object: frame.phase == 8
        case .array: frame.phase == 5
        case .literals(let trie): trie.children[Int(frame.cursor)].isEmpty && isComplete(frame)
        case .number, .choice: false
        }
    }

    private func isComplete(_ frame: GrammarFrame) -> Bool {
        switch grammar.nodes[Int(frame.node)] {
        case .number: Self.numberEnds.contains(frame.phase)
        case .literals(let trie):
            trie.terminal[Int(frame.cursor)] >= 0
                && JSONGrammar.LiteralTrie.bit(Int(trie.terminal[Int(frame.cursor)])) & ~frame.seen != 0
        default: isFinal(frame)
        }
    }
}
