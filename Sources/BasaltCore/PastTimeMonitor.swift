import Foundation

public indirect enum PastFormula: Sendable, Equatable {
    case atom(Int)
    case not(PastFormula)
    case and(PastFormula, PastFormula)
    case or(PastFormula, PastFormula)
    case yesterday(PastFormula)
    case once(PastFormula)
    case historically(PastFormula)
    case since(PastFormula, PastFormula)
}

public struct PastMonitor: Sendable {
    private enum Node: Sendable {
        case atom(Int)
        case not(Int)
        case and(Int, Int)
        case or(Int, Int)
        case yesterday(Int)
        case once(Int)
        case since(Int, Int)

        func value(at index: Int, atoms: UInt64, now: UInt64, previous: UInt64) -> Bool {
            switch self {
            case .atom(let bit): PastMonitor.bit(atoms, bit)
            case .not(let operand): !PastMonitor.bit(now, operand)
            case .and(let lhs, let rhs): PastMonitor.bit(now, lhs) && PastMonitor.bit(now, rhs)
            case .or(let lhs, let rhs): PastMonitor.bit(now, lhs) || PastMonitor.bit(now, rhs)
            case .yesterday(let operand): PastMonitor.bit(previous, operand)
            case .once(let operand): PastMonitor.bit(now, operand) || PastMonitor.bit(previous, index)
            case .since(let lhs, let rhs): PastMonitor.bit(now, rhs) || (PastMonitor.bit(now, lhs) && PastMonitor.bit(previous, index))
            }
        }
    }

    private var nodes: [Node] = []
    private var previous: UInt64 = 0

    public init(_ formula: PastFormula) {
        _ = compile(formula)
        precondition(nodes.count <= UInt64.bitWidth, "a past-time formula has at most 64 subformulas")
    }

    private mutating func compile(_ formula: PastFormula) -> Int {
        let node: Node = switch formula {
        case .atom(let bit): .atom(bit)
        case .not(let operand): .not(compile(operand))
        case .and(let lhs, let rhs): .and(compile(lhs), compile(rhs))
        case .or(let lhs, let rhs): .or(compile(lhs), compile(rhs))
        case .yesterday(let operand): .yesterday(compile(operand))
        case .once(let operand): .once(compile(operand))
        case .historically(let operand): .not(compile(.once(.not(operand))))
        case .since(let lhs, let rhs): .since(compile(lhs), compile(rhs))
        }
        nodes.append(node)
        return nodes.count - 1
    }

    fileprivate static func bit(_ word: UInt64, _ index: Int) -> Bool {
        word >> UInt64(index) & 1 == 1
    }

    public mutating func step(_ atoms: UInt64) -> Bool {
        var now: UInt64 = 0
        for (index, node) in nodes.enumerated() where node.value(at: index, atoms: atoms, now: now, previous: previous) {
            now |= 1 << UInt64(index)
        }
        previous = now
        return Self.bit(now, nodes.count - 1)
    }
}
