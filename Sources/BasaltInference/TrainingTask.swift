import Foundation
import BasaltCore

struct TrainingTask: Sendable, Codable, Equatable {
    var prompt: String
    var answer: String

    static let counts = 2...10_000

    static func load(_ tasks: TrainingTasks) throws(BasaltError) -> [TrainingTask] {
        let loaded = switch tasks {
        case .arithmetic(let count, let seed): arithmetic(count: count, seed: seed)
        case .file(let path): try file(path)
        }
        guard counts.contains(loaded.count) else {
            throw .invalidRequestStructure(details: "train: \(loaded.count) tasks; give \(counts.lowerBound)...\(counts.upperBound)")
        }
        return loaded
    }

    static func arithmetic(count: Int, seed: UInt64) -> [TrainingTask] {
        var generator = SplitMix64(seed: seed)
        return (0 ..< max(count, 0)).map { _ in
            let a = Int.random(in: 2...99, using: &generator)
            let b = Int.random(in: 2...99, using: &generator)
            let (symbol, value) = [("+", a + b), ("-", a - b), ("*", a * b)][Int.random(in: 0...2, using: &generator)]
            return TrainingTask(prompt: "What is \(a) \(symbol) \(b)? Reply with the number only.", answer: String(value))
        }
    }

    static func file(_ path: String) throws(BasaltError) -> [TrainingTask] {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw .invalidRequestStructure(details: "train: cannot read the task file \(url.path)")
        }
        let lines = text.split(whereSeparator: \.isNewline).filter { !$0.allSatisfy(\.isWhitespace) }
        var tasks: [TrainingTask] = []
        for (index, line) in lines.enumerated() {
            guard let task = try? JSONDecoder().decode(TrainingTask.self, from: Data(line.utf8)),
                  !task.prompt.isEmpty, !task.answer.isEmpty else {
                throw .invalidRequestStructure(details: "train: line \(index + 1) of \(url.lastPathComponent) is not {\"prompt\": string, \"answer\": string}")
            }
            tasks.append(task)
        }
        return tasks
    }

    func accepts(_ output: String) -> Bool {
        if let expected = Int(answer.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return Self.finalNumber(output) == expected
        }
        let expected = Self.normalized(answer)
        let final = Self.normalized(output.split(whereSeparator: \.isNewline).last.map(String.init) ?? "")
        return !expected.isEmpty && (final == expected || final.hasSuffix(": " + expected))
    }

    static func finalNumber(_ text: String) -> Int? {
        text.replacing(",", with: "").matches(of: /-?\d+/).last.flatMap { Int($0.output) }
    }

    static func normalized(_ text: String) -> String {
        text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
    }
}
