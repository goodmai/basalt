import Foundation
import CryptoKit
import BasaltCore

struct DistillExample: Sendable, Equatable, Codable {
    var prompt: String
    var answer: String?
}

struct DistillEntry: Sendable, Equatable, Codable {
    var index: Int
    var sample: Int
    var completion: TeacherClient.Completion
}

struct DistillCache: Sendable {
    let directory: URL
    var file: URL { directory.appending(component: "teacher.jsonl") }

    static func directory(for request: DistillRequest, prompts: Data, home: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        let teacher = request.teacher
        let identity = [teacher.url, teacher.model, teacher.provider.rawValue, "\(teacher.maxTokens)", "\(teacher.temperature)", teacher.reasoningEffort ?? ""].joined(separator: "\n")
        let digest = SHA256.hash(data: Data(identity.utf8) + prompts).map { String(format: "%02x", $0) }.joined()
        return home.appending(path: ".basalt/distill/\(digest.prefix(32))")
    }

    func load() -> [DistillEntry] {
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return [] }
        return text.split(whereSeparator: \.isNewline).compactMap { try? JSONDecoder().decode(DistillEntry.self, from: Data($0.utf8)) }
    }

    func append(_ entries: [DistillEntry]) throws {
        guard !entries.isEmpty else { return }
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        if !manager.fileExists(atPath: file.path) {
            manager.createFile(atPath: file.path, contents: nil, attributes: [.posixPermissions: 0o600])
        }
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.seekToEnd()
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        for entry in entries { try handle.write(contentsOf: try encoder.encode(entry) + Data("\n".utf8)) }
    }
}

enum DistillCore {
    static func examples(jsonl: String) throws(BasaltError) -> [DistillExample] {
        let lines = jsonl.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        let examples = try lines.enumerated().map { index, line throws(BasaltError) -> DistillExample in
            guard let example = try? JSONDecoder().decode(DistillExample.self, from: Data(line.utf8)), !example.prompt.isEmpty else {
                throw .invalidRequestStructure(details: "distill: line \(index + 1) of the prompts is not {\"prompt\": \"…\", \"answer\"?: \"…\"}")
            }
            return example
        }
        guard examples.count >= 2 else { throw .invalidRequestStructure(details: "distill: the prompts file needs at least two prompts, one for training and one held out") }
        return examples
    }

    static func heldOutCount(_ total: Int, fraction: Float) -> Int {
        min(total - 1, max(1, Int((Float(total) * fraction).rounded(.up))))
    }

    static func cost(_ completion: TeacherClient.Completion, teacher: TeacherSpec) -> Double {
        (Double(completion.inputTokens) * (teacher.inputPricePerMillion ?? 0) + Double(completion.outputTokens) * (teacher.outputPricePerMillion ?? 0)) / 1e6
    }

    static func worstCase(_ calls: Int, teacher: TeacherSpec) -> Double {
        Double(calls * teacher.maxTokens) * (teacher.outputPricePerMillion ?? 0) / 1e6
    }

    static func rejection(_ completion: TeacherClient.Completion, example: DistillExample, filter: DistillFilter, seen: inout Set<String>) -> String? {
        let text = completion.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return "empty" }
        if completion.truncated { return "truncated" }
        if filter == .verify, let answer = example.answer, !TrainingTask(prompt: example.prompt, answer: answer).accepts(text) { return "unverified" }
        if !seen.insert(example.prompt + "\u{0}" + text).inserted { return "duplicate" }
        return nil
    }

    static func missing(_ entries: [DistillEntry], prompts: Int, samples: Int) -> [(index: Int, sample: Int)] {
        let have = Set(entries.map { "\($0.index):\($0.sample)" })
        return (0 ..< prompts).flatMap { index in (0 ..< samples).map { (index, $0) } }.filter { !have.contains("\($0.0):\($0.1)") }
    }
}
