import Foundation
import BasaltCore

public struct ARCAGITaskItem: Sendable, Equatable, Codable {
    public struct Pair: Sendable, Equatable, Codable {
        public var input: [[Int]]
        public var output: [[Int]]?

        public init(input: [[Int]], output: [[Int]]? = nil) {
            (self.input, self.output) = (input, output)
        }
    }

    public var id: String
    public var train: [Pair]
    public var test: [Pair]

    public init(id: String, train: [Pair], test: [Pair]) {
        (self.id, self.train, self.test) = (id, train, test)
    }
}

public enum ARCGridParser {
    public static func withoutThinking(_ text: String) -> String {
        var result = text
        while let start = result.range(of: "<think>"),
              let end = result.range(of: "</think>", range: start.upperBound ..< result.endIndex) {
            result.removeSubrange(start.lowerBound ..< end.upperBound)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func extractAttemptGrids(from response: String) -> [[[Int]]] {
        let cleaned = withoutThinking(response)
        if let fromObject = parseJSONObjectAttempts(cleaned), !fromObject.isEmpty {
            return Array(fromObject.prefix(2))
        }
        var found: [[[Int]]] = []
        for candidate in candidateBlocks(cleaned) {
            for grid in parseGridsFromSnippet(candidate) where !grid.isEmpty {
                found.append(grid)
                if found.count == 2 { return found }
            }
        }
        return found
    }

    static func parseJSONObjectAttempts(_ text: String) -> [[[Int]]]? {
        for block in candidateBlocks(text) {
            guard let open = block.firstIndex(of: "{"), let close = block.lastIndex(of: "}"), open < close else { continue }
            let slice = String(block[open ... close])
            guard let obj = try? JSONSerialization.jsonObject(with: Data(slice.utf8)) as? [String: Any] else { continue }
            var out: [[[Int]]] = []
            if let a1 = as2DGrid(obj["attempt_1"]) { out.append(a1) }
            if let a2 = as2DGrid(obj["attempt_2"]) { out.append(a2) }
            if out.isEmpty, let list = obj["attempts"] as? [Any] {
                out = list.compactMap(as2DGrid)
            }
            if !out.isEmpty { return out }
        }
        return nil
    }

    static func candidateBlocks(_ text: String) -> [String] {
        var blocks: [String] = []
        let parts = text.components(separatedBy: "```")
        if parts.count >= 3 {
            for idx in stride(from: 1, to: parts.count, by: 2) {
                var body = parts[idx]
                if let newline = body.firstIndex(of: "\n") {
                    let firstLine = body[..<newline].trimmingCharacters(in: .whitespacesAndNewlines)
                    if !firstLine.hasPrefix("[") && !firstLine.hasPrefix("{") {
                        body = String(body[body.index(after: newline)...])
                    }
                }
                blocks.append(body.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        blocks.append(text)
        return blocks
    }

    static func parseGridsFromSnippet(_ snippet: String) -> [[[Int]]] {
        guard let open = snippet.firstIndex(of: "["), let close = snippet.lastIndex(of: "]"), open < close else {
            return []
        }
        let slice = String(snippet[open ... close])
        guard let raw = try? JSONSerialization.jsonObject(with: Data(slice.utf8)) as? [Any] else { return [] }
        if let single = as2DGrid(raw) { return [single] }
        return raw.compactMap(as2DGrid)
    }

    static func as2DGrid(_ any: Any?) -> [[Int]]? {
        guard let rows = any as? [Any], !rows.isEmpty else { return nil }
        var grid: [[Int]] = []
        for rowAny in rows {
            guard let cols = rowAny as? [Any], !cols.isEmpty else { return nil }
            var row: [Int] = []
            for cell in cols {
                guard let number = cell as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
                let value = number.intValue
                guard Double(value) == number.doubleValue else { return nil }
                row.append(value)
            }
            grid.append(row)
        }
        return grid
    }

    public static func scoreAttempts(_ attempts: [[[Int]]], expected: [[Int]]) -> (passed: Bool, score: Double) {
        let matched = attempts.prefix(2).contains(expected)
        return (matched, matched ? 1.0 : 0.0)
    }
}

public struct ARCAGIAdapter: BenchmarkAdapter {
    public let id: String
    public let summary: String
    public var tier: BenchmarkAdapterTier { .pureSwift }
    public var defaultSubmissionTarget: SubmissionTarget? { .arcPrize }

    public init(id: String = "arc-agi", summary: String = "Pure-Swift ARC-AGI 2-attempt grid reasoner and exact-match scorer") {
        (self.id, self.summary) = (id, summary)
    }

    static let smokeTask = ARCAGITaskItem(
        id: "arc-smoke-invert-2x2",
        train: [
            .init(input: [[0, 1], [1, 0]], output: [[1, 0], [0, 1]]),
            .init(input: [[1, 1], [0, 0]], output: [[0, 0], [1, 1]]),
        ],
        test: [.init(input: [[0, 0], [1, 0]], output: [[1, 1], [0, 1]])]
    )

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        let tasks = try loadTasks(path: context.task.dataset)
        var samples: [BenchmarkSampleRecord] = []
        for item in tasks {
            let pairs = item.test.isEmpty ? [ARCAGITaskItem.Pair(input: [], output: nil)] : item.test
            for (idx, pair) in pairs.enumerated() {
                let sampleId = pairs.count > 1 ? "\(item.id)#\(idx)" : item.id
                let prompt = Self.buildPrompt(item, testIndex: idx)
                let payload: [String: Any] = [
                    "model": context.resolvedModelName,
                    "messages": [["role": "user", "content": prompt]],
                    "max_tokens": context.task.params["max_tokens"]?.intValue ?? 2048,
                    "temperature": context.temperature,
                    "seed": context.seed,
                ]
                let (json, reply) = try await context.endpoint.postJSON("/v1/chat/completions", payload: payload)
                let content = BenchmarkStats.extractChatContent(json)
                let usage = BenchmarkStats.extractUsage(json)
                var grids = ARCGridParser.extractAttemptGrids(from: content)
                if grids.count == 1 { grids.append(grids[0]) }
                let verdict: (passed: Bool, score: Double)
                if let expected = pair.output {
                    verdict = ARCGridParser.scoreAttempts(grids, expected: expected)
                } else {
                    let valid = grids.count == 2 && grids.allSatisfy { ARCGridValidator.validate($0, label: sampleId).isEmpty }
                    verdict = (valid, valid ? 1.0 : 0.0)
                }
                var sample = BenchmarkSampleRecord(
                    id: sampleId, adapter: id, model: context.resolvedModelName,
                    prompt: prompt, answer: content, score: verdict.score
                )
                sample.dataset = context.task.dataset
                sample.gridAttempts = grids
                sample.latencyMs = reply.totalMs
                sample.promptTokens = usage.prompt
                sample.completionTokens = usage.completion
                sample.passed = verdict.passed
                samples.append(sample)
            }
        }
        let accuracy = Double(samples.filter(\.passed).count) / Double(max(1, samples.count))
        let metrics: [String: Double] = [
            "accuracy": accuracy,
            "p50_ms": BenchmarkStats.percentile(samples.map(\.latencyMs), 0.5),
        ]
        return BenchmarkAdapterOutput(samples: samples, metrics: metrics)
    }

    func loadTasks(path: String?) throws -> [ARCAGITaskItem] {
        guard let path else { return [Self.smokeTask] }
        let url = URL(fileURLWithPath: path)
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue {
            let files = try FileManager.default.contentsOfDirectory(atPath: path).filter { $0.hasSuffix(".json") }.sorted()
            return try files.map { file in
                let id = String(file.dropLast(".json".count))
                return try decodeSingleTask(id: id, data: Data(contentsOf: url.appendingPathComponent(file)))
            }
        }
        let data = try Data(contentsOf: url)
        if let list = try? JSONDecoder().decode([ARCAGITaskItem].self, from: data), !list.isEmpty {
            return list
        }
        let rawMap = try JSONDecoder().decode([String: RawTaskBody].self, from: data)
        return rawMap.keys.sorted().compactMap { key in
            rawMap[key].map { ARCAGITaskItem(id: key, train: $0.train, test: $0.test) }
        }
    }

    struct RawTaskBody: Codable {
        var train: [ARCAGITaskItem.Pair]
        var test: [ARCAGITaskItem.Pair]
    }

    func decodeSingleTask(id: String, data: Data) throws -> ARCAGITaskItem {
        let body = try JSONDecoder().decode(RawTaskBody.self, from: data)
        return ARCAGITaskItem(id: id, train: body.train, test: body.test)
    }

    static func buildPrompt(_ task: ARCAGITaskItem, testIndex: Int = 0) -> String {
        var lines = ["Solve this ARC-AGI grid transformation puzzle. Cells are integers 0..9."]
        for (idx, pair) in task.train.enumerated() {
            lines.append("Train \(idx + 1) input: \(pair.input)")
            if let out = pair.output { lines.append("Train \(idx + 1) output: \(out)") }
        }
        if testIndex >= 0, testIndex < task.test.count {
            lines.append("Test input: \(task.test[testIndex].input)")
        }
        lines.append("Reply with JSON {\"attempt_1\": [[...]], \"attempt_2\": [[...]]}.")
        return lines.joined(separator: "\n")
    }
}
