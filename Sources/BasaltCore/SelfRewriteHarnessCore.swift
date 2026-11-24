import Foundation

public struct SelfRewriteRepairSpec: Sendable, Codable, Equatable {
    public let tensor: String
    public let rows: [Int]
    public let scale: Float

    public init(tensor: String, rows: [Int], scale: Float) throws(SelfRewriteSessionError) {
        let trimmed = tensor.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !rows.isEmpty, rows.allSatisfy({ $0 >= 0 }), Set(rows).count == rows.count else {
            throw .invalidEditorTurn("repair spec requires non-empty tensor name and unique non-negative row indices")
        }
        guard scale.isFinite, scale > 0 else {
            throw .invalidEditorTurn("repair spec scale must be finite and positive")
        }
        self.tensor = trimmed
        self.rows = rows
        self.scale = scale
    }

    public func damageProgram(
        sessionId: String = "repair-damage",
        valuesKey: String = "damage_rows"
    ) -> EditProgram {
        EditProgram(
            sessionId: sessionId,
            approval: .pending,
            ops: [.rows(tensor: tensor, rowIndices: rows, valuesKey: valuesKey)]
        )
    }
}

public enum SelfRewriteTaskMode: Sendable, Codable, Equatable {
    case improve
    case repair(SelfRewriteRepairSpec)

    public static func parse(_ raw: String) throws(SelfRewriteSessionError) -> SelfRewriteTaskMode {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == "improve" {
            return .improve
        }
        guard trimmed.hasPrefix("repair:") else {
            throw .invalidEditorTurn("invalid --task '\(raw)' (expected 'improve' or 'repair:<tensor>:<rows>:<scale>')")
        }
        let body = String(trimmed.dropFirst("repair:".count))
        let parts = body.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 3 else {
            throw .invalidEditorTurn("repair task must have format 'repair:<tensor>:<row0,row1,...>:<scale>'")
        }
        let rows = parts[1].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard rows.count == parts[1].split(separator: ",").count, !rows.isEmpty else {
            throw .invalidEditorTurn("invalid row indices '\(parts[1])' in repair task")
        }
        guard let scale = Float(parts[2].trimmingCharacters(in: .whitespaces)) else {
            throw .invalidEditorTurn("invalid scale '\(parts[2])' in repair task")
        }
        return .repair(try SelfRewriteRepairSpec(tensor: parts[0], rows: rows, scale: scale))
    }

    public var label: String {
        switch self {
        case .improve: "improve"
        case .repair(let spec): "repair:\(spec.tensor):\(spec.rows.map(String.init).joined(separator: ",")):\(spec.scale)"
        }
    }
}

public enum SelfRewriteDatasetLoader {
    public static func loadProblems(fromJSONL data: Data) throws -> [PairedProblem] {
        let text = String(decoding: data, as: UTF8.self)
        var problems: [PairedProblem] = []
        for (lineIdx, rawLine) in text.split(whereSeparator: \.isNewline).enumerated() {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            guard let obj = try JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any] else {
                throw SelfRewriteSessionError.invalidEditorTurn("dataset line \(lineIdx + 1) is not a JSON object")
            }
            guard let prompt = (obj["prompt"] as? String) ?? (obj["problem"] as? String) ?? (obj["question"] as? String),
                  !prompt.isEmpty else {
                throw SelfRewriteSessionError.invalidEditorTurn("dataset line \(lineIdx + 1) missing non-empty prompt")
            }
            guard let answer = (obj["expected_answer"] as? String) ?? (obj["expectedAnswer"] as? String) ?? (obj["answer"] as? String) ?? (obj["reference"] as? String),
                  !answer.isEmpty else {
                throw SelfRewriteSessionError.invalidEditorTurn("dataset line \(lineIdx + 1) missing expected_answer")
            }
            let id = (obj["id"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "row-\(lineIdx)"
            let sourceText = (obj["source_text"] as? String) ?? (obj["sourceText"] as? String) ?? (obj["source"] as? String)
            problems.append(PairedProblem(id: id, prompt: prompt, expectedAnswer: answer, sourceText: sourceText))
        }
        guard !problems.isEmpty else {
            throw SelfRewriteSessionError.invalidEditorTurn("dataset JSONL contains no problems")
        }
        return problems
    }

    public static func loadProblems(at url: URL) throws -> [PairedProblem] {
        try loadProblems(fromJSONL: try Data(contentsOf: url))
    }

    public static func loadBenchmarkTexts(at url: URL) throws -> [String] {
        let text = try String(contentsOf: url, encoding: .utf8)
        var out: [String] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            if trimmed.hasPrefix("{"),
               let obj = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any],
               let extracted = (obj["prompt"] as? String) ?? (obj["problem"] as? String) ?? (obj["question"] as? String) ?? (obj["source"] as? String) {
                out.append(extracted)
            } else {
                out.append(trimmed)
            }
        }
        return out
    }

    public static func buildSplit(
        practiceURL: URL,
        decontaminateURL: URL? = nil,
        practiceCount: Int? = nil,
        ngramSize: Int = 8,
        seed: UInt64 = 676
    ) throws -> (split: PairedDatasetSplit, removedDuplicates: Int) {
        let candidates = try loadProblems(at: practiceURL)
        let benchmarks = try decontaminateURL.map(loadBenchmarkTexts(at:)) ?? []
        let targetPractice = practiceCount ?? max(1, (candidates.count * 3) / 4)
        let result = PairedDatasetSplit.decontaminated(
            candidates: candidates,
            benchmarks: benchmarks,
            practiceCount: targetPractice,
            ngramSize: ngramSize,
            seed: seed
        )
        return (result.split, result.removedCount)
    }
}

public struct SelfEditSessionMetadata: Sendable, Codable, Equatable {
    public let sessionId: String
    public let modelPath: String
    public let editor: String
    public let taskMode: SelfRewriteTaskMode
    public let transcriptPath: String
    public let startedAtISO8601: String

    public init(
        sessionId: String,
        modelPath: String,
        editor: String,
        taskMode: SelfRewriteTaskMode,
        transcriptPath: String,
        startedAtISO8601: String
    ) {
        self.sessionId = sessionId
        self.modelPath = modelPath
        self.editor = editor
        self.taskMode = taskMode
        self.transcriptPath = transcriptPath
        self.startedAtISO8601 = startedAtISO8601
    }
}

public struct SelfEditSessionRecord: Sendable, Codable, Equatable {
    public let metadata: SelfEditSessionMetadata
    public var phase: EditSessionPhase
    public var practiceCount: Int
    public var reservedCount: Int
    public var removedDuplicates: Int
    public var turnCount: Int
    public var savedAdapterPath: String?

    public init(
        metadata: SelfEditSessionMetadata,
        phase: EditSessionPhase = .running,
        practiceCount: Int,
        reservedCount: Int,
        removedDuplicates: Int = 0,
        turnCount: Int = 0
    ) {
        self.metadata = metadata
        self.phase = phase
        self.practiceCount = practiceCount
        self.reservedCount = reservedCount
        self.removedDuplicates = removedDuplicates
        self.turnCount = turnCount
        self.savedAdapterPath = nil
    }

    public static func defaultDirectory(
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        homeDirectory.appending(path: ".basalt/selfedit")
    }

    public static func defaultTranscriptURL(
        sessionId: String,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        let safe = sessionId.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" ? $0 : "_" }
        return defaultDirectory(homeDirectory: homeDirectory).appending(path: "\(String(safe)).jsonl")
    }

    public static func loadTranscript(at url: URL) throws -> [EditSessionTranscriptEntry] {
        let text = try String(contentsOf: url, encoding: .utf8)
        let decoder = JSONDecoder()
        return try text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { try decoder.decode(EditSessionTranscriptEntry.self, from: Data($0.utf8)) }
    }
}
