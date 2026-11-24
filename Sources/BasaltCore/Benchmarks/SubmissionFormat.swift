import Foundation

public enum SubmissionTarget: String, Sendable, Codable, CaseIterable {
    case arcPrize = "arc-prize"
    case terminalBench = "terminal-bench"
    case sweBench = "swe-bench"
    case hfLeaderboard = "hf-leaderboard"
    case liveCodeBench = "livecodebench"
    case bfcl = "bfcl"
    case inspectAI = "inspect-ai"
    case tauBench = "tau-bench"
    case gaia = "gaia"
    case weaveOTel = "weave-otel"
    case mtLeaderboard = "mt-leaderboard"
}

public struct SubmissionFile: Sendable, Equatable, Codable {
    public var relativePath: String
    public var data: Data
    public var contentType: String

    public init(relativePath: String, data: Data, contentType: String = "application/json") {
        (self.relativePath, self.data, self.contentType) = (relativePath, data, contentType)
    }
}

public struct SubmissionPlan: Sendable, Equatable, Codable {
    public var target: SubmissionTarget
    public var files: [SubmissionFile]
    public var cliCommand: [String]?
    public var httpEndpoint: String?
    public var requiredEnvVar: String?
    public var validationErrors: [String]

    public init(
        target: SubmissionTarget, files: [SubmissionFile] = [],
        cliCommand: [String]? = nil, httpEndpoint: String? = nil,
        requiredEnvVar: String? = nil, validationErrors: [String] = []
    ) {
        (self.target, self.files, self.cliCommand) = (target, files, cliCommand)
        (self.httpEndpoint, self.requiredEnvVar, self.validationErrors) = (httpEndpoint, requiredEnvVar, validationErrors)
    }

    public var isValid: Bool { validationErrors.isEmpty }
}

public enum ARCGridValidator {
    public static let sizeRange = 1 ... 30
    public static let cellRange = 0 ... 9

    public static func validate(_ grid: [[Int]], label: String) -> [String] {
        guard sizeRange.contains(grid.count) else {
            return ["\(label): grid height \(grid.count) must lie in 1...30"]
        }
        let width = grid[0].count
        guard sizeRange.contains(width) else {
            return ["\(label): grid width \(width) must lie in 1...30"]
        }
        var errors: [String] = []
        for (rowIndex, row) in grid.enumerated() {
            if row.count != width {
                errors.append("\(label): ragged row \(rowIndex) has width \(row.count), expected \(width)")
            }
            for (colIndex, cell) in row.enumerated() where !cellRange.contains(cell) {
                errors.append("\(label): cell [\(rowIndex),\(colIndex)] value \(cell) outside 0...9")
            }
        }
        return errors
    }
}

public enum SubmissionFormatter {
    public static func build(
        target: SubmissionTarget,
        artifact: BenchmarkRunArtifact,
        submit: BenchmarkSubmitSpec? = nil
    ) -> SubmissionPlan {
        var plan: SubmissionPlan = switch target {
        case .arcPrize: buildARCPrize(artifact, submit: submit)
        case .terminalBench: buildTerminalBench(artifact, submit: submit)
        case .sweBench: buildSWEBench(artifact, submit: submit)
        case .hfLeaderboard: buildHFLeaderboard(artifact, submit: submit)
        case .liveCodeBench: buildLiveCodeBench(artifact, submit: submit)
        case .bfcl: buildBFCL(artifact, submit: submit)
        case .inspectAI: buildInspectAI(artifact, submit: submit)
        case .tauBench: buildTauBench(artifact, submit: submit)
        case .gaia: buildGAIA(artifact, submit: submit)
        case .weaveOTel: buildWeaveOTel(artifact, submit: submit)
        case .mtLeaderboard: buildMTLeaderboard(artifact, submit: submit)
        }
        if let env = submit?.apiKeyEnv, let issue = BenchmarkSecretGuard.apiKeyEnvIssue(env, path: "submit.api_key_env") {
            plan.validationErrors.append(issue.message)
        }
        if artifact.samples.isEmpty {
            plan.validationErrors.append("\(target.rawValue): artifact has no evaluated samples")
        }
        let provenanceData = encodeCodable(artifact.provenance)
        plan.files.append(SubmissionFile(relativePath: "basalt_provenance.json", data: provenanceData))
        return plan
    }

    static func jsonData(_ object: Any) -> Data {
        guard JSONSerialization.isValidJSONObject(object) else { return Data() }
        return (try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])) ?? Data()
    }

    static func jsonLines(_ objects: [[String: Any]]) -> Data {
        let lines = objects.compactMap { obj -> String? in
            guard JSONSerialization.isValidJSONObject(obj),
                  let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes]) else { return nil }
            return String(decoding: data, as: UTF8.self)
        }
        return Data((lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")).utf8)
    }

    static func encodeCodable<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return (try? encoder.encode(value)) ?? Data()
    }

    static func safeFilename(_ raw: String) -> String {
        let mapped = raw.unicodeScalars.map { scalar -> Character in
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "-" || scalar == "_" || scalar == "." {
                return Character(scalar)
            }
            return "_"
        }
        let cleaned = String(mapped).trimmingCharacters(in: CharacterSet(charactersIn: "."))
        return cleaned.isEmpty ? "sample" : cleaned
    }

    static func normalizeGAIAAnswer(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let lines = trimmed.split(whereSeparator: \.isNewline)
        let last = lines.last.map(String.init) ?? trimmed
        if let range = last.range(of: "FINAL ANSWER:", options: .caseInsensitive) {
            return last[range.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return last.trimmingCharacters(in: CharacterSet(charactersIn: ". "))
    }
}
