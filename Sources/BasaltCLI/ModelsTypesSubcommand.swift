import ArgumentParser
import BasaltInference
import Foundation

struct TypesSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "types",
        abstract: "Print which of the given config.json model_type values this build loads (reads them one per line from standard input when none are given)"
    )

    @Argument(help: "model_type values to check, e.g. qwen3 gemma4 whisper")
    var modelTypes: [String] = []

    func run() async throws {
        let candidates = modelTypes.isEmpty ? Self.standardInputLines() : modelTypes
        for type in await MLXInferenceEngine.loadableModelTypes(among: candidates) {
            print(type)
        }
    }

    static func standardInputLines() -> [String] {
        var lines: [String] = []
        while let line = readLine() {
            let type = line.trimmingCharacters(in: .whitespaces)
            if !type.isEmpty { lines.append(type) }
        }
        return lines
    }
}
