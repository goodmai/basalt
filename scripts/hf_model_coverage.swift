#!/usr/bin/env swift
import Foundation

struct Repo: Decodable {
    struct Config: Decodable { let model_type: String? }
    let id: String
    let downloads: Int?
    let pipeline_tag: String?
    let config: Config?
}

let arguments = CommandLine.arguments.dropFirst()
func option(_ name: String) -> String? {
    arguments.firstIndex(of: name).flatMap { arguments.index(after: $0) < arguments.endIndex ? arguments[arguments.index(after: $0)] : nil }
}
let author = option("--author") ?? "mlx-community"
let basalt = option("--basalt") ?? ".build/release/basalt"
let pipelines: Set<String> = ["text-generation", "image-text-to-text"]

func nextPage(_ response: URLResponse) -> URL? {
    guard let link = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Link"),
          let match = link.firstMatch(of: #/<([^>]+)>;\s*rel="next"/#) else { return nil }
    return URL(string: String(match.1))
}

var repos: [Repo] = []
var page = URL(string: "https://huggingface.co/api/models?author=\(author)&limit=1000&expand[]=config&expand[]=downloads&expand[]=pipeline_tag")
while let url = page {
    var request = URLRequest(url: url)
    request.setValue("basalt-model-coverage", forHTTPHeaderField: "User-Agent")
    let (data, response) = try await URLSession.shared.data(for: request)
    repos += try JSONDecoder().decode([Repo].self, from: data)
    page = nextPage(response)
}

let generative = repos.filter { $0.pipeline_tag.map(pipelines.contains) == true && $0.config?.model_type != nil }
let types = Array(Set(generative.compactMap { $0.config?.model_type })).sorted()

let process = Process()
process.executableURL = URL(fileURLWithPath: basalt)
process.arguments = ["models", "types"]
let input = Pipe(), output = Pipe()
process.standardInput = input
process.standardOutput = output
try process.run()
input.fileHandleForWriting.write(Data(types.joined(separator: "\n").utf8))
try input.fileHandleForWriting.close()
let printed = output.fileHandleForReading.readDataToEndOfFile()
process.waitUntilExit()
guard process.terminationStatus == 0 else { print("\(basalt) models types failed"); exit(1) }
let loadable = Set(String(decoding: printed, as: UTF8.self).split(separator: "\n").map(String.init))

let supported = generative.filter { loadable.contains($0.config?.model_type ?? "") }
let downloads = generative.reduce(0) { $0 + ($1.downloads ?? 0) }
let supportedDownloads = supported.reduce(0) { $0 + ($1.downloads ?? 0) }
let percent = Int((Double(supported.count) / Double(max(generative.count, 1)) * 100).rounded(.down))
let weighted = Double(supportedDownloads) / Double(max(downloads, 1)) * 100

print("\(author): \(repos.count) repos, \(generative.count) text-generation / image-text-to-text with a model_type, \(types.count) distinct types")
print("loadable by basalt: \(supported.count) repos (\(percent)%), \(loadable.count) of \(types.count) types, \(String(format: "%.1f", weighted))% of downloads")
let missing = Dictionary(grouping: generative.filter { !loadable.contains($0.config?.model_type ?? "") }, by: { $0.config?.model_type ?? "" })
    .map { ($0.key, $0.value.count) }.sorted { $0.1 > $1.1 }.prefix(15)
print("largest unsupported types: " + missing.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
print("badge: https://img.shields.io/badge/\(author.replacingOccurrences(of: "-", with: "--"))%20LLM%2FVLM-\(percent)%25%20loadable-0A84FF.svg")
