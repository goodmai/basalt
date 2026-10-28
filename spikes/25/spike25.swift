import Foundation

let scratch = "/private/tmp/claude-501/-Users-alexeiboklag-projects-CAEF/28d39ee8-3b2d-4951-9179-a7c75938e2ef/scratchpad"
let repo = "/Users/alexeiboklag/projects/mlx-25"
let hub = NSHomeDirectory() + "/.cache/huggingface/hub"
let logPath = scratch + "/spike25-run.log"
FileManager.default.createFile(atPath: logPath, contents: nil)
let logHandle = FileHandle(forWritingAtPath: logPath)!
var failures = 0

func log(_ line: String) {
    let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(line)"
    print(stamped)
    fflush(stdout)
    logHandle.write((stamped + "\n").data(using: .utf8)!)
}

func run(_ executable: String, _ arguments: [String], cwd: String? = nil) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    if let cwd { process.currentDirectoryURL = URL(fileURLWithPath: cwd) }
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    do { try process.run() } catch { return (-1, "\(error)") }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

func snapshot(of repoID: String) -> (path: String, revision: String)? {
    let dir = hub + "/models--" + repoID.replacingOccurrences(of: "/", with: "--")
    guard let revision = try? String(contentsOfFile: dir + "/refs/main", encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines) else { return nil }
    let path = dir + "/snapshots/" + revision
    let blobs = (try? FileManager.default.contentsOfDirectory(atPath: dir + "/blobs")) ?? []
    guard !blobs.contains(where: { $0.hasSuffix(".incomplete") }) else { return nil }
    var shards = ["model.safetensors"]
    if let data = FileManager.default.contents(atPath: path + "/model.safetensors.index.json"),
       let index = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let map = index["weight_map"] as? [String: String] {
        shards = Array(Set(map.values))
    }
    for name in shards + ["config.json", "tokenizer.json"]
    where !FileManager.default.fileExists(atPath: path + "/" + name) {
        return nil
    }
    return (path, revision)
}

let commit = run("/usr/bin/git", ["-C", repo, "rev-parse", "HEAD"]).output.trimmingCharacters(in: .whitespacesAndNewlines)
let chip = run("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"]).output.trimmingCharacters(in: .whitespacesAndNewlines)
let memory = run("/usr/sbin/sysctl", ["-n", "hw.memsize"]).output.trimmingCharacters(in: .whitespacesAndNewlines)
let os = ProcessInfo.processInfo.operatingSystemVersionString
log("start commit=\(commit) machine=\(chip) mem=\(memory) os=\(os)")

struct Case {
    let model: String
    let binary: String
    let prompt: String
    let temperature: String
    let lookup: Bool
}

let qwen = "mlx-community/Qwen3-4B-4bit"
let llama = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
var cases: [Case] = []
for model in [qwen, llama] {
    for prompt in ["code", "prose"] {
        for temperature in ["0", "0.7"] {
            for lookup in [false, true] {
                cases.append(Case(model: model, binary: "bench-A", prompt: prompt, temperature: temperature, lookup: lookup))
            }
        }
    }
}
for binary in ["bench-K4", "bench-K16"] {
    for model in [qwen, llama] {
        cases.append(Case(model: model, binary: binary, prompt: "code", temperature: "0", lookup: true))
    }
}
cases.append(Case(model: qwen, binary: "bench-B", prompt: "code", temperature: "0.7", lookup: true))

var revisions: [String: String] = [:]
for model in Set(cases.map(\.model)) {
    guard let found = snapshot(of: model) else {
        log("error snapshot incomplete or missing model=\(model)")
        failures += 1
        continue
    }
    revisions[model] = found.revision
    log("snapshot ok model=\(model) revision=\(found.revision)")
}

for item in cases {
    guard let revision = revisions[item.model] else { continue }
    guard let prompt = try? String(contentsOfFile: scratch + "/prompt-\(item.prompt).txt", encoding: .utf8) else {
        log("error prompt missing name=\(item.prompt)")
        failures += 1
        continue
    }
    var arguments = ["-m", item.model, "-i", "3", "-t", "300", "--temperature", item.temperature, "--seed", "1", "-p", prompt]
    if item.model == qwen { arguments += ["--reasoning-effort", "none"] }
    if item.lookup { arguments.append("--speculative-lookup") }
    let label = "model=\(item.model) rev=\(revision.prefix(12)) bin=\(item.binary) prompt=\(item.prompt) T=\(item.temperature) lookup=\(item.lookup)"
    log("begin \(label)")
    let result = run(scratch + "/" + item.binary, arguments, cwd: repo + "/.build/release")
    let lines = result.output.split(separator: "\n").map(String.init)
    let tps = lines.first { $0.contains("TPS") && $0.contains("avg=") }?.trimmingCharacters(in: .whitespaces) ?? "TPS missing"
    let drafts = lines.first { $0.contains("Drafts") }?.trimmingCharacters(in: .whitespaces) ?? ""
    if result.status != 0 || !tps.contains("avg=") {
        failures += 1
        log("error \(label) status=\(result.status) tail=\(lines.suffix(5).joined(separator: " | "))")
    } else {
        log("result \(label) \(tps) \(drafts)")
    }
}

log("done failures=\(failures)")
exit(failures == 0 ? 0 : 1)
