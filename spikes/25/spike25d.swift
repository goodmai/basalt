import Foundation

let scratch = "/private/tmp/claude-501/-Users-alexeiboklag-projects-CAEF/28d39ee8-3b2d-4951-9179-a7c75938e2ef/scratchpad"
let repo = "/Users/alexeiboklag/projects/mlx-25"
let hub = NSHomeDirectory() + "/.cache/huggingface/hub"
let lockDir = "/private/tmp/claude-501/basalt-gpu.lock"
let ownerFile = lockDir + "/owner"
let pid = ProcessInfo.processInfo.processIdentifier
let ownerLine = "owner=#25 spike25d pid \(pid)"
let logPath = scratch + "/spike25d-run.log"
let maxLoad = 8.0
let quietWait: TimeInterval = 1800
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

func hostLoad() -> (load: Double, ciActive: Bool, text: String) {
    let raw = run("/usr/sbin/sysctl", ["-n", "vm.loadavg"]).output
    let load = raw.split(separator: " ").compactMap { Double($0) }.first ?? .infinity
    let ci = run("/usr/bin/pgrep", ["-f", "Runner.Worker"]).status == 0
    return (load, ci, "load=\(raw.trimmingCharacters(in: .whitespacesAndNewlines)) ci=\(ci)")
}

func acquireLock() {
    while true {
        if FileManager.default.fileExists(atPath: quietFile) {
            Thread.sleep(forTimeInterval: 15)
            continue
        }
        if (try? FileManager.default.createDirectory(atPath: lockDir, withIntermediateDirectories: false)) != nil {
            FileManager.default.createFile(atPath: ownerFile, contents: (ownerLine + " " + Date().description + "\n").data(using: .utf8))
            log("lock acquired \(ownerLine)")
            return
        }
        Thread.sleep(forTimeInterval: 15)
    }
}

func releaseLock() {
    let owner = (try? String(contentsOfFile: ownerFile, encoding: .utf8)) ?? ""
    guard owner.hasPrefix(ownerLine + " ") else {
        log("error lock not ours on release, left untouched owner=\(owner.trimmingCharacters(in: .whitespacesAndNewlines))")
        failures += 1
        return
    }
    try? FileManager.default.removeItem(atPath: ownerFile)
    try? FileManager.default.removeItem(atPath: lockDir)
    log("lock released")
}

let quietFile = "/private/tmp/claude-501/basalt-quiet"
let quietLine = "\(pid) #25"

func ciRunsInProgress() -> Int {
    let result = run("/opt/homebrew/bin/gh", ["run", "list", "-R", "goodmai/basalt", "--status", "in_progress", "--json", "databaseId", "-q", "length"])
    return result.status == 0 ? Int(result.output.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 1 : 1
}

func waitForQuiet() -> Bool {
    let deadline = Date().addingTimeInterval(quietWait)
    while Date() < deadline {
        let host = hostLoad()
        let ci = ciRunsInProgress()
        if host.load <= maxLoad && !host.ciActive && ci == 0 { return true }
        log("waiting for quiet host \(host.text) gh_in_progress=\(ci)")
        Thread.sleep(forTimeInterval: 60)
    }
    return false
}

func raiseQuietFlag() {
    FileManager.default.createFile(atPath: quietFile, contents: (quietLine + "\n").data(using: .utf8))
    log("quiet flag raised \(quietLine)")
}

func lowerQuietFlag() {
    let owner = (try? String(contentsOfFile: quietFile, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard owner == quietLine else {
        log("error quiet flag not ours on removal, left untouched owner=\(owner)")
        failures += 1
        return
    }
    try? FileManager.default.removeItem(atPath: quietFile)
    log("quiet flag removed")
}

func snapshot(of repoID: String) -> String? {
    let dir = hub + "/models--" + repoID.replacingOccurrences(of: "/", with: "--")
    let revision = (try? String(contentsOfFile: dir + "/refs/main", encoding: .utf8))
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) } ?? "main"
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
    return revision
}

func testStep(_ label: String) {
    log("begin tests \(label)")
    let build = run("/usr/bin/swift", ["build", "--build-tests"], cwd: repo)
    guard build.status == 0 else {
        failures += 1
        log("error build \(label) tail=\(build.output.split(separator: "\n").suffix(5).joined(separator: " | "))")
        return
    }
    acquireLock()
    let result = run("/usr/bin/swift", ["test", "--skip-build", "--filter", "Spike25Tests"], cwd: repo)
    releaseLock()
    let lines = result.output.split(separator: "\n").filter { $0.hasPrefix("✔ Test") || $0.hasPrefix("✘ Test") }
    for line in lines { log("tests \(label) \(line.prefix(220))") }
    log("end tests \(label) status=\(result.status)")
}

let commit = run("/usr/bin/git", ["-C", repo, "rev-parse", "HEAD"]).output.trimmingCharacters(in: .whitespacesAndNewlines)
let chip = run("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"]).output.trimmingCharacters(in: .whitespacesAndNewlines)
let memory = run("/usr/sbin/sysctl", ["-n", "hw.memsize"]).output.trimmingCharacters(in: .whitespacesAndNewlines)
log("start commit=\(commit) machine=\(chip) mem=\(memory) os=\(ProcessInfo.processInfo.operatingSystemVersionString)")

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
for (prompt, temperature) in [("code", "0"), ("code", "0.7"), ("prose", "0.7")] {
    for lookup in [false, true] {
        cases.append(Case(model: llama, binary: "bench-A", prompt: prompt, temperature: temperature, lookup: lookup))
    }
}
for model in [qwen, llama] {
    for binary in ["bench-K4", "bench-A", "bench-K16"] {
        cases.append(Case(model: model, binary: binary, prompt: "code", temperature: "0", lookup: true))
    }
    cases.append(Case(model: model, binary: "bench-K16", prompt: "prose", temperature: "0.7", lookup: true))
}

var revisions: [String: String] = [:]
for model in Set(cases.map(\.model)) {
    guard let revision = snapshot(of: model) else {
        failures += 1
        log("error snapshot incomplete or missing model=\(model)")
        continue
    }
    revisions[model] = revision
    log("snapshot ok model=\(model) revision=\(revision)")
}

for item in cases {
    guard let revision = revisions[item.model] else { continue }
    let prompt = try! String(contentsOfFile: scratch + "/prompt-\(item.prompt).txt", encoding: .utf8)
    var arguments = ["-m", item.model, "-i", "3", "-t", "300", "--temperature", item.temperature, "--seed", "1", "-p", prompt]
    if item.model == qwen { arguments += ["--reasoning-effort", "none"] }
    if item.lookup { arguments.append("--speculative-lookup") }
    let label = "model=\(item.model) rev=\(revision.prefix(12)) bin=\(item.binary) prompt=\(item.prompt) T=\(item.temperature) lookup=\(item.lookup)"
    let quiet = waitForQuiet()
    acquireLock()
    raiseQuietFlag()
    let before = hostLoad()
    log("begin \(label) \(before.text)")
    let result = run(scratch + "/" + item.binary, arguments, cwd: repo + "/.build/release")
    let after = hostLoad()
    lowerQuietFlag()
    releaseLock()
    let lines = result.output.split(separator: "\n").map(String.init)
    let tps = lines.first { $0.contains("TPS") && $0.contains("avg=") }?.trimmingCharacters(in: .whitespaces) ?? "TPS missing"
    let drafts = lines.first { $0.contains("Drafts") }?.trimmingCharacters(in: .whitespaces) ?? ""
    let sigmaOK: Bool = {
        guard let avg = tps.range(of: "avg="), let sig = tps.range(of: "σ=") else { return false }
        let mean = Double(tps[avg.upperBound...].prefix { $0 != " " }) ?? 0
        let sigma = Double(tps[sig.upperBound...].prefix { $0 != " " }) ?? .infinity
        return mean > 0 && sigma / mean <= 0.03
    }()
    let valid = sigmaOK && quiet && before.load <= maxLoad && after.load <= maxLoad && !before.ciActive && !after.ciActive
    if result.status != 0 || !tps.contains("avg=") {
        failures += 1
        log("error \(label) status=\(result.status) tail=\(lines.suffix(5).joined(separator: " | "))")
    } else {
        log("result speed=\(valid ? "valid" : "INVALID") \(label) \(tps) \(drafts) after:\(after.text)")
    }
}

log("done failures=\(failures)")
exit(failures == 0 ? 0 : 1)
