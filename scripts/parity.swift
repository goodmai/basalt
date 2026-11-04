#!/usr/bin/env swift
import CryptoKit
import Foundation
import Synchronization

let usage = """
    Usage:
      swift scripts/parity.swift --binary <PerformanceBenchmark>
      swift scripts/parity.swift --binary <PerformanceBenchmark> --record
      swift scripts/parity.swift --binary <PerformanceBenchmark> --against <PerformanceBenchmark>
      swift scripts/parity.swift --self-test

    Greedy 64-token output of every model and mode below on scripts/fixtures/parity-document.txt,
    compared by the md5 and line count of the --- OUTPUT --- block. An empty block fails.

      (default)   compare --binary with scripts/parity-goldens.json
      --record    run --binary and write its blocks into scripts/parity-goldens.json
      --against   compare --binary with another PerformanceBenchmark, no goldens
      --self-test check block extraction and the comparison without loading a model

    The Basalt binary next to --binary checks each snapshot with `models inspect --json` first.
    """

struct Step {
    let model: String
    let mode: String
    let flags: [String]
}

let qwen = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
let gemma = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
let lfm = "LiquidAI/LFM2.5-24B-A2B-MLX-4bit"
let bonsai = "Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
let steps = [
    Step(model: qwen, mode: "plain", flags: []),
    Step(model: qwen, mode: "lookup", flags: ["--speculative-lookup"]),
    Step(model: qwen, mode: "zones", flags: ["--kv-bits", "8", "--kv-zones"]),
    Step(model: qwen, mode: "split", flags: ["--kv-head-split", "0.5"]),
    Step(model: gemma, mode: "plain", flags: []),
    Step(model: gemma, mode: "quest", flags: ["--quest-top-k", "16"]),
    Step(model: bonsai, mode: "plain", flags: []),
    Step(model: bonsai, mode: "mtp", flags: ["--mtp"]),
]

struct Block: Equatable {
    let md5: String
    let lines: Int
    let empty: Bool
}

let emptyBlockMD5 = "df2463dc07f1bf60646a990cfa9b5afe"

struct Golden: Codable, Equatable {
    var model: String
    var mode: String
    var md5: String
    var lines: Int
    var revision: String
    var commit: String

    var block: Block { Block(md5: md5, lines: lines, empty: md5 == emptyBlockMD5) }
}

func md5(_ text: String) -> String {
    Insecure.MD5.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
}

func outputBlock(_ text: String) -> Block? {
    let lines = text.components(separatedBy: "\n")
    guard let start = lines.firstIndex(where: { $0.contains("--- OUTPUT ---") }),
          let end = lines[start...].firstIndex(where: { $0.contains("--- END OUTPUT ---") })
    else { return nil }
    let block = lines[start ... end]
    let empty = block.dropFirst().dropLast().allSatisfy { $0.allSatisfy(\.isWhitespace) }
    return Block(md5: md5(block.map { $0 + "\n" }.joined()), lines: block.count, empty: empty)
}

func mismatch(_ got: Block?, expected: Block?) -> String? {
    guard let got else { return "no output block" }
    guard !got.empty else { return "empty output block" }
    guard let expected else { return "nothing to compare with" }
    guard !expected.empty else { return "the expected block is empty" }
    guard got != expected else { return nil }
    return "md5 \(got.md5) lines \(got.lines), expected \(expected.md5) lines \(expected.lines)"
}

func merged(_ goldens: [Golden], with recorded: [Golden]) -> [Golden] {
    var result = goldens
    for golden in recorded {
        if let index = result.firstIndex(where: { $0.model == golden.model && $0.mode == golden.mode }) {
            result[index] = golden
        } else {
            result.append(golden)
        }
    }
    return result
}

func prompt(document: String) -> String {
    var document = document
    while document.hasSuffix("\n") { document.removeLast() }
    return "Document: \(document) Summarize the text above."
}

func revision(snapshotPath: String) -> String {
    let name = URL(fileURLWithPath: snapshotPath).lastPathComponent
    return name.count == 40 && name.allSatisfy(\.isHexDigit) ? name : "unknown"
}

func selfTest(fixture: String) {
    let wrapped = "log line\n--- OUTPUT ---\nHello\r\n  world\n--- END OUTPUT ---\n  Memory (active): 1 MB\n"
    precondition(outputBlock(wrapped) == Block(md5: "6895ec5a55227850998bfeb166c68458", lines: 4, empty: false))
    precondition(outputBlock("--- OUTPUT ---\n\n--- END OUTPUT ---\n") == Block(md5: emptyBlockMD5, lines: 3, empty: true))
    precondition(outputBlock("--- OUTPUT ---\n \t\n--- END OUTPUT ---") == Block(md5: "da4ef629c31d92b651f1887f46b2ee0b", lines: 3, empty: true))
    precondition(outputBlock("no markers\n") == nil)
    precondition(outputBlock("--- OUTPUT ---\ncut off before the end marker\n") == nil)

    let a = Block(md5: "a", lines: 6, empty: false)
    let empty = Block(md5: emptyBlockMD5, lines: 3, empty: true)
    precondition(mismatch(a, expected: a) == nil)
    precondition(mismatch(a, expected: Block(md5: "b", lines: 6, empty: false)) == "md5 a lines 6, expected b lines 6")
    precondition(mismatch(a, expected: Block(md5: "a", lines: 7, empty: false)) != nil)
    precondition(mismatch(nil, expected: a) == "no output block")
    precondition(mismatch(empty, expected: empty) == "empty output block")
    precondition(mismatch(a, expected: nil) == "nothing to compare with")
    let lfmSeed = Golden(model: lfm, mode: "plain", md5: emptyBlockMD5, lines: 3, revision: "unknown", commit: "5738fe0")
    precondition(mismatch(a, expected: lfmSeed.block) == "the expected block is empty")

    let old = [Golden(model: "m", mode: "plain", md5: "a", lines: 6, revision: "r", commit: "c"), lfmSeed]
    let new = Golden(model: "m", mode: "plain", md5: "b", lines: 7, revision: "r2", commit: "d")
    let extra = Golden(model: "m", mode: "mtp", md5: "c", lines: 5, revision: "r2", commit: "d")
    precondition(merged(old, with: [new, extra]) == [new, lfmSeed, extra])

    precondition(md5(prompt(document: fixture)) == "3153246d6f3f31ada79ea17a174afc5c")
    precondition(prompt(document: "x\n\n") == "Document: x Summarize the text above.")
    precondition(revision(snapshotPath: "/s/dc755c911a7716a1db15cd8df8bab46a9dcfffdc") == "dc755c911a7716a1db15cd8df8bab46a9dcfffdc")
    precondition(revision(snapshotPath: "/s/main") == "unknown")
    print("parity self-test passed")
}

let scripts = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.deletingLastPathComponent().path
let repo = URL(fileURLWithPath: scripts).deletingLastPathComponent().path
let goldensPath = scripts + "/parity-goldens.json"
guard let fixture = try? String(contentsOfFile: scripts + "/fixtures/parity-document.txt", encoding: .utf8) else {
    print("FAIL cannot read \(scripts)/fixtures/parity-document.txt")
    exit(1)
}

struct Options {
    var binary = ""
    var against: String?
    var record = false
}

func parse(_ arguments: [String]) -> Options? {
    var options = Options()
    var rest = arguments[...]
    while let flag = rest.popFirst() {
        switch flag {
        case "--record": options.record = true
        case "--binary": guard let value = rest.popFirst() else { return nil }; options.binary = value
        case "--against": guard let value = rest.popFirst() else { return nil }; options.against = value
        default: return nil
        }
    }
    guard !options.binary.isEmpty, !(options.record && options.against != nil) else { return nil }
    return options
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments == ["--self-test"] {
    selfTest(fixture: fixture)
    exit(0)
}
guard let options = parse(arguments) else {
    print(usage)
    exit(arguments.contains("-h") || arguments.contains("--help") ? 0 : 2)
}

let lockDirectory = "/private/tmp/basalt-gpu.lock"
let pid = String(ProcessInfo.processInfo.processIdentifier)
let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
let logPath = repo + "/logs/parity-\(stamp).log"
let outputDirectory = repo + "/logs/parity-\(stamp)"
try? FileManager.default.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)
FileManager.default.createFile(atPath: logPath, contents: nil)
guard let logHandle = FileHandle(forWritingAtPath: logPath) else {
    print("FAIL cannot open the log file \(logPath)")
    exit(1)
}

final class RunState: Sendable {
    let child = Mutex<Process?>(nil)
    let lockHeld = Mutex(false)
    let failures = Mutex(0)
}

let runState = RunState()

func log(_ line: String) {
    let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(line)"
    print(stamped)
    fflush(stdout)
    try? logHandle.write(contentsOf: Data((stamped + "\n").utf8))
}

func fail(_ step: String, _ detail: String) {
    runState.failures.withLock { $0 += 1 }
    log("FAIL \(step) \(detail)")
}

func trimmed(_ text: String) -> String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

func run(_ executable: String, _ arguments: [String], in directory: String = repo) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    process.currentDirectoryURL = URL(fileURLWithPath: directory)
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    do { try process.run() } catch { return (-1, "\(error)") }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

func lockOwner() -> String {
    trimmed((try? String(contentsOfFile: lockDirectory + "/owner", encoding: .utf8)) ?? "")
}

func releaseLock() {
    guard runState.lockHeld.withLock({ held in defer { held = false }; return held }) else { return }
    guard lockOwner() == pid else {
        fail("release lock", "held by pid \(lockOwner()), left untouched")
        return
    }
    unlink(lockDirectory + "/owner")
    rmdir(lockDirectory)
    log("lock released")
}

func acquireLock() {
    var waiting = false
    while mkdir(lockDirectory, 0o755) != 0 {
        let owner = lockOwner()
        if let holder = Int32(owner), kill(holder, 0) != 0, errno == ESRCH {
            unlink(lockDirectory + "/owner")
            rmdir(lockDirectory)
            log("reclaimed the GPU lock from pid \(holder), which no longer runs")
            continue
        }
        if !waiting { log("waiting for the GPU lock held by pid \(owner.isEmpty ? "unknown" : owner)") }
        waiting = true
        Thread.sleep(forTimeInterval: 15)
    }
    guard (try? pid.write(toFile: lockDirectory + "/owner", atomically: true, encoding: .utf8)) != nil else {
        rmdir(lockDirectory)
        fail("lock", "cannot write the owner file")
        exit(1)
    }
    runState.lockHeld.withLock { $0 = true }
    log("lock acquired by pid \(pid)")
}

signal(SIGPIPE, SIG_IGN)
let signalSources = [SIGINT, SIGTERM, SIGHUP].map { signalNumber in
    signal(signalNumber, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .global())
    source.setEventHandler {
        if let running = runState.child.withLock({ $0 }) {
            running.terminate()
            running.waitUntilExit()
        }
        log("FAIL interrupted by signal \(signalNumber)")
        releaseLock()
        exit(128 + signalNumber)
    }
    source.resume()
    return source
}

func inspect(_ model: String, basalt: String) -> String? {
    let result = run(basalt, ["models", "inspect", model, "--json"])
    guard result.status == 0, let start = result.output.firstIndex(of: "{"),
          let report = try? JSONSerialization.jsonObject(with: Data(result.output[start...].utf8)) as? [String: Any]
    else {
        fail("snapshot \(model)", "inspect exit=\(result.status) \(result.output.suffix(300))")
        return nil
    }
    guard report["intact"] as? Bool == true, let path = report["path"] as? String else {
        fail("snapshot \(model)", "not intact: \(report["errors"] ?? "no errors reported")")
        return nil
    }
    log("snapshot ok model=\(model) path=\(path)")
    return revision(snapshotPath: path)
}

func benchmark(_ step: Step, binary: String, label: String) -> Block? {
    let name = "\(step.model.replacingOccurrences(of: "/", with: "--"))-\(step.mode)-\(label)"
    let outPath = "\(outputDirectory)/\(name).out"
    let errPath = "\(outputDirectory)/\(name).err"
    FileManager.default.createFile(atPath: outPath, contents: nil)
    FileManager.default.createFile(atPath: errPath, contents: nil)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: binary)
    process.arguments = ["-m", step.model, "-i", "1", "-t", "64", "-p", prompt(document: fixture), "--temperature", "0", "--seed", "1",
                         "--reasoning-effort", "none", "--no-warmup", "--print-output"] + step.flags
    process.currentDirectoryURL = URL(fileURLWithPath: binary).deletingLastPathComponent()
    process.standardOutput = FileHandle(forWritingAtPath: outPath)
    process.standardError = FileHandle(forWritingAtPath: errPath)
    runState.child.withLock { $0 = process }
    defer { runState.child.withLock { $0 = nil } }
    do { try process.run() } catch {
        fail("\(step.model) \(step.mode) \(label)", "cannot start \(binary): \(error)")
        return nil
    }
    process.waitUntilExit()
    let output = String(decoding: FileManager.default.contents(atPath: outPath) ?? Data(), as: UTF8.self)
    guard process.terminationStatus == 0 else {
        let detail = ((try? String(contentsOfFile: errPath, encoding: .utf8)) ?? "").suffix(300)
        fail("\(step.model) \(step.mode) \(label)", "exit=\(process.terminationStatus) \(detail) output=\(outPath)")
        return nil
    }
    let block = outputBlock(output)
    log("step model=\(step.model) mode=\(step.mode) binary=\(label) flags=\(step.flags.joined(separator: " ")) "
        + "md5=\(block?.md5 ?? "none") lines=\(block?.lines ?? 0) empty=\(block?.empty ?? true) output=\(outPath)")
    return block
}

struct Row {
    let step: Step
    let revision: String
    let got: Block?
    let expected: Block?
    let result: String
}

func loadGoldens() -> [Golden] {
    guard let data = FileManager.default.contents(atPath: goldensPath) else { return [] }
    do { return try JSONDecoder().decode([Golden].self, from: data) } catch {
        fail("goldens", "cannot decode \(goldensPath): \(error)")
        exit(1)
    }
}

func compare(_ step: Step, revision: String, goldens: [Golden]) -> Row {
    let got = benchmark(step, binary: options.binary, label: "binary")
    if options.record {
        let result = got == nil ? "FAIL no output block" : got?.empty == true ? "FAIL empty output block" : "recorded"
        return Row(step: step, revision: revision, got: got, expected: nil, result: result)
    }
    let golden = goldens.first { $0.model == step.model && $0.mode == step.mode }
    let expected = options.against.map { benchmark(step, binary: $0, label: "against") } ?? golden?.block
    guard let reason = mismatch(got, expected: expected) else {
        return Row(step: step, revision: revision, got: got, expected: expected, result: "SAME")
    }
    let moved = options.against == nil && golden.map { $0.revision != revision && $0.revision != "unknown" && revision != "unknown" } == true
    let note = moved ? " (snapshot changed from \(golden?.revision ?? ""))" : ""
    return Row(step: step, revision: revision, got: got, expected: expected, result: "FAIL \(reason)\(note)")
}

func record(_ rows: [Row], into goldens: [Golden], commit: String) {
    let recorded = rows.compactMap { row -> Golden? in
        guard row.result == "recorded", let got = row.got else { return nil }
        return Golden(model: row.step.model, mode: row.step.mode, md5: got.md5, lines: got.lines, revision: row.revision, commit: commit)
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    do {
        try (encoder.encode(merged(goldens, with: recorded)) + Data("\n".utf8)).write(to: URL(fileURLWithPath: goldensPath))
        log("recorded \(recorded.count) golden(s) into \(goldensPath)")
    } catch {
        fail("goldens", "cannot write \(goldensPath): \(error)")
    }
}

for binary in [options.binary] + (options.against.map { [$0] } ?? []) where !FileManager.default.isExecutableFile(atPath: binary) {
    fail("binary", "not an executable: \(binary)")
    exit(1)
}
let basalt = URL(fileURLWithPath: options.binary).deletingLastPathComponent().appending(path: "Basalt").path
guard FileManager.default.isExecutableFile(atPath: basalt) else {
    fail("inspect", "no Basalt binary next to \(options.binary)")
    exit(1)
}

let commit = trimmed(run("/usr/bin/git", ["rev-parse", "--short", "HEAD"]).output)
let chip = trimmed(run("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"]).output)
let memory = (Int(trimmed(run("/usr/sbin/sysctl", ["-n", "hw.memsize"]).output)) ?? 0) >> 30
let models = steps.map(\.model).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
var revisions: [String: String] = [:]
for model in models { revisions[model] = inspect(model, basalt: basalt) }
let mode = options.record ? "record" : options.against.map { "against \($0)" } ?? "goldens \(goldensPath)"
log("parity commit=\(trimmed(run("/usr/bin/git", ["rev-parse", "HEAD"]).output)) chip=\(chip) memory=\(memory)GB binary=\(options.binary) "
    + "mode=\(mode) revisions=\(models.map { "\($0)@\(revisions[$0] ?? "not intact")" }.joined(separator: ",")) log=\(logPath)")

let goldens = loadGoldens()
var rows: [Row] = []
for model in models {
    guard let revision = revisions[model] else { continue }
    acquireLock()
    for step in steps where step.model == model {
        let row = compare(step, revision: revision, goldens: goldens)
        if row.result.hasPrefix("FAIL") { fail("\(step.model) \(step.mode)", String(row.result.dropFirst(5))) }
        rows.append(row)
    }
    releaseLock()
}
if options.record { record(rows, into: goldens, commit: commit) }

log("summary | model | mode | revision | lines | md5 | expected md5 | result |")
log("summary |---|---|---|---|---|---|---|")
for row in rows {
    let lines = row.got.map { String($0.lines) } ?? "-"
    log("summary | \(row.step.model) | \(row.step.mode) | \(row.revision) | \(lines) | \(row.got?.md5 ?? "-") | \(row.expected?.md5 ?? "-") | \(row.result) |")
}
for model in models where revisions[model] == nil {
    log("summary | \(model) | all | not intact | - | - | - | FAIL not measured |")
}
let failures = runState.failures.withLock { $0 }
log("done failures=\(failures)")
exit(failures == 0 ? 0 : 1)
