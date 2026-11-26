#!/usr/bin/env swift
import Foundation

let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("check_quality_metrics.swift").path

func runChecker(_ args: [String]) -> (status: Int32, output: String) {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    p.arguments = ["swift", script] + args
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = pipe
    try! p.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    return (p.terminationStatus, String(data: data, encoding: .utf8) ?? "")
}

func writeFile(_ root: URL, _ rel: String, _ content: String) {
    let url = root.appendingPathComponent(rel)
    try! FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try! content.write(to: url, atomically: true, encoding: .utf8)
}

func makeCleanFixture() -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qm-test-\(UUID().uuidString)")
    writeFile(dir, "Sources/Core/A.swift", """
    import Foundation
    let s = "fatalError() try! as!"
    let raw = #"{"msg": "fatalError() try! as!"}"#
    // try! as! fatalError()
    /* fatalError("comment") */
    func ok() -> Int { 42 }
    """)
    writeFile(dir, "Tests/CoreTests/ATests.swift", """
    import Testing
    @Suite("A")
    struct ATests {
        @Test("ok")
        func ok() {}
    }
    """)
    writeFile(dir, "scripts/concurrency-hatches.allow", "")
    writeFile(dir, "Localization/en.lproj/Localizable.strings", "\"Hello\" = \"Hello\";\n\"Bye\" = \"Bye\";\n")
    writeFile(dir, "Localization/ru.lproj/Localizable.strings", "\"Hello\" = \"Привет\";\n\"Bye\" = \"Пока\";\n")
    writeFile(dir, ".github/workflows/ci.yml", """
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
      - uses: ./.github/actions/local
    """)
    writeFile(dir, "Package.resolved", "{\"pins\":[{\"identity\":\"a\"},{\"identity\":\"b\"}]}")
    writeFile(dir, "Tests/BasaltTests/Fixtures/golden/m1.json", "{}")
    writeFile(dir, "Tests/BasaltTests/Fixtures/ReferenceLogits/qwen3/case.json", "{}")
    return dir
}

func assertCheck(_ cond: Bool, _ msg: String) {
    if !cond {
        FileHandle.standardError.write(Data("TEST FAILED: \(msg)\n".utf8))
        exit(1)
    }
}

let clean = makeCleanFixture()
defer { try? FileManager.default.removeItem(at: clean) }
let badges = clean.appendingPathComponent("out-badges")
let r1 = runChecker([clean.path, "--osv-ok", "--max-crash-traps", "0", "--write-badges", badges.path])
assertCheck(r1.status == 0, "clean fixture should pass (ignores comments/strings/raw strings): \(r1.output)")
for file in ["concurrency-hatches.json", "swift-testing.json", "file-size.json", "crash-traps.json", "l10n-parity.json", "supply-chain.json", "golden-logits.json", "metrics.json"] {
    let url = badges.appendingPathComponent(file)
    assertCheck(FileManager.default.fileExists(atPath: url.path), "missing badge file \(file)")
    let obj = try! JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
    assertCheck(obj != nil, "invalid json in \(file)")
}
let sc = try! JSONSerialization.jsonObject(with: Data(contentsOf: badges.appendingPathComponent("supply-chain.json"))) as! [String: Any]
assertCheck((sc["message"] as? String) == "1/1 SHA · 0 CVEs (2 deps)", "unexpected supply-chain badge: \(sc)")

let badTrap = makeCleanFixture()
defer { try? FileManager.default.removeItem(at: badTrap) }
writeFile(badTrap, "Sources/Core/B.swift", "let x = try! JSONDecoder().decode(Int.self, from: Data())\n")
let r2 = runChecker([badTrap.path, "--max-crash-traps", "0"])
assertCheck(r2.status != 0 && r2.output.contains("Crash traps"), "real try! above ceiling should fail: \(r2.output)")

let badL10n = makeCleanFixture()
defer { try? FileManager.default.removeItem(at: badL10n) }
writeFile(badL10n, "Localization/ru.lproj/Localizable.strings", "\"Hello\" = \"Привет\";\n\"Hello\" = \"Дубль\";\n")
let r3 = runChecker([badL10n.path])
assertCheck(r3.status != 0 && r3.output.contains("duplicate") && r3.output.contains("missing"), "l10n drift/dup should fail: \(r3.output)")

let badAction = makeCleanFixture()
defer { try? FileManager.default.removeItem(at: badAction) }
writeFile(badAction, ".github/workflows/ci.yml", "steps:\n  - uses: actions/checkout@v4\n")
let r4 = runChecker([badAction.path])
assertCheck(r4.status != 0 && r4.output.contains("Unpinned"), "unpinned action should fail: \(r4.output)")

let badTest = makeCleanFixture()
defer { try? FileManager.default.removeItem(at: badTest) }
writeFile(badTest, "Tests/CoreTests/LegacyTests.swift", "class Legacy: XCTestCase {}\nfunc f() { MLXRandom.seed(1) }\n")
let r5 = runChecker([badTest.path])
assertCheck(r5.status != 0 && r5.output.contains("XCTestCase") && r5.output.contains("MLXRandom.seed"), "legacy XCTest & global seed should fail: \(r5.output)")

let boundary = makeCleanFixture()
defer { try? FileManager.default.removeItem(at: boundary) }
writeFile(boundary, "Sources/Core/Exact600.swift", String(repeating: "let _ = 1\n", count: 600))
let r6a = runChecker([boundary.path, "--max-crash-traps", "0"])
assertCheck(r6a.status == 0, "600-line file with trailing newline must pass (matches check_size_limits.swift): \(r6a.output)")
writeFile(boundary, "Sources/Core/Over600.swift", String(repeating: "let _ = 1\n", count: 601))
let r6b = runChecker([boundary.path, "--max-crash-traps", "0"])
assertCheck(r6b.status != 0 && r6b.output.contains("Over600.swift (601)"), "601-line file must fail: \(r6b.output)")

print("check_quality_metrics_test: all 6 cases passed.")
