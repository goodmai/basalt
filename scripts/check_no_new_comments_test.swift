#!/usr/bin/env swift
import Foundation

let checker = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appending(path: "check_no_new_comments.swift").path

let before = "let kept = 1 // an old comment the PR does not touch\n"

let clean = #"""
let url = "https://example.com/a//b"
let s = "a // b /* c */"
let i = "\(f("//"))"
let m = """
    https://x.y // inside a multi-line string
    /* still a string */
    """
let raw = #"a "quoted" // raw"#
let raw2 = #"\"# + "x // y"
let deeper = ##"x "# // y"##
let closure = "\(xs.map { "\($0) // no" }.joined(separator: ")"))"
let r = #/x//y/#
let bare = /a\/b/
let cls = /[/]x/
let q = a / b
let product = a
    * b

"""#

let commented = [
    "let t = 1 // trailing",
    "// full line",
    "/// doc",
    "/*",
    "   body line without a star",
    "*/",
    "case .x:// after a colon",
    "let n = 1 /* inline */ + 2",
    "/* outer /* nested */ still comment */",
    "let q = a / b // after a division",
]

func shell(_ arguments: [String], in directory: URL) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = arguments
    process.currentDirectoryURL = directory
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try! process.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

func check(_ files: [String: String?], config: [(String, String)] = []) -> (status: Int32, output: String) {
    let directory = FileManager.default.temporaryDirectory.appending(path: "no-comments-\(UUID().uuidString)")
    try! FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    _ = shell(["git", "init", "-q", "-b", "main"], in: directory)
    for (key, value) in [("user.name", "t"), ("user.email", "t@t")] + config {
        _ = shell(["git", "config", key, value], in: directory)
    }
    try! before.write(to: directory.appending(path: "Old.swift"), atomically: true, encoding: .utf8)
    _ = shell(["git", "add", "-A"], in: directory)
    _ = shell(["git", "commit", "-qm", "base"], in: directory)
    _ = shell(["git", "checkout", "-qb", "pr"], in: directory)
    for (name, body) in files {
        let file = directory.appending(path: name)
        if let body { try! body.write(to: file, atomically: true, encoding: .utf8) } else { try! FileManager.default.removeItem(at: file) }
    }
    _ = shell(["git", "add", "-A"], in: directory)
    _ = shell(["git", "commit", "-qm", "pr"], in: directory)
    return shell(["swift", checker, "main"], in: directory)
}

var failures = 0
func expect(_ condition: Bool, _ name: String, _ output: String) {
    if condition { print("ok   \(name)") } else {
        failures += 1
        print("FAIL \(name)\n\(output)")
    }
}

var result = check(["Clean.swift": clean, "Package.swift": "// swift-tools-version: 6.0\nlet p = 1\n"])
expect(result.status == 0 && result.output.contains("No comments added."), "code that only looks like comments passes", result.output)

result = check(["Bad.swift": commented.joined(separator: "\n") + "\n"])
expect(result.status == 1 && (1 ... commented.count).allSatisfy { result.output.contains("Bad.swift:\($0):") }
    && result.output.contains("Bad.swift:1: let t = 1 // trailing")
    && result.output.contains("\n\n\(commented.count) added comment line(s). Move the explanation to the commit message, the PR or a test name (see CONTRIBUTING.md)."),
    "every comment form is reported", result.output)

result = check(["R.swift": "let r = /\"/\n// real comment\nlet s = \"x\"\n"])
expect(result.status == 1 && result.output.contains("R.swift:2:"), "a quote inside a bare regex cannot hide the next line", result.output)

result = check(["Crlf.swift": "let a = 1\r\n// c\r\n", "Other.swift": "// swift-tools-version: 6.0\n"])
expect(result.output.contains("Crlf.swift:2: // c\n") && result.output.contains("Other.swift:1:"), "crlf and directive only in Package.swift", result.output)

result = check(["Old.swift": "", "Renamed.swift": before])
expect(result.status == 1 && result.output.contains("Renamed.swift:1:"), "a renamed file counts as added", result.output)

result = check(["New.swift": "let x = 1\n"])
expect(result.status == 0, "untouched old comments are not reported", result.output)

result = check(["Ünï code.swift": "// hidden?\n"], config: [("diff.noprefix", "true"), ("color.diff", "always")])
expect(result.status == 1 && result.output.contains("Ünï code.swift:1:"), "non-ASCII path and noprefix config cannot hide a comment", result.output)

result = check(["Old.swift": nil, "Moved.swift": before])
expect(result.status == 1 && result.output.contains("Moved.swift:1:"), "a file moved with git mv counts as added", result.output)

result = check(["Old.swift": nil])
expect(result.status == 0, "a deleted file is skipped", result.output)

result = check(["Package.swift": "// swift-tools-version: 6.0\n// swift-tools-version: 6.0\n"])
expect(result.status == 1 && !result.output.contains("Package.swift:1:") && result.output.contains("Package.swift:2:"),
       "the directive is allowed on line 1 of Package.swift only", result.output)

result = check(["Continued.swift": "let m = \"\"\"\n    a \\\n    b\n    \"\"\"\n// real\nlet r = /a\\\n// two\n"])
expect(result.status == 1 && result.output.contains("Continued.swift:5: // real") && result.output.contains("Continued.swift:7: // two"),
       "a backslash before a newline keeps the line count", result.output)

result = check(["Interpolated.swift": "let a = #\"\\#(x) // no\"#\nlet b = \"\\(\"(\") // no\"\nlet c = 1 /*\n\n*/\n"])
expect(result.status == 1 && !result.output.contains("Interpolated.swift:1:") && !result.output.contains("Interpolated.swift:2:")
    && result.output.contains("Interpolated.swift:3:") && result.output.contains("Interpolated.swift:5:"),
    "raw interpolation and parentheses in strings stay code, a block comment spans lines", result.output)

exit(failures == 0 ? 0 : 1)
