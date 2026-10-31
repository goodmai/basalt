import Foundation
import Testing

/// Every target the CLI links must stay headless; only BasaltApp may import a UI framework.
/// The targets are discovered from `Sources/`, so a new one is checked without editing this list.
@Suite("Module boundaries")
struct ModuleBoundaryTests {
    static let sources = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Tests/BasaltTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // repo root
        .appendingPathComponent("Sources")

    // Attributes with arguments (@_spi(X), @preconcurrency), kind-qualified imports
    // (import struct SwiftUI.Color) and the umbrella modules that re-export the frameworks.
    // (?!\w), not \b: Swift's default Unicode word boundary treats "SwiftUI.Color" as one word.
    // A computed property: Regex is not Sendable, so it cannot be a stored static.
    static var banned: Regex<(Substring, Substring?, Substring?, Substring?, Substring?, Substring)> { /^\s*(@\w+(\([^)]*\))?\s+)*import\s+((struct|class|enum|protocol|func|var|let|typealias)\s+)?(SwiftUI|SwiftUICore|AppKit|Cocoa)(?!\w)/
        .anchorsMatchLineEndings() }

    @Test("Every target except BasaltApp is free of SwiftUI, AppKit and Cocoa imports")
    func headless() throws {
        let targets = try FileManager.default.contentsOfDirectory(atPath: Self.sources.path)
            .filter { $0 != "BasaltApp" && !$0.hasPrefix(".") }
        for core in ["BasaltCore", "BasaltInference", "BasaltServer", "BasaltCLI"] {
            #expect(targets.contains(core), "\(core) not found under Sources/")
        }
        for target in targets {
            let dir = Self.sources.appendingPathComponent(target)
            let files = (FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)?
                .compactMap { $0 as? URL } ?? [])
                .filter { $0.pathExtension == "swift" }
            for file in files {
                let source = try String(contentsOf: file, encoding: .utf8)
                #expect(source.firstMatch(of: Self.banned) == nil, "\(target)/\(file.lastPathComponent) imports a UI framework")
            }
        }
    }

    @Test("The pattern catches every import form it claims to")
    func patternCoverage() {
        for line in ["import SwiftUI", "  import AppKit", "@preconcurrency import Cocoa",
                     "@_spi(Private) import SwiftUICore", "import struct SwiftUI.Color"] {
            #expect(line.firstMatch(of: Self.banned) != nil, "\(line)")
        }
        for line in ["import Foundation", "// import SwiftUI is not allowed here", "import SwiftUIHelpers"] {
            #expect(line.firstMatch(of: Self.banned) == nil, "\(line)")
        }
    }

    @Test("No source writes to a fixed path under /tmp (world-readable, shared by every user)")
    func noFixedTmpWrites() throws {
        let files = (FileManager.default.enumerator(at: Self.sources, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL } ?? []).filter { $0.pathExtension == "swift" }
        #expect(!files.isEmpty)
        for file in files {
            let source = try String(contentsOf: file, encoding: .utf8)
            // Any "/tmp literal: toFile:, URL(fileURLWithPath:), createFile(atPath:) and "/tmp" + name
            // all start with it. NSTemporaryDirectory() stays allowed (per-user on macOS).
            #expect(!source.contains("\"/tmp"), "\(file.lastPathComponent) references a fixed /tmp path")
        }
    }

    @Test("The built CLI binary does not dynamically link AppKit, SwiftUI or Cocoa (#65)")
    func binaryFreeOfUIFrameworks() throws {
        let repoRoot = Self.sources.deletingLastPathComponent()
        let releaseBin = repoRoot.appendingPathComponent(".build/out/Products/Release/basalt")
        let debugBin = repoRoot.appendingPathComponent(".build/out/Products/Debug/basalt")
        let targetBin = FileManager.default.fileExists(atPath: releaseBin.path) ? releaseBin : (FileManager.default.fileExists(atPath: debugBin.path) ? debugBin : nil)
        guard let bin = targetBin else { return }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/otool")
        process.arguments = ["-L", bin.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        #expect(!output.contains("AppKit.framework"), "basalt binary links AppKit.framework")
        #expect(!output.contains("SwiftUI.framework"), "basalt binary links SwiftUI.framework")
        #expect(!output.contains("Cocoa.framework"), "basalt binary links Cocoa.framework")
    }
}
