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
}
