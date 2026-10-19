import Foundation
import Testing

/// Every target the CLI links must stay headless; only BasaltApp may import a UI framework.
@Suite("Module boundaries")
struct ModuleBoundaryTests {
    @Test("Headless targets do not import SwiftUI, AppKit or Cocoa",
          arguments: ["BasaltCore", "BasaltInference", "BasaltCloud", "BasaltServer", "BasaltUI", "BasaltCLI", "BasaltBin"])
    func headless(target: String) throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Tests/BasaltTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("Sources/\(target)")
        let files = try #require(FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil))
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "swift" }
        #expect(!files.isEmpty, "no sources found under \(dir.path)")
        // Covers attributes (@preconcurrency), kind-qualified imports (import struct SwiftUI.Color)
        // and the umbrella modules that re-export the frameworks (SwiftUICore, Cocoa).
        let banned = /^\s*(@\w+\s+)*import\s+((struct|class|enum|protocol|func|var|let|typealias)\s+)?(SwiftUI|SwiftUICore|AppKit|Cocoa)\b/
            .anchorsMatchLineEndings()
        for file in files {
            let source = try String(contentsOf: file, encoding: .utf8)
            #expect(source.firstMatch(of: banned) == nil, "\(file.lastPathComponent) imports a UI framework")
        }
    }
}
