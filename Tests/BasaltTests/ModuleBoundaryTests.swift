import Foundation
import Testing

/// Core and inference targets must stay headless: no SwiftUI or AppKit imports.
@Suite("Module boundaries")
struct ModuleBoundaryTests {
    @Test("BasaltCore and BasaltInference do not import SwiftUI or AppKit", arguments: ["BasaltCore", "BasaltInference"])
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
        let banned = /^\s*(@\w+\s+)*import\s+(SwiftUI|AppKit)\b/.anchorsMatchLineEndings()
        for file in files {
            let source = try String(contentsOf: file, encoding: .utf8)
            #expect(source.firstMatch(of: banned) == nil, "\(file.lastPathComponent) imports a UI framework")
        }
    }
}
