import Foundation
import Testing
@testable import BasaltCLI
@testable import BasaltCore

@Suite("Specialized translation models compose recipe (#721)")
struct TranslationComposeTests {
    private static func recipeURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "docs/examples/compose/translate.yaml")
    }

    @Test("translate.yaml parses cleanly for 16GB, 24GB, and 64GB Mac profiles with specialized MLX translation models")
    func translateComposeProfilesParse() throws {
        let url = Self.recipeURL()
        let yaml = try String(contentsOf: url, encoding: .utf8)

        let spec16 = try ComposeFile.parse(
            yaml,
            file: url.path,
            context: ComposeContext(memoryGB: 16, chip: "Apple M4").with(base: url.path)
        )
        #expect(spec16.models["gemma"]?.source == .local("mlx-community/translategemma-4b-it-4bit"))
        #expect(spec16.models["hymt"]?.source == .local("mlx-community/HY-MT1.5-1.8B-4bit"))
        #expect(spec16.pipelines["cjk"] != nil)
        #expect(spec16.pipelines["fast"] != nil)

        let spec24 = try ComposeFile.parse(
            yaml,
            file: url.path,
            context: ComposeContext(memoryGB: 24, chip: "Apple M4 Pro").with(base: url.path)
        )
        #expect(spec24.models["gemma"]?.source == .local("mlx-community/translategemma-12b-it-4bit"))
        #expect(spec24.models["hymt"]?.source == .local("mlx-community/HY-MT1.5-7B-4bit"))

        let spec64 = try ComposeFile.parse(
            yaml,
            file: url.path,
            context: ComposeContext(memoryGB: 64, chip: "Apple M4 Max").with(base: url.path)
        )
        #expect(spec64.models["gemma"]?.source == .local("mlx-community/translategemma-27b-it-4bit"))
        #expect(spec64.models["hymt"]?.source == .local("mlx-community/HY-MT1.5-7B-8bit"))
    }
}
