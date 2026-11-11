// swift-tools-version: 6.4
import PackageDescription

let swiftSettings: [SwiftSetting] = [.swiftLanguageMode(.v6)]

let package = Package(
    name: "Basalt",
    platforms: [
        .macOS(.v27)
    ],
    dependencies: [
        .package(url: "https://github.com/goodmai/mlx-swift",         revision: "6df64251d74bddfa3b958eb99cebd6f817a9f3ae"),
        .package(url: "https://github.com/goodmai/mlx-swift-lm",      revision: "79be1bac429a10648fa2382ea9dca54549e682da"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.28.0"),
        .package(url: "https://github.com/apple/swift-argument-parser",    from: "1.8.2"),
        .package(url: "https://github.com/swift-server/swift-service-lifecycle", from: "2.12.0"),
        .package(url: "https://github.com/apple/swift-http-types.git",     from: "1.4.0"),
        .package(url: "https://github.com/apple/swift-log.git",            from: "1.6.2"),
        .package(url: "https://github.com/apple/swift-nio.git",            from: "2.104.0"),
        // Provides the `Tokenizers` module used by MLXHuggingFace macros
        .package(url: "https://github.com/huggingface/swift-transformers",  from: "1.3.5"),
        .package(url: "https://github.com/huggingface/swift-jinja.git",     from: "3.1.0"),
        // Epic 16: CLI Enhancements
        .package(url: "https://github.com/onevcat/Rainbow.git", from: "4.2.2"),
        .package(url: "https://github.com/apple/swift-markdown.git", from: "0.9.0"),
        .package(url: "https://github.com/jpsim/Yams.git", exact: "6.2.2"),
    ],
    targets: [

        // ── Libraries (layered DAG: Core ← Inference/Cloud/Server/UI ← CLI) ──
        // Core: DTOs, errors, config, logger, profilers, calculators. No MLX, no UI.
        .target(
            name: "BasaltCore",
            dependencies: [],
            path: "Sources/BasaltCore",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltHarness",
            dependencies: ["BasaltCore"],
            path: "Sources/BasaltHarness",
            swiftSettings: swiftSettings
        ),
        // Inference: the MLX-backed engine.
        .target(
            name: "BasaltInference",
            dependencies: [
                "BasaltCore",
                .product(name: "MLX",           package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXLLM",        package: "mlx-swift-lm"),
                // Gemma 4 and other *ForConditionalGeneration checkpoints are only
                // registered in MLXVLM; without it ModelFactoryRegistry cannot load them.
                .product(name: "MLXVLM",        package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon",   package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXGuidedGeneration", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders",  package: "mlx-swift-lm"),
                .product(name: "MLXRerankers",  package: "mlx-swift-lm"),
                .product(name: "Tokenizers",    package: "swift-transformers"),
                .product(name: "Hub",           package: "swift-transformers"),
                .product(name: "Jinja",         package: "swift-jinja"),
            ],
            path: "Sources/BasaltInference",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltSpeech",
            dependencies: ["BasaltCore"],
            path: "Sources/BasaltSpeech",
            swiftSettings: swiftSettings
        ),
        // Cloud: OpenRouter client, router, cost tracking.
        .target(
            name: "BasaltCloud",
            dependencies: ["BasaltCore"],
            path: "Sources/BasaltCloud",
            swiftSettings: swiftSettings
        ),
        // Server: REST (Hummingbird) controllers, middleware, MCP server.
        .target(
            name: "BasaltServer",
            dependencies: [
                "BasaltCore", "BasaltHarness",
                .product(name: "ServiceLifecycle", package: "swift-service-lifecycle"),
                .product(name: "Hummingbird",   package: "hummingbird"),
                .product(name: "HTTPTypes",     package: "swift-http-types"),
                .product(name: "Logging",       package: "swift-log"),
                .product(name: "NIOFoundationEssentialsCompat", package: "swift-nio"),
            ],
            path: "Sources/BasaltServer",
            swiftSettings: swiftSettings
        ),
        // UI: terminal renderers. Headless — the SwiftUI client lives in BasaltApp.
        .target(
            name: "BasaltUI",
            dependencies: [
                "BasaltCore",
                .product(name: "Rainbow",       package: "Rainbow"),
                .product(name: "Markdown",      package: "swift-markdown"),
            ],
            path: "Sources/BasaltUI",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltAppCore",
            dependencies: ["BasaltCore"],
            path: "Sources/BasaltAppCore",
            swiftSettings: swiftSettings
        ),
        // App: the SwiftUI REST chat client. Nothing links it yet (its launch command is #55),
        // so SwiftUI/AppKit stay out of the CLI binary.
        .target(
            name: "BasaltApp",
            dependencies: [
                "BasaltCore", "BasaltAppCore",
                .product(name: "Markdown",      package: "swift-markdown"),
            ],
            path: "Sources/BasaltApp",
            swiftSettings: swiftSettings
        ),
        // CLI: ArgumentParser commands wiring everything together.
        .target(
            name: "BasaltCLI",
            dependencies: [
                "BasaltCore", "BasaltInference", "BasaltCloud", "BasaltServer", "BasaltUI", "BasaltHarness", "BasaltSpeech",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Yams", package: "Yams"),
            ],
            path: "Sources/BasaltCLI",
            swiftSettings: swiftSettings
        ),

        // ── Executable (thin wrapper — just calls the CLI main) ──────────────
        .executableTarget(
            name: "Basalt",
            dependencies: [
                "BasaltCLI",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/BasaltBin",
            swiftSettings: swiftSettings
        ),

        // ── Tests ─────────────────────────────────────────────────────────────
        .testTarget(
            name: "BasaltTests",
            dependencies: [
                .product(name: "ServiceLifecycle", package: "swift-service-lifecycle"),
                "BasaltCore", "BasaltInference", "BasaltCloud", "BasaltServer", "BasaltUI", "BasaltCLI", "BasaltApp", "BasaltAppCore", "BasaltHarness", "BasaltSpeech",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
                .product(name: "MLXGuidedGeneration", package: "mlx-swift-lm"),
                .product(name: "Jinja", package: "swift-jinja"),
            ],
            path: "Tests/BasaltTests",
            exclude: ["Fixtures/ReferenceLogits", "Fixtures/golden"],
            swiftSettings: swiftSettings
        ),
        .executableTarget(
            name: "PerformanceBenchmark",
            dependencies: [
                "BasaltCore", "BasaltInference",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/PerformanceBenchmark",
            swiftSettings: swiftSettings
        ),
    ]
)
