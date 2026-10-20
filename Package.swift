// swift-tools-version: 6.0
import PackageDescription

let swiftSettings: [SwiftSetting] = [.swiftLanguageMode(.v6)]

let package = Package(
    name: "Basalt",
    platforms: [
        .macOS(.v15)
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift",     from: "0.32.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm",  revision: "7e4024e52bb9edc6f5b92e9370c50a9140f5e322"),
        // Capped below 2.28: it pulls swift-distributed-tracing 1.7.x, whose ServiceContextModule target
        // collides with swift-service-context ("target names need to be unique"). Lift once that is resolved.
        .package(url: "https://github.com/hummingbird-project/hummingbird", "2.27.0"..<"2.28.0"),
        .package(url: "https://github.com/apple/swift-argument-parser",    from: "1.8.2"),
        // Provides the `Tokenizers` module used by MLXHuggingFace macros
        .package(url: "https://github.com/huggingface/swift-transformers",  from: "1.3.5"),
        .package(url: "https://github.com/huggingface/swift-jinja.git",     from: "2.4.2"),
        // Epic 16: CLI Enhancements
        .package(url: "https://github.com/onevcat/Rainbow.git", from: "4.2.2"),
        .package(url: "https://github.com/apple/swift-markdown.git", from: "0.9.0"),
        .package(url: "https://github.com/JohnSundell/Splash.git", from: "0.16.0"),
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
                .product(name: "MLXLLM",        package: "mlx-swift-lm"),
                // Gemma 4 and other *ForConditionalGeneration checkpoints are only
                // registered in MLXVLM; without it ModelFactoryRegistry cannot load them.
                .product(name: "MLXVLM",        package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon",   package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXGuidedGeneration", package: "mlx-swift-lm"),
                .product(name: "Tokenizers",    package: "swift-transformers"),
                .product(name: "Hub",           package: "swift-transformers"),
                .product(name: "Jinja",         package: "swift-jinja"),
            ],
            path: "Sources/BasaltInference",
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
                .product(name: "Hummingbird",   package: "hummingbird"),
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
                .product(name: "Splash",        package: "Splash"),
            ],
            path: "Sources/BasaltUI",
            swiftSettings: swiftSettings
        ),
        // App: the SwiftUI REST chat client. Nothing links it yet (its launch command is #55),
        // so SwiftUI/AppKit stay out of the CLI binary.
        .target(
            name: "BasaltApp",
            dependencies: [
                "BasaltCore",
                .product(name: "Markdown",      package: "swift-markdown"),
            ],
            path: "Sources/BasaltApp",
            swiftSettings: swiftSettings
        ),
        // CLI: ArgumentParser commands wiring everything together.
        .target(
            name: "BasaltCLI",
            dependencies: [
                "BasaltCore", "BasaltInference", "BasaltCloud", "BasaltServer", "BasaltUI", "BasaltHarness",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
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
                "BasaltCore", "BasaltInference", "BasaltCloud", "BasaltServer", "BasaltUI", "BasaltCLI", "BasaltApp", "BasaltHarness",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
                .product(name: "MLXGuidedGeneration", package: "mlx-swift-lm"),
                .product(name: "Jinja", package: "swift-jinja"),
            ],
            path: "Tests/BasaltTests",
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
