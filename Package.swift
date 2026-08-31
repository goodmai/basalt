// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Gem",
    platforms: [
        .macOS(.v15)
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift",     from: "0.20.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm",  from: "3.31.4"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.6.0"),
        .package(url: "https://github.com/apple/swift-argument-parser",    from: "1.5.0"),
        // Provides the `Tokenizers` module used by MLXHuggingFace macros
        .package(url: "https://github.com/huggingface/swift-transformers",  from: "1.3.3"),
        // Epic 16: CLI Enhancements
        .package(url: "https://github.com/onevcat/Rainbow.git", from: "4.0.0"),
        .package(url: "https://github.com/vapor/console-kit.git", from: "4.0.0"),
        .package(url: "https://github.com/apple/swift-markdown.git", from: "0.5.0"),
        .package(url: "https://github.com/JohnSundell/Splash.git", from: "0.16.0"),
    ],
    targets: [

        // ── Library (all logic, importable by tests) ─────────────────────────
        .target(
            name: "GemCore",
            dependencies: [
                .product(name: "MLX",           package: "mlx-swift"),
                .product(name: "MLXNN",         package: "mlx-swift"),
                .product(name: "MLXRandom",     package: "mlx-swift"),
                .product(name: "MLXLLM",        package: "mlx-swift-lm"),
                // Gemma 4 and other *ForConditionalGeneration checkpoints are only
                // registered in MLXVLM; without it ModelFactoryRegistry cannot load them.
                .product(name: "MLXVLM",        package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon",   package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Hummingbird",   package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Tokenizers",    package: "swift-transformers"),
                .product(name: "Rainbow",       package: "Rainbow"),
                .product(name: "ConsoleKit",    package: "console-kit"),
                .product(name: "Markdown",      package: "swift-markdown"),
                .product(name: "Splash",        package: "Splash"),
            ],
            path: "Sources/Gem",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),

        // ── Executable (thin wrapper — just calls library main) ───────────────
        .executableTarget(
            name: "Gemm",
            dependencies: ["GemCore"],
            path: "Sources/GemBin",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),

        // ── Tests ─────────────────────────────────────────────────────────────
        .testTarget(
            name: "GemTests",
            dependencies: [
                "GemCore",
            ],
            path: "Tests/GemTests",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),
        .executableTarget(
            name: "PerformanceBenchmark",
            dependencies: [
                "GemCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/PerformanceBenchmark",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),
    ]
)
