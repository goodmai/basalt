// swift-tools-version: 6.4
import PackageDescription

let swiftSettings: [SwiftSetting] = [.swiftLanguageMode(.v6)]

let package = Package(
    name: "Basalt",
    platforms: [
        .macOS(.v27)
    ],
    products: [
        .library(
            name: "BasaltFoundationModels",
            targets: ["BasaltFoundationModels"]
        ),
        .executable(
            name: "Basalt",
            targets: ["Basalt"]
        ),
        .executable(
            name: "BasaltMac",
            targets: ["BasaltMac"]
        ),
        .executable(
            name: "PerformanceBenchmark",
            targets: ["PerformanceBenchmark"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift",         from: "0.33.0"),
        .package(url: "https://github.com/goodmai/mlx-swift-lm",      revision: "e9feb1ec03a79d69eba69373371650980cba463c"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.28.0"),
        .package(url: "https://github.com/apple/swift-argument-parser",    from: "1.8.2"),
        .package(url: "https://github.com/swift-server/swift-service-lifecycle", from: "2.12.0"),
        .package(url: "https://github.com/apple/swift-http-types.git",     from: "1.4.0"),
        .package(url: "https://github.com/apple/swift-log.git",            from: "1.6.2"),
        .package(url: "https://github.com/apple/swift-metrics.git",        exact: "2.13.1"),
        .package(url: "https://github.com/apple/swift-nio.git",            from: "2.104.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "1.0.0", traits: ["Xet"]),
        .package(url: "https://github.com/huggingface/swift-transformers",  from: "1.3.5"),
        .package(url: "https://github.com/huggingface/swift-jinja.git",     from: "3.1.0"),
        .package(url: "https://github.com/onevcat/Rainbow.git", from: "4.2.2"),
        .package(url: "https://github.com/apple/swift-markdown.git", from: "0.9.0"),
        .package(url: "https://github.com/jpsim/Yams.git", exact: "6.2.2"),
        .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.13.0"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.1.1"),
    ],
    targets: [
        .target(
            name: "BasaltCore",
            dependencies: [
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "RealModule", package: "swift-numerics"),
            ],
            path: "Sources/BasaltCore",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltHarness",
            dependencies: ["BasaltCore"],
            path: "Sources/BasaltHarness",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltBench",
            dependencies: ["BasaltCore", "BasaltHarness"],
            path: "Sources/BasaltBench",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltInference",
            dependencies: [
                "BasaltCore",
                .product(name: "MLX",           package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXLLM",        package: "mlx-swift-lm"),
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
        .target(
            name: "BasaltCloud",
            dependencies: ["BasaltCore"],
            path: "Sources/BasaltCloud",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltServer",
            dependencies: [
                "BasaltCore", "BasaltHarness",
                .product(name: "ServiceLifecycle", package: "swift-service-lifecycle"),
                .product(name: "Hummingbird",   package: "hummingbird"),
                .product(name: "HTTPTypes",     package: "swift-http-types"),
                .product(name: "Logging",       package: "swift-log"),
                .product(name: "CoreMetrics",   package: "swift-metrics"),
                .product(name: "NIOFoundationEssentialsCompat", package: "swift-nio"),
                .product(name: "yyjson",        package: "yyjson"),
            ],
            path: "Sources/BasaltServer",
            swiftSettings: swiftSettings
        ),
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
        .target(
            name: "BasaltFoundationModels",
            dependencies: ["BasaltCore", "BasaltAppCore"],
            path: "Sources/BasaltFoundationModels",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltApp",
            dependencies: [
                "BasaltCore", "BasaltAppCore",
                .product(name: "Markdown",      package: "swift-markdown"),
            ],
            path: "Sources/BasaltApp",
            swiftSettings: swiftSettings
        ),
        .executableTarget(
            name: "BasaltMac",
            dependencies: ["BasaltApp", "BasaltAppCore"],
            path: "Sources/BasaltMac",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "BasaltCLI",
            dependencies: [
                "BasaltCore", "BasaltInference", "BasaltCloud", "BasaltServer", "BasaltUI", "BasaltHarness", "BasaltBench", "BasaltSpeech",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Yams", package: "Yams"),
            ],
            path: "Sources/BasaltCLI",
            swiftSettings: swiftSettings
        ),
        .executableTarget(
            name: "Basalt",
            dependencies: [
                "BasaltCLI", "BasaltInference",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/BasaltBin",
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "BasaltTests",
            dependencies: [
                .product(name: "ServiceLifecycle", package: "swift-service-lifecycle"),
                "BasaltCore", "BasaltInference", "BasaltCloud", "BasaltServer", "BasaltUI", "BasaltCLI", "BasaltApp", "BasaltAppCore", "BasaltFoundationModels", "BasaltHarness", "BasaltBench", "BasaltSpeech",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
                .product(name: "CoreMetrics", package: "swift-metrics"),
                .product(name: "MetricsTestKit", package: "swift-metrics"),
                .product(name: "MLXGuidedGeneration", package: "mlx-swift-lm"),
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "RealModule", package: "swift-numerics"),
            ],
            path: "Tests/BasaltTests",
            exclude: ["Fixtures/ReferenceLogits", "Fixtures/golden", "Fixtures/VisualGoldens", "Fixtures/Inkling2"],
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
