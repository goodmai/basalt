// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GemmaServer",
    platforms: [
        .macOS(.v15)
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift",     from: "0.10.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm",  from: "3.31.3"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.6.0"),
        .package(url: "https://github.com/apple/swift-argument-parser",    from: "1.5.0"),
        // Provides the `Tokenizers` module used by MLXHuggingFace macros
        .package(url: "https://github.com/huggingface/swift-transformers",  from: "1.3.0"),
        .package(url: "https://github.com/stephencelis/SQLite.swift.git", from: "0.15.3"),
        .package(url: "https://github.com/vapor/jwt-kit.git", from: "4.13.0"),
    ],
    targets: [

        // ── Library (all logic, importable by tests) ─────────────────────────
        .target(
            name: "GemmaServerCore",
            dependencies: [
                .product(name: "MLX",           package: "mlx-swift"),
                .product(name: "MLXNN",         package: "mlx-swift"),
                .product(name: "MLXRandom",     package: "mlx-swift"),
                .product(name: "MLXLLM",        package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon",   package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Hummingbird",   package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Tokenizers",    package: "swift-transformers"),
                .product(name: "SQLite",        package: "SQLite.swift"),
                .product(name: "JWTKit",        package: "jwt-kit"),
            ],
            path: "Sources/GemmaServer",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),

        // ── Executable (thin wrapper — just calls library main) ───────────────
        .executableTarget(
            name: "GemmaServer",
            dependencies: ["GemmaServerCore"],
            path: "Sources/GemmaServerBin",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),

        // ── Tests ─────────────────────────────────────────────────────────────
        .testTarget(
            name: "GemmaServerTests",
            dependencies: ["GemmaServerCore"],
            path: "Tests/GemmaServerTests",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),
    ]
)
