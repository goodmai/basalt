import Foundation
import ArgumentParser

// MARK: — models (subcommand group)

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct ModelsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "Browse, download and manage MLX models (Gemma, Qwen, …)",
        subcommands: [
            ListSubcommand.self,
            DownloadSubcommand.self,
            InfoSubcommand.self,
            CacheSubcommand.self,
            CheckSubcommand.self,
        ],
        defaultSubcommand: ListSubcommand.self
    )
}

// MARK: — models list

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct ListSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List available MLX models on Hugging Face (Gemma, Qwen, …)"
    )

    @Option(name: .shortAndLong, help: "Search term  (e.g. gemma-4 | Qwen3 | Qwen2.5-Coder)")
    var search: String = ""

    @Option(name: .long, help: "HuggingFace author / org (default: mlx-community)")
    var author: String = "mlx-community"

    @Option(name: .shortAndLong, help: "Filter by quantization: 4bit | 8bit | bf16")
    var quant: String? = nil

    @Option(name: .shortAndLong, help: "Maximum number of results")
    var limit: Int = 20

    mutating func run() async throws {
        let hub = HuggingFaceHub()
        printFetching(author: author, search: search, quant: quant)

        let models: [HFModelInfo]
        do {
            models = try await hub.listModels(author: author, search: search, quant: quant, limit: limit)
        } catch {
            printError(error.localizedDescription)
            throw ExitCode.failure
        }

        guard !models.isEmpty else {
            print("No models found for search='\(search)'" + (quant.map { " quant=\($0)" } ?? ""))
            return
        }

        printModelTable(models)
        printHints(models)
    }
}

// MARK: — models download

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct DownloadSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download a model (interactive picker if no ID given)"
    )

    @Argument(help: "Model repo ID, e.g. mlx-community/gemma-4-e2b-it-4bit (optional — shows picker)")
    var modelId: String? = nil

    @Option(name: .long, help: "HuggingFace author / org (default: mlx-community)")
    var author: String = "mlx-community"

    @Option(name: .long, help: "HuggingFace token for private/gated models")
    var token: String? = nil

    @Option(name: .shortAndLong, help: "Filter picker by quantization: 4bit | 8bit | bf16 | nvfp4 | dq")
    var quant: String? = nil

    @Option(name: .shortAndLong, help: "Search term for interactive picker (e.g. Qwen3 | Qwen2.5-Coder | gemma-4 | Huihui | PRISM)")
    var search: String = ""

    mutating func run() async throws {
        let hub = HuggingFaceHub()

        let repoId: String
        if let explicit = modelId {
            repoId = explicit
        } else {
            // Interactive selection
            guard let selected = try await interactivePicker(hub: hub) else {
                print("Cancelled.")
                return
            }
            repoId = selected
        }

        // Check if already cached
        if ModelCache.isDownloaded(repoId: repoId) {
            let path = ModelCache.cacheDir(for: repoId).path
            print("Already downloaded: \(path)")
            printRunHint(repoId: repoId)
            return
        }

        print("\nDownloading \(bold(repoId))…\n")
        let dest: URL
        do {
            dest = try await hub.download(repoId: repoId, token: token,
                                          onFile: DownloadProgressReporter().callAsFunction)
        } catch {
            print()
            printError(error.localizedDescription)
            throw ExitCode.failure
        }

        print("\n")
        printSuccess("Saved to \(dest.path)")
        printRunHint(repoId: repoId)
    }

    // MARK: — Interactive picker

    private func interactivePicker(hub: HuggingFaceHub) async throws -> String? {
        printFetching(author: author, search: search, quant: quant)
        let models = try await hub.listModels(author: author, search: search, quant: quant, limit: 20)
        guard !models.isEmpty else {
            print("No models found.")
            return nil
        }

        printModelTable(models, numbered: true)
        print()

        let prompt = "Select [\(green("1"))–\(green("\(models.count)"))] or \(red("q")) to quit: "
        print(prompt, terminator: "")
        fflush(stdout)

        guard let input = readLine()?.trimmingCharacters(in: .whitespaces),
              input.lowercased() != "q",
              let idx = Int(input),
              (1...models.count).contains(idx)
        else { return nil }

        return models[idx - 1].id
    }
}

// MARK: — models info

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct InfoSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Show details and file list for a model"
    )

    @Argument(help: "Model repo ID, e.g. mlx-community/gemma-4-e2b-it-4bit")
    var modelId: String

    mutating func run() async throws {
        let hub = HuggingFaceHub()
        print("Fetching info for \(bold(modelId))…\n")
        let details: HFModelDetails
        do {
            details = try await hub.modelDetails(repoId: modelId)
        } catch {
            printError(error.localizedDescription)
            throw ExitCode.failure
        }

        let cached = ModelCache.isDownloaded(repoId: modelId)
        print("  ID:        \(details.id)")
        print("  Downloads: \(details.downloads.map { "\($0)" } ?? "—")")
        print("  Likes:     \(details.likes.map    { "\($0)" } ?? "—")")
        print("  Cached:    \(cached ? green("yes → \(ModelCache.cacheDir(for: modelId).path)") : red("no"))")
        print()
        print("  Files:")
        for file in details.siblings ?? [] {
            let sizeStr = file.size.map { formatBytes(Int64($0)) } ?? ""
            let col = file.rfilename.hasSuffix(".safetensors") ? yellow : { $0 }
            print("    \(col(file.rfilename))\(sizeStr.isEmpty ? "" : "  \(dim(sizeStr))")")
        }
        print()
        if !cached {
            print("  \(dim("gemm models download \(modelId)"))")
        }
    }
}

// MARK: — models cache

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct CacheSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "cache",
        abstract: "List locally downloaded models"
    )

    mutating func run() async throws {
        let root = ModelCache.root
        guard FileManager.default.fileExists(atPath: root.path) else {
            print("Cache empty — no models downloaded yet.")
            print(dim("Cache location: \(root.path)"))
            return
        }

        let entries = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        let modelDirs = entries
            .filter { $0.hasPrefix("models--") }
            .sorted()

        if modelDirs.isEmpty {
            print("No models in cache.")
            return
        }

        print(bold("\nDownloaded models:\n"))
        print("  \(padded("REPO ID", 48))  SIZE")
        print("  " + String(repeating: "─", count: 60))

        for dir in modelDirs {
            let repoId = dir
                .replacingOccurrences(of: "models--", with: "")
                .replacingOccurrences(of: "--", with: "/")
            let fullPath = root.appendingPathComponent(dir)
            let size = directorySize(at: fullPath)
            print("  \(padded(repoId, 48))  \(formatBytes(size))")
        }
        print()
        print(dim("Cache: \(root.path)"))
    }

    private func directorySize(at url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            total += (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { Int64($0) } ?? 0
        }
        return total
    }
}

// MARK: — Shared formatting helpers

func printFetching(author: String = "mlx-community", search: String, quant: String?) {
    let qualifier = quant.map { " [\($0)]" } ?? ""
    print("\nFetching \(bold("\(author)/\(search)*"))\(qualifier) from Hugging Face…\n")
}

func printModelTable(_ models: [HFModelInfo], numbered: Bool = false) {
    let numWidth = numbered ? 4 : 0
    let idWidth = 48
    let quantWidth = 8

    // Header
    let numCol    = numbered ? padded("#",       numWidth) + "  " : ""
    let cached    = "★"
    print("  \(numCol)\(padded("MODEL",         idWidth))  \(padded("QUANT", quantWidth))  \(padded("PARAMS", 7))  \(padded("↓", 7))  \(cached)")
    print("  " + String(repeating: "─", count: idWidth + numWidth + 32))

    for (idx, m) in models.enumerated() {
        let num     = numbered ? padded("\(idx + 1)", numWidth - 1) + "  " : ""
        let isCached = ModelCache.isDownloaded(repoId: m.id) ? green("★") : " "
        let row = "  \(num)\(padded(m.id, idWidth))  \(padded(m.quantization, quantWidth))  \(padded(m.parameterSize, 7))  \(padded(m.formattedDownloads, 7))  \(isCached)"
        print(row)
    }
}

func printHints(_ models: [HFModelInfo]) {
    guard let first = models.first else { return }
    print()
    print(dim("  gemm models download \(first.id)"))
    print(dim("  gemm models download          ← interactive picker"))
    print(dim("  gemm serve --model \(first.id)"))
    print()
}

func printRunHint(repoId: String) {
    print()
    print(dim("  gemm serve --model \(repoId)"))
    print()
}

func printFileProgress(filename: String, downloaded: Int64, total: Int64) {
    let name = padded(filename, 40)
    if total > 0 {
        let pct  = Double(downloaded) / Double(total)
        let bar  = progressBar(fraction: pct, width: 20)
        let pctStr = String(format: "%3.0f%%", pct * 100)
        print("\r  \(name)  \(bar) \(pctStr)  \(formatBytes(downloaded))/\(formatBytes(total))", terminator: "")
    } else {
        print("\r  \(name)  \(formatBytes(downloaded))", terminator: "")
    }
    fflush(stdout)
}

func progressBar(fraction: Double, width: Int) -> String {
    let filled = Int((fraction * Double(width)).rounded())
    let empty  = width - filled
    return "[" + String(repeating: "█", count: max(0, filled)) + String(repeating: "░", count: max(0, empty)) + "]"
}

func printError(_ msg: String) {
    fputs(red("Error: \(msg)") + "\n", stderr)
}

func printSuccess(_ msg: String) {
    print(green("✓ \(msg)"))
}

// MARK: — ANSI helpers

func bold(_ s: String)   -> String { "\u{1B}[1m\(s)\u{1B}[0m" }
func dim(_ s: String)    -> String { "\u{1B}[2m\(s)\u{1B}[0m" }
func green(_ s: String)  -> String { "\u{1B}[32m\(s)\u{1B}[0m" }
func red(_ s: String)    -> String { "\u{1B}[31m\(s)\u{1B}[0m" }
func yellow(_ s: String) -> String { "\u{1B}[33m\(s)\u{1B}[0m" }
func blue(_ s: String)   -> String { "\u{1B}[34m\(s)\u{1B}[0m" }
func cyan(_ s: String)   -> String { "\u{1B}[36m\(s)\u{1B}[0m" }

func padded(_ s: String, _ width: Int) -> String {
    s.count >= width ? String(s.prefix(width)) : s + String(repeating: " ", count: width - s.count)
}

func formatBytes(_ bytes: Int64) -> String {
    switch bytes {
    case 0..<1_024:                    return "\(bytes) B"
    case 0..<1_048_576:                return String(format: "%.1f KB", Double(bytes) / 1_024)
    case 0..<1_073_741_824:            return String(format: "%.1f MB", Double(bytes) / 1_048_576)
    default:                           return String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
    }
}

// MARK: — models check / dry-run

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct CheckSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "check",
        abstract: "Dry-run memory feasibility evaluation for a model",
        aliases: ["dry-run", "eval"]
    )

    @Argument(help: "Model repo ID, e.g. AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit")
    var modelId: String

    mutating func run() async throws {
        let evaluator = MemoryEvaluator()
        print("\n🔍 Evaluating memory feasibility for \(bold(modelId))…\n")
        let assessment = await evaluator.evaluate(modelId: modelId)

        print("  Model:           \(assessment.modelName) (\(assessment.modelId))")
        print("  Required RAM:    \(assessment.requiredRAMFormatted)")
        print("  Available RAM:   \(assessment.availableRAMFormatted)")
        print("  Physical RAM:    \(assessment.totalRAMFormatted)")
        print("  Fit Level:       \(assessment.fitLevel.emoji) \(assessment.fitLevel.rawValue)")
        print("  Context Budget:  \(assessment.maxContextBudgetTokens) tokens")
        print()

        if let warning = assessment.warning {
            print(red("  \(warning)"))
        }
        if let rec = assessment.recommendation {
            print(yellow("  💡 Recommendation: \(rec)"))
        }

        if assessment.fitsInMemory {
            print(green("  ✓ Feasible: Model fits within system memory budget."))
        } else {
            print(red("  ❌ Insufficient Memory: Model exceeds available memory!"))
        }
        print()
    }
}

