# Epic 16: Advanced CLI Enhancements & Rich Terminal UI 🎨

**Version:** v0.6.0  
**Priority:** HIGH  
**Total Effort:** 6-8 weeks  
**Status:** Planned  
**Created:** April 28, 2026

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Competitive Analysis: OpenCLI vs GemmaServer](#competitive-analysis)
3. [Gap Analysis & Feature Extraction](#gap-analysis)
4. [Epic Breakdown](#epic-breakdown)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Success Metrics](#success-metrics)

---

## Executive Summary

### Business Rationale
GemmaServer currently has a functional CLI with basic interactivity (Epic 15), but lacks the **polished, professional terminal UI** that modern CLI tools (like `gh`, `kubectl`, `docker`) provide. After analyzing the competitive landscape (opencli, gemini-cli), we've identified **6 major feature gaps** that prevent GemmaServer from achieving production-grade UX.

### Strategic Goals
1. **Visual Excellence**: Rich colors, tables, diff views, markdown rendering, progress bars
2. **Developer Experience**: Auto-completion, history, clipboard integration, file previews
3. **Task-First Architecture**: Modular commands for specific tasks (asr, tts, ocr, t2i)
4. **Smart Hardware Awareness**: `fit` command for model recommendations based on RAM
5. **Professional Output**: Structured JSON/YAML export, piping support
6. **Cross-Platform Copy/Paste**: Native clipboard integration (macOS, Linux)

---

## Competitive Analysis: OpenCLI vs GemmaServer

### OpenCLI Strengths (Features We Should Adopt)

#### 1. **Task-First Command Architecture** ⭐⭐⭐
**Current State:**
```bash
# OpenCLI
opencli asr < audio.wav        # Speech-to-Text
opencli tts "Hello" > out.wav  # Text-to-Speech
opencli ocr image.png          # OCR
opencli t2i "sunset"           # Text-to-Image
```

**GemmaServer:**
```bash
# Generic chat interface only
gemmaserver chat --model qwen3.5-4b
```

**Gap:** OpenCLI has **15+ specialized task commands** (asr, tts, vad, sts, ocr, vlm, t2i, i2i, t2v, i2v, t2m, sr, rmbg, embedding, rerank, chat). GemmaServer only has `chat` and `serve`.

**Business Value:** Task-first commands make the CLI **discoverable** and **composable** via Unix pipes.

---

#### 2. **`fit` Command: Hardware-Aware Model Selection** ⭐⭐⭐
**OpenCLI Implementation:**
```bash
$ opencli fit

Device: Apple M2 | total 16.0 GB | available 4.6 GB | model budget 3.9 GB
GPU: Apple M2 | backend: metal | unified_memory: true

Recommendations by task:
- [asr] Qwen3-ASR 1.7B 4bit | 🟡 Good | score 86.5 | GPU
- [chat] Qwen3 Chat 1.7B 4bit | 🟠 Marginal | score 82.5 | GPU
- [embedding] Qwen3 Embedding 0.6B 4bit DWQ | 🟢 Perfect | score 74.3 | GPU

Top candidates:
TASK     MODEL                    FIT       SCORE  MODE     QUANT  EST/MEM     LOCAL
----     -----                    ---       -----  ----     -----  -------     -----
asr      Qwen3-ASR 1.7B          Good      86.5   GPU      4bit   1.2/4.6GB   yes
chat     Qwen3 Chat 1.7B         Marginal  82.5   GPU      4bit   1.5/4.6GB   no
```

**Key Features:**
- **Hardware Profiling**: Detects M1/M2/M3/M4/M5, RAM, GPU, unified memory
- **Fit Scoring**: Ranks models by fit level (Perfect/Good/Marginal/TooTight)
- **Context-Aware**: Estimates memory based on context length
- **Rich Tables**: ASCII tables with proper column alignment
- **Color-Coded Output**: Green (Perfect), Yellow (Good), Orange (Marginal), Red (Too Tight)

**GemmaServer:** Has basic `onboard` command with hardware detection, but **no fit scoring** or **model ranking**.

---

#### 3. **Rich Terminal UI & Output Modes** ⭐⭐⭐
**OpenCLI OutputEmitter:**
```swift
struct OutputEmitter {
    let mode: OutputMode  // .json | .pretty | .plain
    
    func emitSuccessData<E: Encodable>(_ data: E)
    func emitFailure(_ error: CLIError)
    func logProgressInline(_ message: String)  // \r\u{1B}[K for inline updates
}

enum OutputMode {
    case json    // Machine-readable, structured
    case pretty  // Human-readable with colors, tables
    case plain   // Simple text for piping
}
```

**Auto-Detection:**
- If `stdout` is a TTY → `.pretty` (colors, tables)
- If `stdout` is a pipe → `.plain` (no colors)
- Explicit `--json` flag → `.json`

**GemmaServer:** Only plain text output, no JSON mode, no pretty mode.

---

#### 4. **Progress Reporting System** ⭐⭐
**OpenCLI:**
```swift
class PercentProgressReporter {
    func update(percent: Double, message: String) {
        // Inline progress bar: [████████░░] 80% Downloading model...
        writeStderrRaw("\r[\(bar)] \(percent)% \(message)\u{1B}[K")
    }
}
```

**Features:**
- Inline progress bars (no newlines, uses `\r`)
- Percentage + visual bar
- Conditional rendering (only on TTY stderr)

**GemmaServer:** Basic file download progress, no unified progress system.

---

#### 5. **Modular Library Architecture** ⭐⭐
**OpenCLI Structure:**
```
OpenCLIKernel/        # Core abstractions (ModelRegistry, Profiler, CacheManager)
  ├── Hardware/       # Profiler, ResourceMatcher, MemoryManager
  ├── ModelHub/       # CacheManager, DownloadManager, HubClient
  └── Registry/       # ModelRegistry, ModelDefinition, ModelSelectionEngine

OpenCLIAudio/         # Audio tasks (ASR, TTS, VAD, STS)
OpenCLIVision/        # Vision tasks (OCR, VLM, I2T)
OpenCLIGen/           # Generation tasks (T2I, I2I, T2V, I2V, T2M)
OpenCLILLM/           # LLM tasks (Chat, Embedding, Rerank)
OpenCLIServe/         # HTTP/MCP server
```

**GemmaServer:** Monolithic `Sources/GemmaServer/` with less clear separation.

---

#### 6. **Clipboard Integration** (Not in OpenCLI, but common in modern CLIs)
**Examples:**
- `gh pr view --web` → Opens PR in browser
- `gh pr view --json | pbcopy` → Copies JSON to clipboard
- `git log --oneline | head -5 | pbcopy` → Copies commits

**GemmaServer:** No clipboard support.

---

### OpenCLI Weaknesses (Areas We Already Excel)

| Feature | OpenCLI | GemmaServer | Winner |
|---------|---------|-------------|--------|
| Interactive Chat | ❌ Basic | ✅ Advanced (TerminalManager, non-blocking) | GemmaServer |
| Streaming Support | ⚠️ Limited | ✅ Full SSE + AsyncStream | GemmaServer |
| Authentication | ❌ None | ✅ JWT + Bcrypt + RBAC | GemmaServer |
| Cloud Integration | ❌ None | ✅ OpenRouter + Hybrid Routing | GemmaServer |
| Security Audit | ❌ None | ✅ 10/10 Score | GemmaServer |
| Test Coverage | ⚠️ Basic | ✅ 101 tests, 100% coverage | GemmaServer |
| Documentation | ⚠️ Minimal | ✅ PLAN.md, SECURITY.md, Epic docs | GemmaServer |

---

## Gap Analysis & Feature Extraction

### High-Priority Gaps (Must-Have for v1.0)

#### Gap 1: No Rich Terminal Output Libraries
**Current:** Manual ANSI escape codes (`\u{1B}[32m`)  
**Needed:** Professional libraries (Rainbow, ConsoleKit)

#### Gap 2: No Table Rendering
**Current:** Plain text  
**Needed:** ASCII tables with column alignment

#### Gap 3: No Markdown Rendering
**Current:** Raw text  
**Needed:** Syntax highlighting, bold, italic, code blocks

#### Gap 4: No Diff Viewer
**Current:** N/A  
**Needed:** Side-by-side diff for code changes

#### Gap 5: No Image Preview
**Current:** N/A  
**Needed:** Terminal image rendering (iTerm2, Kitty protocols)

#### Gap 6: No Clipboard Integration
**Current:** N/A  
**Needed:** `pbcopy`/`xclip` support

---

## Epic Breakdown

### Epic 16.1: Rich Terminal UI Foundation
**Effort:** 1 week  
**Priority:** CRITICAL

#### Tasks
- **16.1.1**: Integrate `Rainbow` library for colors and styles
- **16.1.2**: Create `TerminalUI` module with helper functions
- **16.1.3**: Implement color scheme (success=green, error=red, warning=yellow, info=blue, dim=gray)
- **16.1.4**: Add `--no-color` flag support
- **16.1.5**: Auto-detect TTY and disable colors when piping

#### User Stories
- **US-16.1.1**: As a developer, I want colored output for better readability
- **US-16.1.2**: As a CI/CD pipeline, I want plain text output without ANSI codes
- **US-16.1.3**: As a user, I want consistent color themes across all commands

#### Acceptance Criteria
- [x] Rainbow library integrated
- [ ] Helper functions: `success()`, `error()`, `warning()`, `info()`, `dim()`
- [ ] Environment variable: `NO_COLOR=1` disables colors
- [ ] `--no-color` flag works on all commands
- [ ] Auto-detect `isatty()` for stdout/stderr

#### Technical Design
```swift
// Sources/GemmaServer/UI/TerminalUI.swift
import Rainbow

public enum TerminalUI {
    public static var colorsEnabled: Bool = {
        #if canImport(Darwin)
        return isatty(fileno(stdout)) == 1 && ProcessInfo.processInfo.environment["NO_COLOR"] == nil
        #else
        return false
        #endif
    }()
    
    public static func success(_ text: String) -> String {
        colorsEnabled ? text.green.bold : text
    }
    
    public static func error(_ text: String) -> String {
        colorsEnabled ? text.red.bold : text
    }
    
    public static func warning(_ text: String) -> String {
        colorsEnabled ? text.yellow : text
    }
    
    public static func info(_ text: String) -> String {
        colorsEnabled ? text.blue : text
    }
    
    public static func dim(_ text: String) -> String {
        colorsEnabled ? text.dim : text
    }
    
    public static func code(_ text: String) -> String {
        colorsEnabled ? text.cyan : "`\(text)`"
    }
}
```

#### Test Cases
```swift
@Test("TerminalUI colors work when enabled")
func testColorsEnabled() {
    TerminalUI.colorsEnabled = true
    let output = TerminalUI.success("OK")
    #expect(output.contains("\u{1B}["))  // Contains ANSI codes
}

@Test("TerminalUI plain text when disabled")
func testColorsDisabled() {
    TerminalUI.colorsEnabled = false
    let output = TerminalUI.success("OK")
    #expect(output == "OK")  // No ANSI codes
}
```

#### Example Output
```bash
$ gemmaserver chat --model qwen3.5-4b
✅ Model loaded successfully (2.3 GB)
💬 Gemma > Hello
🤖 Assistant: Hi! How can I help you today?
📊 Stats: 12 in / 8 out | 92.5 TPS | 0.045s TTFT | 2.3GB RAM
```

---

### Epic 16.2: Table Rendering System
**Effort:** 1 week  
**Priority:** HIGH

#### Tasks
- **16.2.1**: Integrate `ConsoleKit` (from Vapor) for tables
- **16.2.2**: Create `TableRenderer` utility
- **16.2.3**: Implement auto-sizing columns
- **16.2.4**: Support borders, headers, footers
- **16.2.5**: Add pagination for large tables

#### User Stories
- **US-16.2.1**: As a user running `fit` command, I want to see model recommendations in a table
- **US-16.2.2**: As a developer, I want to compare multiple models side-by-side
- **US-16.2.3**: As a user, I want tables to auto-adjust to terminal width

#### Acceptance Criteria
- [ ] ConsoleKit integrated
- [ ] `TableRenderer.render(rows:headers:)` function
- [ ] Auto-sizing based on terminal width (via `ioctl` or env `COLUMNS`)
- [ ] Support for alignment (left, right, center)
- [ ] Unicode box-drawing characters (optional, fallback to ASCII)

#### Technical Design
```swift
// Sources/GemmaServer/UI/TableRenderer.swift
import ConsoleKit

public struct TableRenderer {
    public static func render(
        headers: [String],
        rows: [[String]],
        style: TableStyle = .unicode
    ) -> String {
        var output = ""
        
        // Calculate column widths
        var widths = headers.map { $0.count }
        for row in rows {
            for (i, cell) in row.enumerated() {
                widths[i] = max(widths[i], cell.count)
            }
        }
        
        // Render header
        output += renderRow(headers, widths: widths, style: style)
        output += renderSeparator(widths: widths, style: style)
        
        // Render rows
        for row in rows {
            output += renderRow(row, widths: widths, style: style)
        }
        
        return output
    }
    
    private static func renderRow(_ cells: [String], widths: [Int], style: TableStyle) -> String {
        let paddedCells = zip(cells, widths).map { cell, width in
            cell.padding(toLength: width, withPad: " ", startingAt: 0)
        }
        let border = style == .unicode ? "│" : "|"
        return "\(border) \(paddedCells.joined(separator: " \(border) ")) \(border)\n"
    }
    
    private static func renderSeparator(widths: [Int], style: TableStyle) -> String {
        let segments = widths.map { String(repeating: style.horizontalLine, count: $0) }
        let junction = style == .unicode ? "┼" : "+"
        let left = style == .unicode ? "├" : "+"
        let right = style == .unicode ? "┤" : "+"
        return "\(left)\(segments.joined(separator: junction))\(right)\n"
    }
}

public enum TableStyle {
    case unicode  // ┌─┬─┐
    case ascii    // +-+-+
    
    var horizontalLine: String {
        self == .unicode ? "─" : "-"
    }
}
```

#### Example Output
```bash
$ gemmaserver models list --format table

╭─────────────────────────────────┬──────────┬─────────┬──────────╮
│ Model                           │ Size     │ TPS     │ Status   │
├─────────────────────────────────┼──────────┼─────────┼──────────┤
│ mlx-community/Qwen3.5-4B-4bit   │ 2.3 GB   │ 92 TPS  │ ✓ Cached │
│ mlx-community/Qwen3.6-27B-4bit  │ 14.5 GB  │ 11 TPS  │ ✗ Remote │
│ mlx-community/gemma-4-e2b-it    │ 2.7 GB   │ 60 TPS  │ ✓ Cached │
╰─────────────────────────────────┴──────────┴─────────┴──────────╯
```

---

### Epic 16.3: Markdown Rendering & Syntax Highlighting
**Effort:** 2 weeks  
**Priority:** HIGH

#### Tasks
- **16.3.1**: Integrate `swift-markdown` parser
- **16.3.2**: Integrate `Splash` for syntax highlighting
- **16.3.3**: Create `MarkdownRenderer` for terminal output
- **16.3.4**: Support code blocks with language detection
- **16.3.5**: Render tables, lists, headers, links

#### User Stories
- **US-16.3.1**: As a user, I want code blocks in chat responses to be syntax-highlighted
- **US-16.3.2**: As a developer, I want markdown tables to render properly
- **US-16.3.3**: As a user, I want bold/italic text to be visually distinct

#### Acceptance Criteria
- [ ] `swift-markdown` integrated
- [ ] Code blocks syntax-highlighted for Swift, Python, JavaScript, Bash, JSON
- [ ] Bold → bright/bold ANSI
- [ ] Italic → underline (terminal-dependent)
- [ ] Headers → larger/colored
- [ ] Links → blue underline with URL hint

#### Technical Design
```swift
// Sources/GemmaServer/UI/MarkdownRenderer.swift
import Markdown
import Splash

public struct MarkdownRenderer {
    private let highlighter = SyntaxHighlighter(format: ANSIOutputFormat())
    
    public func render(_ markdown: String) -> String {
        let document = Document(parsing: markdown)
        var visitor = TerminalVisitor(highlighter: highlighter)
        return visitor.visit(document)
    }
}

struct TerminalVisitor: MarkupVisitor {
    let highlighter: SyntaxHighlighter
    
    mutating func visitCodeBlock(_ codeBlock: CodeBlock) -> String {
        guard let language = codeBlock.language else {
            return "```\n\(codeBlock.code)\n```\n"
        }
        
        let highlighted = highlighter.highlight(codeBlock.code, as: .init(rawValue: language))
        return "\n\(TerminalUI.dim("```\(language)"))\n\(highlighted)\n\(TerminalUI.dim("```"))\n"
    }
    
    mutating func visitHeading(_ heading: Heading) -> String {
        let text = heading.plainText
        switch heading.level {
        case 1: return "\n\(TerminalUI.success(text).underline)\n"
        case 2: return "\n\(TerminalUI.info(text).bold)\n"
        default: return "\n\(text.bold)\n"
        }
    }
    
    mutating func visitStrong(_ strong: Strong) -> String {
        strong.plainText.bold
    }
    
    mutating func visitEmphasis(_ emphasis: Emphasis) -> String {
        emphasis.plainText.underline
    }
}
```

#### Example Output
```bash
$ gemmaserver chat --model qwen3.5-4b
💬 Gemma > Explain recursion in Swift

🤖 Assistant:

## Recursion in Swift

**Recursion** is when a function calls itself. Here's an example:

```swift
func factorial(_ n: Int) -> Int {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}
```

*Key points*:
- Base case prevents infinite loop
- Each call gets closer to base case
```

---

### Epic 16.4: Code Diff Viewer
**Effort:** 1 week  
**Priority:** MEDIUM

#### Tasks
- **16.4.1**: Implement unified diff parser
- **16.4.2**: Create side-by-side diff renderer
- **16.4.3**: Color-code additions (green), deletions (red), context (gray)
- **16.4.4**: Support file diffs and inline diffs

#### User Stories
- **US-16.4.1**: As a developer, I want to see code changes in a readable format
- **US-16.4.2**: As a user reviewing AI-generated code, I want to see what changed

#### Acceptance Criteria
- [ ] Parse unified diff format (`diff -u`)
- [ ] Render with colors: + green, - red, context gray
- [ ] Side-by-side mode (optional)
- [ ] Inline mode (default)

#### Technical Design
```swift
// Sources/GemmaServer/UI/DiffRenderer.swift

public struct DiffRenderer {
    public static func render(_ diff: String, mode: DiffMode = .inline) -> String {
        let lines = diff.split(separator: "\n")
        var output = ""
        
        for line in lines {
            if line.hasPrefix("+++") || line.hasPrefix("---") {
                output += TerminalUI.dim(String(line)) + "\n"
            } else if line.hasPrefix("+") {
                output += TerminalUI.success(String(line)) + "\n"
            } else if line.hasPrefix("-") {
                output += TerminalUI.error(String(line)) + "\n"
            } else if line.hasPrefix("@@") {
                output += TerminalUI.info(String(line)) + "\n"
            } else {
                output += String(line) + "\n"
            }
        }
        
        return output
    }
}

public enum DiffMode {
    case inline      // Unified diff
    case sideBySide  // Split view
}
```

#### Example Output
```bash
$ gemmaserver chat --model qwen3.5-4b
💬 Gemma > Refactor this function @auth.swift

🤖 Assistant: Here's the refactored version:

--- auth.swift
+++ auth.swift
@@ -12,8 +12,5 @@
 func authenticate(username: String, password: String) -> Bool {
-    let hash = SHA256.hash(data: Data(password.utf8))
-    let hashString = hash.compactMap { String(format: "%02x", $0) }.joined()
-    return database.verify(username, hashString)
+    return BCryptHasher.verify(password, against: database.getHash(username))
 }
```

---

### Epic 16.5: Image Preview in Terminal
**Effort:** 1 week  
**Priority:** LOW

#### Tasks
- **16.5.1**: Detect terminal capabilities (iTerm2, Kitty, Sixel)
- **16.5.2**: Implement iTerm2 inline images protocol
- **16.5.3**: Implement Kitty graphics protocol
- **16.5.4**: Fallback to ASCII art for unsupported terminals
- **16.5.5**: Image scaling and aspect ratio preservation

#### User Stories
- **US-16.5.1**: As a user generating images, I want to preview them in terminal
- **US-16.5.2**: As a developer, I want to see screenshots inline

#### Acceptance Criteria
- [ ] Detect terminal via `$TERM_PROGRAM`
- [ ] iTerm2 protocol: Base64-encoded image with escape sequence
- [ ] Kitty protocol: Direct image rendering
- [ ] ASCII art fallback using ANSI blocks
- [ ] Image resizing to terminal width

#### Technical Design
```swift
// Sources/GemmaServer/UI/ImageRenderer.swift

public struct ImageRenderer {
    public static func render(_ imageURL: URL) -> String {
        let termProgram = ProcessInfo.processInfo.environment["TERM_PROGRAM"]
        
        switch termProgram {
        case "iTerm.app":
            return renderITerm2(imageURL)
        case "kitty":
            return renderKitty(imageURL)
        default:
            return renderASCII(imageURL)
        }
    }
    
    private static func renderITerm2(_ url: URL) -> String {
        guard let data = try? Data(contentsOf: url) else { return "" }
        let base64 = data.base64EncodedString()
        return "\u{1B}]1337;File=inline=1:\(base64)\u{07}"
    }
    
    private static func renderKitty(_ url: URL) -> String {
        // Kitty graphics protocol
        return "\u{1B}_Gf=100,a=T,t=f;\(url.path)\u{1B}\\"
    }
    
    private static func renderASCII(_ url: URL) -> String {
        // Fallback: convert to ASCII art
        return "[Image: \(url.lastPathComponent)]"
    }
}
```

---

### Epic 16.6: Clipboard Integration
**Effort:** 3 days  
**Priority:** MEDIUM

#### Tasks
- **16.6.1**: Create `ClipboardManager` for macOS (`pbcopy`/`pbpaste`)
- **16.6.2**: Support Linux (`xclip`/`xsel`)
- **16.6.3**: Add `--copy` flag to copy output to clipboard
- **16.6.4**: Add `--paste` flag to read input from clipboard

#### User Stories
- **US-16.6.1**: As a user, I want to copy AI responses to clipboard with `--copy`
- **US-16.6.2**: As a developer, I want to paste code from clipboard into prompts

#### Acceptance Criteria
- [ ] `--copy` flag works on all commands
- [ ] `--paste` reads from clipboard
- [ ] Auto-detect clipboard tool (pbcopy, xclip, xsel)
- [ ] Graceful fallback if no clipboard tool available

#### Technical Design
```swift
// Sources/GemmaServer/UI/ClipboardManager.swift

public struct ClipboardManager {
    public static func copy(_ text: String) throws {
        #if os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pbcopy")
        let pipe = Pipe()
        process.standardInput = pipe
        try process.run()
        pipe.fileHandleForWriting.write(Data(text.utf8))
        try pipe.fileHandleForWriting.close()
        process.waitUntilExit()
        #elseif os(Linux)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xclip")
        process.arguments = ["-selection", "clipboard"]
        let pipe = Pipe()
        process.standardInput = pipe
        try process.run()
        pipe.fileHandleForWriting.write(Data(text.utf8))
        try pipe.fileHandleForWriting.close()
        process.waitUntilExit()
        #endif
    }
    
    public static func paste() throws -> String {
        #if os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pbpaste")
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(data: data, encoding: .utf8) ?? ""
        #elseif os(Linux)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xclip")
        process.arguments = ["-selection", "clipboard", "-o"]
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(data: data, encoding: .utf8) ?? ""
        #endif
    }
}
```

#### Example Usage
```bash
# Copy AI response to clipboard
$ gemmaserver chat --model qwen3.5-4b -p "Explain HTTPS" --copy
✅ Response copied to clipboard

# Paste code from clipboard into prompt
$ gemmaserver chat --model qwen3.5-4b -p "Refactor this:" --paste
```

---

### Epic 16.7: `fit` Command Implementation
**Effort:** 1 week  
**Priority:** HIGH

#### Tasks
- **16.7.1**: Port OpenCLI's `Profiler` and `ResourceMatcher`
- **16.7.2**: Create model database with RAM requirements
- **16.7.3**: Implement fit scoring algorithm
- **16.7.4**: Render recommendations table
- **16.7.5**: Support filtering by modality/task

#### User Stories
- **US-16.7.1**: As a new user, I want to know which models fit my hardware
- **US-16.7.2**: As a user with 8GB RAM, I want to see only viable models
- **US-16.7.3**: As a developer, I want JSON output for automation

#### Acceptance Criteria
- [ ] `gemmaserver fit` command works
- [ ] Detects M1/M2/M3/M4/M5 chip
- [ ] Scores models as Perfect/Good/Marginal/TooTight
- [ ] Filters by `--modality` and `--task`
- [ ] `--json` output mode

#### Technical Design
```swift
// Sources/GemmaServer/CLI/FitCommand.swift

struct FitCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fit",
        abstract: "Analyze hardware and recommend optimal models"
    )
    
    @Option(help: "Filter by task (chat, code, vision, audio)")
    var task: String?
    
    @Flag(help: "Output in JSON format")
    var json = false
    
    func run() async throws {
        // 1. Profile hardware
        let profiler = SystemProfiler()
        let profile = await profiler.detectResources()
        
        // 2. Load model database
        let models = ModelDatabase.allModels
        
        // 3. Score models
        let scorer = FitScorer(profile: profile)
        let scored = models.map { scorer.score($0) }
        
        // 4. Rank and filter
        let recommendations = scored
            .sorted { $0.score > $1.score }
            .prefix(10)
        
        // 5. Render output
        if json {
            let output = try JSONEncoder().encode(recommendations)
            print(String(data: output, encoding: .utf8)!)
        } else {
            renderFitReport(profile: profile, recommendations: Array(recommendations))
        }
    }
    
    private func renderFitReport(profile: SystemResources, recommendations: [FitScore]) {
        print(TerminalUI.info("Device: \(profile.chipModel) | \(profile.totalRAM / 1_073_741_824) GB RAM"))
        print("")
        print(TerminalUI.success("Top Recommendations:"))
        
        let headers = ["Model", "Fit", "Score", "RAM", "TPS"]
        let rows = recommendations.map { rec in
            [
                rec.modelName,
                rec.fitLevel.emoji + " " + rec.fitLevel.label,
                String(format: "%.1f", rec.score),
                "\(rec.estimatedRAM / 1024) GB",
                "\(rec.estimatedTPS) TPS"
            ]
        }
        
        print(TableRenderer.render(headers: headers, rows: rows))
    }
}

struct FitScore: Codable {
    let modelName: String
    let fitLevel: FitLevel
    let score: Double
    let estimatedRAM: Int64  // MB
    let estimatedTPS: Int
}

enum FitLevel: String, Codable {
    case perfect = "Perfect"
    case good = "Good"
    case marginal = "Marginal"
    case tooTight = "TooTight"
    
    var emoji: String {
        switch self {
        case .perfect: return "🟢"
        case .good: return "🟡"
        case .marginal: return "🟠"
        case .tooTight: return "🔴"
        }
    }
    
    var label: String { rawValue }
}
```

#### Example Output
```bash
$ gemmaserver fit

Device: Apple M2 Max | 32 GB RAM | 64 GB Unified Memory

Top Recommendations:
╭────────────────────────────┬─────────────┬────────┬─────────┬──────────╮
│ Model                      │ Fit         │ Score  │ RAM     │ TPS      │
├────────────────────────────┼─────────────┼────────┼─────────┼──────────┤
│ Qwen3.5-4B-4bit            │ 🟢 Perfect  │ 95.2   │ 2 GB    │ 92 TPS   │
│ Qwen3.6-27B-4bit           │ 🟡 Good     │ 87.3   │ 14 GB   │ 11 TPS   │
│ Qwen2.5-Coder-7B-4bit      │ 🟢 Perfect  │ 89.1   │ 4 GB    │ 60 TPS   │
╰────────────────────────────┴─────────────┴────────┴─────────┴──────────╯

Run `gemmaserver models download <model>` to install.
```

---

### Epic 16.8: Progress Bar System
**Effort:** 3 days  
**Priority:** MEDIUM

#### Tasks
- **16.8.1**: Create `ProgressBar` component
- **16.8.2**: Support different styles (spinner, bar, percentage)
- **16.8.3**: Inline updates (no newlines)
- **16.8.4**: Multi-task progress (parallel downloads)

#### Technical Design
```swift
// Sources/GemmaServer/UI/ProgressBar.swift

public struct ProgressBar {
    private let total: Int
    private var current: Int = 0
    
    public init(total: Int) {
        self.total = total
    }
    
    public mutating func update(current: Int) {
        self.current = current
        render()
    }
    
    private func render() {
        let percent = Double(current) / Double(total) * 100
        let barWidth = 40
        let filled = Int(Double(barWidth) * Double(current) / Double(total))
        let bar = String(repeating: "█", count: filled) + String(repeating: "░", count: barWidth - filled)
        
        fputs("\r[\(bar)] \(String(format: "%.1f%%", percent))\u{1B}[K", stderr)
        fflush(stderr)
    }
    
    public func finish() {
        fputs("\n", stderr)
    }
}
```

---

### Epic 16.9: Output Mode System (JSON/Plain/Pretty)
**Effort:** 2 days  
**Priority:** HIGH

#### Tasks
- **16.9.1**: Create `OutputMode` enum
- **16.9.2**: Auto-detect TTY
- **16.9.3**: Add `--json`, `--plain`, `--pretty` flags
- **16.9.4**: Structured error responses in JSON mode

#### Technical Design
```swift
// Sources/GemmaServer/CLI/OutputMode.swift

public enum OutputMode: String, Sendable {
    case json
    case plain
    case pretty
    
    public static var auto: OutputMode {
        #if canImport(Darwin)
        return isatty(fileno(stdout)) == 1 ? .pretty : .plain
        #else
        return .plain
        #endif
    }
}

public struct OutputFormatter {
    let mode: OutputMode
    
    public func emit<T: Encodable>(success: T) {
        switch mode {
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            if let data = try? encoder.encode(success),
               let json = String(data: data, encoding: .utf8) {
                print(json)
            }
        case .plain, .pretty:
            print(success)
        }
    }
    
    public func emit(error: Error) {
        switch mode {
        case .json:
            let payload = ["error": error.localizedDescription]
            emit(success: payload)
        case .plain:
            fputs("error: \(error.localizedDescription)\n", stderr)
        case .pretty:
            fputs(TerminalUI.error("Error: \(error.localizedDescription)") + "\n", stderr)
        }
    }
}
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [x] Epic 16.1: Rich Terminal UI Foundation (Rainbow integration)
- [ ] Epic 16.9: Output Mode System

**Deliverable:** Colored output, auto-detect TTY, JSON mode

### Phase 2: Rich Rendering (Weeks 3-4)
- [ ] Epic 16.2: Table Rendering System
- [ ] Epic 16.3: Markdown Rendering & Syntax Highlighting

**Deliverable:** Tables, syntax highlighting, markdown support

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Epic 16.7: `fit` Command Implementation
- [ ] Epic 16.4: Code Diff Viewer
- [ ] Epic 16.8: Progress Bar System

**Deliverable:** Hardware-aware recommendations, diff viewer, progress bars

### Phase 4: Polish (Weeks 7-8)
- [ ] Epic 16.6: Clipboard Integration
- [ ] Epic 16.5: Image Preview in Terminal (optional)
- [ ] Documentation and examples

**Deliverable:** Clipboard support, image preview, comprehensive docs

---

## Success Metrics

### Quantitative
- [ ] 100% test coverage for new UI components
- [ ] Zero regressions in existing CLI commands
- [ ] < 50ms overhead for color rendering
- [ ] JSON output passes `jq` validation
- [ ] Tables render correctly in 80-column and 120-column terminals

### Qualitative
- [ ] CLI feels "professional" (subjective user testing)
- [ ] `fit` command reduces time-to-first-inference by 50%
- [ ] Markdown rendering improves code readability
- [ ] Clipboard integration increases productivity

### User Feedback Metrics (Post-Release)
- [ ] Net Promoter Score (NPS) > 50
- [ ] GitHub stars increase by 30%
- [ ] User retention (DAU/MAU ratio) > 40%

---

## Dependencies

### External Libraries
```swift
// Package.swift additions
dependencies: [
    .package(url: "https://github.com/onevcat/Rainbow", from: "4.0.0"),
    .package(url: "https://github.com/vapor/console-kit", from: "4.0.0"),
    .package(url: "https://github.com/apple/swift-markdown", from: "0.5.0"),
    .package(url: "https://github.com/JohnSundell/Splash", from: "0.16.0"),
]
```

### System Requirements
- macOS 14+ (for Swift 6, ConsoleKit)
- Xcode 16+ (for Swift Testing, Swift 6)

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Rainbow/ConsoleKit incompatibility with Swift 6 | Medium | High | Fork and patch if needed |
| Performance degradation with markdown rendering | Low | Medium | Benchmark and optimize |
| Terminal compatibility issues (iTerm2, Kitty, etc.) | High | Low | Graceful fallbacks |
| Clipboard tools not available on Linux | Medium | Low | Detect and warn user |

---

## References

- [OpenCLI Repository](https://github.com/openclirun/opencli)
- [Rainbow Library](https://github.com/onevcat/Rainbow)
- [ConsoleKit](https://github.com/vapor/console-kit)
- [swift-markdown](https://github.com/apple/swift-markdown)
- [Splash](https://github.com/JohnSundell/Splash)
- [iTerm2 Inline Images](https://iterm2.com/documentation-images.html)
- [Kitty Graphics Protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/)

---

**Status:** Planned  
**Next Steps:** Review with team, prioritize epics, create GitHub issues  
**Owner:** BA Team + CLI Engineering  
**Last Updated:** April 28, 2026
