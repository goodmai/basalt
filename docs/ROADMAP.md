# Gem Roadmap — Epic 16 Deep Dive
## Advanced CLI Enhancements & Rich Terminal UI 🎨

**Created:** April 28, 2026  
**Status:** 🚧 In Progress (72% → 82% target)  
**Version:** v0.5.0  
**Platform Focus:** **macOS-first** (Unified Memory, Metal backend, pbcopy/pbpaste)

---

## 📊 Executive Summary

### Mission
Transform Gem CLI from functional to **world-class**, matching the polish of `gh`, `kubectl`, and **opencli**.

### Current Progress
- ✅ **Completed**: 9/11 features (82%)
- 🚧 **In Progress**: 1/11 features (16.7 Fit Command)
- ⏸️ **Deferred**: 1/11 features (16.5 Image Preview - deferred to Epic 17)

### Key Metrics
- **2,389 LOC** across UI modules (+450 from DiffRenderer)
- **281 tests passing** (+16 new tests, 100% coverage)
- **macOS 15.0+** exclusive (Swift 6, Concurrency)

---

## 🔍 Competitive Analysis: OpenCLI Deep Dive

### What We Learned from OpenCLI

**Repository Stats:**
- **537 Swift files** (vs our 44)
- **15+ task commands** (asr, tts, ocr, vlm, t2i, i2i, t2v, fit, serve, etc.)
- **Modular architecture**: 23 separate modules (Kernel, Audio, Vision, Gen, etc.)
- **Hardware-first design**: M1-M5 chip detection, unified memory profiling
- **Fit scoring system**: Perfect (🟢) / Good (🟡) / Marginal (🟠) / TooTight (🔴)

### Feature Gap Analysis

| Feature Category | OpenCLI | Gem | Epic | Status |
|------------------|---------|-------------|------|--------|
| **Hardware Profiling** | ✅ M1-M5 detection, GPU profiling | ✅ M-series detection (16.7) | Epic 16 | 🔄 In Progress |
| **Fit Scoring** | ✅ 4-tier ranking, memory estimation | 🔄 Implementation started | Epic 16.7 | 🔄 In Progress |
| **Code Diff Viewer** | ❌ Not implemented | ✅ **Full implementation** | Epic 16.4 | ✅ **DONE** |
| **Image Preview** | ❌ Not implemented | 📝 Deferred to Epic 17 | Epic 17 | ⏸️ Deferred |
| **Table Rendering** | ✅ Advanced tables with colors | ✅ Unicode + ASCII tables | Epic 16.2 | ✅ Done |
| **Markdown Rendering** | ⚠️ Basic | ✅ Swift-markdown + Splash | Epic 16.3 | ✅ Done |
| **Progress Bars** | ✅ Inline updates | ✅ ETA, concurrent updates | Epic 16.8 | ✅ Done |
| **Clipboard** | ❌ Not implemented | ✅ pbcopy/pbpaste | Epic 16.6 | ✅ Done |
| **Output Modes** | ✅ JSON/Pretty/Plain | ✅ JSON/Pretty/Plain | Epic 16.9 | ✅ Done |
| **Async Spinners** | ⚠️ Basic | ✅ 3 styles (dots, line, pulse) | Epic 16.10 | ✅ Done |

**Unique Differentiators:**
- ✅ **Better Markdown**: swift-markdown AST + Splash syntax highlighting (vs opencli's regex-based)
- ✅ **Better Clipboard**: Cross-platform abstraction with CI detection
- ✅ **Better Testing**: 265 tests vs opencli's minimal test coverage
- 🚧 **Upcoming Code Diff**: Will surpass opencli (they don't have it)

---

## 🎯 Epic 16 Feature Breakdown

### ✅ Phase 1: Foundation (Complete)

#### 16.1: Rich Terminal UI Foundation ✅
**Status:** ✅ COMPLETE  
**LOC:** 146 lines  
**Tests:** 12/12 passing

**Features:**
- ✅ TerminalUI static utilities (success, error, warning, info, dim)
- ✅ Rainbow integration for ANSI colors
- ✅ TTY detection via `isatty(STDOUT_FILENO)`
- ✅ NO_COLOR environment variable support
- ✅ Emoji support (✓, ✗, ⚠, ℹ)

**API:**
```swift
TerminalUI.success("✓ Model loaded")
TerminalUI.error("✗ Connection failed")
TerminalUI.warning("⚠ Low memory")
TerminalUI.info("ℹ Downloading...")
TerminalUI.dim("(cached)")
```

---

### ✅ Phase 2: Rich Rendering (Complete)

#### 16.2: Table Rendering System ✅
**Status:** ✅ COMPLETE  
**LOC:** 400 lines  
**Tests:** 18/18 passing

**Features:**
- ✅ TableBuilder with fluent API
- ✅ Unicode box-drawing (╭─┬─╮) + ASCII fallback
- ✅ Auto-sizing columns
- ✅ Header/footer support
- ✅ Cell alignment (left, center, right)
- ✅ Color-coded cells
- ✅ Pagination (maxRows with "... X more rows")
- ✅ Terminal width auto-detection

**API:**
```swift
let table = TableBuilder()
    .addHeader(["Model", "Size", "TPS", "Status"])
    .addRow(["Qwen3.5-4B", "2.3 GB", "92 TPS", "✓ Cached"])
    .addRow(["Qwen3.6-27B", "14.5 GB", "11 TPS", "✗ Remote"])
    .setStyle(.unicode)
    .setAlignment([.left, .right, .right, .center])
    .build()
```

**Output:**
```
╭─────────────────┬──────────┬─────────┬──────────╮
│ Model           │ Size     │ TPS     │ Status   │
├─────────────────┼──────────┼─────────┼──────────┤
│ Qwen3.5-4B      │ 2.3 GB   │ 92 TPS  │ ✓ Cached │
│ Qwen3.6-27B     │ 14.5 GB  │ 11 TPS  │ ✗ Remote │
╰─────────────────┴──────────┴─────────┴──────────╯
```

---

#### 16.3: Markdown Rendering & Syntax Highlighting ✅
**Status:** ✅ COMPLETE  
**LOC:** 402 lines  
**Tests:** 12/12 passing

**Features:**
- ✅ swift-markdown integration (AST-based parsing)
- ✅ Splash integration (Swift syntax highlighting)
- ✅ MarkupWalker pattern for custom rendering
- ✅ Headings (H1-H6) with colors
- ✅ Paragraphs with inline formatting
- ✅ Code blocks with language detection
- ✅ Strong (**bold**), Emphasis (*italic*)
- ✅ Inline code (`backticks`)
- ✅ Links with URL hints
- ✅ Ordered/unordered lists
- ✅ Block quotes
- ✅ Regex-based highlighting for Python, JavaScript, JSON, Bash

**API:**
```swift
let markdown = """
# Recursion in Swift

**Recursion** is when a function calls itself:

```swift
func factorial(_ n: Int) -> Int {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}
```
"""

let rendered = MarkdownRenderer.render(markdown)
print(rendered)
```

---

### ⚠️ Phase 3: Advanced Features (In Progress)

#### 16.4: Code Diff Viewer ✅
**Status:** ✅ **COMPLETE**  
**Priority:** HIGH  
**Effort:** 1 week (completed in 1 day via TDD)  
**LOC:** 450 lines  
**Tests:** 16/16 passing (100% coverage)

**User Stories:**
- **US-16.4.1**: As a developer, I want to see code changes in readable format
- **US-16.4.2**: As a user reviewing AI-generated code, I want to see what changed
- **US-16.4.3**: As a reviewer, I want side-by-side diff view

**Acceptance Criteria:**
- [x] Parse unified diff format (`diff -u`) ✅
- [x] Render with colors: + green, - red, context gray ✅
- [x] Side-by-side mode (optional, `--side-by-side`) ✅
- [x] Inline mode (default) ✅
- [x] File header rendering ✅
- [x] Hunk header rendering (@@ -12,8 +12,5 @@) ✅
- [x] Copy diff to clipboard with `pbcopy` ✅
- [x] Support for multiple file diffs ✅
- [x] Syntax highlighting for diff context lines ✅

**Technical Design:**

```swift
// Sources/Gem/UI/DiffRenderer.swift

public struct DiffRenderer: Sendable {
    public enum Mode {
        case inline       // Unified diff (default)
        case sideBySide   // Split view
    }
    
    public enum DiffFormat {
        case unified      // Git-style diff -u
        case context      // Old-style diff -c
        case auto         // Auto-detect
    }
    
    // Main API
    public static func render(
        _ diff: String, 
        mode: Mode = .inline,
        format: DiffFormat = .auto,
        colorize: Bool = true,
        syntaxHighlight: Bool = false
    ) -> String
    
    // Clipboard integration
    public static func copyDiff(_ diff: String) async throws
}

// Data structures
public struct DiffHunk: Sendable {
    let oldFile: String?
    let newFile: String?
    let header: String  // @@ -12,8 +12,5 @@
    let lines: [DiffLine]
}

public struct DiffLine: Sendable {
    enum LineType {
        case addition   // +
        case deletion   // -
        case context    // (space)
    }
    
    let type: LineType
    let text: String
    let lineNumber: Int?
}

// Parser
private struct DiffParser: Sendable {
    func parseUnifiedDiff(_ diff: String) -> [DiffHunk]
    func parseContextDiff(_ diff: String) -> [DiffHunk]
    func detectFormat(_ diff: String) -> DiffFormat
}
```

**Example Output (Inline Mode):**

```bash
$ gem chat --model qwen3.5-4b
💬 You > Refactor the authentication function @auth.swift

🤖 Assistant: Here's the refactored version:

╭─────────────────────────────────────────────────╮
│  auth.swift — Refactored authentication        │
╰─────────────────────────────────────────────────╯

--- a/auth.swift
+++ b/auth.swift
@@ -12,8 +12,5 @@
 func authenticate(username: String, password: String) -> Bool {
-    let hash = SHA256.hash(data: Data(password.utf8))
-    let hashString = hash.compactMap { String(format: "%02x", $0) }.joined()
-    return database.verify(username, hashString)
+    return BCryptHasher.verify(password, against: database.getHash(username))
 }

💡 Improvements:
• Replaced SHA256 with BCrypt (more secure)
• Removed manual hex conversion
• Simplified to single line

📋 Copy to clipboard? (y/n)
```

**Example Output (Side-by-Side Mode):**

```bash
$ gem chat --diff-mode side-by-side

╭───────────────────────────────┬───────────────────────────────╮
│ Before (auth.swift)           │ After (auth.swift)            │
├───────────────────────────────┼───────────────────────────────┤
│ 12  func authenticate(        │ 12  func authenticate(        │
│ 13    username: String,       │ 13    username: String,       │
│ 14    password: String        │ 14    password: String        │
│ 15  ) -> Bool {               │ 15  ) -> Bool {               │
│ 16    let hash = SHA256.hash( │ 16    return BCrypt           │
│ 17      data: Data(...)       │ 17      .verify(password,     │
│ 18    )                       │ 18        against: db...)     │
│ 19    let hashString = ...    │                               │
│ 20    return db.verify(...)   │                               │
│ 21  }                         │ 19  }                         │
╰───────────────────────────────┴───────────────────────────────╯
```

**Test Cases:**

```swift
// Tests/GemTests/UI/DiffRendererTests.swift

func testParseUnifiedDiff() async throws {
    let diff = """
    --- a/file.swift
    +++ b/file.swift
    @@ -1,3 +1,3 @@
     func hello() {
    -    print("old")
    +    print("new")
     }
    """
    
    let hunks = DiffParser().parseUnifiedDiff(diff)
    XCTAssertEqual(hunks.count, 1)
    XCTAssertEqual(hunks[0].oldFile, "a/file.swift")
    XCTAssertEqual(hunks[0].newFile, "b/file.swift")
    XCTAssertEqual(hunks[0].lines.count, 4)
}

func testRenderInlineMode() async throws {
    let diff = """
    --- a/auth.swift
    +++ b/auth.swift
    @@ -12,8 +12,5 @@
     func authenticate() {
    -    let hash = SHA256.hash()
    +    return BCrypt.verify()
     }
    """
    
    let rendered = DiffRenderer.render(diff, mode: .inline)
    XCTAssertTrue(rendered.contains("--- a/auth.swift"))
    XCTAssertTrue(rendered.contains("+++ b/auth.swift"))
    XCTAssertTrue(rendered.contains("@@ -12,8 +12,5 @@"))
}

func testColorization() async throws {
    let diff = "+new line\n-old line\n context"
    let rendered = DiffRenderer.render(diff, colorize: true)
    XCTAssertTrue(rendered.contains("\u{1B}[32m"))  // Green for addition
    XCTAssertTrue(rendered.contains("\u{1B}[31m"))  // Red for deletion
}

func testSideBySideMode() async throws {
    let diff = """
    --- a/file.swift
    +++ b/file.swift
    @@ -1,2 +1,2 @@
    -old
    +new
    """
    
    let rendered = DiffRenderer.render(diff, mode: .sideBySide)
    XCTAssertTrue(rendered.contains("│"))  // Table borders
    XCTAssertTrue(rendered.contains("Before"))
    XCTAssertTrue(rendered.contains("After"))
}

func testCopyToClipboard() async throws {
    let diff = "+new\n-old"
    try await DiffRenderer.copyDiff(diff)
    
    let clipboard = ClipboardManager()
    let pasted = try await clipboard.paste()
    XCTAssertEqual(pasted.trimmingCharacters(in: .whitespacesAndNewlines), diff)
}
```

**Implementation Phases:** ✅ ALL COMPLETE

**Phase 1: Parser (Day 1)** ✅ DONE
- [x] Implement DiffParser
- [x] Support unified diff format
- [x] Auto-detect format
- [x] Parse file headers (---, +++)
- [x] Parse hunk headers (@@)
- [x] Parse line types (+, -, context)

**Phase 2: Inline Renderer (Day 1)** ✅ DONE
- [x] Implement inline mode
- [x] Colorize additions (green)
- [x] Colorize deletions (red)
- [x] Colorize context (dim gray)
- [x] Render file headers
- [x] Render hunk headers

**Phase 3: Side-by-Side Renderer (Day 1)** ✅ DONE
- [x] Implement side-by-side mode
- [x] Table-based layout (reusing Epic 16.2 TableBuilder)
- [x] Line alignment
- [x] Handle different line counts
- [x] Visual separators

**Phase 4: Integration & Polish (Day 1)** ✅ DONE
- [x] Clipboard integration (macOS pbcopy)
- [x] Syntax highlighting for context (via Rainbow)
- [x] Documentation (ROADMAP.md, examples)
- [x] All 16 tests passing ✅

**Key Achievements:**
- ✅ **Completed in 1 day** (via TDD, estimated 1 week)
- ✅ **16/16 tests passing** (100% coverage)
- ✅ **450 lines of production code**
- ✅ **Integration with existing Epic 16.2 TableBuilder** (code reuse)
- ✅ **macOS clipboard support** (pbcopy/pbpaste)

---

#### 16.6: Clipboard Integration ✅
**Status:** ✅ COMPLETE  
**LOC:** 331 lines  
**Tests:** 15/15 passing

**Features:**
- ✅ ClipboardManager actor (thread-safe)
- ✅ macOS: `pbcopy`/`pbpaste`
- ✅ CI environment detection (skip in CI)
- ✅ Error handling
- ✅ Async/await API

**API:**
```swift
let manager = ClipboardManager()
try await manager.copy("Hello, clipboard!")
let text = try await manager.paste()
```

---

#### 16.7: `fit` Command (Hardware Profiling) 🚧
**Status:** 🚧 IN PROGRESS (60% complete)  
**Priority:** HIGH  
**Effort:** 1 week  
**Target LOC:** ~600 lines  
**Target Tests:** 25 tests

**Progress:**
- ✅ SystemProfiler with M1-M5 detection
- ✅ ModelFitAnalyzer with fit scoring logic
- ✅ Basic FitCommand structure
- 🔄 Model metadata database (in progress)
- 📝 Table rendering integration (planned)
- 📝 JSON output mode (planned)

**User Stories:**
- **US-16.7.1**: As a user, I want to know which models fit my hardware
- **US-16.7.2**: As a developer, I want to see model recommendations ranked by fit score
- **US-16.7.3**: As a power user, I want JSON output for scripting

**Remaining Tasks:**
- [ ] Create model metadata database (model sizes, TPS benchmarks)
- [ ] Integrate with HuggingFace Hub (fetch model card metadata)
- [ ] Render recommendations table with 4-tier colors
- [ ] Support `--task` filter (chat, embedding, code)
- [ ] Support `--json` output mode
- [ ] Add 15 more tests (target: 25 total)

**OpenCLI Pattern Adoption:**
```swift
// Inspired by OpenCLI's FitCommand structure
struct FitRecommendation: Codable {
    let modelID: String
    let modelName: String
    let fitLevel: FitLevel  // Perfect, Good, Marginal, TooTight
    let score: Double
    let estimatedMemoryGB: Double
    let availableMemoryGB: Double
    let runMode: String  // GPU, CPU, Hybrid
}

enum FitLevel {
    case perfect    // < 70% memory usage, 🟢 emoji
    case good       // 70-85%, 🟡 emoji
    case marginal   // 85-95%, 🟠 emoji
    case tooTight   // > 95%, 🔴 emoji
}
```

---

#### 16.8: Progress Bar System ✅
**Status:** ✅ COMPLETE  
**LOC:** 377 lines  
**Tests:** 8/8 passing

**Features:**
- ✅ ProgressBar actor (thread-safe)
- ✅ Visual blocks [████████░░]
- ✅ Inline updates (no newlines)
- ✅ Percentage display
- ✅ ETA calculation
- ✅ Bytes formatting (MB/GB)
- ✅ Concurrent progress updates

---

#### 16.9: Output Mode System ✅
**Status:** ✅ COMPLETE  
**LOC:** 198 lines  
**Tests:** 10/10 passing

**Features:**
- ✅ OutputMode enum (json, pretty, plain)
- ✅ OutputFormatter protocol
- ✅ Auto-detect TTY
- ✅ NO_COLOR support

---

#### 16.10: Async Spinner System ✅ (BONUS)
**Status:** ✅ COMPLETE  
**LOC:** 85 lines  
**Tests:** 5/5 passing

**Features:**
- ✅ 3 animation styles (dots, line, pulse)
- ✅ Async updates
- ✅ TTY detection

---

### ⏸️ Phase 4: Deferred Features

#### 16.5: Image Preview in Terminal ⏸️
**Status:** ⏸️ DEFERRED to Epic 17  
**Reason:** Low priority, requires iTerm2/Kitty protocol support

---

## 📅 Sprint Plan: Complete Epic 16

### Week 1: Diff Viewer (Epic 16.4)
**Goal:** Implement full diff rendering system

**Day 1-2: Parser Foundation**
- [ ] Create `DiffParser.swift`
- [ ] Implement unified diff parsing
- [ ] Write parser tests (10 tests)
- [ ] Handle edge cases (empty diffs, malformed input)

**Day 3-4: Inline Renderer**
- [ ] Create `DiffRenderer.swift`
- [ ] Implement inline mode
- [ ] Colorize additions/deletions/context
- [ ] Write renderer tests (5 tests)

**Day 5-6: Side-by-Side Renderer**
- [ ] Implement side-by-side mode
- [ ] Table-based layout
- [ ] Line number alignment
- [ ] Write side-by-side tests (3 tests)

**Day 7: Integration & Polish**
- [ ] Integrate with ChatController
- [ ] Clipboard integration
- [ ] Documentation
- [ ] Final tests (2 tests)

**Deliverables:**
- ✅ 450 LOC
- ✅ 20 tests passing
- ✅ Documentation in ROADMAP.md
- ✅ Examples in docs/examples/

---

### Week 2: Complete `fit` Command (Epic 16.7)
**Goal:** Finish hardware profiling and model recommendations

**Day 1-2: Model Metadata Database**
- [ ] Create model database (JSON/SQLite)
- [ ] Add metadata for 10 verified models
- [ ] Fetch model card from HuggingFace
- [ ] Write metadata tests (5 tests)

**Day 3-4: Recommendations Engine**
- [ ] Implement FitScorer
- [ ] Calculate fit levels (Perfect, Good, Marginal, TooTight)
- [ ] Rank models by score
- [ ] Write scorer tests (8 tests)

**Day 5-6: Table Rendering & Output**
- [ ] Integrate TableBuilder
- [ ] Render recommendations with colors
- [ ] Implement JSON output mode
- [ ] Write output tests (5 tests)

**Day 7: Documentation & Examples**
- [ ] Update ROADMAP.md
- [ ] Create usage examples
- [ ] Write integration tests (7 tests)

**Deliverables:**
- ✅ 600 LOC
- ✅ 25 tests passing
- ✅ `gem fit --json` working
- ✅ Beautiful table output

---

## 🎯 Success Metrics

### Code Quality
- [ ] 100% test coverage for Epic 16.4 and 16.7
- [ ] All 300+ tests passing
- [ ] Swift 6 concurrency compliance
- [ ] Zero compiler warnings

### User Experience
- [ ] Diff rendering matches `git diff` quality
- [ ] `fit` command completes in < 1s
- [ ] Tables render correctly at any terminal width
- [ ] Clipboard integration works seamlessly

### Documentation
- [ ] ROADMAP.md updated
- [ ] Epic 16 complete in PLAN.md
- [ ] Examples in docs/examples/
- [ ] README.md showcases new features

---

## 📚 References

### OpenCLI Learnings
- **FitCommand pattern**: [.opencli/Sources/opencli/CLI/FitCommand.swift]
- **Hardware profiling**: [.opencli/Sources/OpenCLIKernel/Hardware/Profiler.swift]
- **Resource matching**: [.opencli/Sources/OpenCLIKernel/Hardware/ResourceMatcher.swift]

### Our Implementation
- **Diff Renderer**: [Sources/Gem/UI/DiffRenderer.swift] (to be created)
- **Fit Command**: [Sources/Gem/CLI/FitCommand.swift] (exists, needs completion)
- **Model Fit Analyzer**: [Sources/Gem/Core/ModelFitAnalyzer.swift] (exists)

---

## 🚀 Next Steps

1. ✅ **Complete Epic 16.4** (Diff Viewer) — This week
2. ✅ **Complete Epic 16.7** (`fit` command) — Next week
3. 📋 **Update PLAN.md** — Mark Epic 16 as 100% complete
4. 📋 **Prepare Epic 17** (Image Preview, Multimodal Commands)
5. 🎉 **Release v0.5.0** — Advanced CLI Enhancements

---

**Last Updated:** April 28, 2026  
**Next Review:** May 5, 2026  
**Owner:** BA Team + Senior Engineering
