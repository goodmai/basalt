# Epic 16 UI Components - Best Practices & Architecture

## 🏗️ Architecture Overview

Epic 16 introduces a comprehensive terminal UI system built with Swift 6 concurrency in mind. All components follow best practices for thread safety and modern Swift design.

## 📦 Components

### 1. TerminalUI (Epic 16.1)

**Purpose**: Colored and styled terminal output

**Thread Safety**: ✅ Full Swift 6 compliance
- Uses `Actor` for configuration management
- Legacy `nonisolated(unsafe)` accessor for backward compatibility
- Environment variable detection (NO_COLOR, FORCE_COLOR)

**Usage**:
```swift
// Synchronous (legacy, but safe)
print(TerminalUI.success("✅ Operation complete"))
print(TerminalUI.error("❌ Failed"))

// Actor-based (recommended for new code)
let config = TerminalUI.Configuration.shared
await config.setColorsEnabled(true)
```

**Design Decisions**:
- ✅ Enum for namespace (no instances needed)
- ✅ Actor for thread-safe configuration
- ✅ Rainbow library for ANSI colors
- ✅ Automatic TTY detection
- ✅ Standards compliance (NO_COLOR)

### 2. TableRenderer (Epic 16.2)

**Purpose**: ASCII/Unicode table rendering

**Thread Safety**: ✅ Fully `Sendable`
- Pure value types
- No mutable state
- Safe for concurrent use

**Usage**:
```swift
let headers = ["Model", "Size", "TPS"]
let rows = [
    ["Qwen3.5-4B", "2.3 GB", "92"],
    ["Qwen3.6-27B", "14.5 GB", "11"]
]

let table = TableRenderer.render(
    headers: headers,
    rows: rows,
    style: .unicode
)
```

**Design Decisions**:
- ✅ Pure Swift implementation (no external deps)
- ✅ 3 styles: Unicode, ASCII, Minimal
- ✅ Builder pattern for complex tables
- ✅ Auto-sizing columns
- ✅ Column alignment support

### 3. MarkdownRenderer (Epic 16.3)

**Purpose**: Terminal-optimized markdown rendering

**Thread Safety**: ✅ Safe mutable visitor pattern
- Visitor is local to each render call
- No shared state
- `Sendable` conformance

**Usage**:
```swift
let markdown = """
# Title
**Bold** and *italic* text
```swift
func hello() { print("Hi") }
\```
"""

let output = MarkdownRenderer.render(markdown)
print(output)
```

**Design Decisions**:
- ✅ swift-markdown for parsing
- ✅ Splash for Swift syntax highlighting
- ✅ Custom highlighters for Python/JS/JSON/Bash
- ✅ Graceful degradation without colors
- ✅ Local mutable state in visitor

### 4. Spinner (Epic 16.8)

**Purpose**: Animated loading indicators

**Thread Safety**: ✅ Actor-based implementation
- All state managed by actor
- Safe concurrent access
- Proper cleanup in deinit

**Usage**:
```swift
let spinner = Spinner(style: .dots, message: "Loading...")
await spinner.start()

// Do work
try await someAsyncOperation()

await spinner.stop(finalMessage: "✅ Done")
```

**Design Decisions**:
- ✅ Actor for thread-safe state
- ✅ Modern async/await with Task.sleep
- ✅ No Timer/RunLoop complexity
- ✅ Automatic cursor management
- ✅ Proper cleanup on deinit

### 5. TerminalStatus (Epic 16.8)

**Purpose**: Rich loading states and context statistics

**Thread Safety**: ✅ Pure value types
- Immutable structs
- `Sendable` conformance
- Thread-safe by design

**Usage**:
```swift
// Loading states
let state = LoadingState.thinking
print(state.coloredDescription)

// Context stats
let stats = ContextStats(files: 3, mcpServers: 2, skills: 5)
print(stats.coloredDescription)
```

**Design Decisions**:
- ✅ Enums for type safety
- ✅ Structs for data
- ✅ CustomStringConvertible for nice printing
- ✅ Colored variants for terminal output

## 🎯 Swift 6 Concurrency Best Practices

### ✅ DO:
1. **Use actors for mutable state**
   ```swift
   public actor Configuration {
       private var _colorsEnabled: Bool
       public func setColorsEnabled(_ enabled: Bool) {
           _colorsEnabled = enabled
       }
   }
   ```

2. **Make value types Sendable**
   ```swift
   public struct ContextStats: Sendable { ... }
   public enum LoadingState: Sendable { ... }
   ```

3. **Use modern async/await**
   ```swift
   try await Task.sleep(for: .milliseconds(100))
   ```

4. **Proper cleanup in deinit**
   ```swift
   deinit {
       isRunning = false
       fputs("\u{1B}[?25h", stderr)
   }
   ```

### ❌ DON'T:
1. **Avoid `@unchecked Sendable`**
   - Use actors instead
   - Only when absolutely necessary (and document why)

2. **Don't use global mutable state**
   - Bad: `static var sharedState: SomeClass`
   - Good: `actor SharedState { ... }`

3. **Don't mix old concurrency (DispatchQueue) with new**
   - Bad: `DispatchQueue.global().async { ... }`
   - Good: `Task { await ... }`

4. **Don't ignore Sendable warnings**
   - Fix root cause, don't suppress

## 🧪 Testing Best Practices

### Unit Tests:
```swift
@Suite("Feature Tests")
struct FeatureTests {
    @Test("Description of what we're testing")
    func testSomething() async {
        // Arrange
        let sut = SystemUnderTest()
        
        // Act
        let result = await sut.doSomething()
        
        // Assert
        #expect(result == expectedValue)
    }
}
```

### Integration Tests:
```swift
@Test("Integration with multiple components")
func testIntegration() async {
    let spinner = Spinner()
    await spinner.start()
    
    let markdown = MarkdownRenderer.render("# Test")
    
    await spinner.stop()
    
    #expect(!markdown.isEmpty)
}
```

## 📊 Performance Considerations

### TableRenderer:
- ✅ O(n) column width calculation
- ✅ Minimal string allocations
- ✅ Efficient for 100+ rows

### MarkdownRenderer:
- ✅ Single-pass visitor pattern
- ✅ Lazy evaluation where possible
- ✅ Efficient for large documents

### Spinner:
- ✅ Low CPU usage with Task.sleep
- ✅ No blocking RunLoop
- ✅ Proper cleanup

## 🔒 Security Considerations

### Input Validation:
```swift
// TableRenderer handles mismatched rows
let rows = [
    ["1", "2"],     // Missing column
    ["3", "4", "5"] // Full row
]
// Safe: auto-fills missing columns
```

### ANSI Injection Protection:
```swift
// TerminalUI sanitizes input
let userInput = "\u{1B}[31mHACK"
let safe = TerminalUI.code(userInput)
// Safe: wrapped in our own ANSI codes
```

### Environment Variables:
```swift
// NO_COLOR respected for accessibility
if ProcessInfo.processInfo.environment["NO_COLOR"] != nil {
    // Disable all colors
}
```

## 📚 Further Reading

- [Swift Concurrency](https://docs.swift.org/swift-book/LanguageGuide/Concurrency.html)
- [Sendable Types](https://github.com/apple/swift-evolution/blob/main/proposals/0302-concurrent-value-and-concurrent-closures.md)
- [NO_COLOR Standard](https://no-color.org/)
- [ANSI Escape Codes](https://en.wikipedia.org/wiki/ANSI_escape_code)

## 🎓 Migration Guide

### From old code to Epic 16:

**Before**:
```swift
print("Success: \(message)")
```

**After**:
```swift
print(TerminalUI.success("Success: \(message)"))
```

**Before**:
```swift
// Manual table formatting
print("Model        | Size")
print("-------------|------")
print("Qwen3.5-4B   | 2.3GB")
```

**After**:
```swift
let table = TableRenderer.render(
    headers: ["Model", "Size"],
    rows: [["Qwen3.5-4B", "2.3 GB"]],
    style: .unicode
)
print(table)
```

## ✅ Checklist for New UI Components

When adding new UI components:

- [ ] Mark as `Sendable` where appropriate
- [ ] Use actors for mutable state
- [ ] Add comprehensive unit tests
- [ ] Document thread safety guarantees
- [ ] Support NO_COLOR environment variable
- [ ] Handle edge cases gracefully
- [ ] Add usage examples
- [ ] Update this document

---

**Prepared by**: Senior Engineering Team
**Date**: April 28, 2026
**Epic**: 16 (CLI Enhancements)
**Version**: 0.6.0
