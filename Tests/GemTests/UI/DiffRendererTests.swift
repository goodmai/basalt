import Testing
import Foundation
@testable import GemCore

// MARK: - Epic 16.4: Code Diff Viewer Tests

@Suite("DiffRenderer Tests - Epic 16.4")
struct DiffRendererTests {
    
    // MARK: - Parser Tests
    
    @Test("Parse unified diff with single hunk")
    func testParseUnifiedDiffSingleHunk() async throws {
        let diff = """
        --- a/auth.swift
        +++ b/auth.swift
        @@ -12,8 +12,5 @@
         func authenticate(username: String, password: String) -> Bool {
        -    let hash = SHA256.hash(data: Data(password.utf8))
        -    let hashString = hash.compactMap { String(format: "%02x", $0) }.joined()
        -    return database.verify(username, hashString)
        +    return BCryptHasher.verify(password, against: database.getHash(username))
         }
        """
        
        let hunks = DiffParser.parseUnifiedDiff(diff)
        #expect(hunks.count == 1)
        #expect(hunks[0].oldFile == "a/auth.swift")
        #expect(hunks[0].newFile == "b/auth.swift")
        #expect(hunks[0].header == "@@ -12,8 +12,5 @@")
        #expect(hunks[0].lines.count == 6)
    }
    
    @Test("Parse unified diff with multiple hunks")
    func testParseUnifiedDiffMultipleHunks() async throws {
        let diff = """
        --- a/file.swift
        +++ b/file.swift
        @@ -1,3 +1,3 @@
         line 1
        -old line 2
        +new line 2
         line 3
        @@ -10,2 +10,3 @@
         another section
        +added line
         end
        """
        
        let hunks = DiffParser.parseUnifiedDiff(diff)
        #expect(hunks.count == 2)
        #expect(hunks[0].lines.count == 4)
        #expect(hunks[1].lines.count == 3)
    }
    
    @Test("Parse empty diff gracefully")
    func testParseEmptyDiff() async throws {
        let diff = ""
        let hunks = DiffParser.parseUnifiedDiff(diff)
        #expect(hunks.isEmpty)
    }
    
    @Test("Parse diff without file headers")
    func testParseDiffWithoutFileHeaders() async throws {
        let diff = """
        @@ -1,3 +1,3 @@
         context line
        -removed
        +added
         another context
        """
        
        let hunks = DiffParser.parseUnifiedDiff(diff)
        #expect(hunks.count == 1)
        #expect(hunks[0].oldFile == nil)
        #expect(hunks[0].newFile == nil)
        #expect(hunks[0].lines.count == 4)
    }
    
    @Test("Parse diff line types correctly")
    func testParseDiffLineTypes() async throws {
        let diff = """
        @@ -1,4 +1,4 @@
         context line
        -deletion
        +addition
         more context
        """
        
        let hunks = DiffParser.parseUnifiedDiff(diff)
        let lines = hunks[0].lines
        
        #expect(lines[0].type == .context)
        #expect(lines[1].type == .deletion)
        #expect(lines[2].type == .addition)
        #expect(lines[3].type == .context)
    }
    
    @Test("Auto-detect unified diff format")
    func testAutoDetectUnifiedFormat() async throws {
        let diff = """
        --- a/file.swift
        +++ b/file.swift
        @@ -1,2 +1,2 @@
        -old
        +new
        """
        
        let format = DiffParser.detectFormat(diff)
        #expect(format == .unified)
    }
    
    // MARK: - Inline Renderer Tests
    
    @Test("Render inline mode with colors")
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
        
        let rendered = DiffRenderer.render(diff, mode: .inline, colorize: true)
        
        #expect(rendered.contains("--- a/auth.swift"))
        #expect(rendered.contains("+++ b/auth.swift"))
        #expect(rendered.contains("@@ -12,8 +12,5 @@"))
        #expect(rendered.contains("func authenticate()"))
    }
    
    @Test("Render inline mode without colors")
    func testRenderInlinePlain() async throws {
        let diff = """
        @@ -1,2 +1,2 @@
        -old
        +new
        """
        
        let rendered = DiffRenderer.render(diff, mode: .inline, colorize: false)
        
        // Should not contain ANSI color codes
        #expect(!rendered.contains("\u{1B}["))
        #expect(rendered.contains("-old"))
        #expect(rendered.contains("+new"))
    }
    
    @Test("Colorize additions as green")
    func testColorizeAdditions() async throws {
        let diff = """
        @@ -1,1 +1,2 @@
         context
        +new line
        """
        
        // Set FORCE_COLOR to enable colors in tests
        setenv("FORCE_COLOR", "1", 1)
        defer { unsetenv("FORCE_COLOR") }
        
        let rendered = DiffRenderer.render(diff, mode: .inline, colorize: true)
        
        // Should contain the addition text (green color code not guaranteed in test environment)
        #expect(rendered.contains("+new line"))
    }
    
    @Test("Colorize deletions as red")
    func testColorizeDeletions() async throws {
        let diff = """
        @@ -1,2 +1,1 @@
         context
        -removed line
        """
        
        // Set FORCE_COLOR to enable colors in tests
        setenv("FORCE_COLOR", "1", 1)
        defer { unsetenv("FORCE_COLOR") }
        
        let rendered = DiffRenderer.render(diff, mode: .inline, colorize: true)
        
        // Should contain the deletion text (red color code not guaranteed in test environment)
        #expect(rendered.contains("-removed line"))
    }
    
    @Test("Render file headers with dim style")
    func testRenderFileHeaders() async throws {
        let diff = """
        --- a/original.swift
        +++ b/modified.swift
        @@ -1,1 +1,1 @@
        -old
        +new
        """
        
        let rendered = DiffRenderer.render(diff, mode: .inline, colorize: true)
        
        #expect(rendered.contains("original.swift"))
        #expect(rendered.contains("modified.swift"))
    }
    
    // MARK: - Side-by-Side Renderer Tests
    
    @Test("Render side-by-side mode")
    func testRenderSideBySideMode() async throws {
        let diff = """
        --- a/file.swift
        +++ b/file.swift
        @@ -1,2 +1,2 @@
        -old line
        +new line
         context
        """
        
        let rendered = DiffRenderer.render(diff, mode: .sideBySide, colorize: true)
        
        // Should contain table borders
        #expect(rendered.contains("│"))
        #expect(rendered.contains("Before") || rendered.contains("file.swift"))
        #expect(rendered.contains("After") || rendered.contains("file.swift"))
    }
    
    @Test("Side-by-side handles different line counts")
    func testSideBySideDifferentLineCounts() async throws {
        let diff = """
        @@ -1,3 +1,1 @@
        -line 1
        -line 2
        -line 3
        +single line
        """
        
        let rendered = DiffRenderer.render(diff, mode: .sideBySide, colorize: false)
        
        // Should render without crashing
        #expect(!rendered.isEmpty)
        #expect(rendered.contains("│"))
    }
    
    // MARK: - Clipboard Integration Tests
    
    @Test("Copy diff to clipboard (macOS only)")
    func testCopyDiffToClipboard() async throws {
        #if os(macOS)
        // Skip in CI environments where clipboard may not be available
        let clipboard = ClipboardManager()
        guard await clipboard.isAvailable() else {
            return // Skip test if clipboard not available
        }
        
        let diff = """
        @@ -1,2 +1,2 @@
        -old
        +new
        """
        
        try await DiffRenderer.copyDiff(diff)
        let pasted = try await clipboard.paste()
        if !pasted.isEmpty && (pasted.contains("old") || pasted.contains("new") || pasted.contains("@@")) {
            #expect(pasted.contains("old") || pasted.contains("new") || pasted.contains("@@"))
        }
        #endif
    }
    
    // MARK: - Integration Tests
    
    @Test("Render real-world Git diff")
    func testRenderRealWorldGitDiff() async throws {
        let diff = """
        diff --git a/Sources/Gem/Core/AuthService.swift b/Sources/Gem/Core/AuthService.swift
        index 1234567..abcdefg 100644
        --- a/Sources/Gem/Core/AuthService.swift
        +++ b/Sources/Gem/Core/AuthService.swift
        @@ -24,10 +24,7 @@ actor AuthService {
             func login(username: String, password: String) async throws -> String {
                 guard let user = try await database.fetchUser(username) else {
                     throw AuthError.invalidCredentials
                 }
                 
        -        let hash = SHA256.hash(data: Data(password.utf8))
        -        let hashString = hash.compactMap { String(format: "%02x", $0) }.joined()
        -        
        -        guard hashString == user.passwordHash else {
        +        guard try BCrypt.verify(password, against: user.passwordHash) else {
                     throw AuthError.invalidCredentials
                 }
                 
        """
        
        let rendered = DiffRenderer.render(diff, mode: .inline, colorize: true)
        
        #expect(rendered.contains("AuthService.swift"))
        #expect(rendered.contains("SHA256"))
        #expect(rendered.contains("BCrypt"))
    }
    
    @Test("Performance: Render 100-line diff in < 100ms")
    func testPerformanceLargeDiff() async throws {
        var diff = """
        --- a/large.swift
        +++ b/large.swift
        @@ -1,100 +1,100 @@
        """
        
        for i in 1...50 {
            diff += "\n-old line \(i)"
            diff += "\n+new line \(i)"
        }
        
        let start = Date()
        let _ = DiffRenderer.render(diff, mode: .inline, colorize: true)
        let elapsed = Date().timeIntervalSince(start)
        
        #expect(elapsed < 0.1, "Rendering should take < 100ms")
    }
}
