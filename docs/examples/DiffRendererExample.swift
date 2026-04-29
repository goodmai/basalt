import Foundation
import GemCore

/// Example demonstrating Epic 16.4: Code Diff Viewer
/// Shows how to render unified diffs with inline and side-by-side modes

print("═══════════════════════════════════════════════════════════")
print("Epic 16.4: Code Diff Viewer Examples")
print("═══════════════════════════════════════════════════════════\n")

// MARK: - Example 1: Simple Inline Diff

print("Example 1: Simple Inline Diff (with colors)")
print("─────────────────────────────────────────────────────────\n")

let simpleDiff = """
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

// Enable colors for demonstration
setenv("FORCE_COLOR", "1", 1)

let rendered1 = DiffRenderer.render(simpleDiff, mode: .inline, colorize: true)
print(rendered1)

// MARK: - Example 2: Plain Text (No Colors)

print("\nExample 2: Same Diff Without Colors (plain mode)")
print("─────────────────────────────────────────────────────────\n")

let rendered2 = DiffRenderer.render(simpleDiff, mode: .inline, colorize: false)
print(rendered2)

// MARK: - Example 3: Side-by-Side Mode

print("\nExample 3: Side-by-Side Diff (split view)")
print("─────────────────────────────────────────────────────────\n")

let sideBySideDiff = """
--- a/calculator.swift
+++ b/calculator.swift
@@ -1,5 +1,3 @@
 func add(a: Int, b: Int) -> Int {
-    let result = a + b
-    print("Result: \\(result)")
-    return result
+    return a + b
 }
"""

let rendered3 = DiffRenderer.render(sideBySideDiff, mode: .sideBySide, colorize: true)
print(rendered3)

// MARK: - Example 4: Real-World Git Diff

print("\nExample 4: Real-World Git Diff (with git headers)")
print("─────────────────────────────────────────────────────────\n")

let gitDiff = """
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

let rendered4 = DiffRenderer.render(gitDiff, mode: .inline, colorize: true)
print(rendered4)

// MARK: - Example 5: Multiple Hunks

print("\nExample 5: Diff with Multiple Hunks")
print("─────────────────────────────────────────────────────────\n")

let multiHunkDiff = """
--- a/models.swift
+++ b/models.swift
@@ -10,3 +10,3 @@
 struct User {
-    let id: Int
+    let id: UUID
     let name: String
@@ -50,5 +50,3 @@
 func fetchUser() -> User? {
-    let id = generateID()
-    let name = getName()
-    return User(id: id, name: name)
+    return User(id: UUID(), name: "Unknown")
 }
"""

let rendered5 = DiffRenderer.render(multiHunkDiff, mode: .inline, colorize: true)
print(rendered5)

// MARK: - Example 6: Clipboard Integration (macOS)

print("\nExample 6: Copy Diff to Clipboard (macOS only)")
print("─────────────────────────────────────────────────────────\n")

#if os(macOS)
Task {
    do {
        try await DiffRenderer.copyDiff(simpleDiff)
        print("✓ Diff copied to clipboard successfully!")
        print("  You can now paste it anywhere with ⌘V")
        
        // Verify by pasting
        let clipboard = ClipboardManager()
        let pasted = try await clipboard.paste()
        print("\n📋 Clipboard contents:")
        print(pasted.prefix(200) + "...")
    } catch {
        print("✗ Failed to copy to clipboard: \(error)")
    }
}
#else
print("⚠ Clipboard integration is macOS-only")
#endif

// MARK: - Example 7: Parser API

print("\n\nExample 7: Using DiffParser Directly")
print("─────────────────────────────────────────────────────────\n")

let hunks = DiffParser.parseUnifiedDiff(simpleDiff)
print("Parsed \(hunks.count) hunk(s):")

for (index, hunk) in hunks.enumerated() {
    print("\nHunk #\(index + 1):")
    print("  Old file: \(hunk.oldFile ?? "N/A")")
    print("  New file: \(hunk.newFile ?? "N/A")")
    print("  Header: \(hunk.header)")
    print("  Lines: \(hunk.lines.count)")
    
    let additions = hunk.lines.filter { $0.type == .addition }.count
    let deletions = hunk.lines.filter { $0.type == .deletion }.count
    let context = hunk.lines.filter { $0.type == .context }.count
    
    print("    +\(additions) additions")
    print("    -\(deletions) deletions")
    print("     \(context) context lines")
}

print("\n═══════════════════════════════════════════════════════════")
print("All examples completed successfully! ✓")
print("═══════════════════════════════════════════════════════════")
