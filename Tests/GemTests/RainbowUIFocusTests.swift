import Testing
import SwiftUI
import AppKit
@testable import GemCore

/// TDD Red→Green cycle for the "TextField not focused, can't type" bug.
///
/// Root cause (confirmed):
///   SwiftUI TextField inside NSHostingView → win.makeFirstResponder(tf) returns true
///   but win.firstResponder ends up being the NSHostingView, NOT the NSTextField.
///   The AppKit field editor is never attached.
///
/// Fix: Replace SwiftUI TextField with NativeInputField (NSTextField via NSViewRepresentable).
///   NSTextField gets added to the view hierarchy as a real AppKit control, and
///   makeFirstResponder works correctly through the normal AppKit responder chain.
@Suite("Rainbow UI Focus - Bugfix TDD")
@MainActor
struct RainbowUIInputBugTests {

    // MARK: - Helpers

    private func makeWindow(for state: RainbowUIState) -> (NSWindow, NSHostingView<RainbowChatView>) {
        let view = RainbowChatView(state: state)
        let hosting = NSHostingView(rootView: view)
        hosting.frame = NSRect(x: 0, y: 0, width: 800, height: 600)

        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
            styleMask: [.titled, .resizable],
            backing: .buffered,
            defer: false
        )
        win.contentView = hosting
        win.makeKeyAndOrderFront(nil)
        hosting.layout()
        return (win, hosting)
    }

    /// Recursively find any NSTextField (works for both SwiftUI and native wraps)
    private func findTextField(in view: NSView) -> NSTextField? {
        // Skip NSHostingView wrappers — we want the real control
        if let tf = view as? NSTextField { return tf }
        for sub in view.subviews {
            if let found = findTextField(in: sub) { return found }
        }
        return nil
    }

    private func findMTK(in view: NSView) -> RainbowMTKView? {
        if let m = view as? RainbowMTKView { return m }
        for sub in view.subviews { if let found = findMTK(in: sub) { return found } }
        return nil
    }

    // MARK: - Test 1: NativeInputField produces a real NSTextField in hierarchy

    @Test("NativeInputField produces a real NSTextField in the NSView hierarchy")
    func nativeTextFieldExistsInHierarchy() throws {
        let state = RainbowUIState()
        let (_, hosting) = makeWindow(for: state)

        let tf = findTextField(in: hosting)
        #expect(tf != nil, "NSTextField must be in the hierarchy — SwiftUI TextField doesn't appear as one")
    }

    // MARK: - Test 2: TextField has real size

    @Test("NativeInputField has non-zero frame (is clickable)")
    func nativeTextFieldHasNonZeroFrame() throws {
        let state = RainbowUIState()
        let (_, hosting) = makeWindow(for: state)
        let tf = try #require(findTextField(in: hosting))
        #expect(tf.frame.width > 100, "width=\(tf.frame.width) must be > 100")
        #expect(tf.frame.height > 10, "height=\(tf.frame.height) must be > 10")
    }

    // MARK: - Test 3: TextField is visible

    @Test("NativeInputField is visible and opaque")
    func nativeTextFieldIsVisible() throws {
        let state = RainbowUIState()
        let (_, hosting) = makeWindow(for: state)
        let tf = try #require(findTextField(in: hosting))
        #expect(!tf.isHidden, "NSTextField must not be hidden")
        #expect(tf.alphaValue > 0.0, "alphaValue=\(tf.alphaValue) must be > 0")
    }

    // MARK: - Test 4 (THE KEY BUG): makeFirstResponder succeeds AND sticks

    @Test("makeFirstResponder on NativeInputField actually sticks (field editor is attached)")
    func makeFirstResponderActuallySticks() throws {
        let state = RainbowUIState()
        let (win, hosting) = makeWindow(for: state)
        let tf = try #require(findTextField(in: hosting), "NSTextField must exist")

        let success = win.makeFirstResponder(tf)
        #expect(success, "makeFirstResponder must return true")

        // THE REAL CHECK: field editor (NSTextView) must be the firstResponder
        // and its nextResponder must be our NSTextField.
        // This is exactly what fails with SwiftUI TextField — the field editor never attaches.
        let fr = win.firstResponder
        let isFieldEditorForTF = (fr === tf) ||
                                  (fr?.nextResponder === tf) ||
                                  (win.fieldEditor(false, for: tf) != nil && fr is NSTextView)
        let frDescription = String(describing: type(of: fr))
        #expect(isFieldEditorForTF, "firstResponder class=\(frDescription) must be NSTextField or its field editor NSTextView")
    }

    // MARK: - Test 5: Typing English text is reflected in state.inputText

    @Test("Typing English text updates state.inputText via controlTextDidChange")
    func typingEnglishUpdatesState() throws {
        let state = RainbowUIState()
        let (win, hosting) = makeWindow(for: state)
        let tf = try #require(findTextField(in: hosting))

        win.makeFirstResponder(tf)

        // Simulate typing "hello" via the NSTextField delegate path
        tf.stringValue = "hello"
        // The NSTextField coordinator calls controlTextDidChange, but in tests
        // we trigger it manually via the notification
        NotificationCenter.default.post(
            name: NSControl.textDidChangeNotification,
            object: tf
        )
        hosting.layout()

        #expect(state.inputText == "hello",
                "state.inputText='\(state.inputText)' must be 'hello' after typing")
    }

    // MARK: - Test 6: Typing Russian (Cyrillic) text is handled correctly

    @Test("Typing Russian/Cyrillic text updates state.inputText correctly")
    func typingRussianUpdatesState() throws {
        let state = RainbowUIState()
        let (win, hosting) = makeWindow(for: state)
        let tf = try #require(findTextField(in: hosting))

        win.makeFirstResponder(tf)

        // Cyrillic text — simulates IME composition that NSTextField handles natively
        let russianText = "Привет мир"
        tf.stringValue = russianText
        NotificationCenter.default.post(
            name: NSControl.textDidChangeNotification,
            object: tf
        )
        hosting.layout()

        #expect(state.inputText == russianText,
                "state.inputText='\(state.inputText)' must be '\(russianText)' after typing Russian")
        // Verify character encoding is preserved
        #expect(state.inputText.unicodeScalars.allSatisfy { $0.value < 0x10FFFF },
                "All Cyrillic characters must be valid Unicode")
    }

    // MARK: - Test 8: Return key calls onSubmit and clears the field

    @Test("Return key triggers onSubmit and clears input field")
    func returnKeySubmitsMessage() throws {
        let state = RainbowUIState()
        let (win, hosting) = makeWindow(for: state)
        let tf = try #require(findTextField(in: hosting))

        win.makeFirstResponder(tf)
        tf.stringValue = "test message"
        NotificationCenter.default.post(name: NSControl.textDidChangeNotification, object: tf)
        hosting.layout()

        if let coordinator = tf.delegate as? NativeInputField.Coordinator {
            let handled = coordinator.control(tf, textView: NSTextView(),
                                              doCommandBy: #selector(NSResponder.insertNewline(_:)))
            #expect(handled, "Coordinator must handle insertNewline")
        }

        #expect(state.messages.count == 2, "Must have 2 messages after submit (user + system error), got \(state.messages.count)")
        #expect(state.inputText == "", "inputText must be cleared after submit")
    }

    // MARK: - Test 8.5: updateNSView does not break cursor position (RED test for ghosting/duplication)
    
    @Test("updateNSView does not forcefully overwrite fieldEditor during active typing")
    func updateNSViewPreservesCursorPosition() throws {
        let state = RainbowUIState()
        let (win, hosting) = makeWindow(for: state)
        let tf = try #require(findTextField(in: hosting))

        win.makeFirstResponder(tf)
        
        let fieldEditor = try #require(win.fieldEditor(false, for: tf) as? NSTextView, "Field editor must be active")
        
        // User types "п"
        fieldEditor.string = "п"
        // Move cursor to the end (index 1)
        fieldEditor.setSelectedRange(NSRange(location: 1, length: 0))
        
        // This triggers controlTextDidChange -> state.inputText = "п"
        NotificationCenter.default.post(name: NSControl.textDidChangeNotification, object: tf)
        
        // Let SwiftUI evaluate the state change and call updateNSView
        // In AppKit testing, we must spin the runloop to force SwiftUI to apply state changes to NSViewRepresentable
        RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.1))
        
        // The bug: buggy updateNSView forcefully does `fieldEditor.string = text`.
        // This resets the cursor selection to (0, 0), which breaks macOS IME composition and causes duplication.
        let selection = fieldEditor.selectedRange()
        #expect(selection.location == 1, "Cursor MUST remain at index 1. If it reset to 0, updateNSView is breaking IME composition!")
    }

    // MARK: - Test 9: MTKView does NOT accept first responder

    @Test("RainbowMTKView.acceptsFirstResponder is false (doesn't steal keyboard)")
    func mtkViewDoesNotStealFocus() throws {
        let state = RainbowUIState()
        let (_, hosting) = makeWindow(for: state)
        let mtk = try #require(findMTK(in: hosting), "RainbowMTKView must be in hierarchy")
        #expect(!mtk.acceptsFirstResponder,
                "MTKView.acceptsFirstResponder must be false to avoid stealing keyboard focus")
    }
}

// MARK: - Inference generation bug tests

/// Bug: After submitting a message, state transitions to .processing but no generation happens.
/// Root cause: Task {} inherits @MainActor context → calling generateStream (actor-isolated)
/// from @MainActor blocks the main thread while waiting for the actor, causing a deadlock
/// in the runloop. Fix: use Task.detached so inference runs off the main thread.
@Suite("Rainbow UI Generation Flow - TDD")
@MainActor
struct RainbowUIGenerationTests {

    // MARK: - Test 9: Submitting a message transitions through processing→streaming→finished

    @Test("submitMessage drives state through processing→streaming→finished")
    func submitMessageDrivesStateTransitions() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 512)
        try await orchestrator.loadModel(path: "/fake/path")

        let state = RainbowUIState()
        let view = RainbowChatView(state: state, orchestrator: orchestrator, maxTokens: 512)

        // Verify initial state
        #expect(state.currentMode == .idle)
        #expect(state.messages.isEmpty)

        // Build the hosting view so the view is initialized (not strictly necessary here)
        _ = NSHostingView(rootView: view)

        // Directly call submit via state manipulation (simulates what NativeInputField does)
        state.inputText = "привет"

        // We need access to submitMessage — it's private, so we go through the coordinator
        let hosting = NSHostingView(rootView: view)
        hosting.frame = NSRect(x: 0, y: 0, width: 800, height: 600)
        hosting.layout()

        // Find the NSTextField and trigger submit via the coordinator
        func findTF(_ v: NSView) -> NSTextField? {
            if let t = v as? NSTextField { return t }
            for s in v.subviews { if let f = findTF(s) { return f } }
            return nil
        }
        let tf = try #require(findTF(hosting), "Must find NSTextField")
        tf.stringValue = "привет"
        NotificationCenter.default.post(name: NSControl.textDidChangeNotification, object: tf)

        // Simulate Return key
        if let coordinator = tf.delegate as? NativeInputField.Coordinator {
            _ = coordinator.control(tf, textView: NSTextView(), doCommandBy: #selector(NSResponder.insertNewline(_:)))
        }

        // At this point state should immediately be .processing (sync part)
        #expect(state.currentMode == .processing || state.currentMode == .streaming,
                "After submit state must be processing or streaming, got \(state.currentMode)")

        // Give the detached task time to complete
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5s

        #expect(state.messages.count >= 2,
                "Must have user + assistant messages, got \(state.messages.count)")
        #expect(state.messages.last?.role == .assistant,
                "Last message must be from assistant")
        #expect(!(state.messages.last?.text.isEmpty ?? true),
                "Assistant message must not be empty")

        let finalMode = state.currentMode
        #expect(finalMode == .finished || finalMode == .streaming,
                "Mode must be finished or still streaming, got \(finalMode)")
    }

    // MARK: - Test 10: inputText is cleared after submit

    @Test("inputText is cleared after submit (no ghost text)")
    func inputTextClearedAfterSubmit() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 512)
        try await orchestrator.loadModel(path: "/fake/path")

        let state = RainbowUIState()
        let view = RainbowChatView(state: state, orchestrator: orchestrator, maxTokens: 512)
        let hosting = NSHostingView(rootView: view)
        hosting.frame = NSRect(x: 0, y: 0, width: 800, height: 600)
        hosting.layout()

        func findTF(_ v: NSView) -> NSTextField? {
            if let t = v as? NSTextField { return t }
            for s in v.subviews { if let f = findTF(s) { return f } }
            return nil
        }
        let tf = try #require(findTF(hosting))
        tf.stringValue = "test input"
        NotificationCenter.default.post(name: NSControl.textDidChangeNotification, object: tf)

        if let coordinator = tf.delegate as? NativeInputField.Coordinator {
            _ = coordinator.control(tf, textView: NSTextView(), doCommandBy: #selector(NSResponder.insertNewline(_:)))
        }

        // inputText must be cleared immediately (sync, before async generation)
        #expect(state.inputText == "", "inputText='\(state.inputText)' must be empty after submit")
    }

    // MARK: - Test 10.5: /btw parallel generation flag
    
    @Test("/btw flag allows submitting while processing and triggers parallel stream")
    func btwFlagParallelGeneration() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 512)
        try await orchestrator.loadModel(path: "/fake/path")

        let state = RainbowUIState()
        let view = RainbowChatView(state: state, orchestrator: orchestrator, maxTokens: 512)
        let hosting = NSHostingView(rootView: view)
        hosting.frame = NSRect(x: 0, y: 0, width: 800, height: 600)
        hosting.layout()

        func findTF(_ v: NSView) -> NSTextField? {
            if let t = v as? NSTextField { return t }
            for s in v.subviews { if let f = findTF(s) { return f } }
            return nil
        }
        let tf = try #require(findTF(hosting))
        
        // Simulate already processing state
        await MainActor.run { state.setMode(.processing) }
        
        // Submit normally while processing -> SHOULD BE IGNORED
        tf.stringValue = "blocked"
        NotificationCenter.default.post(name: NSControl.textDidChangeNotification, object: tf)
        if let coordinator = tf.delegate as? NativeInputField.Coordinator {
            _ = coordinator.control(tf, textView: NSTextView(), doCommandBy: #selector(NSResponder.insertNewline(_:)))
        }
        #expect(state.messages.isEmpty, "Normal message should be blocked while processing")
        
        // Submit with /btw -> SHOULD BE ALLOWED
        tf.stringValue = "parallel task /btw"
        NotificationCenter.default.post(name: NSControl.textDidChangeNotification, object: tf)
        if let coordinator = tf.delegate as? NativeInputField.Coordinator {
            _ = coordinator.control(tf, textView: NSTextView(), doCommandBy: #selector(NSResponder.insertNewline(_:)))
        }
        
        #expect(state.messages.count == 1, "Message with /btw should bypass state check")
        #expect(state.messages.first?.text == "parallel task", "/btw should be stripped from the prompt")
        #expect(state.currentMode == .processing, "Mode should remain processing")
    }

    // MARK: - Test 10.6: Cancellation
    
    @Test("Cancel generation via MVI intent stops generation and returns to finished")
    func cancelGenerationMVIIntent() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 512)
        try await orchestrator.loadModel(path: "/fake/path")

        let state = RainbowUIState()
        let view = RainbowChatView(state: state, orchestrator: orchestrator, maxTokens: 512)
        let hosting = NSHostingView(rootView: view)
        hosting.frame = NSRect(x: 0, y: 0, width: 800, height: 600)
        hosting.layout()

        func findTF(_ v: NSView) -> NSTextField? {
            if let t = v as? NSTextField { return t }
            for s in v.subviews { if let f = findTF(s) { return f } }
            return nil
        }
        let tf = try #require(findTF(hosting))
        
        // Submit normally
        tf.stringValue = "long task"
        NotificationCenter.default.post(name: NSControl.textDidChangeNotification, object: tf)
        if let coordinator = tf.delegate as? NativeInputField.Coordinator {
            _ = coordinator.control(tf, textView: NSTextView(), doCommandBy: #selector(NSResponder.insertNewline(_:)))
        }
        
        // Wait briefly for the detached task to start
        try await Task.sleep(nanoseconds: 10_000_000)
        
        #expect(state.activeGenerationTask != nil, "activeGenerationTask must be set after submit")
        
        // Cancel it!
        await MainActor.run {
            state.cancelGeneration()
        }
        
        #expect(state.activeGenerationTask == nil, "activeGenerationTask must be cleared")
        #expect(state.currentMode == .finished, "Mode must switch to finished on cancel")
    }

    // MARK: - Test 11: Real AsyncStream yielding does not starve MainActor
    
    @Test("AsyncStream yields correctly update the MainActor without starvation")
    func asyncStreamYieldsWithoutStarvingMainActor() async throws {
        let state = RainbowUIState()
        
        // We simulate what MLXInferenceEngine does: an AsyncStream evaluated on a background thread.
        let stream = AsyncStream<String> { continuation in
            Task.detached {
                // Simulate fast generation of 5 tokens
                for i in 1...5 {
                    try? await Task.sleep(nanoseconds: 10_000_000) // 10ms per token
                    continuation.yield("token\(i) ")
                }
                continuation.finish()
            }
        }
        
        // This simulates RainbowChatView's Task.detached logic
        let task = Task.detached {
            await MainActor.run { state.setMode(.streaming) }
            
            for await chunk in stream {
                // If MainActor is starved, this will hang or timeout
                await MainActor.run {
                    state.appendToLastAssistant(chunk)
                }
            }
            
            await MainActor.run { state.setMode(.finished) }
        }
        
        // Wait for the stream to complete, with a 2-second timeout to prevent infinite test hangs
        let result = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await task.value
                return true
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                return false
            }
            let first = await group.next()
            group.cancelAll()
            return first ?? false
        }
        
        #expect(result == true, "The generation task must finish within the timeout (MainActor must not be starved)")
        
        // Now verify the state on the MainActor
        await MainActor.run {
            #expect(state.currentMode == .finished, "Mode should be finished")
            #expect(state.messages.count == 1, "There should be 1 assistant message")
            #expect(state.messages.first?.text == "token1 token2 token3 token4 token5 ", "Text should contain all chunks")
        }
    }
}
