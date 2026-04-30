# Epic 21: Rainbow Metal Rendering Bugfix

> **Status**: ✅ DONE  
> **Priority**: CRITICAL  
> **Depends on**: Epic 18, Epic 20  

## Problem Statement

After implementing the MVI Event Bus (Epic 20), the data pipeline works correctly —
user messages are added to `state.messages`, the MLX inference engine streams 100+ tokens,
and the `RenderCoordinator` dispatches `RenderState` updates at 30 FPS. **However, none of
this content appears on screen.** The Metal renderer draws only the rainbow background,
header, and footer — the chat message area remains empty.

## Root Cause Analysis

### Bug 1: Messages not rendered in Metal chat area
**Severity**: CRITICAL  
**File**: `RainbowRenderer.swift`, `renderUI()` method  

The `renderUI()` method iterates over `uiState.messages` and creates textures for each line
via `getOrCreateTexture()`. The issue is **twofold**:

1. **Stale texture cache**: During streaming, `messages[lastIndex].text` is mutated every
   ~33ms by the `RenderCoordinator`. The cache key uses `text.hashValue`, but the Metal
   draw loop runs at 60 FPS while the `@Published` property change notifications may not
   trigger a re-render of the Metal layer (MTKView uses its own draw loop, not SwiftUI's
   observation).

2. **Content overflow clipping**: Line 296 clips messages when
   `yOffset > viewportSize.height - footerHeight - padding`, but with enough messages
   and no proper scroll offset tracking, content can be invisible from the start.

### Bug 2: State shows IDLE after generation completes
**Severity**: MEDIUM  
**File**: `RainbowUIState.swift`, `submit(state:)` method  

The `submit(state:)` method sets a 3-second timer to transition from `.finished` → `.idle`.
This works, but the mode indicator shows "● IDLE" even though messages should be visible.
This is cosmetic but confusing.

### Bug 3: Input text not echoed in chat area
**Severity**: HIGH  
**File**: `RainbowChatView.swift`, `submitMessage()` method  

The user message IS added to `state.messages` (confirmed by logs), but the Metal renderer
does not see the updated array because `MTKView.draw()` reads `uiState.messages` on the
render thread while mutations happen on `@MainActor`. There may be a thread-safety issue
or the `ObservableObject` changes don't trigger `MTKView` redraws.

## Fix Strategy

### Task 1: Force MTKView needsDisplay on state changes ✅
Wire `objectWillChange` from `RainbowUIState` to `mtkView.needsDisplay = true`
so every `@Published` change triggers a Metal redraw.

### Task 2: Invalidate message textures on content change ✅  
When streaming, don't cache assistant message textures — always regenerate
during active streaming to reflect the latest `text` content.

### Task 3: Auto-scroll to latest message ✅
After adding a user or assistant message, auto-adjust `scrollOffset` so the
latest content is always visible within the viewport.

### Task 4: Integration tests ✅
- Red test: verify `messages.count > 0` after submit → must see user message in state
- Red test: verify assistant response text matches streamed content
- Red test: verify render state transitions through streaming → finished → idle

### Task 5: XCUITest E2E Architecture (Zero-Python) ✅
Implemented a native end-to-end UI testing framework directly in XCTest using the macOS Accessibility Tree (`NSAccessibility`). This completely eliminates the need for POSIX-emulators (like Python PTY) and performs topological and text-based validation against the Metal `MTKView`'s accessibility children.

## Test Plan

```
swift test --filter RainbowUIGenerationTests
swift test --filter RainbowUIRenderingTests  
swift build -c release  # zero warnings
```

## Acceptance Criteria

- [ ] User message appears in the Metal chat area immediately after Enter
- [ ] Assistant response streams visibly in real-time (30 FPS text updates)
- [ ] State indicator shows correct mode during generation
- [ ] Zero compiler warnings in release build
- [ ] All integration tests pass
