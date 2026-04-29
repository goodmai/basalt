# Epic 17: Advanced CLI Layout & UX Enhancements

## Goal
Transform the Gemm CLI into a professional terminal interface with a fixed bottom-anchored UI, improved keyboard navigation, and robust model support.

## Decomposition

*(Note: This epic focuses on the terminal CLI layout. For the next generation Metal-based Graphical UI, see [Epic 18: Rainbow AI Chat UI](EPIC_18_RAINBOW_UI.md))*

### 17.1: Fixed-Height Bottom Layout (5 lines)
- [ ] Implement a reserved 5-line region at the bottom of the terminal.
- [ ] Row 1: Divider line.
- [ ] Row 2: Secondary status / Debug info (Modes, Tokens, TPS, Memory).
- [ ] Row 3: Main Input area (Gemma > ...).
- [ ] Row 4-5: Footer with system info (Workspace, Branch, Chip, Model ID).
- [ ] Use ANSI escape codes for stable rendering without screen flicker or duplication.

### 17.2: Keyboard Interrupts & Navigation
- [ ] **Esc**: Interrupt current generation immediately (cancel active task).
- [ ] **Ctrl+C**: Graceful exit of the application.
- [ ] **Tab**: Queue/Stage input (optional/already partially exists).

### 17.3: Top-Level Header
- [ ] Implement a persistent or semi-persistent header at the top of the session.

### 17.4: Gemma 4 31B Compatibility Restore (MoE Fix)
- [ ] Investigate `broadcast_shapes` error in `mlx-swift-lm`.
- [ ] Adjust Attention parameters or KV-cache configuration to prevent tensor shape mismatches.
- [ ] Validate MoE weight loading sequence.

## Implementation Plan

1. **TerminalManager Update**: Complete rewrite of `renderUI` and `refreshLine` to use relative cursor positioning to protect the 5-line footer.
2. **Input Handling**: Update `readEvents` to capture `Esc` (ASCII 27) and `Ctrl+C` (ASCII 3).
3. **Inference Engine Patch**: Add safety checks for tensor shapes in MoE models.

## TODO
- [ ] Restore Gemma 4 31B functionality.
- [ ] Verify scrolling history doesn't overwrite the footer.
