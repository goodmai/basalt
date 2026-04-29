# Bug Epic: CLI & Inference Defects

This document tracks identified bugs, visual regressions, and technical issues discovered during active testing of the Gemm CLI.

## Active Bugs

### 1. [Critical] Gemma 4 31B `broadcast_shapes` Failure
- **Description**: Model crashes with `Fatal error: [broadcast_shapes] Shapes (1,16,4,512) and (1,4,16,512) cannot be broadcast` during the first token generation.
- **Root Cause**: Potential mismatch in Attention head configuration or KV-cache indexing for MoE architecture in `mlx-swift-lm`.
- **Status**: TO_INVESTIGATE (Identified as GQA head/seq transposition in full_attention layers)

## Fixed Bugs
### 2. [Visual] Cursor Positioning in Multi-line Prompts
- **Fix**: Updated `TerminalManager` to calculate total footer lines based on input wrap and use relative cursor positioning.

### 3. [UX] Slow Git Branch Detection
- **Fix**: Moved Git branch detection to an asynchronous background task in `TerminalManager`.

### 4. [Technical] Memory Leak in AsyncStream
- **Fix**: Added `onTermination` handler to `AsyncStream` in `MLXInferenceEngine` to correctly cancel the underlying generation task.

