# Epic 15: Interactive Non-Blocking CLI Prompt ⌨️

**Version:** v0.6.0  
**Priority:** HIGH  
**Effort:** 2 weeks  
**Status:** In Progress

## Overview
Implement a highly interactive, non-blocking command-line interface inspired by `gemini-cli`. The CLI should allow users to continue typing while the model is thinking or streaming, support request queuing, handle graceful cancellations (Ctrl+C), and provide a rich syntax for file insertions (using `@` references).

---

## Business Analysis (BA) Decomposition

### 1. Use Cases
- **UC1: Non-Blocking Input & Queueing**
  - **Actor:** Developer
  - **Scenario:** The user submits a prompt that takes 10 seconds to generate. While the response is streaming, the user starts typing the next prompt and presses `Enter` (or `Tab` for queueing). The CLI queues the next prompt and automatically executes it once the current generation completes.
- **UC2: Graceful Cancellation**
  - **Actor:** Developer
  - **Scenario:** The user triggers a long-running task or generation. Realizing a mistake, the user presses `Ctrl+C`. The active generation is aborted immediately, the stream stops, and the prompt returns to a ready state without crashing the server.
- **UC3: Context Injection via `@` References**
  - **Actor:** Developer
  - **Scenario:** The user types `Explain the logic in @Sources/Gem/Core/AuthService.swift`. The CLI intercepts the `@` symbol, resolves the file path, reads its contents, and silently injects it into the LLM context before routing the request.
- **UC4: Auto-Execution (CLI Flags)**
  - **Actor:** CI/CD Pipeline / Script
  - **Scenario:** The server is launched with a direct prompt flag (e.g., `gemini -p "Summarize @README.md"`). The CLI executes the prompt immediately, streams the result, and exits.

---

### 2. Tasks & Deliverables

#### Task 15.1: Non-Blocking I/O & Terminal Management
- Switch from standard `readLine()` to a custom non-blocking terminal input reader (e.g., raw mode via `termios`).
- Implement an input buffer that safely updates the screen while background asynchronous streams (from `ModelRouter`) print their chunks.

#### Task 15.2: Request Queueing & Cancellation (Task Management)
- Introduce a `PromptQueue` or an `Actor`-based task manager.
- Handle `SIGINT` (Ctrl+C) to cancel Swift `Task`s.
- Pass `Task.isCancelled` checks down to the `InferenceEngine` and `CloudAPIClient` to abort network streams or Metal evaluations.

#### Task 15.3: `@` File References (Context Injection)
- Build a regex parser for `@/path/to/file` or `@filename`.
- Implement a `ContextBuilder` that reads resolved files, limits their size, and appends them to the actual prompt.
- Handle missing files and permission errors gracefully with UI hints.

#### Task 15.4: UI Enhancements & Hints
- Implement dynamic CLI prompts (e.g., `Gemma > ` vs `Gemma (busy) > `).
- Add colorized output for user input, model responses, and system warnings.

---

### 3. Test Cases (TDD/BDD)

| ID | Module | Description | Expected Result |
|---|---|---|---|
| TC-15.1 | Terminal | Input buffer handles concurrent background prints | Text typed by user is restored at the bottom after background prints. |
| TC-15.2 | Queue | Submit two prompts rapidly | Second prompt is queued and executed after the first finishes. |
| TC-15.3 | Cancel | Send `SIGINT` during generation | Stream yields `finishReason: .cancel` and input loop resumes. |
| TC-15.4 | Context | Parse `Fix @main.swift` | Payload sent to orchestrator contains `main.swift` contents. |
| TC-15.5 | Context | Parse invalid `@missing.txt` | CLI warns user about missing file and halts execution. |

---

## Architectural Impact

- **`ChatCommand.swift`**: Needs a complete rewrite of the `run()` loop from synchronous `readLine()` to an asynchronous `TaskGroup` reading from standard input in raw mode.
- **`ModelOrchestratorActor`**: Must check `Task.isCancelled` during the token generation loop.
- **`CloudAPIClient`**: `performStreamingRequest` should respect Swift's cooperative cancellation.