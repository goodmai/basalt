# Gemm Development Guidelines

This document outlines the core technical stack, development methodologies, AI skills, and testing strategies required for the Gemm application ecosystem.

## Tech Stack
*   **Language:** Swift 6.
*   **Client UI Framework:** SwiftUI.
*   **Backend Server/API:** Swift on Server (`Hummingbird`), OpenAPI REST API (`/api/v1/...`).
*   **Machine Learning / Inference:** `MLX-Swift`, `MLX-Swift-LM`.
*   **Concurrency:** Modern Swift Concurrency (`async`/`await`, `Actor` isolation, `Task` groups).
*   **Testing:** Swift Testing framework (replacing `XCTest`).

## Development Style & Architectural Principles
1.  **Decoupled Architecture:** 
    *   The UI must act strictly as a REST client. It should NEVER directly access backend actors (like `ModelOrchestratorActor`) or `MLXInferenceEngine`.
    *   Communication must go over the network via standard JSON/SSE protocols.
2.  **Safety & Hygiene:** 
    *   Zero warnings policy.
    *   Follow modern Swift API guidelines and structure files logically.
3.  **Concurrency Safety:**
    *   Ensure `@MainActor` is used exclusively for UI updates.
    *   Avoid blocking the main thread; use `Task.detached` or specific actors for heavy processing.

## Required AI Skills
The workspace relies heavily on specific expert-level skills installed locally. The agent must strictly adhere to the guidelines provided by these skills:
1.  **SwiftUI Pro (`swiftui-pro`):** Use for building responsive, maintainable, and accessible UI clients based on the latest Apple Human Interface Guidelines.
2.  **Swift Concurrency Pro (`swift-concurrency-pro`):** Use when writing any multi-threaded code to avoid data races, deadlocks, and improper use of structured/unstructured tasks.
3.  **Swift Testing Pro (`swift-testing-pro`):** Use to ensure all unit and integration tests are robust, using the modern `@Test`, `#expect`, and parametrized testing methodologies.

## Testing Strategy
1.  **Real Integration Tests:** 
    *   Code must be backed by genuine E2E tests, verifying that the client properly interfaces with a live REST backend, processes tokens, and handles HTTP errors correctly.
2.  **Vision Screenshots:**
    *   Visual aspects of the UI must be verified via screenshots.
    *   *Mandatory Requirement:* **Never mark a UI task as completed without attaching a screenshot confirming the final rendered state.** All changes to the user interface must be visually validated.
