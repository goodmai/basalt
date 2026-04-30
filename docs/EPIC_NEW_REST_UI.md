# Epic: New REST UI Client

## Objective
Replace the experimental, tightly-coupled Rainbow Metal UI with a robust, modern UI client. The new UI must be completely decoupled from the internal `MLXInferenceEngine` and `ModelOrchestratorActor` and communicate with the Gemm backend strictly via the OpenAPI REST interface (`/api/v1/generate`, `/api/v1/auth/login`, etc.).

## Requirements
1. **Architecture:**
   - Develop a separate SwiftUI client (or module) acting as a front-end.
   - All communication with the AI model must go through HTTP/REST endpoints.
   - Support streaming generation via Server-Sent Events (SSE) or chunked responses over REST.

2. **Features:**
   - Graceful connection handling (login, auth token storage).
   - Chat interface with input field, history, and markdown/code block rendering.
   - Display generation metadata (Tokens per second, Time to first token, Total time).

3. **Tech Stack:**
   - SwiftUI for the presentation layer.
   - Swift 6 with modern concurrency (`async`/`await`).
   - Swift Testing for validation.

4. **Testing Strategy:**
   - **Real Integration Tests:** E2E tests against the running REST server.
   - **Vision Screenshots:** UI functionality must be confirmed visually through screenshots during automated tests.

## Milestones
- [ ] M1: Initial Setup of the SwiftUI project structure.
- [ ] M2: REST Client integration (auth, simple text generation).
- [ ] M3: Streaming generation support.
- [ ] M4: UI Polish and Markdown rendering.
- [ ] M5: Automated Vision E2E tests.