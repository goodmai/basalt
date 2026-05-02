import Foundation
import Hummingbird

// MARK: — Model Management Endpoints
//
// GET  /v1/models          — list locally cached HuggingFace models (OpenAI-compatible)
// GET  /v1/models/current  — currently loaded model + status
// POST /v1/models/load     — hot-swap the loaded model at runtime
//
// ── Claude Code integration ─────────────────────────────────────────────────
//
// Recommended setup (add to ~/.zshrc):
//
//   function gemm-claude() {
//     ANTHROPIC_BASE_URL=http://localhost:8080       \
//     ANTHROPIC_AUTH_TOKEN=local                    \
//     ANTHROPIC_DEFAULT_HAIKU_MODEL=mlx-community/gemma-4-e4b-it-4bit    \
//     ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-community/Qwen3.5-4B-4bit       \
//     ANTHROPIC_DEFAULT_OPUS_MODEL=mlx-community/gemma-4-31b-it-4bit     \
//     claude "$@"
//   }
//
//   gemm-claude                   # sonnet alias → Qwen 4B
//   gemm-claude --model haiku     # haiku alias  → Gemma 4B (fastest)
//   gemm-claude --model opus      # opus alias   → Gemma 31B (strongest)
//
// Key: ANTHROPIC_AUTH_TOKEN avoids touching ANTHROPIC_API_KEY (different header).
//
// ── Model discovery ──────────────────────────────────────────────────────────
//
// Claude Code (v2.1.126+) queries GET /v1/models at startup and adds returned
// models to the /model picker — BUT only if the ID starts with "claude" or "anthropic".
//
// Strategy: /v1/models returns IDs with a "claude-local/" prefix so discovery
// works. The prefix is stripped when the ID arrives in a generate request, so
// switchModel resolves the actual HuggingFace repo ID.
//
//   claude-local/mlx-community--Qwen3.5-4B-4bit  →  mlx-community/Qwen3.5-4B-4bit
//
// Users can also pass HF IDs directly (e.g. via ANTHROPIC_MODEL) — the
// orchestrator handles both forms.

struct ModelsController: Sendable {

    let orchestrator: ModelOrchestratorActor
    let defaultModelId: String

    // MARK: — GET /v1/models

    /// Returns all locally cached models in OpenAI-compatible list format.
    ///
    /// IDs are returned in `claude-local/<hf-id-encoded>` form so Claude Code's
    /// automatic model discovery adds them to the /model picker.
    /// `display_name` carries the original HuggingFace repo ID for readability.
    @Sendable
    func listModels(request: Request, context: BasicRequestContext) async throws -> Response {
        let cached   = await orchestrator.cachedModels()
        let loadedId = await orchestrator.currentModelId
        let now      = Int(Date().timeIntervalSince1970)

        var entries: [[String: Any]]

        if cached.isEmpty {
            let hfId     = loadedId.isEmpty ? defaultModelId : loadedId
            entries = [[
                "id":           claudeLocalId(from: hfId),
                "display_name": hfId,
                "object":       "model",
                "created":      now,
                "owned_by":     "gemm",
                "is_loaded":    true,
            ]]
        } else {
            let sorted = cached.sorted { a, b in
                if a.isLoaded != b.isLoaded { return a.isLoaded }
                return a.id < b.id
            }
            entries = sorted.map { m -> [String: Any] in
                [
                    "id":           claudeLocalId(from: m.id),
                    "display_name": m.id,
                    "object":       "model",
                    "created":      now,
                    "owned_by":     "gemm",
                    "is_loaded":    m.isLoaded,
                    "size_bytes":   m.sizeBytes,
                    "size_human":   m.sizeFormatted,
                ]
            }
        }

        return makeJSONResponse(["object": "list", "data": entries])
    }

    // MARK: — GET /v1/models/current

    @Sendable
    func currentModel(request: Request, context: BasicRequestContext) async throws -> Response {
        let snapshot = await orchestrator.healthSnapshot(modelId: nil)
        let loadedId = await orchestrator.currentModelId
        let cached   = await orchestrator.cachedModels()
        let sizeInfo = cached.first(where: { $0.id == loadedId })
        let hfId     = loadedId.isEmpty ? defaultModelId : loadedId

        return makeJSONResponse([
            "id":           claudeLocalId(from: hfId),
            "display_name": hfId,
            "object":       "model",
            "created":      Int(Date().timeIntervalSince1970),
            "owned_by":     "gemm",
            "is_ready":     snapshot.isReady,
            "status":       snapshot.status,
            "size_bytes":   sizeInfo?.sizeBytes ?? 0,
            "size_human":   sizeInfo?.sizeFormatted ?? "unknown",
        ])
    }

    // MARK: — POST /v1/models/load

    /// Hot-swap the loaded model.
    ///
    /// Accepts both forms:
    ///   { "model": "mlx-community/Qwen3.5-4B-4bit" }          ← HF repo ID
    ///   { "model": "claude-local/mlx-community--Qwen3.5-4B-4bit" }  ← discovery ID
    ///
    /// Blocks until loaded (10–60 s). All queued generate requests run after.
    @Sendable
    func loadModel(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: 1 * 1024 * 1024)
        let data   = Data(buffer.readableBytesView)

        struct LoadRequest: Codable { let model: String }

        guard let dto = try? JSONDecoder().decode(LoadRequest.self, from: data),
              !dto.model.isEmpty else {
            return makeErrorResponse(status: .badRequest,
                                     message: "Request body must be JSON with a non-empty \"model\" field.")
        }

        let hfId = hfRepoId(from: dto.model)
        fputs("🔄 [Models] Load: \(dto.model) → \(hfId)\n", stderr)

        do {
            try await orchestrator.switchModel(to: hfId)
        } catch {
            return makeErrorResponse(status: HTTPResponse.Status(code: error.httpStatus),
                                     message: error.errorDescription ?? error.localizedDescription)
        }

        let loadedId = await orchestrator.currentModelId
        fputs("✅ [Models] Loaded: \(loadedId)\n", stderr)
        return makeJSONResponse([
            "id":       claudeLocalId(from: loadedId),
            "display_name": loadedId,
            "object":   "model",
            "created":  Int(Date().timeIntervalSince1970),
            "owned_by": "gemm",
            "status":   "loaded",
        ])
    }

    // MARK: — ID helpers

    /// Converts an HF repo ID to the claude-local/ form that Claude Code's
    /// model discovery filter accepts:
    ///   "mlx-community/Qwen3.5-4B-4bit"  →  "claude-local/mlx-community--Qwen3.5-4B-4bit"
    static func claudeLocalId(from hfId: String) -> String {
        "claude-local/" + hfId.replacingOccurrences(of: "/", with: "--")
    }

    private func claudeLocalId(from hfId: String) -> String {
        Self.claudeLocalId(from: hfId)
    }

    /// Inverse of claudeLocalId — strips prefix and restores "/" separator.
    /// Also passes through plain HF IDs (those containing "/") unchanged.
    static func hfRepoId(from modelId: String) -> String {
        if modelId.hasPrefix("claude-local/") {
            let body = String(modelId.dropFirst("claude-local/".count))
            return body.replacingOccurrences(of: "--", with: "/")
        }
        return modelId   // already a plain HF ID or an alias
    }

    private func hfRepoId(from modelId: String) -> String {
        Self.hfRepoId(from: modelId)
    }

    // MARK: — Response helpers

    private func makeJSONResponse(_ body: [String: Any], status: HTTPResponse.Status = .ok) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        var buf  = ByteBuffer()
        buf.writeBytes(data)
        return Response(status: status,
                        headers: [.contentType: "application/json"],
                        body: ResponseBody(byteBuffer: buf))
    }

    private func makeErrorResponse(status: HTTPResponse.Status, message: String) -> Response {
        makeJSONResponse(["error": ["message": message, "type": "invalid_request_error"]], status: status)
    }
}

private extension HTTPResponse.Status {
    init(code: Int) { self = HTTPResponse.Status(code: code, reasonPhrase: "") }
}
