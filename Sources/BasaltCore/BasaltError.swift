import Foundation

public enum BasaltError: Error, Sendable, LocalizedError {
    case modelNotFound(identifier: String)
    case weightsCorrupted(path: String, reason: String)
    case modelConfigurationInvalid(path: String, reason: String)
    case modelProbeFailed(path: String, check: String, reason: String)

    case outOfMemory(requestedMB: Int, availableMB: Int)
    case inferenceHardwareFailure(reason: String)

    case invalidRequestStructure(details: String)
    case contextLengthExceeded(maxTokens: Int, requested: Int)
    case rateLimitExceeded(retryAfter: String?)
    case authenticationFailed(details: String)
    case modelInferenceError(details: String)
    case nonFiniteLogits

    case inferenceError(String)
    case modelNotCached(identifier: String)
    case modelSwitching(from: String, to: String)
    case serverBusy(retryAfterSeconds: Int?)
    case memoryPressure(retryAfterSeconds: Int)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Model '\(id)' not found in storage."
        case .weightsCorrupted(let path, let reason):
            return "Weights at '\(path)' are corrupted: \(reason)"
        case .modelConfigurationInvalid(let path, let reason):
            return "The model configuration at '\(path)' cannot be used: \(reason)"
        case .modelProbeFailed(let path, let check, let reason):
            return "The model at '\(path)' failed its load probe (\(check)): \(reason). Start with --no-model-probe to load it anyway."
        case .outOfMemory(let req, let avail):
            return "OOM: requested \(req) MB, available \(avail) MB."
        case .inferenceHardwareFailure(let reason):
            return "Metal/MLX failure: \(reason)"
        case .invalidRequestStructure(let details):
            return "Invalid request: \(details)"
        case .contextLengthExceeded(let max, let req):
            return "Context too long: max \(max) tokens, requested \(req)."
        case .rateLimitExceeded(let retryAfter):
            return "Rate limit exceeded. Try again later. (Retry-After: \(retryAfter ?? "unknown"))"
        case .authenticationFailed(let details):
            return "Authentication failed: \(details)"
        case .modelInferenceError(let details):
            return "Model inference error: \(details)"
        case .nonFiniteLogits:
            return "The model produced NaN or infinite logits, so its output is unusable; the weights or their quantization are likely broken."
        case .inferenceError(let msg):
            return "Inference error: \(msg)"
        case .modelNotCached(let id):
            return "Model '\(id)' is not in the local cache. Run: basalt models download \(id)"
        case .modelSwitching(let from, let to):
            return "Switching model from '\(from)' to '\(to)'. Retry in a few seconds."
        case .serverBusy(let retryAfter):
            return retryAfter.map { "Server is busy: every generation slot and queue place is taken. Retry in \($0) s." }
                ?? "Server is shutting down."
        case .memoryPressure(let retryAfter):
            return "The machine is under critical memory pressure: long prompts are refused until it eases. Retry in \(retryAfter) s."
        }
    }

    public var wire: (status: Int, openAIType: String, anthropicType: String, retryAfter: String?) {
        switch self {
        case .modelNotFound, .modelNotCached: (404, "invalid_request_error", "not_found_error", nil)
        case .invalidRequestStructure: (400, "invalid_request_error", "invalid_request_error", nil)
        case .contextLengthExceeded: (413, "invalid_request_error", "invalid_request_error", nil)
        case .modelConfigurationInvalid, .modelProbeFailed: (422, "invalid_request_error", "invalid_request_error", nil)
        case .authenticationFailed: (401, "authentication_error", "authentication_error", nil)
        case .rateLimitExceeded(let retryAfter): (429, "rate_limit_error", "rate_limit_error", retryAfter ?? "1")
        case .weightsCorrupted, .inferenceError, .nonFiniteLogits: (500, "server_error", "api_error", nil)
        case .modelInferenceError: (502, "server_error", "api_error", nil)
        case .inferenceHardwareFailure, .modelSwitching: (503, "server_error", "overloaded_error", "1")
        case .serverBusy(let seconds): (503, "server_error", "overloaded_error", seconds.map(String.init) ?? "1")
        case .memoryPressure(let seconds): (503, "server_error", "overloaded_error", String(seconds))
        case .outOfMemory: (507, "server_error", "overloaded_error", "1")
        }
    }

    public var httpStatus: Int { wire.status }
}
