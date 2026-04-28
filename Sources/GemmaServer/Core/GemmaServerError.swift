import Foundation

// Алгебраический тип ошибок — полное множество состояний отказа системы.
// E_total = E_io ∪ E_memory ∪ E_inference ∪ E_protocol
public enum GemmaServerError: Error, Sendable, LocalizedError {
    // MARK: — IO / Model loading
    case modelNotFound(identifier: String)
    case weightsCorrupted(path: String, reason: String)

    // MARK: — Memory / Hardware
    case outOfMemory(requestedMB: Int, availableMB: Int)
    case inferenceHardwareFailure(reason: String)

    // MARK: — Protocol / Request
    case invalidRequestStructure(details: String)
    case contextLengthExceeded(maxTokens: Int, requested: Int)
    case rateLimitExceeded(retryAfter: String?)
    case authenticationFailed(details: String)
    case modelInferenceError(details: String)

    // MARK: — LocalizedError
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Model '\(id)' not found in storage."
        case .weightsCorrupted(let path, let reason):
            return "Weights at '\(path)' are corrupted: \(reason)"
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
        }
    }

    // Maps to HTTP status codes for the REST layer.
    public var httpStatus: Int {
        switch self {
        case .modelNotFound:            return 404
        case .weightsCorrupted:         return 500
        case .outOfMemory:              return 507  // Insufficient Storage
        case .inferenceHardwareFailure: return 503
        case .invalidRequestStructure:  return 400
        case .contextLengthExceeded:    return 413  // Payload Too Large
        case .rateLimitExceeded:        return 429  // Too Many Requests
        case .authenticationFailed:     return 401
        case .modelInferenceError:      return 502  // Bad Gateway
        }
    }
}
