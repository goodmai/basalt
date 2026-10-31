import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

package enum ProviderErrorShape: Sendable {
    case anthropic
    case openAI
    case native

    package static func forPath(_ path: String) -> ProviderErrorShape {
        if path == "/v1/messages" || path.hasPrefix("/v1/messages/") {
            return .anthropic
        } else if path.hasPrefix("/v1/") {
            return .openAI
        } else {
            return .native
        }
    }
}

package enum ProviderErrorResponse: Sendable {
    package static func response(
        for error: any Error,
        shape: ProviderErrorShape,
        overrideMessage: String? = nil,
        retryAfter: String? = nil
    ) -> Response {
        let statusCode: Int
        if let basalt = error as? BasaltError {
            statusCode = basalt.httpStatus
        } else if error is DecodingError {
            statusCode = 400
        } else if let httpErr = error as? HTTPError {
            statusCode = httpErr.status.code
        } else {
            statusCode = 500
        }
        let status = HTTPResponse.Status(code: statusCode)
        let message = overrideMessage ?? ModelsController.publicMessage(for: error)
        return response(status: status, message: message, error: error, shape: shape, retryAfter: retryAfter)
    }

    package static func response(
        for error: any Error,
        path: String,
        overrideMessage: String? = nil,
        retryAfter: String? = nil
    ) -> Response {
        let shape = ProviderErrorShape.forPath(path)
        return response(for: error, shape: shape, overrideMessage: overrideMessage, retryAfter: retryAfter)
    }

    package static func response(
        status: HTTPResponse.Status,
        message: String,
        error: (any Error)? = nil,
        shape: ProviderErrorShape,
        retryAfter: String? = nil
    ) -> Response {
        var headers: HTTPFields = [.contentType: "application/json"]
        if status == .unauthorized {
            headers[.wwwAuthenticate] = "Bearer"
        }
        let effectiveRetry = resolveRetryAfter(status: status, error: error, explicit: retryAfter)
        if let effectiveRetry {
            headers[.retryAfter] = effectiveRetry
        }

        let bodyBytes: Data
        switch shape {
        case .anthropic:
            let errType = anthropicErrorType(status: status, error: error)
            let payload: [String: Any] = [
                "type": "error",
                "error": [
                    "type": errType,
                    "message": message
                ]
            ]
            bodyBytes = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()

        case .openAI:
            let errType = openAIErrorType(status: status, error: error)
            let payload: [String: Any] = [
                "type": "error",
                "error": [
                    "type": errType,
                    "message": message
                ]
            ]
            bodyBytes = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()

        case .native:
            let envelope = ErrorResponse(error: message, code: status.code)
            bodyBytes = (try? JSONEncoder().encode(envelope)) ?? Data()
        }

        var buffer = ByteBuffer()
        buffer.writeBytes(bodyBytes)
        return Response(status: status, headers: headers, body: ResponseBody(byteBuffer: buffer))
    }

    package static func response(
        status: HTTPResponse.Status,
        message: String,
        error: (any Error)? = nil,
        path: String,
        retryAfter: String? = nil
    ) -> Response {
        let shape = ProviderErrorShape.forPath(path)
        return response(status: status, message: message, error: error, shape: shape, retryAfter: retryAfter)
    }

    private static func resolveRetryAfter(
        status: HTTPResponse.Status,
        error: (any Error)?,
        explicit: String?
    ) -> String? {
        if let explicit { return explicit }
        if let basalt = error as? BasaltError {
            switch basalt {
            case .rateLimitExceeded(let r):
                return r ?? "1"
            case .serverBusy(let s):
                return s.map(String.init) ?? "1"
            case .modelSwitching, .inferenceHardwareFailure, .outOfMemory:
                return "1"
            default:
                break
            }
        }
        if status == .tooManyRequests || status == .serviceUnavailable {
            return "1"
        }
        return nil
    }

    private static func anthropicErrorType(
        status: HTTPResponse.Status,
        error: (any Error)?
    ) -> String {
        if let basalt = error as? BasaltError {
            switch basalt {
            case .modelNotFound, .modelNotCached:
                return "not_found_error"
            case .authenticationFailed:
                return "authentication_error"
            case .rateLimitExceeded:
                return "rate_limit_error"
            case .modelSwitching, .serverBusy, .inferenceHardwareFailure, .outOfMemory:
                return "overloaded_error"
            case .weightsCorrupted, .modelInferenceError, .inferenceError:
                return "api_error"
            case .invalidRequestStructure, .modelConfigurationInvalid, .contextLengthExceeded:
                return "invalid_request_error"
            }
        }
        switch status.code {
        case 401: return "authentication_error"
        case 403: return "permission_error"
        case 404: return "not_found_error"
        case 429: return "rate_limit_error"
        case 503, 529, 507: return "overloaded_error"
        case 500, 502: return "api_error"
        default:
            return status.code >= 500 ? "api_error" : "invalid_request_error"
        }
    }

    private static func openAIErrorType(
        status: HTTPResponse.Status,
        error: (any Error)?
    ) -> String {
        if let basalt = error as? BasaltError {
            switch basalt {
            case .authenticationFailed:
                return "authentication_error"
            case .rateLimitExceeded:
                return "rate_limit_error"
            case .modelNotFound, .modelNotCached, .invalidRequestStructure,
                 .modelConfigurationInvalid, .contextLengthExceeded:
                return "invalid_request_error"
            case .modelSwitching, .serverBusy, .inferenceHardwareFailure,
                 .outOfMemory, .weightsCorrupted, .modelInferenceError, .inferenceError:
                return "server_error"
            }
        }
        switch status.code {
        case 401: return "authentication_error"
        case 429: return "rate_limit_error"
        default:
            return status.code >= 500 ? "server_error" : "invalid_request_error"
        }
    }
}

package extension HTTPResponse.Status {
    init(code: Int) {
        self = HTTPResponse.Status(code: code, reasonPhrase: "")
    }
}
