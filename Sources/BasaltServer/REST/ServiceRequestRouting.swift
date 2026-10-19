import Foundation
import HTTPTypes
import BasaltCore

extension ServiceRequestDetector {
    /// Shared by every generation endpoint: classify the request (`x-service-request: true` says so
    /// explicitly, otherwise the heuristic decides), and compact its prompt only when the client
    /// opts in with `x-compact-prompt: true` or `x-service-request: true` — rewriting a prompt on a
    /// guess would silently corrupt ordinary traffic.
    static func prepare(prompt: String, maxTokens: Int?, headers: HTTPFields) -> (prompt: String, isService: Bool) {
        let declared = headers.isFlagTrue("x-service-request")
        let isService = declared || isTitleOrServiceRequest(prompt: prompt, maxTokens: maxTokens)
        let optIn = headers.isFlagTrue("x-compact-prompt") || declared
        return (isService && optIn ? compactTitlePrompt(prompt) : prompt, isService)
    }
}

extension HTTPFields {
    func isFlagTrue(_ name: String) -> Bool {
        guard let fieldName = HTTPField.Name(name) else { return false }
        return self[fieldName] == "true"
    }
}

enum PromptCacheKey {
    static let shared = "shared"

    static func resolve(isService: Bool) -> String? {
        isService ? nil : shared
    }

    static func scoped(_ request: GenerationRequest) -> GenerationRequest {
        request.withCacheKey(resolve(isService: request.isServiceRequest == true))
    }
}
