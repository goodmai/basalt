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
    static let sessionHeader = "x-session-id"
    static let hashedPrefixLength = 512

    static func resolve(declared: String?, headers: HTTPFields, prompt: String, isService: Bool) -> String? {
        guard !isService else { return nil }
        if let declared, !declared.isEmpty { return "key:\(declared)" }
        if let name = HTTPField.Name(sessionHeader), let session = headers[name], !session.isEmpty {
            return "session:\(session)"
        }
        return "prefix:\(prompt.prefix(hashedPrefixLength).hashValue)"
    }
}
