import HTTPTypes

extension ServiceRequestDetector {
    /// Shared by every generation endpoint: classify the request, and compact its prompt only
    /// when the client opts in with `x-compact-prompt: true` or `x-service-request: true`
    /// (rewriting a prompt on a guess would silently corrupt ordinary traffic).
    static func prepare(prompt: String, maxTokens: Int?, headers: HTTPFields) -> (prompt: String, isService: Bool) {
        let isService = isTitleOrServiceRequest(prompt: prompt, maxTokens: maxTokens)
        let optIn = headers.isFlagTrue("x-compact-prompt") || headers.isFlagTrue("x-service-request")
        return (isService && optIn ? compactTitlePrompt(prompt) : prompt, isService)
    }
}

extension HTTPFields {
    func isFlagTrue(_ name: String) -> Bool {
        guard let fieldName = HTTPField.Name(name) else { return false }
        return self[fieldName] == "true"
    }
}
