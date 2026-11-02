import Foundation
import HTTPTypes
import BasaltCore

extension ServiceRequestDetector {
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
    static let clientPrefix = "key:"
    static let maxClientKeyBytes = 256
    static var header: HTTPField.Name { HTTPField.Name("x-basalt-cache-key")! }

    static func resolve(isService: Bool, headers: HTTPFields = [:]) -> String? {
        guard !isService else { return nil }
        guard let key = headers[header], !key.isEmpty, key.utf8.count <= maxClientKeyBytes else { return shared }
        return clientPrefix + key
    }

    static func scoped(_ request: GenerationRequest, headers: HTTPFields) -> GenerationRequest {
        request.withCacheKey(resolve(isService: request.isServiceRequest == true, headers: headers))
    }
}
