import Foundation
import CryptoKit
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
    static let anonymous = "anonymous"

    static func resolve(headers: HTTPFields, isService: Bool) -> String? {
        guard !isService else { return nil }
        return credential(in: headers).map { "credential:\(digest($0))" } ?? anonymous
    }

    static func scoped(_ request: GenerationRequest, headers: HTTPFields) -> GenerationRequest {
        request.withCacheKey(resolve(headers: headers, isService: request.isServiceRequest == true))
    }

    static func credential(in headers: HTTPFields) -> String? {
        if let authorization = headers[.authorization] {
            let parts = authorization.split(separator: " ", maxSplits: 1)
            if parts.count == 2, parts[0].lowercased() == "bearer" { return String(parts[1]) }
        }
        return HTTPField.Name("x-api-key").flatMap { headers[$0] }
    }

    static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }
}
