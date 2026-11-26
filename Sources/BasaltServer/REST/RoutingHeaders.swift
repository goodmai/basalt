import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

extension GenerationRequest {
    static var sessionHeader: HTTPField.Name { HTTPField.Name("x-session-id")! }
    static var basaltSessionHeader: HTTPField.Name { HTTPField.Name("x-basalt-session")! }
    static var adapterHeader: HTTPField.Name { HTTPField.Name("x-basalt-adapter")! }
    static var anthropicVersionHeader: HTTPField.Name { HTTPField.Name("anthropic-version")! }

    static func sessionId(in headers: HTTPFields) -> String? {
        (headers[basaltSessionHeader] ?? headers[sessionHeader]).flatMap { $0.isEmpty ? nil : String($0.prefix(128)) }
    }

    func routed(by headers: HTTPFields) -> GenerationRequest {
        let decision = HarnessRoute.resolve(
            HarnessRequestFingerprint(
                userAgent: headers[.userAgent],
                endpointPath: "",
                explicitAdapterHeader: headers[Self.adapterHeader],
                systemPromptPrefix: nil,
                anthropicVersionHeader: headers[Self.anthropicVersionHeader]
            ),
            table: .empty
        )
        return routed(by: headers, decision: decision)
    }

    func routed(by headers: HTTPFields, decision: HarnessRouteDecision) -> GenerationRequest {
        withRouting(
            sessionId: Self.sessionId(in: headers),
            budgetKey: headers[APIKeyAuthenticator<BasicRequestContext>.budgetKeyHeader]
        ).withAdapter(decision.adapterName)
    }
}
