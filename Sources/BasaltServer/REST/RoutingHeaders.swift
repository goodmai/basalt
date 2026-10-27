import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

extension GenerationRequest {
    static var sessionHeader: HTTPField.Name { HTTPField.Name("x-session-id")! }

    func routed(by headers: HTTPFields) -> GenerationRequest {
        withRouting(
            sessionId: headers[Self.sessionHeader].flatMap { $0.isEmpty ? nil : String($0.prefix(128)) },
            budgetKey: headers[APIKeyAuthenticator<BasicRequestContext>.budgetKeyHeader])
    }
}
