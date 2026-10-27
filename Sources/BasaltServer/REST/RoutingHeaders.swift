import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

extension GenerationRequest {
    static var sessionHeader: HTTPField.Name { HTTPField.Name("x-session-id")! }
    static var basaltSessionHeader: HTTPField.Name { HTTPField.Name("x-basalt-session")! }

    static func sessionId(in headers: HTTPFields) -> String? {
        (headers[basaltSessionHeader] ?? headers[sessionHeader]).flatMap { $0.isEmpty ? nil : String($0.prefix(128)) }
    }

    func routed(by headers: HTTPFields) -> GenerationRequest {
        withRouting(
            sessionId: Self.sessionId(in: headers),
            budgetKey: headers[APIKeyAuthenticator<BasicRequestContext>.budgetKeyHeader])
    }
}
