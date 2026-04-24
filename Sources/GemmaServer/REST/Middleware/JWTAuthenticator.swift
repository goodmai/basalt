import Hummingbird
import HummingbirdAuth
import Foundation

public struct User: Sendable, Equatable {
    public let username: String
}

public struct GemmaRequestContext: AuthRequestContext {
    public var coreContext: CoreRequestContextStorage
    public var identity: User?

    public init(source: Source) {
        self.coreContext = .init(source: source)
        self.identity = nil
    }
}

public struct JWTAuthenticator: AuthenticatorMiddleware {
    public typealias Input = Request
    public typealias Output = Response
    public typealias Context = GemmaRequestContext

    let authService: AuthService

    public init(authService: AuthService) {
        self.authService = authService
    }

    public func authenticate(
        request: Request,
        context: GemmaRequestContext
    ) async throws -> User? {
        guard let authHeader = request.headers[.authorization],
              authHeader.hasPrefix("Bearer ") else {
            return nil
        }
        
        let token = String(authHeader.dropFirst("Bearer ".count))
        
        do {
            let username = try await authService.verify(token: token)
            return User(username: username)
        } catch {
            return nil
        }
    }
}
