import Foundation
import Hummingbird

struct LoginRequestDTO: Codable {
    let username: String
    let password: String
}

struct AuthController: Sendable {
    let authService: AuthService
    
    // MARK: — POST /api/v1/auth/login
    
    @Sendable
    func login(request: Request, context: GemmaRequestContext) async throws -> Response {
        let dto: LoginRequestDTO
        do {
            let buffer = try await request.body.collect(upTo: 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = try JSONDecoder().decode(LoginRequestDTO.self, from: data)
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }
        
        do {
            let token = try await authService.login(user: dto.username, pass: dto.password)
            return try jsonResponse(status: .ok, body: ["token": token])
        } catch {
            return errorResponse(status: .unauthorized, message: "Invalid username or password", code: 401)
        }
    }
    
    // MARK: — POST /api/v1/auth/logout
    
    @Sendable
    func logout(request: Request, context: GemmaRequestContext) async throws -> Response {
        guard let authHeader = request.headers[.authorization],
              authHeader.hasPrefix("Bearer ") else {
            return errorResponse(status: .unauthorized, message: "Missing or invalid Authorization header", code: 401)
        }
        
        let token = String(authHeader.dropFirst("Bearer ".count))
        
        do {
            try await authService.logout(token: token)
            return try jsonResponse(status: .ok, body: ["status": "ok"])
        } catch {
            return errorResponse(status: .unauthorized, message: "Invalid token", code: 401)
        }
    }
    
    // MARK: — Helpers

    private func jsonResponse<T: Encodable>(status: HTTPResponse.Status, body: T) throws -> Response {
        let data = try JSONEncoder().encode(body)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: buffer)
        )
    }

    private func errorResponse(status: HTTPResponse.Status, message: String, code: Int) -> Response {
        let envelope = ErrorResponse(error: message, code: code)
        let data = (try? JSONEncoder().encode(envelope)) ?? Data()
        var buffer = ByteBuffer()
        buffer.writeBytes(data)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: buffer)
        )
    }
}
