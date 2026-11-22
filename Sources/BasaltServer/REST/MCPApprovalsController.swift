import BasaltCore
import Foundation
import HTTPTypes
import Hummingbird

struct MCPApprovalsController: Sendable {
    let approvals: HITLApprovalActor
    let clock: @Sendable () -> Double

    init(approvals: HITLApprovalActor, clock: @escaping @Sendable () -> Double = { Date().timeIntervalSince1970 }) {
        self.approvals = approvals
        self.clock = clock
    }

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.get("/mcp/approvals", use: listApprovals)
        group.post("/mcp/approvals/:id/approve", use: approve)
        group.post("/mcp/approvals/:id/reject", use: reject)
    }

    @Sendable
    func listApprovals(request _: Request, context _: BasicRequestContext) async throws -> Response {
        _ = await approvals.expire(now: clock())
        let payload = ApprovalsSnapshot(
            pending: await approvals.pendingRequests(),
            audit: await approvals.auditLog()
        )
        return try jsonResponse(payload, status: .ok)
    }

    @Sendable
    func approve(request: Request, context: BasicRequestContext) async throws -> Response {
        try await resolve(request: request, context: context, approved: true)
    }

    @Sendable
    func reject(request: Request, context: BasicRequestContext) async throws -> Response {
        try await resolve(request: request, context: context, approved: false)
    }

    private func resolve(request: Request, context: BasicRequestContext, approved: Bool) async throws -> Response {
        guard let id = context.parameters.get("id"), !id.isEmpty else {
            return ProviderErrorResponse.response(status: .badRequest, message: "Missing approval id", shape: .openAI)
        }
        let body = try await decodeDecisionBody(request)
        let decision: HITLApprovalDecision
        do throws(BasaltError) {
            decision = try await approvals.decide(
                id: id,
                approved: approved,
                reviewer: body.reviewer ?? "operator",
                reason: body.reason ?? (approved ? "approved" : "rejected"),
                now: clock()
            )
        } catch {
            return ProviderErrorResponse.response(status: .notFound, message: error.localizedDescription, error: error, shape: .openAI)
        }
        return try jsonResponse(decision, status: .ok)
    }

    private func decodeDecisionBody(_ request: Request) async throws -> DecisionBody {
        let buffer = try await request.body.collect(upTo: 65_536)
        guard buffer.readableBytes > 0 else { return DecisionBody() }
        return (try? JSONDecoder().decode(DecisionBody.self, from: Data(buffer.readableBytesView))) ?? DecisionBody()
    }

    private func jsonResponse<T: Encodable>(_ value: T, status: HTTPResponse.Status) throws -> Response {
        let data = try JSONEncoder().encode(value)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private struct ApprovalsSnapshot: Encodable {
        let pending: [HITLApprovalRequest]
        let audit: [HITLAuditEntry]
    }

    private struct DecisionBody: Decodable {
        var reviewer: String?
        var reason: String?
    }
}
