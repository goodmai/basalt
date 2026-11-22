import Foundation

public enum HITLApprovalStatus: String, Sendable, Codable, Equatable {
    case pending
    case approved
    case rejected
    case timedOut = "timed_out"
}

public struct HITLToolPolicy: Sendable, Codable, Equatable {
    public static let defaultMutatingTools: Set<String> = [
        "switch_model",
        "basalt_add_knowledge",
        "gemma_add_knowledge",
        "weights_delta",
        "overlay_undo",
        "adapter_apply",
        "tensor_pack",
        "steer_commit",
        "intervene",
    ]

    public var mutatingTools: Set<String>
    public var defaultTimeoutSeconds: Double
    public var maxPendingRequests: Int

    public init(
        mutatingTools: Set<String> = HITLToolPolicy.defaultMutatingTools,
        defaultTimeoutSeconds: Double = 300,
        maxPendingRequests: Int = 64
    ) {
        self.mutatingTools = mutatingTools
        self.defaultTimeoutSeconds = defaultTimeoutSeconds
        self.maxPendingRequests = maxPendingRequests
    }

    public func requiresApproval(toolName: String) -> Bool {
        mutatingTools.contains(toolName)
    }

    public func validate() throws(BasaltError) {
        guard defaultTimeoutSeconds.isFinite, defaultTimeoutSeconds > 0 else {
            throw .invalidRequestStructure(details: "HITL policy: defaultTimeoutSeconds must be positive")
        }
        guard maxPendingRequests >= 1 else {
            throw .invalidRequestStructure(details: "HITL policy: maxPendingRequests must be >= 1")
        }
    }
}

public struct HITLApprovalRequest: Sendable, Codable, Equatable {
    public var id: String
    public var toolName: String
    public var argumentsSummary: String
    public var createdAt: Double
    public var expiresAt: Double
    public var status: HITLApprovalStatus

    public init(
        id: String,
        toolName: String,
        argumentsSummary: String,
        createdAt: Double,
        expiresAt: Double,
        status: HITLApprovalStatus = .pending
    ) {
        self.id = id
        self.toolName = toolName
        self.argumentsSummary = argumentsSummary
        self.createdAt = createdAt
        self.expiresAt = expiresAt
        self.status = status
    }
}

public struct HITLApprovalDecision: Sendable, Codable, Equatable {
    public var requestId: String
    public var approved: Bool
    public var reviewer: String
    public var reason: String
    public var decidedAt: Double

    public init(requestId: String, approved: Bool, reviewer: String, reason: String, decidedAt: Double) {
        self.requestId = requestId
        self.approved = approved
        self.reviewer = reviewer
        self.reason = reason
        self.decidedAt = decidedAt
    }
}

public struct HITLAuditEntry: Sendable, Codable, Equatable {
    public var requestId: String
    public var toolName: String
    public var status: HITLApprovalStatus
    public var reviewer: String
    public var reason: String
    public var timestamp: Double

    public init(
        requestId: String,
        toolName: String,
        status: HITLApprovalStatus,
        reviewer: String,
        reason: String,
        timestamp: Double
    ) {
        self.requestId = requestId
        self.toolName = toolName
        self.status = status
        self.reviewer = reviewer
        self.reason = reason
        self.timestamp = timestamp
    }
}

public struct HITLStateStore: Sendable, Equatable {
    public var policy: HITLToolPolicy
    private var pending: [String: HITLApprovalRequest]
    public private(set) var auditLog: [HITLAuditEntry]

    public init(policy: HITLToolPolicy = .init()) {
        self.policy = policy
        self.pending = [:]
        self.auditLog = []
    }

    public func pendingRequests() -> [HITLApprovalRequest] {
        pending.values.sorted { $0.createdAt < $1.createdAt }
    }

    public mutating func submit(
        id: String,
        toolName: String,
        argumentsSummary: String,
        now: Double
    ) throws(BasaltError) -> HITLApprovalRequest {
        try policy.validate()
        _ = expirePending(now: now)
        guard pending[id] == nil else {
            throw .invalidRequestStructure(details: "HITL approval '\(id)' already exists")
        }
        guard pending.count < policy.maxPendingRequests else {
            throw .invalidRequestStructure(details: "HITL approval queue full (\(policy.maxPendingRequests))")
        }
        let req = HITLApprovalRequest(
            id: id,
            toolName: toolName,
            argumentsSummary: argumentsSummary,
            createdAt: now,
            expiresAt: now + policy.defaultTimeoutSeconds,
            status: .pending
        )
        pending[id] = req
        return req
    }

    public mutating func decide(
        id: String,
        approved: Bool,
        reviewer: String,
        reason: String,
        now: Double
    ) throws(BasaltError) -> HITLApprovalDecision {
        _ = expirePending(now: now)
        guard let req = pending.removeValue(forKey: id) else {
            throw .invalidRequestStructure(details: "HITL approval '\(id)' not found or already resolved")
        }
        let status: HITLApprovalStatus = approved ? .approved : .rejected
        auditLog.append(HITLAuditEntry(
            requestId: id, toolName: req.toolName, status: status, reviewer: reviewer, reason: reason, timestamp: now
        ))
        return HITLApprovalDecision(requestId: id, approved: approved, reviewer: reviewer, reason: reason, decidedAt: now)
    }

    public mutating func expirePending(now: Double) -> [HITLApprovalDecision] {
        let expiredIds = pending.values.filter { now >= $0.expiresAt }.map(\.id).sorted()
        var decisions: [HITLApprovalDecision] = []
        for id in expiredIds {
            guard let req = pending.removeValue(forKey: id) else { continue }
            auditLog.append(HITLAuditEntry(
                requestId: id, toolName: req.toolName, status: .timedOut, reviewer: "system", reason: "approval_timeout", timestamp: now
            ))
            decisions.append(HITLApprovalDecision(
                requestId: id, approved: false, reviewer: "system", reason: "approval_timeout", decidedAt: now
            ))
        }
        return decisions
    }
}

public actor HITLApprovalActor {
    private var store: HITLStateStore
    private var waiters: [String: [CheckedContinuation<HITLApprovalDecision, Never>]] = [:]
    private var resolved: [String: HITLApprovalDecision] = [:]

    public init(policy: HITLToolPolicy = .init()) {
        self.store = HITLStateStore(policy: policy)
    }

    public func requiresApproval(toolName: String) -> Bool {
        store.policy.requiresApproval(toolName: toolName)
    }

    public func pendingRequests() -> [HITLApprovalRequest] {
        store.pendingRequests()
    }

    public func auditLog() -> [HITLAuditEntry] {
        store.auditLog
    }

    public func enqueue(
        id: String,
        toolName: String,
        argumentsSummary: String,
        now: Double
    ) throws(BasaltError) -> HITLApprovalRequest {
        _ = expire(now: now)
        return try store.submit(id: id, toolName: toolName, argumentsSummary: argumentsSummary, now: now)
    }

    public func awaitDecision(id: String) async -> HITLApprovalDecision {
        if let decision = resolved[id] {
            return decision
        }
        let deadline = store.pendingRequests().first(where: { $0.id == id })?.expiresAt
        let wallNow = Date().timeIntervalSince1970
        let timerTask: Task<Void, Never>? = deadline.flatMap { expiresAt in
            guard expiresAt > wallNow else { return nil }
            return Task { [weak self] in
                let remaining = max(0, expiresAt - Date().timeIntervalSince1970)
                try? await Task.sleep(for: .seconds(remaining))
                _ = await self?.expire(now: max(expiresAt, Date().timeIntervalSince1970))
            }
        }
        defer { timerTask?.cancel() }
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if let decision = resolved[id] {
                    continuation.resume(returning: decision)
                } else {
                    waiters[id, default: []].append(continuation)
                }
            }
        } onCancel: {
            Task { await self.cancelWaiters(for: id) }
        }
    }

    private func cancelWaiters(for id: String) {
        if let continuations = waiters.removeValue(forKey: id) {
            let decision = resolved[id] ?? HITLApprovalDecision(
                requestId: id,
                approved: false,
                reviewer: "system",
                reason: "cancelled",
                decidedAt: Date().timeIntervalSince1970
            )
            for continuation in continuations {
                continuation.resume(returning: decision)
            }
        }
    }

    public func decide(
        id: String,
        approved: Bool,
        reviewer: String,
        reason: String,
        now: Double
    ) throws(BasaltError) -> HITLApprovalDecision {
        _ = expire(now: now)
        let decision = try store.decide(id: id, approved: approved, reviewer: reviewer, reason: reason, now: now)
        finish(decision)
        return decision
    }

    public func expire(now: Double) -> [HITLApprovalDecision] {
        let decisions = store.expirePending(now: now)
        for decision in decisions {
            finish(decision)
        }
        return decisions
    }

    private func finish(_ decision: HITLApprovalDecision) {
        resolved[decision.requestId] = decision
        if let continuations = waiters.removeValue(forKey: decision.requestId) {
            for continuation in continuations {
                continuation.resume(returning: decision)
            }
        }
    }
}
