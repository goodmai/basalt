import BasaltCore
import Foundation

public actor SelfEditSessionStore {
    private var records: [String: SelfEditSessionRecord] = [:]

    public init() {}

    public func register(_ record: SelfEditSessionRecord) {
        records[record.metadata.sessionId] = record
    }

    public func updatePhase(
        sessionId: String,
        phase: EditSessionPhase,
        turnCount: Int,
        savedAdapterPath: String?
    ) {
        guard var existing = records[sessionId] else { return }
        existing.phase = phase
        existing.turnCount = turnCount
        existing.savedAdapterPath = savedAdapterPath
        records[sessionId] = existing
    }

    public func record(for sessionId: String) -> SelfEditSessionRecord? {
        records[sessionId]
    }

    public func allRecords() -> [SelfEditSessionRecord] {
        records.values.sorted { $0.metadata.startedAtISO8601 > $1.metadata.startedAtISO8601 }
    }
}

struct SelfEditStatusOutput: Codable, Sendable {
    struct PersistedTranscript: Codable, Sendable {
        let sessionId: String
        let path: String
        let entries: Int
    }

    let sessions: [SelfEditSessionRecord]
    let transcripts: [PersistedTranscript]
}

extension MCPToolCatalog {
    static let selfEditStartDescriptor = MCPToolDescriptor(
        name: "selfedit_start",
        description: "Experiment: validate and register a copy-on-write self-rewriting session (#676, #770). Requires serve --research-tools and explicit experimentalSelfRewrite=true (plus allowCloudEditor=true for cloud:gemini/*).",
        inputSchema: ResearchTool.object(
            [
                "experimentalSelfRewrite": ["type": "boolean", "description": "Must be true to opt into experimental self-rewriting"],
                "practicePath": ["type": "string", "description": "Path to JSONL practice dataset"],
                "decontaminatePath": ["type": "string", "description": "Optional benchmark JSONL/text path for n-gram decontamination"],
                "practiceCount": ["type": "integer", "minimum": 1, "description": "Optional practice split size (rest stays private reserved)"],
                "editor": ["type": "string", "description": "Editor mode: 'local' (default) or 'cloud:gemini/<model>'"],
                "allowCloudEditor": ["type": "boolean", "description": "Explicit opt-in required for cloud:gemini/* editors"],
                "task": ["type": "string", "description": "'improve' (default) or 'repair:<tensor>:<rows>:<scale>'"],
                "scope": ["type": "string", "description": "'text_backbone_only' (default) or 'include_multimodal'"],
                "maxForwardPasses": ["type": "integer", "minimum": 1, "description": "Forward-pass budget (default 20)"],
                "maxToolCalls": ["type": "integer", "minimum": 1, "description": "Tool-call budget (default 200)"],
                "maxCloudUSD": ["type": "number", "minimum": 0, "description": "Maximum cloud spend in USD (default 2.0)"],
            ],
            required: ["experimentalSelfRewrite", "practicePath"]
        )
    )

    static let selfEditStatusDescriptor = MCPToolDescriptor(
        name: "selfedit_status",
        description: "Experiment: report status of registered copy-on-write self-rewriting sessions and persisted ~/.basalt/selfedit/*.jsonl transcripts (#676, #770).",
        inputSchema: ResearchTool.object([
            "sessionId": ["type": "string", "description": "Optional session id filter"],
        ])
    )
}

extension MCPServer {
    func selfEdit(_ tool: MCPTool, _ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        switch tool {
        case .selfEditStart: try await selfEditStart(args)
        default: try await selfEditStatus(args)
        }
    }

    private static let maxSelfEditDatasetBytes = 32 * 1024 * 1024

    func selfEditStart(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard (try MCPArguments.optionalBoolean("experimentalSelfRewrite", in: args)) == true else {
            throw .invalidParams("selfedit_start requires experimentalSelfRewrite=true")
        }
        let practicePath = try MCPArguments.required("practicePath", MCPArguments.string("practicePath", in: args))
        let editor = (try MCPArguments.string("editor", in: args)) ?? "local"
        let allowCloud = (try MCPArguments.optionalBoolean("allowCloudEditor", in: args)) ?? false
        let taskRaw = (try MCPArguments.string("task", in: args)) ?? "improve"
        let scopeRaw = (try MCPArguments.string("scope", in: args)) ?? EditTargetScope.textBackboneOnly.rawValue
        let decontamPath = try MCPArguments.string("decontaminatePath", in: args)
        let practiceCount = try MCPArguments.integer("practiceCount", in: args, 1 ... 100_000)
        let maxPasses = (try MCPArguments.integer("maxForwardPasses", in: args, 1 ... 10_000)) ?? 20
        let maxTools = (try MCPArguments.integer("maxToolCalls", in: args, 1 ... 100_000)) ?? 200
        let maxCloudUSD = (try MCPArguments.double("maxCloudUSD", in: args)) ?? 2.0
        guard maxCloudUSD.isFinite, maxCloudUSD >= 0 else {
            throw .invalidParams("maxCloudUSD must be a finite non-negative number")
        }

        try Self.validateSelfEditEditor(editor, allowCloud: allowCloud)
        guard EditTargetScope(rawValue: scopeRaw) != nil else {
            throw .invalidParams("invalid scope '\(scopeRaw)'")
        }
        let taskMode = try Self.parseSelfEditTask(taskRaw)
        _ = try Self.validateSelfEditBudget(passes: maxPasses, tools: maxTools, editor: editor, maxCloudUSD: maxCloudUSD)
        let allowedRoots = harnessRoots + [selfEditDirectory]
        let practiceURL = try Self.validatedSelfEditFile(practicePath, label: "practicePath", allowedRoots: allowedRoots)
        let decontamURL: URL? = if let decontamPath {
            try Self.validatedSelfEditFile(decontamPath, label: "decontaminatePath", allowedRoots: allowedRoots)
        } else {
            nil
        }
        let splitInfo = try Self.loadSelfEditSplit(
            practiceURL: practiceURL,
            decontamURL: decontamURL,
            practiceCount: practiceCount
        )
        let sessionId = "selfedit-\(UUID().uuidString.prefix(8))"
        let loaded = await orchestrator.currentModelId
        let transcriptURL = selfEditDirectory.appending(path: "\(sessionId).jsonl")
        let metadata = SelfEditSessionMetadata(
            sessionId: sessionId,
            modelPath: loaded.isEmpty ? (modelId ?? "unloaded") : loaded,
            editor: editor,
            taskMode: taskMode,
            transcriptPath: transcriptURL.path,
            startedAtISO8601: ISO8601DateFormatter().string(from: Date())
        )
        let record = SelfEditSessionRecord(
            metadata: metadata,
            phase: .running,
            practiceCount: splitInfo.split.practice.count,
            reservedCount: splitInfo.split.reservedCount,
            removedDuplicates: splitInfo.removedDuplicates
        )
        await selfEditSessions.register(record)
        return .json(record)
    }

    func selfEditStatus(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        let filterId = try MCPArguments.string("sessionId", in: args)
        let sessions: [SelfEditSessionRecord]
        if let filterId, !filterId.isEmpty {
            sessions = await selfEditSessions.record(for: filterId).map { [$0] } ?? []
        } else {
            sessions = await selfEditSessions.allRecords()
        }
        let transcripts = Self.scanSelfEditTranscripts(in: selfEditDirectory, filterId: filterId)
        return .json(SelfEditStatusOutput(sessions: sessions, transcripts: transcripts))
    }

    private static func validateSelfEditEditor(_ editor: String, allowCloud: Bool) throws(MCPFailure) {
        if editor == "local" { return }
        guard editor.hasPrefix("cloud:gemini/"), editor.count > "cloud:gemini/".count else {
            throw .invalidParams("invalid editor '\(editor)' (expected 'local' or 'cloud:gemini/<model>')")
        }
        guard allowCloud else {
            throw .invalidParams(SelfRewriteSessionError.cloudEditorDisabled.description)
        }
    }

    private static func parseSelfEditTask(_ raw: String) throws(MCPFailure) -> SelfRewriteTaskMode {
        do {
            return try SelfRewriteTaskMode.parse(raw)
        } catch {
            throw .invalidParams(error.description)
        }
    }

    private static func validateSelfEditBudget(
        passes: Int,
        tools: Int,
        editor: String,
        maxCloudUSD: Double
    ) throws(MCPFailure) -> SelfRewriteBudget {
        do {
            return try SelfRewriteBudget(
                maxForwardPasses: passes,
                maxToolCalls: tools,
                maxReservedTests: 1,
                maxEditApplies: 32,
                maxCloudUSD: editor.hasPrefix("cloud:") ? maxCloudUSD : nil
            )
        } catch {
            throw .invalidParams(error.description)
        }
    }
}

extension MCPServer {
    static func validatedSelfEditFile(
        _ rawPath: String,
        label: String,
        allowedRoots: [URL]
    ) throws(MCPFailure) -> URL {
        let resolved = URL(fileURLWithPath: rawPath).resolvingSymlinksInPath().standardizedFileURL
        let allowed = allowedRoots.map { $0.resolvingSymlinksInPath().standardizedFileURL.path }
        let insideAllowed = allowed.contains { root in
            let prefix = root.hasSuffix("/") ? root : root + "/"
            return resolved.path.hasPrefix(prefix)
        }
        guard insideAllowed else {
            throw .invalidParams("\(label) must reside inside an allowed root")
        }
        guard let values = try? resolved.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true else {
            throw .invalidParams("\(label) must refer to a readable regular file")
        }
        if let bytes = values.fileSize, bytes > maxSelfEditDatasetBytes {
            throw .invalidParams("\(label) exceeds \(maxSelfEditDatasetBytes) byte limit")
        }
        return resolved
    }

    private static func loadSelfEditSplit(
        practiceURL: URL,
        decontamURL: URL?,
        practiceCount: Int?
    ) throws(MCPFailure) -> (split: PairedDatasetSplit, removedDuplicates: Int) {
        do {
            return try SelfRewriteDatasetLoader.buildSplit(
                practiceURL: practiceURL,
                decontaminateURL: decontamURL,
                practiceCount: practiceCount
            )
        } catch {
            throw .invalidParams("failed to load practice dataset: \(error)")
        }
    }

    private static func scanSelfEditTranscripts(
        in directory: URL,
        filterId: String?
    ) -> [SelfEditStatusOutput.PersistedTranscript] {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        return urls
            .filter { $0.pathExtension == "jsonl" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { url -> SelfEditStatusOutput.PersistedTranscript? in
                let sessionId = url.deletingPathExtension().lastPathComponent
                if let filterId, !filterId.isEmpty, sessionId != filterId { return nil }
                let count = (try? SelfEditSessionRecord.loadTranscript(at: url).count) ?? 0
                return SelfEditStatusOutput.PersistedTranscript(sessionId: sessionId, path: url.path, entries: count)
            }
    }
}
