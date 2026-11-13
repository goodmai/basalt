import BasaltCore
import Foundation
import SwiftUI
@testable import BasaltApp
@testable import BasaltAppCore

public struct VisualSceneMatrixEntry: Hashable, Sendable, CustomStringConvertible {
    public let spec: VisualSceneSpec
    public let variant: TraitVariant

    public var description: String { "\(spec.id.rawValue) [\(variant.slug)]" }
}

public enum VisualSceneCatalog {
    public static let specs: [(spec: VisualSceneSpec, variants: [TraitVariant])] = [
        (VisualSceneSpec(id: .rootEmpty, width: 1040, height: 720), [.lightStandard, .darkStandard, .darkReducedTransparency]),
        (VisualSceneSpec(id: .rootBannerFailed, width: 1040, height: 720), [.lightStandard, .darkStandard]),
        (VisualSceneSpec(id: .chatConversationRich, width: 1040, height: 760), [.lightStandard, .darkStandard, .darkReducedTransparency, .darkRussian]),
        (VisualSceneSpec(id: .chatInspectorOpen, width: 1200, height: 760), [.lightStandard, .darkStandard]),
        (VisualSceneSpec(id: .quickAskCompact, width: 680, height: 160), [.lightStandard, .darkStandard, .darkReducedTransparency]),
        (VisualSceneSpec(id: .quickAskStreaming, width: 680, height: 440), [.lightStandard, .darkStandard]),
        (VisualSceneSpec(id: .menuBarPopover, width: 300, height: 420), [.lightStandard, .darkStandard]),
        (VisualSceneSpec(id: .libraryMixed, width: 680, height: 480), [.lightStandard, .darkStandard]),
        (VisualSceneSpec(id: .monitorCompose, width: 480, height: 560), [.lightStandard, .darkStandard]),
        (VisualSceneSpec(id: .settingsTabs, width: 520, height: 380), [.lightStandard, .darkStandard]),
    ]

    public static var entries: [VisualSceneMatrixEntry] {
        specs.flatMap { item in
            item.variants.map { VisualSceneMatrixEntry(spec: item.spec, variant: $0) }
        }
    }

    @MainActor
    public static func render(spec: VisualSceneSpec, variant: TraitVariant) async throws(VisualTestError) -> RenderedBitmap {
        switch spec.id {
        case .rootEmpty:
            let model = try await VisualSceneFixtures.makeAppModel(phase: .ready(port: 58_100), withSummaries: false)
            return try OffscreenSceneRenderer.render(RootView(model: model), spec: spec, variant: variant)
        case .rootBannerFailed:
            let phase = EngineState.Phase.failed(log: ["error: Metal residency ceiling exceeded on shard 4"])
            let model = try await VisualSceneFixtures.makeAppModel(phase: phase, withSummaries: true)
            return try OffscreenSceneRenderer.render(RootView(model: model), spec: spec, variant: variant)
        case .chatConversationRich:
            let (store, _) = VisualSceneFixtures.makeInspectorChatStore()
            return try OffscreenSceneRenderer.render(ChatView(store: store), spec: spec, variant: variant)
        case .chatInspectorOpen:
            let (store, answerID) = VisualSceneFixtures.makeInspectorChatStore()
            let view = HStack(spacing: 0) {
                ChatView(store: store)
                Divider()
                InspectorView(store: store, selected: answerID).frame(width: 320)
            }
            return try OffscreenSceneRenderer.render(view, spec: spec, variant: variant)
        case .quickAskCompact:
            let store = VisualSceneFixtures.makeQuickAskStore(streaming: false)
            return try OffscreenSceneRenderer.render(
                QuickAskView(store: store) {}, spec: spec, variant: variant, desktopBackdrop: true
            )
        case .quickAskStreaming:
            let store = VisualSceneFixtures.makeQuickAskStore(streaming: true)
            return try OffscreenSceneRenderer.render(
                QuickAskView(store: store) {}, spec: spec, variant: variant, desktopBackdrop: true
            )
        case .menuBarPopover:
            let model = try await VisualSceneFixtures.makeAppModel(phase: .ready(port: 58_100), withSummaries: true)
            return try OffscreenSceneRenderer.render(MenuBarContent(model: model) {}, spec: spec, variant: variant)
        case .libraryMixed:
            let store = VisualSceneFixtures.makeLibraryStore()
            return try OffscreenSceneRenderer.render(LibraryView(store: store), spec: spec, variant: variant)
        case .monitorCompose:
            let store = VisualSceneFixtures.makeMonitorStore()
            return try OffscreenSceneRenderer.render(MonitorView(store: store), spec: spec, variant: variant)
        case .settingsTabs:
            let keychain = KeychainClient.memory()
            _ = keychain.write("OPENROUTER_API_KEY", "sk-or-fixture")
            return try OffscreenSceneRenderer.render(SettingsView(keychain: keychain), spec: spec, variant: variant)
        default:
            throw .renderFailed(spec.id)
        }
    }
}

@MainActor
enum VisualSceneFixtures {
    static func fixedUUID(_ index: Int) -> UUID {
        UUID(uuidString: String(format: "00000000-0000-4000-8000-%012d", index)) ?? UUID()
    }

    static func makeAppModel(phase: EngineState.Phase, withSummaries: Bool) async throws(VisualTestError) -> AppModel {
        let model: AppModel
        do {
            model = try AppModel(
                launch: HelperLaunch(executable: URL(fileURLWithPath: "/usr/bin/false"), arguments: []),
                model: "mlx-community/gemma-4-12B-it-qat-4bit",
                storeURL: nil,
                drafts: DraftStore(load: { _ in "" }, save: { _, _ in })
            )
        } catch {
            throw .ioFailed(error.localizedDescription)
        }
        if withSummaries {
            await seedSummaries(in: model)
        }
        var engineState = EngineState()
        engineState.phase = phase
        await model.engineChanged(engineState)
        return model
    }

    private static func seedSummaries(in model: AppModel) async {
        let titles = [
            "Out-of-core MoE slot table math",
            "Liquid Glass scroll-pocket hierarchy",
            "Swift 6.4 RawSpan safetensors parser",
            "Prompt cache prefix reuse audit",
            "Speculative lookup p95 latency",
        ]
        for (offset, title) in titles.enumerated() {
            var snapshot = ConversationSnapshot(id: ConversationID(fixedUUID(100 + offset)), model: model.model)
            snapshot.title = title
            snapshot.messages = [MessageSnapshot(id: MessageID(fixedUUID(200 + offset)), role: .user, parts: [.text(title)])]
            try? await model.repository.save(snapshot)
        }
        await model.reload()
    }

    static func makeInspectorChatStore() -> (ChatStore, MessageID) {
        var conversation = ConversationSnapshot(
            id: ConversationID(fixedUUID(1)),
            model: "mlx-community/Qwen3.6-35B-A3B-8bit",
            systemPrompt: "Answer concisely with exact measurements and zero code comments."
        )
        conversation.title = "MoE Expert Streaming & Liquid Glass"
        let q1 = MessageSnapshot(
            id: MessageID(fixedUUID(10)),
            role: .user,
            parts: [
                .attachment(AttachmentPart(name: "invoice-q4.pdf", kind: "pdf", perceived: "Subtotal $1,240.00 · Metal Cloud $0.0042")),
                .attachment(AttachmentPart(name: "voice-note.wav", kind: "audio", perceived: "Check expert cache hit ratio on 24 GB Mac.")),
                .text("Summarize the Q4 compute receipt and verify the 16 KB slot alignment."),
            ]
        )
        let answerID = MessageID(fixedUUID(11))
        var a1 = MessageSnapshot(
            id: answerID,
            role: .assistant,
            parts: [
                .reasoning("Checking page-aligned 16 KB slot strides and verifying cloud perception cost."),
                .toolCall(ToolCallOutput(id: "call_1", name: "kv_inspect", arguments: #"{"layer":0}"#)),
                .text("All **40 MoE layers** align to `16,384`-byte page boundaries with **8.77 tok/s** decode and `0` swap-outs."),
            ]
        )
        var receipt = Receipt(model: "mlx-community/Qwen3.6-35B-A3B-8bit")
        receipt.promptTokens = 612
        receipt.completionTokens = 94
        receipt.timeToFirstToken = .milliseconds(420)
        receipt.tokensPerSecond = 8.8
        receipt.stages = [
            StageReceipt(stage: "perceive", model: "apple-vision", ms: 64),
            StageReceipt(stage: "verify", model: "openrouter/qwen-max", ms: 310, cost: 0.0042),
        ]
        a1.receipt = receipt
        conversation.messages = [q1, a1]
        var state = ChatState(conversation: conversation)
        state.draft = "Compare warm-start vs cold-start hit rate"
        if let chip = ChatAttachment(name: "trace.png", data: Data([0x89, 0x50, 0x4E, 0x47]), id: fixedUUID(30)) {
            state.attachments = [chip]
        }
        return (ChatStore(state, reduce: ChatReducer.reduce, handle: { _ in .none }), answerID)
    }

    static func makeQuickAskStore(streaming: Bool) -> ChatStore {
        var conversation = ConversationSnapshot(id: ConversationID(fixedUUID(40)), model: "mlx-community/gemma-4-12B-it-qat-4bit")
        var state = ChatState(conversation: conversation)
        if streaming {
            let run = RunID(fixedUUID(41))
            conversation.messages = [
                MessageSnapshot(id: MessageID(fixedUUID(42)), role: .user, parts: [.text("How does GlassEffectContainer optimize backdrop blur?")]),
                MessageSnapshot(
                    id: MessageID(fixedUUID(43)),
                    role: .assistant,
                    parts: [.text("`GlassEffectContainer` shares a single backdrop sampling and SDF refraction pass across sibling glass shapes, keeping 2,000-message streaming well inside the **16.7 ms** 60 fps frame budget.")],
                    state: .streaming
                ),
            ]
            state = ChatState(conversation: conversation)
            state.draft = "How does GlassEffectContainer optimize backdrop blur?"
            state.run = .streaming(run)
        } else {
            state.draft = "Explain zero-copy RawSpan pixel diffing"
        }
        return ChatStore(state, reduce: ChatReducer.reduce, handle: { _ in .none })
    }

    static func makeLibraryStore() -> LibraryStore {
        var library = LibraryState(pending: ["mlx-community/Qwen3.6-35B-A3B-8bit"])
        _ = LibraryReducer.reduce(&library, .listed([
            LibraryModel(id: "mlx-community/gemma-4-12B-it-qat-4bit", sizeBytes: 10_200_547_328, isLoaded: true),
            LibraryModel(id: "LiquidAI/LFM2.5-8B-A1B-MLX-4bit", sizeBytes: 4_834_029_941, isLoaded: false),
            LibraryModel(id: "mlx-community/Qwen3-235B-A22B-8bit", sizeBytes: 250_000_000_000, isLoaded: false),
        ]))
        _ = LibraryReducer.reduce(&library, .fitted(
            "mlx-community/gemma-4-12B-it-qat-4bit",
            ModelFit(fits: true, level: "comfortable", requiredBytes: 11_000_000_000, availableBytes: 24_000_000_000, warning: nil)
        ))
        _ = LibraryReducer.reduce(&library, .fitted(
            "LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
            ModelFit(fits: true, level: "comfortable", requiredBytes: 5_400_000_000, availableBytes: 24_000_000_000, warning: nil)
        ))
        _ = LibraryReducer.reduce(&library, .fitted(
            "mlx-community/Qwen3-235B-A22B-8bit",
            ModelFit(fits: false, level: "insufficient", requiredBytes: 260_000_000_000, availableBytes: 24_000_000_000, warning: "Exceeds unified memory")
        ))
        var activeDownload = Download()
        activeDownload.done = 15_842_989_260
        activeDownload.total = 37_721_403_002
        activeDownload.active = true
        library.downloads["mlx-community/Qwen3.6-35B-A3B-8bit"] = activeDownload
        library.problem = "Mirror fallback active: hf-mirror reachable"
        return Store(library, reduce: LibraryReducer.reduce, handle: { _ in .none })
    }

    static func makeMonitorStore() -> MonitorStore {
        var snapshot = ServerTelemetrySnapshot()
        snapshot.tokensPerSecond = 78.4
        snapshot.timeToFirstToken = 0.24
        snapshot.totalRequests = 42
        snapshot.activeRequests = 1
        snapshot.totalTokensGenerated = 18_420
        snapshot.modelRAMMB = 9_840
        snapshot.gpuActiveBytes = 10_317_987_840
        snapshot.gpuPeakBytes = 11_274_289_152
        snapshot.availableRAMGB = 12.6
        snapshot.totalRAMGB = 24
        snapshot.chipModel = "Apple M4 Pro"
        snapshot.gpuName = "Apple M4 Pro GPU (20-core)"
        let composeJSON = """
        {
          "name": "hybrid-desk",
          "pipelines": ["perceive-and-answer", "code-audit"],
          "spent_today": 0.0184,
          "models": [
            {"name": "gemma-4-12b", "source": "hf:mlx-community/gemma-4-12B-it-qat-4bit", "keep": "always", "state": "ready"},
            {"name": "qwen-cloud", "source": "cloud:openrouter/qwen-max", "keep": "on-demand", "state": "ready", "spent_today": 0.0184, "limit": 2.00}
          ]
        }
        """
        let compose = try? JSONDecoder().decode(ComposeInfo.self, from: Data(composeJSON.utf8))
        var state = MonitorState()
        state.polling = true
        state.snapshot = snapshot
        state.compose = compose
        return Store(state, reduce: MonitorReducer.reduce, handle: { _ in .none })
    }
}
