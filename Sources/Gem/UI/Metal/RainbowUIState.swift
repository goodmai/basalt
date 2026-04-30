import Foundation

/// A single message in the chat history
public struct ChatMessage: Identifiable, Sendable {
    public let id = UUID()
    public let role: Role
    public var text: String
    public let timestamp: Date
    
    public enum Role: Sendable {
        case user
        case assistant
        case system
    }
    
    public init(role: Role, text: String) {
        self.role = role
        self.text = text
        self.timestamp = Date()
    }
}

@MainActor
public class RainbowUIState: ObservableObject, RenderPipeline {
    public enum Mode: Int, Sendable {
        case idle = 0
        case processing = 1
        case streaming = 2
        case error = 3
        case finished = 4
    }
    
    @Published public var currentMode: Mode = .idle
    @Published public var inputText: String = ""
    @Published public var placeholderHint: String = "Введите ваш запрос…"
    @Published public var messages: [ChatMessage] = []
    @Published public var modelName: String = "No model loaded"
    @Published public var tokensPerSecond: Double = 0
    @Published public var scrollOffset: CGFloat = 0
    
    /// State transition log for agentic testing
    @Published public var stateLog: [(Date, Mode, Mode)] = []
    
    /// Trigger for screenshot capture
    @Published public var captureScreenshotURL: URL?
    
    /// Flag for automated agentic testing
    public var isAgentMode: Bool = false
    
    /// MVI Event Coordinator
    public let coordinator: RenderCoordinator
    
    /// Reference to the active generation task for cancellation
    public var activeGenerationTask: Task<Void, Never>?
    
    private let logger = GemLogger(module: "RainbowUIState")
    
    public init() {
        self.coordinator = RenderCoordinator()
        logger.debug("Initialized RainbowUIState")
        
        Task {
            await coordinator.setRenderer(self)
        }
    }
    
    public func cancelGeneration() {
        if let task = activeGenerationTask {
            logger.info("MVI Intent: Cancelling active generation task")
            task.cancel()
            activeGenerationTask = nil
            setMode(.finished)
        }
    }
    
    public func setMode(_ mode: Mode) {
        let previous = currentMode
        self.currentMode = mode
        stateLog.append((Date(), previous, mode))
        
        logger.info("State transition: \(previous) -> \(mode)")
        logger.trace("Detailed state update: from=\(previous) to=\(mode), stack=\(Thread.callStackSymbols.prefix(3))")
        
        switch mode {
        case .idle:
            placeholderHint = "Введите ваш запрос…"
        case .processing:
            placeholderHint = "Генерирую ответ…"
        case .streaming:
            placeholderHint = "Печатаю…"
        case .error:
            placeholderHint = "Ошибка соединения"
        case .finished:
            placeholderHint = "Готово. Спросите ещё"
        }
    }
    
    public func exportHistory() -> String? {
        guard !messages.isEmpty else { return nil }
        
        var md = "# Chat Export\n\n"
        for msg in messages {
            let roleStr = msg.role == .user ? "**User**" : "**Assistant**"
            md += "\(roleStr):\n\(msg.text)\n\n"
        }
        
        let fileManager = FileManager.default
        let desktop = fileManager.urls(for: .desktopDirectory, in: .userDomainMask).first!
        let formatter = ISO8601DateFormatter()
        let filename = "chat_export_\(formatter.string(from: Date())).md"
        let fileURL = desktop.appendingPathComponent(filename)
        
        do {
            try md.write(to: fileURL, atomically: true, encoding: .utf8)
            logger.info("Chat exported to \(fileURL.path)")
            return fileURL.path
        } catch {
            logger.error("Failed to export chat: \(error.localizedDescription)")
            return nil
        }
    }

    public func clearHistory() {
        messages.removeAll()
        inputText = ""
        currentMode = .idle
        logger.info("Chat history cleared via /clear")
    }

    public func addUserMessage(_ text: String) {
        messages.append(ChatMessage(role: .user, text: text))
        logger.info("User Message added: \(text)")
    }
    
    public func addAssistantMessage(_ text: String) {
        messages.append(ChatMessage(role: .assistant, text: text))
        logger.info("Assistant Message added: \(text)")
    }
    
    public func appendToLastAssistant(_ chunk: String) {
        guard let lastIndex = messages.indices.last,
              messages[lastIndex].role == .assistant else {
            messages.append(ChatMessage(role: .assistant, text: chunk))
            return
        }
        messages[lastIndex].text += chunk
    }
    
    // MARK: - MVI RenderPipeline
    public func submit(state: RenderState) {
        // Here we map the MVI RenderState to the UI's View State
        // For example, update the current streaming message content:
        if state.isGenerating {
            if currentMode != .streaming && currentMode != .processing {
                setMode(.streaming)
            }
            // In a full implementation, RenderState would contain the full chat history or the delta.
            // For now we assume RenderState.content is the current active assistant response.
            // If the last message isn't assistant, add it.
            guard let lastIndex = messages.indices.last, messages[lastIndex].role == .assistant else {
                addAssistantMessage(state.content)
                return
            }
            // If content is empty or we are resetting, just replace it
            messages[lastIndex].text = state.content
        } else {
            if currentMode == .streaming || currentMode == .processing {
                setMode(.finished)
            }
        }
    }
    
    /// Export state log for agentic testing verification
    public func exportStateLog() -> String {
        stateLog.map { date, from, to in
            let formatter = ISO8601DateFormatter()
            return "\(formatter.string(from: date)): \(from) -> \(to)"
        }.joined(separator: "\n")
    }
}
