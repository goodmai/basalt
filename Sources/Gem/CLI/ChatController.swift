import Foundation

@available(macOS 10.15, iOS 13, *)
actor ChatController {
    private let terminal: TerminalManager
    private let orchestrator: ModelOrchestratorActor
    private let maxTokens: Int
    
    private var messageQueue: [String] = []
    private var activeTask: Task<Void, Never>? = nil
    private var isRunning = true
    
    init(orchestrator: ModelOrchestratorActor, maxTokens: Int) {
        self.terminal = TerminalManager()
        self.orchestrator = orchestrator
        self.maxTokens = maxTokens
    }
    
    func start() async {
        // Start queue processor
        Task {
            await processQueue()
        }
        
        // Setup initial model info
        let modelInfo = await orchestrator.modelInfo
        await terminal.updateInfo(model: modelInfo)
        
        for await event in await terminal.readEvents() {
            guard isRunning else { break }
            
            switch event {
            case .lineSubmitted(let line):
                if line.lowercased() == "exit" || line.lowercased() == "quit" {
                    isRunning = false
                    activeTask?.cancel()
                    break
                }
                messageQueue.append(line)
                
            case .interrupt:
                if let task = activeTask {
                    task.cancel()
                    activeTask = nil
                    await terminal.setBusy(false)
                    await terminal.updateInfo(debug: TerminalUI.dim("Interrupted"))
                }
                
            case .exit:
                isRunning = false
                activeTask?.cancel()
                break
                
            case .EOF:
                isRunning = false
                activeTask?.cancel()
                break
            }
        }
    }
    
    private func processQueue() async {
        while isRunning {
            if !messageQueue.isEmpty, activeTask == nil {
                let prompt = messageQueue.removeFirst()
                
                let task = Task {
                    await terminal.setBusy(true)
                    
                    do {
                        let finalPrompt = try await PromptContextBuilder.build(prompt: prompt)
                        await terminal.updateInfo(debug: "Thinking...")
                        
                        let request = GenerationRequest(prompt: finalPrompt, maxTokens: maxTokens)
                        let stream = try await orchestrator.generateStream(request: request)
                        
                        await terminal.printOutput("\n" + TerminalUI.info("Gemma:") + " ")
                        
                        var statsResponse: GenerationResponse?
                        
                        for try await chunk in stream {
                            if Task.isCancelled {
                                await terminal.printOutput(TerminalUI.dim("[Interrupted]"))
                                break
                            }
                            
                            switch chunk {
                            case .text(let t):
                                await terminal.printOutput(t)
                            case .metadata(let m):
                                statsResponse = m
                                let statsLine = "TPS: \(String(format: "%.1f", m.tokensPerSecond)) | In: \(m.promptTokens) | Out: \(m.completionTokens) | RAM: \(m.memory.activeBytes / 1024 / 1024)MB"
                                await terminal.updateInfo(debug: statsLine)
                            }
                        }
                        
                        await terminal.printOutput("\n")
                    } catch {
                        await terminal.updateInfo(debug: TerminalUI.error("Error: \(error.localizedDescription)"))
                    }
                    
                    if !Task.isCancelled {
                        await self.clearActiveTask()
                    }
                }
                
                self.activeTask = task
            }
            
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }
    
    private func clearActiveTask() async {
        self.activeTask = nil
        await terminal.setBusy(false)
    }
}
