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
        
        for await event in await terminal.readEvents() {
            guard isRunning else { break }
            
            switch event {
            case .lineSubmitted(let line):
                if line.lowercased() == "exit" || line.lowercased() == "quit" {
                    isRunning = false
                    activeTask?.cancel()
                    break
                }
                if activeTask != nil {
                    messageQueue.append(line)
                    await terminal.printOutput("\u{1B}[2m[Queued]\u{1B}[0m\n")
                } else {
                    messageQueue.append(line)
                }
                
            case .lineQueued(let line):
                messageQueue.append(line)
                await terminal.printOutput("\u{1B}[2m[Queued]\u{1B}[0m\n")
                
            case .interrupt:
                if let task = activeTask {
                    task.cancel()
                    activeTask = nil
                    await terminal.setBusy(false)
                } else {
                    isRunning = false
                    break
                }
                
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
                    await terminal.printOutput("\u{1B}[34mGemma:\u{1B}[0m ")
                    
                    do {
                        let finalPrompt = try await PromptContextBuilder.build(prompt: prompt)
                        let request = GenerationRequest(prompt: finalPrompt, maxTokens: maxTokens)
                        let stream = try await orchestrator.generateStream(request: request)
                        
                        var stats: GenerationResponse?
                        for try await chunk in stream {
                            if Task.isCancelled {
                                await terminal.printOutput("\n[Cancelled]\n")
                                break
                            }
                            switch chunk {
                            case .text(let t):
                                await terminal.printOutput(t)
                            case .metadata(let m):
                                stats = m
                            }
                        }
                        
                        await terminal.printOutput("\n\n")
                        if let stats = stats, !Task.isCancelled {
                            let statsString = "\u{1B}[2mTokens: \(stats.promptTokens) in / \(stats.completionTokens) out | TPS: \(String(format: "%.2f", stats.tokensPerSecond)) | TTFT: \(String(format: "%.3fs", stats.timeToFirstToken)) | Memory: \(stats.memory.activeBytes / 1024 / 1024)MB\u{1B}[0m"
                            await terminal.printOutput(statsString + "\n")
                        }
                    } catch {
                        await terminal.printOutput("\u{1B}[31mError:\u{1B}[0m \(error.localizedDescription)\n")
                    }
                    
                    if !Task.isCancelled {
                        await self.clearActiveTask()
                    }
                }
                
                self.activeTask = task
            }
            
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms polling
        }
    }
    
    private func clearActiveTask() async {
        self.activeTask = nil
        await terminal.setBusy(false)
    }
}
