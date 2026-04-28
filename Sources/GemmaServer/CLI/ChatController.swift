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
                    await terminal.printOutput(TerminalUI.dim("[Queued]\n"))
                } else {
                    messageQueue.append(line)
                }
                
            case .lineQueued(let line):
                messageQueue.append(line)
                await terminal.printOutput(TerminalUI.dim("[Queued]\n"))
                
            case .interrupt:
                if let task = activeTask {
                    task.cancel()
                    activeTask = nil
                    await terminal.setBusy(false)
                    await terminal.stopSpinner()
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
                    
                    // 1. Collect Context Stats
                    let files = PromptContextBuilder.extractFileReferences(from: prompt)
                    var stats = ContextStats(
                        files: files.count,
                        systemPrompts: 1, // Default system prompt
                        mcpServers: 0,    // Placeholder
                        skills: 0         // Placeholder
                    )
                    
                    // 2. Start Spinner with Initial State
                    await terminal.startSpinner(state: stats.files > 0 ? .readingFiles : .thinking, stats: stats)
                    
                    do {
                        // 3. Process Context (Reading Files)
                        let finalPrompt = try await PromptContextBuilder.build(prompt: prompt)
                        
                        // Update state to thinking after files are read
                        await terminal.updateSpinner(state: .thinking)
                        
                        let request = GenerationRequest(prompt: finalPrompt, maxTokens: maxTokens)
                        let stream = try await orchestrator.generateStream(request: request)
                        
                        // Move to new line and print speaker label
                        await terminal.printOutput(TerminalUI.info("Gemma:") + " ")
                        
                        var statsResponse: GenerationResponse?
                        var receivedFirstToken = false
                        
                        for try await chunk in stream {
                            if Task.isCancelled {
                                await terminal.printOutput("\n" + TerminalUI.dim("[Cancelled]") + "\n")
                                break
                            }
                            
                            switch chunk {
                            case .text(let t):
                                if !receivedFirstToken {
                                    // Switch to generating state briefly or stop
                                    await terminal.updateSpinner(state: .generating)
                                    try? await Task.sleep(nanoseconds: 100_000_000) // Small visual feedback
                                    await terminal.stopSpinner()
                                    receivedFirstToken = true
                                }
                                await terminal.printOutput(t)
                            case .metadata(let m):
                                statsResponse = m
                            }
                        }
                        
                        await terminal.printOutput("\n\n")
                        if let statsResponse = statsResponse, !Task.isCancelled {
                            let statsString = TerminalUI.dim("Tokens: \(statsResponse.promptTokens) in / \(statsResponse.completionTokens) out | TPS: \(String(format: "%.2f", statsResponse.tokensPerSecond)) | TTFT: \(String(format: "%.3fs", statsResponse.timeToFirstToken)) | Memory: \(statsResponse.memory.activeBytes / 1024 / 1024)MB")
                            await terminal.printOutput(statsString + "\n")
                        }
                    } catch {
                        await terminal.stopSpinner()
                        await terminal.printOutput(TerminalUI.error("Error:") + " \(error.localizedDescription)\n")
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
        await terminal.stopSpinner()
        await terminal.setBusy(false)
    }
}
