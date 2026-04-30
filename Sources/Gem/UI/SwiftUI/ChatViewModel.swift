import SwiftUI

@available(macOS 14.0, *)
@Observable
@MainActor
class ChatViewModel {
    var messages: [ChatMessage] = []
    var currentInput: String = ""
    var isGenerating: Bool = false
    var errorMessage: String?
    
    var statsTokensPerSecond: Double = 0
    var statsGenerationTime: Double = 0
    var statsTTFT: Double = 0
    var statsPromptTokens: Int = 0
    var statsCompletionTokens: Int = 0
    
    private var activeGenerationTask: Task<Void, Never>?
    
    func sendMessage(restClient: RESTClient) {
        let prompt = currentInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        
        messages.append(ChatMessage(role: .user, text: prompt))
        currentInput = ""
        isGenerating = true
        errorMessage = nil
        
        let assistantIndex = messages.count
        messages.append(ChatMessage(role: .assistant, text: ""))
        
        activeGenerationTask = Task {
            do {
                let stream = await restClient.generateStream(prompt: prompt, maxTokens: 65536)
                
                for try await chunk in stream {
                    if Task.isCancelled { break }
                    switch chunk {
                    case .text(let t):
                        messages[assistantIndex].text += t
                    case .metadata(let m):
                        statsTokensPerSecond = m.tokensPerSecond
                        statsGenerationTime = m.generationTime
                        statsTTFT = m.timeToFirstToken
                        statsPromptTokens = m.promptTokens
                        statsCompletionTokens = m.completionTokens
                    }
                }
                self.isGenerating = false
            } catch {
                if !Task.isCancelled {
                    self.errorMessage = error.localizedDescription
                    self.isGenerating = false
                }
            }
        }
    }
    
    func cancelGeneration() {
        activeGenerationTask?.cancel()
        activeGenerationTask = nil
        isGenerating = false
    }
}

// Re-using a basic structure, since GemCore.ChatMessage might not be identical or available,
// we define a simple local struct for UI binding.
public struct ChatMessage: Identifiable, Equatable {
    public let id = UUID()
    public let role: Role
    public var text: String
    
    public enum Role: String, Equatable {
        case user, assistant, system
    }
    
    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }
}
