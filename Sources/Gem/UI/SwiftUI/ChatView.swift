import SwiftUI
import Markdown

@available(macOS 14.0, *)
struct ChatView: View {
    @Environment(AppState.self) private var appState
    @State private var viewModel = ChatViewModel()
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Gemm REST Client")
                    .font(.headline)
                Spacer()
                
                if viewModel.statsTokensPerSecond > 0 {
                    Text(String(format: "%.1f tok/s | %.2fs TTFT", viewModel.statsTokensPerSecond, viewModel.statsTTFT))
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.trailing, 8)
                }
                
                Button("Logout") {
                    appState.logout()
                }
                .buttonStyle(.borderless)
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()
            
            // Message List
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        ForEach(viewModel.messages) { msg in
                            MessageRow(message: msg)
                                .id(msg.id)
                        }
                        
                        if let error = viewModel.errorMessage {
                            Text("Error: \(error)")
                                .foregroundColor(.red)
                                .padding()
                                .id("error")
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages) { _, messages in
                    if let last = messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            
            Divider()
            
            // Input Area
            HStack(spacing: 12) {
                TextField("Message...", text: $viewModel.currentInput, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...10)
                    .focused($isInputFocused)
                    .onSubmit {
                        if !viewModel.isGenerating {
                            viewModel.sendMessage(restClient: appState.restClient)
                        }
                    }
                
                if viewModel.isGenerating {
                    Button(action: {
                        viewModel.cancelGeneration()
                    }) {
                        Image(systemName: "stop.circle.fill")
                            .foregroundColor(.red)
                    }
                    .buttonStyle(.plain)
                    .font(.title2)
                } else {
                    Button(action: {
                        viewModel.sendMessage(restClient: appState.restClient)
                    }) {
                        Image(systemName: "paperplane.fill")
                            .foregroundColor(viewModel.currentInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .secondary : .accentColor)
                    }
                    .buttonStyle(.plain)
                    .font(.title2)
                    .disabled(viewModel.currentInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
        }
        .onAppear {
            isInputFocused = true
        }
    }
}

@available(macOS 14.0, *)
struct MessageRow: View {
    let message: ChatMessage
    
    var body: some View {
        HStack(alignment: .top) {
            if message.role == .user {
                Spacer()
                Text(message.text)
                    .padding(12)
                    .background(Color.accentColor)
                    .foregroundColor(.white)
                    .cornerRadius(16)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Gemma")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    // Note: Ideally use a real Markdown rendering view here,
                    // but standard Text works for basic text. SwiftUI Text supports basic markdown.
                    Text(.init(message.text))
                        .padding(12)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(16)
                }
                Spacer()
            }
        }
    }
}
