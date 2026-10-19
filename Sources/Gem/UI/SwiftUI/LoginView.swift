import SwiftUI

@available(macOS 14.0, *)
struct LoginView: View {
    @Environment(AppState.self) private var appState
    
    // The server has no login route: it checks its API key (Bearer) on every request.
    @State private var apiKey = ""
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "lock.shield.fill")
                .resizable()
                .scaledToFit()
                .frame(width: 60, height: 60)
                .foregroundColor(.accentColor)
            
            Text("Basalt REST Client")
                .font(.largeTitle)
                .bold()
            
            Text("Connect to \(appState.baseURL)")
                .foregroundColor(.secondary)
            
            VStack(spacing: 12) {
                SecureField("API key (empty for a loopback server)", text: $apiKey)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(connect)
                
                Button(action: connect) {
                    Text("Connect")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .padding(.top, 8)
            }
            .frame(maxWidth: 300)
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(12)
            .shadow(radius: 5)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(NSColor.windowBackgroundColor))
    }
    
    private func connect() {
        appState.setToken(apiKey.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
