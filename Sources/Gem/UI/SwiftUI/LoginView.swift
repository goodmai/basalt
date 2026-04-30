import SwiftUI

@available(macOS 14.0, *)
struct LoginView: View {
    @Environment(AppState.self) private var appState
    
    @State private var username = "admin"
    @State private var password = ""
    @State private var errorMessage: String?
    @State private var isLoading = false
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "lock.shield.fill")
                .resizable()
                .scaledToFit()
                .frame(width: 60, height: 60)
                .foregroundColor(.accentColor)
            
            Text("Gemm REST Client")
                .font(.largeTitle)
                .bold()
            
            Text("Connect to \(appState.baseURL)")
                .foregroundColor(.secondary)
            
            VStack(spacing: 12) {
                TextField("Username", text: $username)
                    .textFieldStyle(.roundedBorder)
                    .disabled(isLoading)
                
                SecureField("Password", text: $password)
                    .textFieldStyle(.roundedBorder)
                    .disabled(isLoading)
                    .onSubmit {
                        login()
                    }
                
                if let errorMessage {
                    Text(errorMessage)
                        .foregroundColor(.red)
                        .font(.callout)
                }
                
                Button(action: login) {
                    if isLoading {
                        ProgressView()
                            .controlSize(.small)
                            .frame(maxWidth: .infinity)
                    } else {
                        Text("Connect")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isLoading || username.isEmpty || password.isEmpty)
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
    
    private func login() {
        guard !username.isEmpty, !password.isEmpty else { return }
        
        isLoading = true
        errorMessage = nil
        
        Task {
            do {
                let token = try await appState.restClient.login(username: username, password: password)
                await MainActor.run {
                    appState.setToken(token)
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }
}
