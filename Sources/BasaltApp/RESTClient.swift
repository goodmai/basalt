import Foundation
import BasaltCore

@available(macOS 14.0, *)
public actor RESTClient {
    private let baseURL: String
    private var token: String?
    private let urlSession: URLSession
    
    public init(baseURL: String) {
        self.baseURL = baseURL
        self.urlSession = URLSession(configuration: .default)
    }
    
    public func setToken(_ newToken: String?) {
        self.token = newToken
    }
    
    public func generateStream(prompt: String, maxTokens: Int? = nil) -> AsyncThrowingStream<StreamChunk, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    guard let url = URL(string: "\(baseURL)/api/v1/generate/stream") else {
                        throw URLError(.badURL)
                    }
                    
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    if let token = self.token, !token.isEmpty {
                        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                    }
                    
                    let reqBody = GenerationRequest(prompt: prompt, maxTokens: maxTokens)
                    request.httpBody = try JSONEncoder().encode(reqBody)
                    
                    let (asyncBytes, response) = try await urlSession.bytes(for: request)
                    
                    guard let httpRes = response as? HTTPURLResponse, httpRes.statusCode == 200 else {
                        throw URLError(.badServerResponse)
                    }
                    
                    for try await line in asyncBytes.lines {
                        if Task.isCancelled { break }
                        guard line.hasPrefix("data: ") else { continue }
                        let jsonStr = line.dropFirst(6).trimmingCharacters(in: .whitespaces)
                        
                        if jsonStr == "[DONE]" {
                            break
                        }
                        
                        if let data = jsonStr.data(using: .utf8) {
                            let chunk = try JSONDecoder().decode(StreamChunk.self, from: data)
                            continuation.yield(chunk)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
