import Foundation
import BasaltCore

struct TeacherClient: Sendable {
    struct Reply: Decodable {
        struct Choice: Decodable {
            struct Message: Decodable { var content: String? }
            var message: Message
        }
        var choices: [Choice]
    }

    static let keyVariable = "BASALT_TEACHER_API_KEY"
    let endpoint: URL
    let model: String

    init(_ teacher: String, model: String?) throws(BasaltError) {
        guard var url = URL(string: teacher), ["http", "https"].contains(url.scheme?.lowercased() ?? ""), url.host != nil else {
            throw .invalidRequestStructure(details: "train: teacher must be the http(s) URL of an OpenAI-compatible server, e.g. http://127.0.0.1:8080/v1")
        }
        if !url.path.hasSuffix("chat/completions") { url.append(path: "chat/completions") }
        endpoint = url
        self.model = model ?? "default"
    }

    func answer(_ prompt: String) async throws(BasaltError) -> String {
        var request = URLRequest(url: endpoint, timeoutInterval: 120)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = ProcessInfo.processInfo.environment[Self.keyVariable] { request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        let body: [String: Any] = ["model": model, "messages": [["role": "user", "content": prompt]], "temperature": 0, "max_tokens": 256]
        let status: Int
        let data: Data
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (received, response) = try await URLSession.shared.data(for: request)
            (data, status) = (received, (response as? HTTPURLResponse)?.statusCode ?? 0)
        } catch {
            throw .inferenceError("the teacher at \(endpoint.absoluteString) did not answer: \(error.localizedDescription)")
        }
        guard status == 200, let reply = try? JSONDecoder().decode(Reply.self, from: data) else {
            throw .inferenceError("the teacher at \(endpoint.absoluteString) returned HTTP \(status) without an OpenAI chat completion")
        }
        return reply.choices.first?.message.content ?? ""
    }
}
