import Foundation

let json = "{\"prompt\": \"привет\", \"maxTokens\": 50}"
let data = json.data(using: .utf8)!

struct GenerationRequest: Codable {
    let prompt: String
    let maxTokens: Int
}

do {
    let dto = try JSONDecoder().decode(GenerationRequest.self, from: data)
    print("Success: \(dto)")
} catch {
    print("Error: \(error)")
}
