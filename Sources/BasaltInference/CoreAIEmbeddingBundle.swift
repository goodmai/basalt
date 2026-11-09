import Foundation
import BasaltCore

struct CoreAIEmbeddingBundle: Sendable, Equatable {
    struct Function: Sendable, Equatable {
        var name: String
        var length: Int
    }

    var id: String
    var dimension: Int
    var matryoshka: [Int]
    var tokenTable: String
    var encoder: String
    var functions: [Function]
    var masked: Float
    var maxTokens: Int
    var bos: Int?
    var eos: Int?
    var pad: Int
    var documentPrompt: String

    static func isBundle(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appending(component: "metadata.json").path)
            && FileManager.default.fileExists(atPath: directory.appending(component: "encoder.aimodel").path)
    }

    init(directory: URL) throws(BasaltError) {
        let url = directory.appending(component: "metadata.json")
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedding = root["embedding"] as? [String: Any]
        else { throw .modelConfigurationInvalid(path: directory.path, reason: "metadata.json has no \"embedding\" section of a Core AI embedding bundle") }
        try self.init(embedding: embedding, path: directory.path)
    }

    init(embedding: [String: Any], path: String) throws(BasaltError) {
        let components = embedding["components"] as? [String: Any] ?? [:]
        let table = components["token_table"] as? [String: Any]
        let encoder = components["encoder"] as? [String: Any]
        let text = embedding["text"] as? [String: Any] ?? [:]
        let functions = (encoder?["functions"] as? [[String: Any]] ?? []).compactMap { entry -> Function? in
            guard let name = entry["function"] as? String, let length = entry["length"] as? Int, entry["batch"] as? Int ?? 1 == 1 else { return nil }
            return Function(name: name, length: length)
        }.sorted { $0.length < $1.length }
        guard let tableAsset = table?["asset"] as? String, let encoderAsset = encoder?["asset"] as? String, !functions.isEmpty,
              let dimension = embedding["dimension"] as? Int
        else { throw .modelConfigurationInvalid(path: path, reason: "the bundle needs a token_table and an encoder with batch-1 functions of fixed length (export it with coreai.embedding.export --platform iOS --text-only)") }
        let space = embedding["space"] as? [String: Any]
        id = space?["id"] as? String ?? URL(fileURLWithPath: path).lastPathComponent
        self.dimension = dimension
        matryoshka = embedding["matryoshka_dimensions"] as? [Int] ?? [dimension]
        tokenTable = tableAsset
        self.encoder = encoderAsset
        self.functions = functions
        masked = Float((encoder?["bias"] as? [String: Any])?["masked"] as? Double ?? -30000)
        maxTokens = min((text["truncation"] as? [String: Any])?["max_tokens"] as? Int ?? functions[functions.count - 1].length, functions[functions.count - 1].length)
        bos = (text["add_bos_token"] as? Bool ?? false) ? text["bos_token_id"] as? Int : nil
        eos = (text["add_eos_token"] as? Bool ?? false) ? text["eos_token_id"] as? Int : nil
        pad = text["pad_token_id"] as? Int ?? 0
        documentPrompt = (text["prompts"] as? [String: Any])?["document"] as? String ?? ""
    }

    func tokens(_ body: [Int]) -> [Int] {
        let room = maxTokens - (bos == nil ? 0 : 1) - (eos == nil ? 0 : 1)
        return [bos].compactMap { $0 } + body.prefix(max(0, room)) + [eos].compactMap { $0 }
    }

    func function(for count: Int) -> Function {
        functions.first { $0.length >= count } ?? functions[functions.count - 1]
    }

    func padded(_ tokens: [Int], to length: Int) -> [Int32] {
        (tokens.prefix(length) + Array(repeating: pad, count: max(0, length - tokens.count))).map(Int32.init)
    }

    func bias(real: Int, length: Int) -> [Float] {
        (0 ..< length).flatMap { key in Array(repeating: key < real ? 0 : masked, count: length) }
    }

    func poolWeights(real: Int, length: Int) -> [Float] {
        (0 ..< length).map { $0 < real ? 1 / Float(max(real, 1)) : 0 }
    }

    func truncated(_ vector: [Float], to dimensions: Int?) throws(BasaltError) -> [Float] {
        guard let dimensions, dimensions != vector.count else { return vector }
        guard matryoshka.contains(dimensions) else {
            throw .invalidRequestStructure(details: "dimensions must be one of \(matryoshka.map(String.init).joined(separator: ", ")) for \(id)")
        }
        let head = Array(vector.prefix(dimensions))
        let norm = head.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? head.map { $0 / norm } : head
    }
}
