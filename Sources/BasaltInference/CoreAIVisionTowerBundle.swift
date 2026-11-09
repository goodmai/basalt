import Foundation
import BasaltCore
import MLXLMCommon

struct CoreAIVisionTowerBundle: Sendable, Equatable {
    struct Function: Sendable, Equatable {
        var name: String
        var tokens: Int
    }

    var asset: String
    var masked: Float
    var merge: Int
    var hidden: Int
    var textHidden: Int
    var headDim: Int
    var functions: [Function]

    static func isBundle(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appending(component: "metadata.json").path)
            && FileManager.default.fileExists(atPath: directory.appending(component: "tower.aimodel").path)
    }

    init(directory: URL) throws(BasaltError) {
        guard let data = try? Data(contentsOf: directory.appending(component: "metadata.json")),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tower = root["paddle_vision_tower"] as? [String: Any]
        else { throw .modelConfigurationInvalid(path: directory.path, reason: "metadata.json has no \"paddle_vision_tower\" section of a Core AI vision tower bundle") }
        try self.init(tower: tower, path: directory.path)
    }

    init(tower: [String: Any], path: String) throws(BasaltError) {
        guard let asset = tower["asset"] as? String, let merge = tower["merge"] as? Int, merge > 0,
              let hidden = tower["hidden"] as? Int, let textHidden = tower["text_hidden"] as? Int, let headDim = tower["head_dim"] as? Int
        else { throw .modelConfigurationInvalid(path: path, reason: "the vision tower bundle needs asset, merge, hidden, text_hidden and head_dim") }
        let functions = (tower["functions"] as? [[String: Any]] ?? []).compactMap { entry -> Function? in
            guard let name = entry["function"] as? String, let tokens = entry["tokens"] as? Int, tokens > 0, tokens % (merge * merge) == 0 else { return nil }
            return Function(name: name, tokens: tokens)
        }.sorted { $0.tokens < $1.tokens }
        guard !functions.isEmpty else { throw .modelConfigurationInvalid(path: path, reason: "the vision tower bundle has no function whose token count is a multiple of \(merge * merge)") }
        self.asset = asset
        masked = Float(tower["masked"] as? Double ?? -30000)
        self.merge = merge
        self.hidden = hidden
        self.textHidden = textHidden
        self.headDim = headDim
        self.functions = functions
    }

    func function(for frame: THW) -> Function? {
        guard frame.t == 1, frame.h % merge == 0, frame.w % merge == 0 else { return nil }
        return functions.first { $0.tokens >= frame.h * frame.w }
    }

    func order(_ frame: THW) -> [Int32] {
        (0 ..< frame.h / merge).flatMap { row in
            (0 ..< frame.w / merge).flatMap { column in
                (0 ..< merge * merge).map { Int32((row * merge + $0 / merge) * frame.w + column * merge + $0 % merge) }
            }
        }
    }

    func bias(real: Int, length: Int) -> [Float] {
        (0 ..< length).map { $0 < real ? 0 : masked }
    }
}
