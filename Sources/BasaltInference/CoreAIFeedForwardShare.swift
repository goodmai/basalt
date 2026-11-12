import BasaltCore
import CoreAI
import Foundation

public struct CoreAIFeedForwardShare: FeedForwardShare {
    struct Manifest: Decodable {
        struct Split: Decodable {
            let hidden: Int
            let intermediate: Int
            let columns: Int
            let weights: String
            let activation: String
            let lengths: [Int]
            let layers: [String: String]
        }

        let split: Split

        enum CodingKeys: String, CodingKey { case split = "ffn_split" }
    }

    let columns: Int
    let activation: FeedForwardActivation
    let lengths: [Int]
    let functions: [Int: [Int: InferenceFunction]]

    static func manifest(_ directory: URL) -> Manifest.Split? {
        (try? Data(contentsOf: directory.appending(component: "manifest.json"))).flatMap { try? JSONDecoder().decode(Manifest.self, from: $0).split }
    }

    public static func isBundle(_ directory: URL) -> Bool { manifest(directory) != nil }

    static func load(_ directory: URL) async throws(BasaltError) -> CoreAIFeedForwardShare {
        guard let split = manifest(directory) else {
            throw .modelConfigurationInvalid(path: directory.path, reason: "no manifest.json with ffn_split; build it with swift scripts/ffn_split_export.swift")
        }
        var functions: [Int: [Int: InferenceFunction]] = [:]
        do {
            for (key, file) in split.layers {
                guard let layer = Int(key) else { continue }
                let model = try await AIModel.specialize(
                    contentsOf: directory.appending(component: file), options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
                for length in split.lengths {
                    guard let function = try model.loadFunction(named: "len\(length)") else {
                        throw BasaltError.modelConfigurationInvalid(path: file, reason: "no function len\(length)")
                    }
                    functions[layer, default: [:]][length] = function
                }
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelConfigurationInvalid(path: directory.path, reason: "Core AI could not load the feed-forward share: \(error.localizedDescription)")
        }
        return CoreAIFeedForwardShare(columns: split.columns, activation: FeedForwardActivation(split.activation), lengths: split.lengths, functions: functions)
    }

    func supports(layer: Int, tokens: Int) -> Bool { functions[layer]?[tokens] != nil }

    func run(layer: Int, planar: NDArray, tokens: Int) async throws -> NDArray {
        guard let function = functions[layer]?[tokens] else { throw BasaltError.inferenceError("no ANE share for layer \(layer) at \(tokens) tokens") }
        var outputs = try await function.run(inputs: ["x": planar])
        guard let y = outputs.remove("y")?.ndArray else { throw BasaltError.inferenceError("the feed-forward share returned no y") }
        return y
    }
}
