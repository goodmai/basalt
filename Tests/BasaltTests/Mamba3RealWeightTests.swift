import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("Mamba-3 on the real state-spaces/mamba3-mimo-187m weights: strict load, finite logits, stepped decode equals the chunked pass (#847)", .serialized)
struct Mamba3RealWeightTests {
    static var path: String? { ProcessInfo.processInfo.environment["BASALT_E2E_MAMBA3_PATH"] }
    static var tokens: MLXArray { MLXArray((0 ..< 24).map { Int32(($0 * 1009 + 17) % 128_000) })[.newAxis] }

    static func model(_ dtype: DType) throws -> Mamba3Model {
        let directory = URL(fileURLWithPath: try #require(path))
        let model = try Mamba3Model(configurationData: Data(contentsOf: directory.appending(component: "config.json")))
        let weights = try loadArrays(url: directory.appending(component: "model.safetensors")).mapValues { $0.asType(dtype) }
        try model.update(parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)), verify: .all)
        return model
    }

    @Test("Every checkpoint tensor loads with verify .all and the native bf16 logits are finite",
          .enabled(if: path != nil, "set BASALT_E2E_MAMBA3_PATH to a converted mamba3-mimo-187m folder"))
    func loadsInBF16() throws {
        let logits = try Self.model(.bfloat16)(Self.tokens, cache: nil).asType(.float32)
        #expect(isNaN(logits).any().item(Bool.self) == false && abs(logits).max().item(Float.self).isFinite)
    }

    @Test("In float32, one-token steps across the 16-token chunk boundary reproduce the chunked full-sequence logits and argmax",
          .enabled(if: path != nil, "set BASALT_E2E_MAMBA3_PATH to a converted mamba3-mimo-187m folder"))
    func steppedEqualsChunked() throws {
        let model = try Self.model(.float32)
        let full = model(Self.tokens, cache: nil)[0]
        let cache = model.newCache(parameters: nil)
        let stepped = concatenated((0 ..< 24).map { model(Self.tokens[0..., $0 ..< ($0 + 1)], cache: cache)[0] }, axis: 0)
        #expect(abs(stepped - full).max().item(Float.self) < 1e-3)
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(full, axis: -1).asArray(Int32.self))
    }
}
