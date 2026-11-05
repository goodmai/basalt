import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Tensor pack: one linear layer re-packed live, its error measured, undone bit for bit (#332)", .serialized)
struct TensorPackTests {
    static let layerPath = "model.layers.0.self_attn.q_proj"
    static let projection = layerPath + ".weight"

    static func engine(quantized dtype: DType? = nil) async throws -> MLXInferenceEngine {
        let model = try TinyModel.make(.qwen3)
        if let dtype {
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            quantize(model: model, groupSize: 32, bits: 4)
            eval(model)
        }
        let engine = MLXInferenceEngine()
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen3"), model: model, processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
        await engine.skipJournalForDeltaTests()
        return engine
    }

    static func logits(_ container: ModelContainer) async throws -> [Float] {
        try await container.perform { context in
            let input = MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init)).reshaped(1, -1)
            return context.model(input, cache: try context.model.newCache(parameters: nil)).asType(.float32).asArray(Float.self)
        }
    }

    static func layerType(_ container: ModelContainer) async -> String {
        await container.perform { context in
            context.model.namedModules().first(where: { $0.0 == Self.layerPath }).map { "\(type(of: $0.1))" } ?? "none"
        }
    }

    static func ternaryCodes(_ container: ModelContainer) async -> (bits: Int, onGrid: Bool, zeros: Int, nonzeros: Int)? {
        await container.perform { context in
            guard let layer = context.model.namedModules().first(where: { $0.0 == Self.layerPath })?.1 as? QuantizedLinear else { return nil }
            let groups = TensorPacker.dense(layer).asType(.float32).reshaped(layer.weight.dim(0), -1, layer.groupSize)
            let scale = layer.scales.asType(.float32)[0..., 0..., .newAxis]
            let onGrid = all((groups .== scale) .|| (groups .== 0) .|| (groups .== -scale)).item(Bool.self)
            let zeros = Int((groups .== 0).asType(.int32).sum().item(Int32.self))
            return (layer.bits, onGrid, zeros, groups.size - zeros)
        }
    }

    @Test("Affine 8-bit has a small error and fewer bytes, 2-bit a larger one; a packed layer refuses a second pack; undo restores the logits bit for bit")
    func affine() async throws {
        let engine = try await Self.engine()
        let container = try #require(await engine.container)
        let logits = try await Self.logits(container).map(\.bitPattern)
        let eight = try await engine.pack(PackRequest(tensor: Self.projection, format: .affine(bits: 8, groupSize: 64)))
        #expect(eight.relativeError > 0 && eight.relativeError < 0.02)
        #expect(eight.maxAbsError >= eight.rmse && eight.rmse > 0)
        #expect(eight.bytesAfter < eight.bytesBefore)
        #expect(await engine.modelModified)
        #expect(await Self.layerType(container) == "QuantizedLinear")
        #expect(try await Self.logits(container).map(\.bitPattern) != logits)
        await #expect(throws: BasaltError.self) { try await engine.pack(PackRequest(tensor: Self.projection, format: .ternary)) }
        try await engine.undo(eight.overlay.id)
        #expect(await Self.layerType(container) == "Linear")
        #expect(try await Self.logits(container).map(\.bitPattern) == logits)
        #expect(await !engine.modelModified)
        let two = try await engine.pack(PackRequest(tensor: Self.projection, format: .affine(bits: 2, groupSize: 64)))
        #expect(two.relativeError > 10 * eight.relativeError && two.relativeError < 0.6)
        #expect(two.bytesAfter < eight.bytesAfter)
    }

    @Test("An mxfp4 pack is a 4-bit microscaled layer with a moderate error")
    func mxfp4() async throws {
        let engine = try await Self.engine()
        let report = try await engine.pack(PackRequest(tensor: Self.projection, format: .mxfp4))
        #expect(report.relativeError > 0 && report.relativeError < 0.3)
        #expect(report.bytesAfter < report.bytesBefore)
        #expect(report.overlay.description.contains("mxfp4"))
    }

    @Test("A ternary pack holds scale·{-1, 0, 1} per group of 64 as 2-bit codes, with the error of absmean rounding")
    func ternary() async throws {
        let engine = try await Self.engine()
        let container = try #require(await engine.container)
        let report = try await engine.pack(PackRequest(tensor: Self.projection, format: .ternary))
        let codes = try #require(await Self.ternaryCodes(container))
        #expect(codes.bits == 2 && codes.onGrid)
        #expect(codes.zeros > 0 && codes.nonzeros > codes.zeros)
        #expect(report.relativeError > 0.2 && report.relativeError < 0.8)
    }

    @Test("Unpacking a quantized layer to its scale type is exact: zero error, a dense layer, the same logits up to kernel rounding; undo brings the quantized layer back bit for bit")
    func unpack() async throws {
        let engine = try await Self.engine(quantized: .float16)
        let container = try #require(await engine.container)
        let logits = try await Self.logits(container)
        let report = try await engine.pack(PackRequest(tensor: Self.projection, format: .float16))
        #expect(report.rmse == 0 && report.maxAbsError == 0 && report.relativeError == 0)
        #expect(report.bytesAfter > report.bytesBefore)
        #expect(await Self.layerType(container) == "Linear")
        let unpacked = try await Self.logits(container)
        let scale = logits.map(\.magnitude).max() ?? 1
        #expect(unpacked.count == logits.count && zip(unpacked, logits).allSatisfy { ($0 - $1).magnitude <= 0.02 * scale })
        try await engine.undo(report.overlay.id)
        #expect(await Self.layerType(container) == "QuantizedLinear")
        #expect(try await Self.logits(container).map(\.bitPattern) == logits.map(\.bitPattern))
    }

    @Test("Unknown tensors, weights outside a linear layer, 7 bits and a group size of 48 are refused and leave the model unmodified", arguments: [
        PackRequest(tensor: "model.nope.weight", format: .float16),
        PackRequest(tensor: "model.norm.weight", format: .ternary),
        PackRequest(tensor: "model.layers.0.self_attn.q_proj.weight", format: .affine(bits: 7, groupSize: 64)),
        PackRequest(tensor: "model.layers.0.self_attn.q_proj.weight", format: .affine(bits: 4, groupSize: 48)),
    ])
    func refused(request: PackRequest) async throws {
        let engine = try await Self.engine()
        let error = await #expect(throws: BasaltError.self) { try await engine.pack(request) }
        guard case .invalidRequestStructure(let details)? = error else { Issue.record("unexpected \(String(describing: error))"); return }
        #expect(details.hasPrefix("tensor_pack: "), "\(details)")
        #expect(await !engine.modelModified)
        #expect(await engine.activeDeltas.isEmpty)
    }

    @Test("tensor_pack defaults to affine 4-bit g64 and tensor_unpack to bfloat16; each refuses the other's formats and bits outside 2...8")
    func mcpArguments() throws {
        #expect(try ResearchCall(.tensorPack, ["tensor": "w"]) == .pack(PackRequest(tensor: "w", format: .affine(bits: 4, groupSize: 64))))
        #expect(try ResearchCall(.tensorPack, ["tensor": "w", "format": "ternary"]) == .pack(PackRequest(tensor: "w", format: .ternary)))
        #expect(try ResearchCall(.tensorUnpack, ["tensor": "w"]) == .pack(PackRequest(tensor: "w", format: .bfloat16)))
        let cases: [(ResearchTool, [String: Any])] = [
            (.tensorUnpack, ["tensor": "w", "format": "ternary"]),
            (.tensorPack, ["tensor": "w", "format": "float16"]),
            (.tensorPack, ["tensor": "w", "bits": 9]),
            (.tensorPack, ["format": "affine"]),
        ]
        for (tool, args) in cases {
            let failure = #expect(throws: MCPFailure.self) { try ResearchCall(tool, args) }
            #expect(failure?.code == -32602, "\(tool) \(args)")
        }
    }
}
