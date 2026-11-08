import Testing
import Foundation
@testable import BasaltCore

@Suite("ModelInspector verdict core, no file system")
struct ModelInspectorVerdictTests {

    typealias Tensor = ModelInspector.Tensor

    static func header(_ tensors: [String: Tensor]) -> ModelInspector.SafetensorsHeader {
        ModelInspector.SafetensorsHeader(tensors: tensors, metadata: [:])
    }

    static func quantLayer(_ name: String, bits: Int) -> [String: Tensor] {
        [
            "\(name).weight": Tensor(dtype: "U32", shape: [8, 4096 * bits / 32], end: 0),
            "\(name).scales": Tensor(dtype: "BF16", shape: [8, 64], end: 0),
        ]
    }

    @Test("Each shard outcome lands as the same error, size and validity")
    func shardScanRecordsOutcomes() {
        var scan = ModelInspector.ShardScan()
        scan.record("a.safetensors", .missingBlob)
        scan.record("b.safetensors", .present(sizeBytes: 0, header: .failure(.malformedHeader)))
        scan.record("c.safetensors", .present(sizeBytes: 100, header: .failure(.malformedHeader)))
        scan.record("d.npz", .present(sizeBytes: 50, header: nil))
        scan.record("e.safetensors", .present(sizeBytes: 2048, header: .success(Self.header(["w": Tensor(dtype: "BF16", shape: [2], end: 4)]))))

        #expect(scan.errors == [
            "Weight shard points to a missing blob: a.safetensors",
            "Weight shard is empty (0 bytes): b.safetensors",
            "Invalid or corrupted safetensors header: c.safetensors",
        ])
        #expect(scan.shards.map(\.valid) == [false, false, false, true, true])
        #expect(scan.shards.map(\.sizeBytes) == [0, 0, 100, 50, 2048])
        #expect(scan.totalSizeBytes == 2198)
        #expect(scan.dtypes == ["BF16"])
        #expect(scan.tensors.keys.sorted() == ["w"])
    }

    @Test("A tensor seen in two shards keeps the first shard's entry")
    func firstShardWins() {
        var scan = ModelInspector.ShardScan()
        scan.record("1", .present(sizeBytes: 8, header: .success(Self.header(["w": Tensor(dtype: "F16", shape: [1], end: 2)]))))
        scan.record("2", .present(sizeBytes: 8, header: .success(Self.header(["w": Tensor(dtype: "F32", shape: [1], end: 4)]))))
        #expect(scan.tensors["w"]?.dtype == "F16")
        #expect(scan.dtypes == ["F16", "F32"])
    }

    @Test("Weight shard filter skips training_args.bin and non-weight files")
    func weightShardFilter() {
        let names = ["model.safetensors", "w.NPZ", "pytorch_model-00001.bin", "training_args.bin", "config.json"]
        #expect(names.filter { ModelInspector.isWeightShard(URL(fileURLWithPath: "/m/\($0)")) }
            == ["model.safetensors", "w.NPZ", "pytorch_model-00001.bin"])
    }

    @Test("Index lists required shards; anything else is malformed")
    func requiredShards() {
        let index = Data(#"{"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors", "c": "m-1.safetensors"}}"#.utf8)
        #expect(ModelInspector.requiredShards(index: index) == ["m-1.safetensors", "m-2.safetensors"])
        #expect(ModelInspector.requiredShards(index: Data(#"{"weight_map": [1]}"#.utf8)) == nil)
        #expect(ModelInspector.requiredShards(index: Data("not json".utf8)) == nil)
    }

    @Test("MLX quantization dict with per-layer overrides and mode default group sizes")
    func mlxQuantizationConfig() {
        let config = ModelInspector.QuantizationConfig(json: [
            "quantization": ["bits": 4, "mode": "NVFP4", "lm_head": ["bits": 8, "group_size": 32]] as [String: Any],
            "torch_dtype": "bfloat16",
        ])
        #expect(config.bits == 4)
        #expect(config.groupSize == nil)
        #expect(config.defaultGroupSize == 16)
        #expect(config.groupSize(forLayer: "lm_head") == 32)
        #expect(config.groupSize(forLayer: "model.layers.0.mlp") == 16)
        #expect(config.dtype == "bfloat16")
    }

    @Test("HF quantization_config falls back to weight_bits and quant_method")
    func hfQuantizationConfig() {
        let config = ModelInspector.QuantizationConfig(json: ["quantization_config": ["weight_bits": 3, "group_size": 128, "quant_method": "mxfp4"] as [String: Any]])
        #expect(config.bits == 3)
        #expect(config.groupSize(forLayer: "x") == 128)
        #expect(ModelInspector.QuantizationConfig(json: ["quantization_config": ["quant_method": "mxfp8"]]).defaultGroupSize == 32)
        #expect(ModelInspector.QuantizationConfig(json: nil) == ModelInspector.QuantizationConfig(json: [:]))
    }

    @Test("Measured bits win over config, and a disagreeing claim is a mismatch")
    func measuredBitWidth() {
        let tensors = Self.quantLayer("a", bits: 4).merging(Self.quantLayer("b", bits: 4)) { $1 }.merging(Self.quantLayer("c", bits: 8)) { $1 }
        let claimed = ModelInspector.QuantizationConfig(json: ["quantization": ["bits": 6]])
        let verdict = ModelInspector.bitWidth(tensors: tensors, dtypes: ["U32", "BF16"], config: claimed, modelId: "m")
        #expect(verdict == .init(
            bits: 4,
            description: "mixed 4/8-bit quantized (measured over 3 layers)",
            mismatch: "config.json claims 6-bit but the weights are 4/8-bit"
        ))
        let uniform = ModelInspector.bitWidth(tensors: Self.quantLayer("a", bits: 4), dtypes: [], config: .init(json: ["quantization": ["bits": 4, "mode": "affine"]]), modelId: "m")
        #expect(uniform == .init(bits: 4, description: "4-bit quantized (measured from weights, affine)"))
    }

    @Test("An nf4 checkpoint, mode nf4 or the older quant_method nf4, measures its bias-free layers as 4-bit whatever their group, and its affine layers from the config group", arguments: [false, true])
    func nf4BitWidth(legacy: Bool) {
        let json: [String: Any] = ["quantization": legacy
            ? ["bits": 4, "group_size": 64, "mode": "affine", "quant_method": "nf4"] as [String: Any]
            : ["bits": 4, "group_size": 64, "mode": "nf4"]]
        let nf4g32: [String: Tensor] = [
            "a.weight": Tensor(dtype: "U32", shape: [8, 512], end: 0), "a.scales": Tensor(dtype: "F16", shape: [8, 128], end: 0),
        ]
        let affine6 = Self.quantLayer("b", bits: 6).merging(["b.biases": Tensor(dtype: "BF16", shape: [8, 64], end: 0)]) { $1 }
        let verdict = ModelInspector.bitWidth(tensors: nf4g32.merging(affine6) { $1 }, dtypes: ["U32"], config: .init(json: json), modelId: "m")
        #expect(verdict.mismatch == nil && verdict.description == "mixed 4/6-bit quantized (measured over 2 layers)", "\(verdict)")
    }

    @Test("Without packed layers: config, then dtypes, then repo name, then 16-bit")
    func unmeasuredBitWidthOrder() {
        func verdict(_ dtypes: Set<String>, _ json: [String: Any]? = nil, _ id: String = "m") -> ModelInspector.BitWidth {
            ModelInspector.bitWidth(tensors: [:], dtypes: dtypes, config: .init(json: json), modelId: id)
        }
        #expect(verdict(["F32"], ["quantization": ["bits": 5]]).description == "5-bit quantized (from config.json; no packed layers to measure)")
        #expect(verdict(["U8", "F16"]).bits == 8)
        #expect(verdict(["F32"], ["torch_dtype": "bfloat16"]).description == "16-bit unquantized (FP16/BF16)")
        #expect(verdict(["F32"]) == .init(bits: 32, description: "32-bit floating point"))
        #expect(verdict([], nil, "org/X-4BIT") == .init(bits: 4, description: "4-bit (from repo name, unverified)"))
        #expect(verdict([], nil, "org/x-8bit") == .init(bits: 8, description: "8-bit (from repo name, unverified)"))
        #expect(verdict([]) == .init(bits: 16, description: "16-bit (default)"))
    }

    @Test("Report: intact needs shards and no errors; config and mismatch errors come last")
    func reportVerdict() {
        var scan = ModelInspector.ShardScan()
        scan.record("m.safetensors", .present(sizeBytes: 2048, header: .success(Self.header(Self.quantLayer("a", bits: 4)))))
        let intact = ModelInspector.Report(modelId: "m", path: "/m", scan: scan, config: ["quantization": ["bits": 4]])
        #expect(intact.intact)
        #expect(intact.detectedBitWidth == 4)
        #expect(intact.tensorDtype == "BF16, U32")

        scan.errors = ["Missing shard referenced in index: x"]
        let broken = ModelInspector.Report(modelId: "m", path: "/m", scan: scan, config: nil)
        #expect(!broken.intact)
        #expect(broken.errors == ["Missing shard referenced in index: x", "config.json is missing or not valid JSON"])

        let empty = ModelInspector.Report(modelId: "m", path: "/m", scan: ModelInspector.ShardScan(), config: ["torch_dtype": "float16"])
        #expect(!empty.intact)
        #expect(empty.errors.isEmpty)
        #expect(empty.tensorDtype == "float16")
    }

    @Test("Missing directory report")
    func missingDirectory() {
        let report = ModelInspector.Report(missingDirectory: "/nowhere", modelId: "org/m")
        #expect(!report.intact)
        #expect(report.errors == ["Model directory not found at: /nowhere"])
        #expect(report.bitWidthDescription == "Not downloaded / missing directory")
    }
}
