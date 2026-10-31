import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("ModelInspector integrity and bit width")
struct ModelInspectorTests {

    struct T {
        let name: String
        let dtype: String
        let shape: [Int]
    }

    /// Writes a well-formed safetensors file: each tensor gets 8 data bytes, offsets consecutive.
    /// `truncateBy` drops bytes off the end; `headerLenOverride` lies about the header length.
    @discardableResult
    private func writeShard(_ url: URL, _ tensors: [T], truncateBy: Int = 0, headerLenOverride: UInt64? = nil) throws -> URL {
        let entries = tensors.enumerated().map { i, t in
            "\"\(t.name)\": {\"dtype\": \"\(t.dtype)\", \"shape\": \(t.shape), \"data_offsets\": [\(i * 8), \((i + 1) * 8)]}"
        }.joined(separator: ", ")
        let header = Data("{\(entries)}".utf8)
        let len = (headerLenOverride ?? UInt64(header.count)).littleEndian
        let lenBytes: [UInt8] = (0..<8).map { UInt8((len >> ($0 * 8)) & 0xff) }
        var data = Data(lenBytes)
        data.append(header)
        data.append(Data(repeating: 0, count: tensors.count * 8 - truncateBy))
        try data.write(to: url)
        return url
    }

    /// A model directory with a minimal valid config.json (serve refuses a dir without one).
    private func modelDir(_ name: String = "model", withConfig: Bool = true) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if withConfig { try config(dir, "{}") }
        return dir
    }

    private func cleanup(_ dir: URL) { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

    private func config(_ dir: URL, _ json: String) throws {
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
    }

    /// An MLX quantized linear layer: in=4096, group_size=64 → 64 groups; packed width = 4096·bits/32.
    private func quantLayer(_ name: String, bits: Int) -> [T] {
        [T(name: "\(name).weight", dtype: "U32", shape: [8, 4096 * bits / 32]),
         T(name: "\(name).scales", dtype: "BF16", shape: [8, 64]),
         T(name: "\(name).biases", dtype: "BF16", shape: [8, 64])]
    }

    @Test("A complete shard is intact")
    func completeShard() async throws {
        let dir = try modelDir()
        defer { cleanup(dir) }
        try writeShard(dir.appendingPathComponent("model.safetensors"), [T(name: "w", dtype: "F16", shape: [2, 2])])
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(report.intact)
        #expect(report.errors.isEmpty)
    }

    @Test("Complete shards without a parseable config.json are not intact (serve could not load them)")
    func missingConfig() async throws {
        let dir = try modelDir(withConfig: false)
        defer { cleanup(dir) }
        try writeShard(dir.appendingPathComponent("model.safetensors"), [T(name: "w", dtype: "F16", shape: [2, 2])])
        #expect(await ModelInspector().inspect(modelIdOrPath: dir.path).intact == false)
        try Data("{\"model_type\":".utf8).write(to: dir.appendingPathComponent("config.json"))
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(!report.intact)
        #expect(report.errors.contains { $0.contains("config.json") })
    }

    @Test("A truncated download keeps a valid header but is reported broken")
    func truncatedShard() async throws {
        let dir = try modelDir()
        defer { cleanup(dir) }
        try writeShard(dir.appendingPathComponent("model.safetensors"),
                       [T(name: "a", dtype: "F16", shape: [2, 2]), T(name: "b", dtype: "F16", shape: [2, 2])],
                       truncateBy: 5)
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(!report.intact)
        #expect(report.shardFiles.first?.valid == false)
        #expect(report.errors.contains { $0.contains("Truncated") })
    }

    @Test("Bit width is measured from packed weights, for 4-bit and 8-bit alike")
    func measuredBits() async throws {
        for bits in [4, 8] {
            let dir = try modelDir("plain-name")
            defer { cleanup(dir) }
            try writeShard(dir.appendingPathComponent("model.safetensors"),
                           quantLayer("model.layers.0.mlp", bits: bits) + quantLayer("model.layers.1.mlp", bits: bits))
            let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
            #expect(report.detectedBitWidth == bits, "\(bits)-bit weights")
            #expect(report.intact)
        }
    }

    @Test("Mixed per-layer widths (DQ) are reported as mixed")
    func mixedBits() async throws {
        let dir = try modelDir()
        defer { cleanup(dir) }
        try writeShard(dir.appendingPathComponent("model.safetensors"),
                       quantLayer("l0", bits: 3) + quantLayer("l1", bits: 3) + quantLayer("l2", bits: 6))
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(report.detectedBitWidth == 3)
        #expect(report.bitWidthDescription.contains("mixed 3/6-bit"))
    }

    @Test("A config.json bit claim that the weights contradict is an error")
    func configContradictsWeights() async throws {
        let dir = try modelDir()
        defer { cleanup(dir) }
        try config(dir, #"{"quantization": {"bits": 4, "group_size": 64}}"#)
        try writeShard(dir.appendingPathComponent("model.safetensors"), quantLayer("l0", bits: 8))
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(report.detectedBitWidth == 8)
        #expect(!report.intact)
        #expect(report.errors.contains { $0.contains("claims 4-bit") })
    }

    @Test("Tensor dtypes beat a misleading repo name: F16 weights in a '4bit' dir are 16-bit")
    func dtypeBeatsName() async throws {
        let dir = try modelDir("Fake-Model-4bit")
        defer { cleanup(dir) }
        try writeShard(dir.appendingPathComponent("model.safetensors"), [T(name: "w", dtype: "F16", shape: [2, 2])])
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(report.detectedBitWidth == 16)
    }

    @Test("A header length over the 100 MB format limit is rejected even when the file is larger")
    func oversizedHeaderIsInvalid() async throws {
        let dir = try modelDir()
        defer { cleanup(dir) }
        let url = try writeShard(dir.appendingPathComponent("model.safetensors"),
                                 [T(name: "w", dtype: "F16", shape: [2])], headerLenOverride: 120_000_000)
        // Sparse-extend past the claimed header so only the 100 MB cap can reject it.
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 150_000_000)
        try handle.close()
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(!report.intact)
        #expect(report.shardFiles.first?.valid == false)
    }

    @Test("A shard named in the index but absent, and a dangling blob link, are both errors")
    func missingShards() async throws {
        let dir = try modelDir()
        defer { cleanup(dir) }
        try writeShard(dir.appendingPathComponent("model-00001-of-00002.safetensors"), [T(name: "a", dtype: "F16", shape: [2])])
        try Data(#"{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}"#.utf8)
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent("extra.safetensors"),
                                                   withDestinationURL: dir.appendingPathComponent("blobs/missing"))
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(!report.intact)
        #expect(report.errors.contains { $0.contains("Missing shard referenced in index: model-00002-of-00002.safetensors") })
        #expect(report.errors.contains { $0.contains("missing blob: extra.safetensors") })
    }

    @Test("training_args.bin is not counted as a weight shard")
    func trainingArgsIgnored() async throws {
        let dir = try modelDir()
        defer { cleanup(dir) }
        try writeShard(dir.appendingPathComponent("model.safetensors"), [T(name: "w", dtype: "F16", shape: [2])])
        try Data([1, 2, 3]).write(to: dir.appendingPathComponent("training_args.bin"))
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(report.totalShards == 1)
    }
}
