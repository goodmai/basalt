import Testing
import Foundation
@testable import GemCore

@Suite("ModelInspector bit width and header validation")
struct ModelInspectorTests {

    /// A model dir named `name` with one safetensors shard whose header lists `dtypes`.
    private func makeModel(name: String, dtypes: [String], headerLenOverride: UInt64? = nil) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let tensors = dtypes.enumerated().map { i, dt in
            "\"t\(i)\": {\"dtype\": \"\(dt)\", \"shape\": [2], \"data_offsets\": [0, 8]}"
        }.joined(separator: ", ")
        let header = Data("{\(tensors)}".utf8)
        var len = (headerLenOverride ?? UInt64(header.count)).littleEndian
        var data = Data()
        withUnsafeBytes(of: &len) { data.append(contentsOf: $0) }
        data.append(header)
        data.append(Data(repeating: 0, count: 8))
        try data.write(to: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    @Test("U32 tensors (MLX packed weights) are detected as 4-bit without any config")
    func packedUInt32IsFourBit() async throws {
        let dir = try makeModel(name: "plain-model", dtypes: ["U32", "BF16"])
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(report.intact)
        #expect(report.detectedBitWidth == 4)
    }

    @Test("Tensor dtypes beat a misleading repo name: F16 weights in a '4bit' dir are 16-bit")
    func dtypeBeatsName() async throws {
        let dir = try makeModel(name: "Fake-Model-4bit", dtypes: ["F16"])
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(report.detectedBitWidth == 16)
    }

    @Test("A header length beyond the format limit is rejected, not read")
    func oversizedHeaderIsInvalid() async throws {
        let dir = try makeModel(name: "corrupt", dtypes: ["F16"], headerLenOverride: 200_000_000)
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }
        let report = await ModelInspector().inspect(modelIdOrPath: dir.path)
        #expect(!report.intact)
        #expect(report.shardFiles.first?.valid == false)
    }
}
