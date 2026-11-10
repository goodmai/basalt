import Testing
import Foundation
@testable import BasaltCore

@Suite("GGUF reader and dequantizers against gguf-py, for every type the GSQ-RCO checkpoints use (#408)")
struct GGUFDequantTests {
    static let types: [String: GGMLType] = [
        "f16": .f16, "bf16": .bf16, "q8_0": .q8_0, "iq4NL": .iq4NL, "q2K": .q2K, "q4K": .q4K, "q6K": .q6K,
        "iq4XS": .iq4XS, "iq3S": .iq3S, "iq3XXS": .iq3XXS, "iq2S": .iq2S, "iq2XS": .iq2XS, "iq2XXS": .iq2XXS, "iq1M": .iq1M,
    ]

    static func bytes(_ hex: String) -> [UInt8] {
        stride(from: 0, to: hex.count, by: 2).map { offset in
            let start = hex.index(hex.startIndex, offsetBy: offset)
            return UInt8(hex[start ..< hex.index(start, offsetBy: 2)], radix: 16) ?? 0
        }
    }

    @Test("Two random blocks of each type decode to gguf-py's values", arguments: Array(types.keys).sorted())
    func golden(_ name: String) throws {
        let (hex, expected) = try #require(ggufDequantGolden[name])
        let type = try #require(Self.types[name])
        let raw = Self.bytes(hex)
        #expect(raw.count == 2 * type.block.bytes)
        let values = raw.withUnsafeBytes { GGUFDequant.floats($0, type: type, count: 2 * type.block.elements) }
        #expect(values.count == expected.count)
        let worst = zip(values, expected).map { abs($0 - $1) / max(1, abs($1)) }.max() ?? 0
        #expect(worst <= 1e-6, "\(name): worst relative gap \(worst)")
    }

    @Test("A truncated or foreign file is refused before any tensor is read")
    func refusals() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "gguf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for (name, data) in [("foreign", Data("NOTGGUF!".utf8)), ("short", Data([0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0, 1]))] {
            let url = directory.appending(path: name)
            try data.write(to: url)
            #expect(throws: BasaltError.self) { _ = try GGUFFile(url: url) }
        }
    }

    static let env = ProcessInfo.processInfo.environment
    static let real = env["BASALT_E2E"] == "1" ? env["BASALT_E2E_GGUF_PATH"].flatMap { path in env["BASALT_E2E_GGUF_REFERENCE"].map { (path, $0) } } : nil

    @Test("The Qwen3.8-27B GSQ-RCO file: one tensor of each type matches gguf-py on its first 4096 values and on the sums over the whole tensor",
          .enabled(if: real != nil, "set BASALT_E2E=1, BASALT_E2E_GGUF_PATH and BASALT_E2E_GGUF_REFERENCE"))
    func realFile() throws {
        let (path, reference) = try #require(Self.real)
        let file = try GGUFFile(url: URL(fileURLWithPath: path))
        let expected = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: reference))) as? [String: [String: Any]])
        #expect(file.tensors.count == 851 && file.version == 3)
        for (name, entry) in expected.sorted(by: { $0.key < $1.key }) {
            let tensor = try #require(file.tensor(name), "\(name) is missing")
            let values = file.floats(tensor)
            let first = try #require(entry["first"] as? [Double])
            let gap = zip(values, first).map { abs(Double($0) - $1) }.max() ?? 0
            let (sum, absolute) = (values.reduce(0.0) { $0 + Double($1) }, values.reduce(0.0) { $0 + abs(Double($1)) })
            let expectedAbs = try #require(entry["abs"] as? Double)
            print("GGUF \(name) \(tensor.type) count \(values.count) first-gap \(gap) sum \(sum) abs \(absolute) vs \(expectedAbs)")
            #expect(values.count == entry["count"] as? Int && gap <= 1e-6)
            #expect(abs(absolute - expectedAbs) <= 1e-4 * expectedAbs && abs(sum - (entry["sum"] as? Double ?? 0)) <= 1e-4 * expectedAbs)
        }
    }
}
