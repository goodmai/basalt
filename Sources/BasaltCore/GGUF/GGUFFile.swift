import Foundation

public enum GGMLType: UInt32, Sendable, Equatable, CaseIterable {
    case f32 = 0, f16 = 1, q8_0 = 8, q2K = 10, q4K = 12, q6K = 14
    case iq2XXS = 16, iq2XS = 17, iq3XXS = 18, iq4NL = 20, iq3S = 21, iq2S = 22, iq4XS = 23, iq1M = 29, bf16 = 30

    public var block: (elements: Int, bytes: Int) {
        switch self {
        case .f32: (1, 4)
        case .f16, .bf16: (1, 2)
        case .q8_0: (32, 34)
        case .iq4NL: (32, 18)
        case .q2K: (256, 84)
        case .q4K: (256, 144)
        case .q6K: (256, 210)
        case .iq2XXS: (256, 66)
        case .iq2XS: (256, 74)
        case .iq2S: (256, 82)
        case .iq3XXS: (256, 98)
        case .iq3S: (256, 110)
        case .iq4XS: (256, 136)
        case .iq1M: (256, 56)
        }
    }

    public var bitsPerWeight: Double { Double(block.bytes * 8) / Double(block.elements) }

    public var name: String { String(describing: self) }

    public init?(name: String) {
        guard let match = Self.allCases.first(where: { $0.name == name }) else { return nil }
        self = match
    }
}

public struct GGUFTensor: Sendable, Equatable {
    public let name: String
    public let type: GGMLType
    public let dims: [Int]
    public let offset: Int

    public var count: Int { dims.reduce(1, *) }
    public var rowLength: Int { dims.first ?? 0 }
    public var byteCount: Int { count / type.block.elements * type.block.bytes }
}

public enum GGUFValue: Sendable, Equatable {
    case int(Int64)
    case uint(UInt64)
    case float(Double)
    case bool(Bool)
    case string(String)
    case array([GGUFValue])
}

public struct GGUFFile: Sendable {
    public static let magic: UInt32 = 0x4655_4747

    public let url: URL
    public let version: UInt32
    public let metadata: [String: GGUFValue]
    public let tensors: [GGUFTensor]
    let data: Data

    public init(url: URL) throws(BasaltError) {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .alwaysMapped)
        } catch {
            throw .modelConfigurationInvalid(path: url.path, reason: "cannot map the GGUF file: \(error.localizedDescription)")
        }
        var reader = GGUFReader(data: data, path: url.path)
        guard try reader.u32() == Self.magic else { throw .modelConfigurationInvalid(path: url.path, reason: "not a GGUF file (no GGUF magic)") }
        let version = try reader.u32()
        guard version == 2 || version == 3 else { throw .modelConfigurationInvalid(path: url.path, reason: "GGUF version \(version) is not supported (2 or 3)") }
        let (tensorCount, keyCount) = (try reader.count(), try reader.count())
        var metadata: [String: GGUFValue] = [:]
        for _ in 0 ..< keyCount { metadata[try reader.string()] = try reader.value(type: try reader.u32()) }
        var infos: [(name: String, type: GGMLType, dims: [Int], offset: Int)] = []
        for _ in 0 ..< tensorCount { infos.append(try reader.tensorInfo()) }
        let alignment = metadata["general.alignment"].flatMap { if case .uint(let value) = $0 { Int(value) } else { nil } } ?? 32
        let start = (reader.position + alignment - 1) / alignment * alignment
        let tensors = infos.map { GGUFTensor(name: $0.name, type: $0.type, dims: $0.dims, offset: start + $0.offset) }
        for tensor in tensors where tensor.offset + tensor.byteCount > data.count || tensor.count % tensor.type.block.elements != 0 {
            throw .modelConfigurationInvalid(path: url.path, reason: "tensor \(tensor.name) does not fit the file or its blocks")
        }
        (self.url, self.version, self.metadata, self.tensors, self.data) = (url, version, metadata, tensors, data)
    }

    public func tensor(_ name: String) -> GGUFTensor? { tensors.first { $0.name == name } }

    public func payload(_ tensor: GGUFTensor) -> Data { data.subdata(in: tensor.offset ..< tensor.offset + tensor.byteCount) }

    public func floats(_ tensor: GGUFTensor) -> [Float] {
        data.withUnsafeBytes { raw in
            GGUFDequant.floats(UnsafeRawBufferPointer(rebasing: raw[tensor.offset ..< tensor.offset + tensor.byteCount]), type: tensor.type, count: tensor.count)
        }
    }
}

struct GGUFReader {
    let data: Data
    let path: String
    var position = 0

    mutating func bytes(_ count: Int) throws(BasaltError) -> Data {
        guard count >= 0, position + count <= data.count else { throw .modelConfigurationInvalid(path: path, reason: "the GGUF header is truncated") }
        defer { position += count }
        return data.subdata(in: position ..< position + count)
    }

    mutating func integer<T: FixedWidthInteger>(_: T.Type) throws(BasaltError) -> T {
        let raw = try bytes(MemoryLayout<T>.size)
        return raw.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(as: T.self)) }
    }

    mutating func u32() throws(BasaltError) -> UInt32 { try integer(UInt32.self) }

    mutating func count() throws(BasaltError) -> Int {
        let value = try integer(UInt64.self)
        guard value < 1 << 32 else { throw .modelConfigurationInvalid(path: path, reason: "a GGUF count of \(value) is not plausible") }
        return Int(value)
    }

    mutating func string() throws(BasaltError) -> String {
        String(decoding: try bytes(try count()), as: UTF8.self)
    }

    mutating func value(type: UInt32) throws(BasaltError) -> GGUFValue {
        switch type {
        case 0: .uint(UInt64(try integer(UInt8.self)))
        case 1: .int(Int64(try integer(Int8.self)))
        case 2: .uint(UInt64(try integer(UInt16.self)))
        case 3: .int(Int64(try integer(Int16.self)))
        case 4: .uint(UInt64(try integer(UInt32.self)))
        case 5: .int(Int64(try integer(Int32.self)))
        case 6: .float(Double(Float(bitPattern: try integer(UInt32.self))))
        case 7: .bool(try integer(UInt8.self) != 0)
        case 8: .string(try string())
        case 9: try array()
        case 10: .uint(try integer(UInt64.self))
        case 11: .int(try integer(Int64.self))
        case 12: .float(Double(bitPattern: try integer(UInt64.self)))
        default: throw .modelConfigurationInvalid(path: path, reason: "unknown GGUF value type \(type)")
        }
    }

    mutating func array() throws(BasaltError) -> GGUFValue {
        let (type, count) = (try u32(), try count())
        var values: [GGUFValue] = []
        for _ in 0 ..< count { values.append(try value(type: type)) }
        return .array(values)
    }

    mutating func tensorInfo() throws(BasaltError) -> (name: String, type: GGMLType, dims: [Int], offset: Int) {
        let name = try string()
        let rank = Int(try u32())
        guard (1 ... 4).contains(rank) else { throw .modelConfigurationInvalid(path: path, reason: "tensor \(name) has \(rank) dimensions") }
        var dims: [Int] = []
        for _ in 0 ..< rank { dims.append(try count()) }
        let raw = try u32()
        guard let type = GGMLType(rawValue: raw) else {
            throw .modelConfigurationInvalid(path: path, reason: "tensor \(name) has GGML type \(raw), which basalt does not dequantize")
        }
        return (name, type, dims, Int(try integer(UInt64.self)))
    }
}
