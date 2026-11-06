import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

struct ExpertTensor: Sendable, Equatable {
    let file: Int
    let offset: Int64
    let expertBytes: Int
    let shape: [Int]
    let dtype: DType
}

struct ExpertLayout: Sendable {
    let path: String
    let numExperts: Int
    let weight: ExpertTensor
    var scales: ExpertTensor?
    var biases: ExpertTensor?
    var bias: ExpertTensor?
    var quantization: (groupSize: Int, bits: Int, mode: QuantizationMode)?

    var expertBytes: ByteCount {
        ByteCount([weight, scales, biases, bias].compactMap { $0?.expertBytes }.reduce(0, +))
    }

    var totalBytes: ByteCount { expertBytes * Int64(numExperts) }

    var dimensions: (input: Int, output: Int) {
        let columns = weight.shape.last ?? 0
        return (quantization.map { columns * 32 / $0.bits } ?? columns, weight.shape.first ?? 0)
    }
}

struct ExpertArrays {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let bias: MLXArray?

    static func stacked(_ experts: [ExpertArrays]) -> ExpertArrays {
        func stack(_ part: (ExpertArrays) -> MLXArray?) -> MLXArray? {
            let parts = experts.compactMap(part)
            return parts.isEmpty ? nil : MLX.stacked(parts)
        }
        return ExpertArrays(weight: MLX.stacked(experts.map(\.weight)), scales: stack(\.scales), biases: stack(\.biases), bias: stack(\.bias))
    }
}

final class ExpertStore {
    let layouts: [ExpertLayout]
    private let files: [FileHandle]
    private var lru: ExpertLRU
    private var resident: [ExpertSlot: ExpertArrays] = [:]

    init(files urls: [URL], layouts: [ExpertLayout], capacity: ByteCount) throws {
        files = try urls.map { url in
            let handle = try FileHandle(forReadingFrom: url)
            _ = fcntl(handle.fileDescriptor, F_NOCACHE, 1)
            return handle
        }
        self.layouts = layouts
        lru = ExpertLRU(capacity: capacity)
    }

    var stats: ExpertCacheStats { lru.stats }
    var capacity: ByteCount { lru.capacity }
    var expertBytes: ByteCount { layouts.reduce(.zero) { $0 + $1.totalBytes } }
    var reserve: ByteCount { capacity + (layouts.map(\.totalBytes).max() ?? .zero) * 3 }
    var unclaimedReserve: ByteCount { max(.zero, reserve - lru.resident) }

    func experts(module: Int, _ ids: [Int]) -> [ExpertArrays] {
        let wanted = Set(ids.map { ExpertSlot(module: module, expert: $0) })
        let (missing, evicted) = lru.admit(wanted, bytes: layouts[module].expertBytes)
        for slot in evicted { resident[slot] = nil }
        for slot in missing { resident[slot] = read(slot) }
        return ids.map { resident[ExpertSlot(module: module, expert: $0)]! }
    }

    private func read(_ slot: ExpertSlot) -> ExpertArrays {
        let layout = layouts[slot.module]
        return ExpertArrays(
            weight: read(layout.weight, expert: slot.expert), scales: layout.scales.map { read($0, expert: slot.expert) },
            biases: layout.biases.map { read($0, expert: slot.expert) }, bias: layout.bias.map { read($0, expert: slot.expert) })
    }

    private func read(_ tensor: ExpertTensor, expert: Int) -> MLXArray {
        let offset = tensor.offset + Int64(expert * tensor.expertBytes)
        var data = Data(count: tensor.expertBytes)
        let read = data.withUnsafeMutableBytes { Self.pread(files[tensor.file].fileDescriptor, into: $0, at: offset) }
        precondition(read == tensor.expertBytes, "expert streaming: read \(read) of \(tensor.expertBytes) bytes at \(offset): \(String(cString: strerror(errno)))")
        return MLXArray(data, tensor.shape, dtype: tensor.dtype)
    }

    private static func pread(_ descriptor: Int32, into buffer: UnsafeMutableRawBufferPointer, at offset: Int64) -> Int {
        var done = 0
        while done < buffer.count {
            let count = Darwin.pread(descriptor, buffer.baseAddress! + done, buffer.count - done, off_t(offset) + off_t(done))
            if count < 0, errno == EINTR { continue }
            guard count > 0 else { return done }
            done += count
        }
        return done
    }
}

final class StreamedSwitchLinear: SwitchLinear {
    let store: ExpertStore
    let module: Int

    init(store: ExpertStore, module: Int) {
        self.store = store
        self.module = module
        let layout = store.layouts[module]
        let dimensions = layout.dimensions
        super.init(
            inputDims: dimensions.input, outputDims: dimensions.output, numExperts: layout.numExperts,
            weight: MLXArray.zeros([1], dtype: layout.weight.dtype))
        freeze()
    }

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        let routed = indices.asType(.int32).asArray(Int32.self).map(Int.init)
        let ids = Array(Set(routed)).sorted()
        var rank = [Int](repeating: 0, count: store.layouts[module].numExperts)
        for (position, id) in ids.enumerated() { rank[id] = position }
        let local = MLXArray(routed.map { UInt32(rank[$0]) }, indices.shape)
        return project(x, ExpertArrays.stacked(store.experts(module: module, ids)), local, sortedIndices: sortedIndices)
    }

    static func residentKernel(sorted: Bool, assignments: Int, experts: Int, quantized: Bool) -> Bool {
        quantized ? sorted && assignments / max(1, experts) >= 4 : sorted
    }

    private func project(_ x: MLXArray, _ experts: ExpertArrays, _ indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        let layout = store.layouts[module]
        let quantization = experts.scales == nil ? nil : layout.quantization
        let sorted = Self.residentKernel(sorted: sortedIndices, assignments: indices.size, experts: layout.numExperts, quantized: quantization != nil)
        var result =
            if let quantization, let scales = experts.scales {
                MLX.gatherQuantizedMM(
                    x, experts.weight, scales: scales, biases: experts.biases, rhsIndices: indices, transpose: true,
                    groupSize: quantization.groupSize, bits: quantization.bits, mode: quantization.mode, sortedIndices: sorted)
            } else {
                MLX.gatherMM(x, experts.weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
            }
        if let bias = experts.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        return result
    }
}
